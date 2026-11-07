"""Piper training — the dashboard's last mile from a ported dataset to a voice.

The trainer is a long child process, so the tests pin the parts that fail
silently in production:

* the **command line** for both trainers (piper1-gpl and legacy rhasspy), because
  one wrong flag wastes an hour and the error arrives from deep inside
  Lightning;
* the **dataset shape** — piper1-gpl joins ``--data.audio_dir`` onto the CSV's
  first column, so ``wav/0001.wav`` must become ``0001.wav`` for the run;
* **progress/cancel** — the panel must never sit at "running" after a crash or
  a stop, and a cancel must actually terminate the child;
* **run.json** — a finished run stays attachable after a dashboard restart.

No real training happens here: the process spawner is faked and every path is a
tmp directory.
"""
from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets
from tts import kokoro_setup, piper_setup, voice_lab
from tts import piper_train as pt
from tts import voice_port as vp

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "profiles").mkdir()
    monkeypatch.setattr(piper_setup, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(kokoro_setup, "DEFAULT_VOICES_DIR", tmp_path / "voices" / "kokoro")
    # Most tests are about argv/state, not audio length; they use tiny fake
    # WAVs. The duration gate gets its own tests below, which raise this back.
    monkeypatch.setattr(vp, "MIN_TRAIN_SECONDS", 0.0)
    pt.clear_probe_cache()
    return tmp_path


def make_package(name: str = "Luna BR", clips: int = 4, lang: str = "pt",
                 folder: str | None = None, wav_seconds: float | None = None) -> Path:
    """A Piper port exactly as ``voice_port.build_package`` leaves it.

    ``wav_seconds`` writes real WAVs of that length (the duration gate reads
    them); the default keeps the tiny placeholder bytes the argv tests need.
    """
    pkg = vp.ported_dir() / (folder or f"{vp.slug(name)}-piper")
    wav_dir = pkg / "dataset" / "wav"
    wav_dir.mkdir(parents=True)
    rows = []
    for i in range(1, clips + 1):
        path = wav_dir / f"{i:04d}.wav"
        if wav_seconds is None:
            path.write_bytes(b"RIFF")
        else:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes(b"\x00\x01" * int(22050 * wav_seconds))
        rows.append(f"wav/{i:04d}.wav|line number {i}")
    (pkg / "dataset" / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {
        "format": "wallie.port",
        "version": 1,
        "created_at": "2026-10-19T00:00:00+00:00",
        "target": {"id": "piper", "sample_rate": 22050},
        "source": {"name": "luna", "provider": "fish", "voice_id": "v1"},
        "name": name,
        "lang": {"id": lang, "label": "pt-BR" if lang == "pt" else "en-US",
                 "espeak": "pt-br" if lang == "pt" else "en-us"},
        "clips": clips,
        "seconds": float(clips),
        "capture_sample_rate": 24000,
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pkg


def fake_env(flavor: str = "piper1-gpl", python: str = "") -> dict:
    return {
        "python": python or "python",
        "version": "3.12.10",
        "flavor": flavor,
        "flavor_label": flavor,
        "torch": True,
        "lightning": True,
        "monotonic": True,
        "piper_root": "",
        "warmstart": True,
        "export_missing": False,
        "missing": [],
        "can_train": True,
        "error": "",
        "hint": "ready",
        "setup_command": "python -m pip install piper-tts[train]",
    }


class _Reader:
    def __init__(self, chunks: list[bytes]):
        self.chunks = list(chunks)

    async def read(self, n: int) -> bytes:
        await asyncio.sleep(0)
        return self.chunks.pop(0) if self.chunks else b""


class _Proc:
    """Stand-in for asyncio's Process — enough of it for ``_stream``."""

    def __init__(self, chunks: list[bytes], rc: int = 0):
        self.stdout = _Reader(chunks)
        self.returncode = rc
        self.pid = 999_999          # never a live process: terminate must be harmless
        self.terminated = False

    async def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9


class _SilentReader:
    async def read(self, n: int) -> bytes:
        await asyncio.sleep(10)     # only cancellation ends this read
        return b""


class _SilentProc(_Proc):
    """A trainer that never prints — instead of EOF on the first read."""

    def __init__(self) -> None:
        super().__init__([])
        self.stdout = _SilentReader()
        self.terminated = False


def _checkpoint_and_config(argv: list[str]) -> None:
    """Write what a real training step would leave behind."""
    if "--trainer.default_root_dir" in argv:
        run_dir = Path(argv[argv.index("--trainer.default_root_dir") + 1])
    else:
        run_dir = Path(argv[argv.index("--default_root_dir") + 1])
    ckpt = run_dir / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"CKPT")
    if "--data.config_path" in argv:
        Path(argv[argv.index("--data.config_path") + 1]).write_text('{"audio": {}}', encoding="utf-8")
    else:
        dataset = run_dir / "dataset"
        dataset.mkdir(exist_ok=True)
        (dataset / "config.json").write_text('{"audio": {}}', encoding="utf-8")


def _fake_spawner(monkeypatch, seen: dict | None = None, train_rc: int = 0,
                  silent: bool = False):
    async def spawn(argv, *, cwd, env):
        argv = list(argv)
        if seen is not None:
            seen.setdefault("argv", []).append(argv)
        if "piper_reference.py" in " ".join(argv):
            # The converter runs with the training interpreter; stand in for it.
            Path(argv[argv.index("--out") + 1]).write_bytes(b"REF-CKPT")
            return _Proc([b"[reference] recovered 12 weight tensor(s)\n"])
        if "export_onnx" in " ".join(argv):
            out = Path(argv[argv.index("--output-file") + 1]) if "--output-file" in argv \
                else Path(argv[-1])
            out.write_bytes(b"ONNX")
            return _Proc([b"exporting weights...\n", b"saved\n"])
        if "piper_train.preprocess" in " ".join(argv):
            run_dir = Path(argv[argv.index("--output-dir") + 1])
            (run_dir / "dataset").mkdir(parents=True, exist_ok=True)
            (run_dir / "dataset" / "config.json").write_text('{"audio": {}}', encoding="utf-8")
            return _Proc([b"caching dataset...\n", b"done\n"])
        _checkpoint_and_config(argv)
        if silent:
            return _SilentProc()
        if train_rc:
            return _Proc([b"Traceback (most recent call last):\n",
                          b"RuntimeError: CUDA out of memory\n"], rc=train_rc)
        return _Proc([
            b"Epoch 0: 100%|\xe2\x96\x88| 2/2 [00:01<00:00, 2.0it/s]\n",
            b"Epoch 1:  50%|\xe2\x96\x88 | 1/2 [00:00<00:00, 2.0it/s]\n",
            b"Epoch 1: 100%|\xe2\x96\x88| 2/2 [00:00<00:00, 2.0it/s]\n",
        ])

    monkeypatch.setattr(pt, "_spawn", spawn)


@pytest.fixture()
def fast_probe(monkeypatch):
    async def _probe(python="", *, force=False):
        return fake_env(python=pt.interpreter(python))

    monkeypatch.setattr(pt, "probe", _probe)


@pytest.fixture()
def in_process_launch(monkeypatch):
    """Route tests run the trainer in the test's loop instead of forking.

    The detached launch is covered by its own test; keeping this one in-process
    makes the route assertions (409/400/wait-for-done) deterministic.
    """

    def start(job, options):
        pt.begin(job, kind="train", package=options["package"],
                 voice_name=options["voice_name"], epochs=options["epochs"],
                 python=options["python"], resumed_from=options.get("resume") or {})
        asyncio.create_task(pt.run_train(
            job,
            package=options["package"],
            voice_name=options["voice_name"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            accelerator=options["accelerator"],
            python=options["python"],
            checkpoint=options.get("checkpoint", ""),
            resume_from=options.get("resume_from", ""),
        ))
        return pt.snapshot(job)

    monkeypatch.setattr(pt, "start_run", start)


@pytest.fixture()
def profile(dirs):
    import config
    from config import activate_profile, save_profile
    save_profile(AppConfig(profile_name="luna"), "luna")
    activate_profile("luna")
    return "luna"


# ---------------------------------------------------------------------------
# Environment + defaults
# ---------------------------------------------------------------------------

def test_default_voice_name_satisfies_pipers_loader():
    manifest = {"name": "Luna BR", "lang": {"id": "pt"}, "target": {"sample_rate": 22050}}
    name = pt.default_voice_name(manifest)
    assert name == "pt_BR-luna_br-medium"
    assert piper_setup.is_valid_voice_name(name)
    assert pt.default_voice_name({"name": "Amy", "lang": {"id": "en"}}) == "en_US-amy-medium"
    # A name that cannot be slugged still produces a loader-valid fallback.
    weird = pt.default_voice_name({"name": "///", "lang": {"id": "pt"}})
    assert piper_setup.is_valid_voice_name(weird)


def test_env_payload_reports_what_is_missing():
    ready = pt._env_payload("py", {"version": "3.12", "gpl": True, "torch": True,
                                   "lightning": True, "onnxscript": True})
    assert ready["can_train"] and ready["flavor"] == "piper1-gpl"
    assert ready["missing"] == []

    no_lightning = pt._env_payload("py", {"version": "3.12", "gpl": True, "torch": True})
    assert not no_lightning["can_train"]
    assert "lightning" in no_lightning["missing"]
    assert "piper-tts[train]" in no_lightning["hint"]

    none = pt._env_payload("py", {})
    assert none["flavor"] == ""
    assert "piper-tts[train]" in none["missing"]
    assert "no Piper trainer" in none["hint"]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

def test_piper1_gpl_recipe_normalises_the_csv(dirs):
    pkg = make_package()
    run = pt.new_run_dir(pkg)
    prepared = pt.prepare_run(pkg, run, "piper1-gpl", "pt_BR-luna_br-medium")

    assert prepared["csv_path"] == str(run / "train.csv")
    rows = (run / "train.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "0001.wav|line number 1"
    # The package's own CSV is untouched (the legacy preprocessor reads it).
    original = (pkg / "dataset" / "metadata.csv").read_text(encoding="utf-8")
    assert original.startswith("wav/0001.wav|")

    env = fake_env("piper1-gpl", "py")
    argv = pt.train_argv(
        env, prepared, run, voice_name="pt_BR-luna_br-medium", epochs=50,
        batch_size=4, accelerator="cpu", checkpoint="/tmp/base.ckpt",
    )
    joined = " ".join(argv)
    assert argv[:4] == ["py", "-m", "piper.train", "fit"]
    assert "--data.csv_path" in argv and str(run / "train.csv") in argv
    assert "--data.audio_dir" in argv and prepared["audio_dir"] in argv
    assert "--model.sample_rate" in argv and "22050" in argv
    assert "--data.espeak_voice" in argv and "pt-br" in argv
    assert "--data.config_path" in argv
    assert argv[argv.index("--data.batch_size") + 1] == "4"
    # 4 clips: 1/n would floor to an empty val split, so it asks for 1.5 clips.
    assert float(argv[argv.index("--data.validation_split") + 1]) == pytest.approx(0.375)
    assert argv[argv.index("--data.num_test_examples") + 1] == "0"
    assert argv[argv.index("--data.num_workers") + 1] == "0"
    assert argv[argv.index("--trainer.max_epochs") + 1] == "50"
    assert argv[argv.index("--trainer.accelerator") + 1] == "cpu"
    assert argv[argv.index("--trainer.default_root_dir") + 1] == str(run)
    assert argv[argv.index("--ckpt_path") + 1] == "/tmp/base.ckpt"
    assert "piper_train" not in joined


def test_legacy_recipe_preprocesses_then_exports_positionally(dirs):
    pkg = make_package(clips=3)
    run = pt.new_run_dir(pkg)
    prepared = pt.prepare_run(pkg, run, "legacy", "pt_BR-luna-medium")
    assert prepared["csv_path"] == ""       # legacy reads the package CSV itself
    assert prepared["clips"] == 3

    env = fake_env("legacy", "py")
    pre = pt.preprocess_argv(env, prepared, run)
    assert pre[:3] == ["py", "-m", "piper_train.preprocess"]
    assert pre[pre.index("--input-dir") + 1] == prepared["dataset_dir"]
    assert pre[pre.index("--output-dir") + 1] == str(run / "dataset")
    assert "--single-speaker" in pre
    assert pre[pre.index("--sample-rate") + 1] == "22050"

    train = pt.train_argv(env, prepared, run, voice_name="pt_BR-luna-medium",
                          epochs=10, batch_size=2, accelerator="auto",
                          checkpoint="/tmp/x.ckpt")
    assert train[:3] == ["py", "-m", "piper_train"]
    assert train[train.index("--dataset-dir") + 1] == str(run / "dataset")
    assert train[train.index("--resume_from_checkpoint") + 1] == "/tmp/x.ckpt"
    assert train[train.index("--validation-split") + 1] == "0.0"

    ckpt = Path("/tmp/last.ckpt")
    export = pt.export_argv(env, ckpt, run / "v.onnx")
    assert export[:3] == ["py", "-m", "piper_train.export_onnx"]
    assert export[3:] == [str(ckpt), str(run / "v.onnx")]

    gpl_export = pt.export_argv(fake_env("piper1-gpl", "py"),
                                Path("/tmp/last.ckpt"), run / "v.onnx")
    assert gpl_export[:3] == ["py", "-m", "piper.train.export_onnx"]
    assert gpl_export[gpl_export.index("--output-file") + 1] == str(run / "v.onnx")


def test_normalize_csv_skips_rows_without_text(tmp_path):
    src = tmp_path / "metadata.csv"
    src.write_text(
        "wav/0001.wav|hello\n"
        "not-a-row\n"
        "wav/0002.wav|\n"
        "wav/0003.wav|third line\n",
        encoding="utf-8",
    )
    dest = tmp_path / "train.csv"
    assert pt.normalize_csv(src, dest) == 2
    assert dest.read_text(encoding="utf-8").splitlines() == [
        "0001.wav|hello", "0003.wav|third line",
    ]

    empty = tmp_path / "empty.csv"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(pt.TrainError):
        pt.normalize_csv(empty, dest)


# ---------------------------------------------------------------------------
# Progress + checkpoints
# ---------------------------------------------------------------------------

def test_validation_split_always_holds_out_a_clip():
    assert pt._validation_split({"clips": 8}) == pytest.approx(0.1875)
    assert pt._validation_split({"clips": 30}) == pytest.approx(0.1)
    assert pt._validation_split({"clips": 100}) == pytest.approx(0.1)
    assert pt._validation_split({"clips": 3}) == pytest.approx(0.5)


def test_validation_split_never_floors_to_an_empty_loader():
    """piper does int(n * split): asking for 1/n is 0 for 3 clips."""
    for clips in (3, 4, 5, 7, 8, 9, 10, 30):
        split = pt._validation_split({"clips": clips})
        assert int(clips * split) >= 1, clips


def test_parse_epoch_reads_the_lightning_bar():
    line = "Epoch 3: 100%|\xe2\x96\x88\xe2\x96\x88| 8/8 [00:02<00:00, 3.2it/s]"
    parsed = pt.parse_epoch(line, 200)
    assert parsed["epoch"] == 3
    assert parsed["percent"] == pytest.approx(2.0)

    half = pt.parse_epoch("Epoch 1:  50%|\xe2\x96\x88 | 4/8 [00:00<00:00]", 4)
    assert half["epoch"] == 1
    assert half["percent"] == pytest.approx(37.5)

    assert pt.parse_epoch("Downloading checkpoint...", 10) is None
    assert pt.parse_epoch("", 10) is None


def test_find_checkpoint_prefers_last(tmp_path):
    older = tmp_path / "a" / "epoch=1.ckpt"
    newer = tmp_path / "a" / "epoch=2.ckpt"
    last = tmp_path / "a" / "last.ckpt"
    for p in (older, newer, last):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    os.utime(last, (150, 150))
    # last.ckpt is the resumable one even though it is not the newest file.
    assert pt.find_checkpoint(tmp_path) == last
    (last).unlink()
    assert pt.find_checkpoint(tmp_path) == newer
    # A warm-start reference is not a training result — never export/resume it.
    (tmp_path / pt.REFERENCE_FILE).write_bytes(b"ref")
    (newer).unlink()
    assert pt.find_checkpoint(tmp_path) == older
    (older).unlink()
    assert pt.find_checkpoint(tmp_path) is None
    assert pt.find_checkpoint(tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# Package selection + validation
# ---------------------------------------------------------------------------

def make_stopped_run(pkg: Path, *, epoch: int = 3, epochs: int = 10,
                     status: str = "cancelled", ckpt: bool = True) -> Path:
    """A run that stopped mid-training, as cancel/error leaves it on disk."""
    run = pkg / "train" / "run-20260101-010101"
    run.mkdir(parents=True)
    record = {
        "format": "wallie.train", "version": 1, "status": status,
        "package": str(pkg), "voice_name": "pt_BR-luna_br-medium",
        "epochs": epochs, "epoch": epoch, "exported": {}, "steps": [],
    }
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    if ckpt:
        path = run / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"CKPT")
    return run


def test_latest_resume_finds_a_stopped_run(dirs):
    pkg = make_package()
    run = make_stopped_run(pkg, epoch=3, epochs=10)
    res = pt.latest_resume(pkg)
    assert res["run_dir"] == str(run)
    assert res["checkpoint"].endswith("last.ckpt")
    assert res["epoch"] == 3 and res["epochs"] == 10
    assert res["voice_name"] == "pt_BR-luna_br-medium"

    # A finished run is not something to resume.
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    record["status"] = "done"
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    assert pt.latest_resume(pkg) is None


def test_latest_resume_skips_a_run_without_checkpoint(dirs):
    pkg = make_package()
    make_stopped_run(pkg, ckpt=False)
    assert pt.latest_resume(pkg) is None


def test_a_newer_finished_run_supersedes_an_older_checkpoint(dirs):
    """After a successful resume, the old cancelled branch is no longer offered."""
    pkg = make_package()
    old = make_stopped_run(pkg)
    done = pkg / "train" / "run-20260202-000000"
    done.mkdir()
    onnx = done / "v.onnx"
    onnx.write_bytes(b"ONNX")
    cfg = done / "v.onnx.json"
    cfg.write_text("{}", encoding="utf-8")
    (done / "run.json").write_text(json.dumps({
        "status": "done", "exported": {"onnx": str(onnx), "config": str(cfg)},
    }), encoding="utf-8")
    os.utime(old, (100, 100))
    os.utime(done, (200, 200))

    assert pt.latest_resume(pkg) is None
    assert pt.read_run(pkg)["status"] == "done"


def test_resolve_resume_guards_and_prefers_the_checkpoint(dirs):
    pkg = make_package()
    run = make_stopped_run(pkg)

    res = pt.resolve_resume(pkg, str(run))
    assert res["checkpoint"].endswith("last.ckpt")
    assert res["epoch"] == 3

    with pytest.raises(pt.TrainError) as e:
        pt.resolve_resume(pkg, str(REPO))
    assert "not one of this package" in str(e.value)

    with pytest.raises(pt.TrainError) as e:
        pt.resolve_resume(pkg, str(run.parent / "run-20260101-020202"))
    assert "no .ckpt" in str(e.value)

    opts = pt.validate_start(
        package=str(pkg), voice_name="", epochs=10, batch_size=2,
        accelerator="cpu", python="", checkpoint="/ignored.ckpt",
        resume_from=str(run),
    )
    assert opts["checkpoint"] == res["checkpoint"]
    assert opts["voice_name"] == "pt_BR-luna_br-medium"
    assert opts["resume"]["epoch"] == 3


def test_a_finished_run_cannot_be_resumed(dirs):
    pkg = make_package()
    run = make_stopped_run(pkg, status="done")
    with pytest.raises(pt.TrainError) as e:
        pt.resolve_resume(pkg, str(run))
    assert "already finished" in str(e.value)


def test_read_run_prefers_the_newest_finished_run(dirs):
    """A failed newest attempt must not hide an older model that did finish."""
    pkg = make_package()
    train = pkg / "train"
    good = train / "run-20260101-000000"
    good.mkdir(parents=True)
    onnx = good / "v.onnx"
    onnx.write_bytes(b"ONNX")
    cfg = good / "v.onnx.json"
    cfg.write_text("{}", encoding="utf-8")
    (good / "run.json").write_text(json.dumps({
        "status": "done", "voice_name": "v",
        "exported": {"onnx": str(onnx), "config": str(cfg)},
    }), encoding="utf-8")
    bad = train / "run-20260102-000000"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({"status": "error", "error": "boom"}),
                                  encoding="utf-8")
    os.utime(good, (100, 100))
    os.utime(bad, (200, 200))

    run = pt.read_run(pkg)
    assert run["status"] == "done" and run["attachable"] is True

    (good / "v.onnx").unlink()
    os.utime(good, (100, 100))     # unlinking bumped the folder's mtime
    run = pt.read_run(pkg)
    assert run["status"] == "error"
    assert run["attachable"] is False


def test_packages_list_only_piper_ports(dirs):
    piper = make_package("Luna BR")
    kokoro = vp.ported_dir() / "Luna-kokoro"
    kokoro.mkdir(parents=True)
    (kokoro / "manifest.json").write_text(json.dumps({
        "format": "wallie.port", "name": "Luna", "target": {"id": "kokoro"},
    }), encoding="utf-8")

    packages = pt.list_packages()
    assert [Path(p["package"]) for p in packages] == [piper]
    assert packages[0]["voice_name"] == "pt_BR-luna_br-medium"
    assert packages[0]["clips"] == 4

    with pytest.raises(pt.TrainError) as e:
        pt.resolve_package(str(kokoro))
    assert "only a Piper package" in str(e.value)


def test_validate_start_clamps_and_validates(dirs, tmp_path):
    pkg = make_package()
    opts = pt.validate_start(
        package=str(pkg), voice_name="", epochs=99999, batch_size=0,
        accelerator="auto", python="", checkpoint="",
    )
    assert opts["voice_name"] == "pt_BR-luna_br-medium"
    assert opts["epochs"] == pt.MAX_EPOCHS
    assert opts["batch_size"] == pt.MIN_BATCH

    with pytest.raises(pt.TrainError):
        pt.validate_start(package=str(pkg), voice_name="luna", epochs=10,
                          batch_size=2, accelerator="auto", python="", checkpoint="")
    with pytest.raises(pt.TrainError):
        pt.validate_start(package=str(pkg), voice_name="", epochs=10, batch_size=2,
                          accelerator="tpu", python="", checkpoint="")
    with pytest.raises(pt.TrainError):
        pt.validate_start(package=str(pkg), voice_name="", epochs=10, batch_size=2,
                          accelerator="auto", python="", checkpoint="/nope.ckpt")
    with pytest.raises(pt.TrainError):
        pt.validate_start(package="", voice_name="", epochs=10, batch_size=2,
                          accelerator="auto", python="", checkpoint="")


def test_onnx_reference_is_a_warm_start_not_a_resume(dirs, tmp_path):
    """A .onnx can seed the weights, but it is not a checkpoint: no optimizer,
    no epoch — so it must reach piper as a warm start, never as --ckpt_path."""
    pkg = make_package()
    ref = tmp_path / "pt_BR-faber-medium.onnx"
    ref.write_bytes(b"ONNX")

    opts = pt.validate_start(package=str(pkg), voice_name="", epochs=10, batch_size=2,
                             accelerator="cpu", python="", checkpoint=str(ref))
    assert opts["warmstart"] is True
    assert opts["checkpoint"] == str(ref.resolve())

    run = pt.new_run_dir(pkg)
    prepared = pt.prepare_run(pkg, run, "piper1-gpl", opts["voice_name"])
    argv = pt.train_argv(fake_env("piper1-gpl", "py"), prepared, run,
                         voice_name=opts["voice_name"], epochs=10, batch_size=2,
                         accelerator="cpu", checkpoint=opts["checkpoint"], warmstart=True)
    assert argv[argv.index("--model.warmstart_ckpt") + 1] == opts["checkpoint"]
    assert "--ckpt_path" not in argv


def test_a_ckpt_still_resumes_strictly(dirs, tmp_path):
    pkg = make_package()
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"CKPT")
    opts = pt.validate_start(package=str(pkg), voice_name="", epochs=10, batch_size=2,
                             accelerator="cpu", python="", checkpoint=str(ckpt))
    assert opts["warmstart"] is False


def test_other_reference_types_are_refused(dirs, tmp_path):
    pkg = make_package()
    bogus = tmp_path / "voice.pt"
    bogus.write_bytes(b"PT")
    with pytest.raises(pt.TrainError) as e:
        pt.validate_start(package=str(pkg), voice_name="", epochs=10, batch_size=2,
                          accelerator="cpu", python="", checkpoint=str(bogus))
    assert ".ckpt" in str(e.value) and ".onnx" in str(e.value)


def test_the_legacy_trainer_has_no_warm_start(dirs):
    pkg = make_package()
    run = pt.new_run_dir(pkg)
    prepared = pt.prepare_run(pkg, run, "legacy", "pt_BR-luna-medium")
    with pytest.raises(pt.TrainError) as e:
        pt.train_argv(fake_env("legacy", "py"), prepared, run,
                      voice_name="pt_BR-luna-medium", epochs=5, batch_size=2,
                      accelerator="cpu", checkpoint="/tmp/ref.ckpt", warmstart=True)
    assert "legacy" in str(e.value)


# ---------------------------------------------------------------------------
# The run itself (fake process)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_trains_exports_and_persists(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    seen: dict = {}
    _fake_spawner(monkeypatch, seen)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), voice_name="pt_BR-luna_br-medium",
             epochs=2, python="py")
    await pt.run_train(
        job, package=str(pkg), voice_name="pt_BR-luna_br-medium", epochs=2,
        batch_size=2, accelerator="cpu", python="",
    )

    assert job["status"] == "done", job["error"]
    assert job["progress"] == 100.0
    assert job["running"] is False
    assert "Epoch 1" in job["log"]
    assert job["exported"]["onnx"].endswith("pt_BR-luna_br-medium.onnx")
    onnx = Path(job["exported"]["onnx"])
    assert onnx.read_bytes() == b"ONNX"
    assert Path(job["exported"]["config"]).read_text(encoding="utf-8") == '{"audio": {}}'

    run = Path(job["run_dir"])
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "done"
    assert record["flavor"] == "piper1-gpl"
    assert record["exported"]["onnx"] == str(onnx)
    assert (run / "train.log").read_text(encoding="utf-8").count("Epoch") >= 2
    # The exported pair shares the stem Piper's loader expects.
    assert Path(record["exported"]["config"]).name == onnx.name + ".json"

    exe = pt.interpreter("")
    args = seen["argv"]
    assert args[0][:4] == [exe, "-m", "piper.train", "fit"]
    assert args[-1][:3] == [exe, "-m", "piper.train.export_onnx"]


@pytest.mark.asyncio
async def test_run_converts_the_onnx_reference_and_warms_up(dirs, fast_probe, monkeypatch):
    """The converted checkpoint travels with the run, and piper gets it as the
    warm start — the raw .onnx is never handed to the trainer."""
    pkg = make_package()
    ref = dirs / "pt_BR-faber-medium.onnx"
    ref.write_bytes(b"ONNX")
    seen: dict = {}
    _fake_spawner(monkeypatch, seen)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), voice_name="pt_BR-luna_br-medium",
             epochs=2, python="py")
    await pt.run_train(
        job, package=str(pkg), voice_name="pt_BR-luna_br-medium", epochs=2,
        batch_size=2, accelerator="cpu", python="", checkpoint=str(ref),
    )

    assert job["status"] == "done", job["error"]
    converted = Path(job["run_dir"]) / pt.REFERENCE_FILE
    assert converted.read_bytes() == b"REF-CKPT"
    assert job["reference"] == {"onnx": str(ref.resolve()), "checkpoint": str(converted)}
    assert "[reference] recovered 12" in job["log"]

    args = seen["argv"]
    converter = [a for a in args if "piper_reference.py" in " ".join(a)][0]
    assert converter[0] == pt.interpreter("")
    assert converter[converter.index("--onnx") + 1] == str(ref.resolve())
    train = [a for a in args if a[:3] == [pt.interpreter(""), "-m", "piper.train"]][0]
    assert train[train.index("--model.warmstart_ckpt") + 1] == str(converted)
    assert "--ckpt_path" not in train

    record = json.loads((converted.parent / "run.json").read_text(encoding="utf-8"))
    assert record["warmstart"] is True
    assert record["reference"]["onnx"] == str(ref.resolve())


@pytest.mark.asyncio
async def test_run_refuses_an_onnx_on_the_legacy_trainer(dirs, monkeypatch):
    pkg = make_package()
    ref = dirs / "voice.onnx"
    ref.write_bytes(b"ONNX")

    async def legacy_probe(python="", *, force=False):
        env = fake_env("legacy", python=pt.interpreter(python))
        return env

    monkeypatch.setattr(pt, "probe", legacy_probe)
    seen: dict = {}
    _fake_spawner(monkeypatch, seen)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), voice_name="pt_BR-luna_br-medium",
             epochs=2, python="py")
    await pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                       epochs=2, batch_size=2, accelerator="cpu", checkpoint=str(ref))

    assert job["status"] == "error"
    assert "piper1-gpl" in job["error"]
    assert not seen.get("argv"), "nothing should be spawned for a reference it cannot use"


@pytest.mark.asyncio
async def test_run_resumes_from_the_stopped_checkpoint(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    previous = make_stopped_run(pkg, epoch=3, epochs=10)
    old_ckpt = previous / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt"
    seen: dict = {}
    _fake_spawner(monkeypatch, seen)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), voice_name="pt_BR-luna_br-medium",
             epochs=10, python="py", resumed_from=pt.resolve_resume(pkg, str(previous)))
    await pt.run_train(
        job, package=str(pkg), epochs=10, batch_size=2, accelerator="cpu",
        resume_from=str(previous),
    )

    assert job["status"] == "done", job["error"]
    assert job["resumed_from"]["run_dir"] == str(previous)
    assert "resuming from epoch 3" in job["log"]
    train_argv = seen["argv"][0]
    assert train_argv[train_argv.index("--ckpt_path") + 1] == str(old_ckpt)
    # A resumed run gets its own folder; the checkpoints it was based on stay put.
    assert Path(job["run_dir"]) != previous
    assert old_ckpt.is_file()
    record = json.loads((Path(job["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert record["resumed_from"]["epoch"] == 3
    assert record["status"] == "done"


@pytest.mark.asyncio
async def test_run_can_be_cancelled(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    _fake_spawner(monkeypatch, silent=True)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=100, python="py")
    task = asyncio.create_task(pt.run_train(
        job, package=str(pkg), voice_name="pt_BR-luna_br-medium", epochs=100,
        batch_size=2, accelerator="cpu", python="",
    ))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if job.get("_proc") is not None:
            break
    assert await pt.cancel(job) is True
    await asyncio.wait_for(task, timeout=10)

    assert job["status"] == "cancelled"
    assert job["running"] is False
    assert "cancelled" in job["log"]
    run = Path(job["run_dir"])
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["status"] == "cancelled"
    # Nothing to attach: no exported model was written.
    assert job["exported"] == {}
    assert await pt.cancel(job) is False


@pytest.mark.asyncio
async def test_a_killed_child_with_the_cancel_marker_is_a_stop_not_an_error(
    dirs, fast_probe, monkeypatch,
):
    """cancel_job terminates the trainer directly; the process can be gone before
    the supervisor's watcher copies the flag onto the job. The marker is the
    truth, so a stop must not be reported as a training error."""
    pkg = make_package()

    async def spawn(argv, *, cwd, env):
        (Path(cwd) / pt.CANCEL_FILE).write_text("1", encoding="utf-8")
        _checkpoint_and_config(list(argv))
        return _Proc([b"Epoch 0: 100%|x| 1/1 [00:00<00:00, 1.0it/s]\n"], rc=1)

    monkeypatch.setattr(pt, "_spawn", spawn)
    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=2, python="py")
    await pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                       epochs=2, batch_size=2, accelerator="cpu", python="")

    assert job["status"] == "cancelled"
    assert job["error"] == ""


@pytest.mark.asyncio
async def test_a_failed_run_records_the_error_and_stops(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    _fake_spawner(monkeypatch, train_rc=1)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=2, python="py")
    await pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                       epochs=2, batch_size=2, accelerator="cpu")

    assert job["status"] == "error"
    assert job["running"] is False
    assert "training exited with an error" in job["error"]
    assert "CUDA out of memory" in job["log"]
    record = json.loads((Path(job["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "error" and record["error"]


@pytest.mark.asyncio
async def test_a_dataset_too_small_to_learn_from_is_refused(dirs, fast_probe, monkeypatch):
    """8 seconds of audio produced a robotic-noise .onnx — refuse before spawning."""
    pkg = make_package(clips=3, wav_seconds=1.0)
    monkeypatch.setattr(vp, "MIN_TRAIN_SECONDS", 60.0)
    called = {"n": 0}

    async def spawn(*a, **kw):  # pragma: no cover - must not be reached
        called["n"] += 1

    monkeypatch.setattr(pt, "_spawn", spawn)
    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=2, python="py")
    await pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                       epochs=2, batch_size=8, accelerator="cpu")

    assert called["n"] == 0
    assert job["status"] == "error"
    assert "too small" in job["error"] and "noise" in job["error"]
    assert job["seconds"] == pytest.approx(3.0, abs=0.1)
    run = Path(job["run_dir"])
    # The verdict is on disk, not only in the dashboard's memory: the log tail
    # and run.json survive a restarted dashboard.
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "TRAINING FAILED" in log and "noise" in log
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["seconds"] == pytest.approx(3.0, abs=0.1)
    assert record["error"] == job["error"]


@pytest.mark.asyncio
async def test_batch_size_is_capped_to_the_train_split(dirs, fast_probe, monkeypatch):
    pkg = make_package(clips=4)
    seen: dict = {}
    _fake_spawner(monkeypatch, seen)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=2, python="py")
    await pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                       epochs=2, batch_size=8, accelerator="cpu")

    assert job["status"] == "done", job["error"]
    argv = seen["argv"][0]
    # 4 clips, one held out → 3 training clips, so a batch of 8 becomes 3.
    assert argv[argv.index("--data.batch_size") + 1] == "3"
    record = json.loads((Path(job["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert record["batch_size"] == 3
    assert "batch size 8 is larger than the 3-clip train split" in job["log"]


@pytest.mark.asyncio
async def test_a_tiny_dataset_is_refused(dirs, fast_probe, monkeypatch):
    pkg = make_package(clips=2)
    _fake_spawner(monkeypatch)
    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=1, python="py")
    await pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                       epochs=1, batch_size=1, accelerator="cpu")
    assert job["status"] == "error"
    assert "at least 3 clips" in job["error"]


@pytest.mark.asyncio
async def test_a_missing_trainer_fails_before_spawning(dirs, monkeypatch):
    pkg = make_package()

    async def no_trainer(python="", *, force=False):
        env = fake_env("")
        env.update(can_train=False, missing=["piper-tts[train]"],
                   hint="no Piper trainer behind this interpreter")
        return env

    monkeypatch.setattr(pt, "probe", no_trainer)
    called = {"n": 0}

    async def spawn(*a, **kw):  # pragma: no cover - must not be reached
        called["n"] += 1

    monkeypatch.setattr(pt, "_spawn", spawn)
    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=1, python="py")
    await pt.run_train(job, package=str(pkg), epochs=1, batch_size=1, accelerator="cpu")

    assert called["n"] == 0
    assert job["status"] == "error"
    assert "no Piper trainer" in job["error"]


# ---------------------------------------------------------------------------
# Attach + routes
# ---------------------------------------------------------------------------

def _app(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "profiles").mkdir(exist_ok=True)
    from config import activate_profile, save_profile
    cfg = AppConfig(profile_name="luna")
    save_profile(cfg, "luna")
    activate_profile("luna")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=cfg, secrets=Secrets(fish_api_key="k")),
    )
    from dashboard.server import DashboardState, _build_app
    from fastapi.testclient import TestClient
    state = DashboardState()
    return TestClient(_build_app(state, None)), state


def test_a_run_trained_on_too_little_audio_cannot_be_attached(dirs, monkeypatch, profile):
    """The old 8s run must not install a noise model under a real voice name."""
    pkg = make_package(clips=3, wav_seconds=1.0)
    run = pkg / "train" / "run-20260101-010101"
    run.mkdir(parents=True)
    onnx = run / "pt_BR-luna_br-medium.onnx"
    onnx.write_bytes(b"ONNX")
    cfg = run / "pt_BR-luna_br-medium.onnx.json"
    cfg.write_text("{}", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({
        "status": "done", "voice_name": "pt_BR-luna_br-medium", "seconds": 3.0,
        "exported": {"onnx": str(onnx), "config": str(cfg)},
    }), encoding="utf-8")
    monkeypatch.setattr(vp, "MIN_TRAIN_SECONDS", 60.0)

    with pytest.raises(pt.TrainError) as e:
        pt.attach_run(profile="luna", package=str(pkg), run_dir=str(run))
    assert "noise" in str(e.value) and "3.0s" in str(e.value)
    assert voice_lab.load_library("luna") == []
    assert not (piper_setup.voices_dir() / "pt_BR-luna_br-medium.onnx").exists()


def test_attach_run_installs_the_trained_pair(dirs, fast_probe, monkeypatch, profile):
    pkg = make_package()
    _fake_spawner(monkeypatch, )

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=1, python="py")
    asyncio.run(pt.run_train(job, package=str(pkg), voice_name="pt_BR-luna_br-medium",
                             epochs=1, batch_size=1, accelerator="cpu"))

    res = pt.attach_run(
        profile="luna", package=str(pkg), run_dir=job["run_dir"],
        display_name="Luna local",
    )
    assert res["ok"] and res["target"] == "piper"
    assert Path(res["path"]).name == "pt_BR-luna_br-medium.onnx"
    assert (piper_setup.voices_dir() / "pt_BR-luna_br-medium.onnx.json").is_file()
    assert [v.name for v in voice_lab.load_library("luna")] == ["Luna local"]

    # Pointing the attach at an arbitrary folder must be refused.
    with pytest.raises(pt.TrainError):
        pt.attach_run(profile="luna", package=str(pkg),
                      run_dir=str(REPO), name="pt_BR-x-medium")


@pytest.mark.asyncio
async def test_setup_installs_the_train_extra_and_reprobes(dirs, monkeypatch):
    seen: list[list[str]] = []
    probes = {"n": 0}

    async def probe(python="", *, force=False):
        probes["n"] += 1
        env = fake_env("" if probes["n"] == 1 else "piper1-gpl")
        if probes["n"] == 1:
            env.update(can_train=False, missing=["piper-tts[train]"],
                       hint="no Piper trainer behind this interpreter")
        return env

    async def spawn(argv, *, cwd, env):
        seen.append(list(argv))
        return _Proc([b"Collecting lightning...\n", b"Successfully installed\n"])

    monkeypatch.setattr(pt, "probe", probe)
    monkeypatch.setattr(pt, "_spawn", spawn)

    job = pt.new_job()
    pt.begin(job, kind="setup", python=pt.interpreter(""), epochs=0)
    await pt.run_setup(job, "")

    assert job["status"] == "done", job["error"]
    assert probes["n"] == 2, "a successful install must re-probe the interpreter"
    assert seen and seen[0][1:] == ["-m", "pip", "install", "--upgrade", pt.TRAIN_EXTRA]
    assert "pip install" in job["log"]


def test_train_status_lists_packages_and_the_job(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    make_stopped_run(pkg, epoch=3, epochs=10)
    app, state = _app(dirs, monkeypatch)
    with app as client:
        r = client.get("/api/voices/port/train")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["env"]["can_train"] is True
        assert [Path(p["package"]) for p in data["packages"]] == [pkg]
        assert data["selected"]["voice_name"] == "pt_BR-luna_br-medium"
        assert data["selected"]["resume"]["epoch"] == 3
        assert data["packages"][0]["resume"]["run_dir"].endswith("run-20260101-010101")
        assert data["job"]["running"] is False
        assert data["defaults"]["epochs"] == pt.DEFAULT_EPOCHS

        r = client.get("/api/voices/port/train", params={"package": str(pkg)})
        assert r.status_code == 200
        bad = client.get("/api/voices/port/train", params={"package": "/etc"})
        assert bad.status_code == 400


def test_references_list_every_onnx_voice_on_disk(dirs):
    """The Train panel offers the models already here, not just a path box."""
    root = piper_setup.DEFAULT_VOICES_DIR
    installed = root / "pt_BR-faber-medium.onnx"
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_bytes(b"ONNX")
    (root / "pt_BR-faber-medium.onnx.json").write_text("{}", encoding="utf-8")
    run_export = root / "ported" / "luna-piper" / "train" / "run-20260101-010101" / "pt_BR-luna-medium.onnx"
    run_export.parent.mkdir(parents=True)
    run_export.write_bytes(b"ONNX")
    os.utime(run_export, (1_700_000_000, 1_700_000_000))   # older than the download

    refs = pt.list_references()

    assert [Path(r["path"]).name for r in refs] == [
        "pt_BR-faber-medium.onnx", "pt_BR-luna-medium.onnx",
    ]
    assert refs[0]["config"] is True and refs[0]["trained"] is False
    assert refs[1]["config"] is False and refs[1]["trained"] is True


def test_status_route_reconnects_to_a_detached_run(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    run = pkg / "train" / "run-20260304-000000"
    run.mkdir(parents=True)
    (run / "train.log").write_text("Epoch 4: 100%|##| 8/8\n", encoding="utf-8")
    (run / pt.JOB_FILE).write_text(json.dumps({
        "status": "running", "pid": os.getpid(), "epoch": 4, "epochs": 10,
        "progress": 42.0, "package": str(pkg), "run_dir": str(run),
        "updated_at": 5.0,
    }), encoding="utf-8")
    app, state = _app(dirs, monkeypatch)
    with app as client:
        d = client.get("/api/voices/port/train").json()
        assert d["job"]["running"] is True
        assert d["job"]["epoch"] == 4
        assert d["job"]["run_dir"] == str(run)
        assert "Epoch 4" in d["job"]["log"]
        # The panel selects the package that is actually training.
        assert d["selected"]["package"] == str(pkg)


def test_train_route_starts_and_attaches(dirs, fast_probe, monkeypatch, in_process_launch):
    pkg = make_package()
    _fake_spawner(monkeypatch)
    app, state = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/train", json={
            "package": str(pkg), "epochs": 2, "batch_size": 2, "accelerator": "cpu",
        })
        assert r.status_code == 200, r.text
        assert r.json()["job"]["running"] is True
        for _ in range(300):
            if not state.voice_train["running"]:
                break
            import time as _t
            _t.sleep(0.02)
        assert state.voice_train["status"] == "done", state.voice_train["error"]
        run_dir = state.voice_train["run_dir"]

        r = client.post("/api/voices/port/train/attach", json={
            "package": str(pkg), "run_dir": run_dir, "display_name": "Luna trained",
        })
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "Luna trained"
        assert "Luna trained" in [p["name"] for p in r.json()["presets"]]


def test_train_route_accepts_resume_from(dirs, fast_probe, monkeypatch, in_process_launch):
    pkg = make_package()
    stopped = make_stopped_run(pkg, epoch=2, epochs=5)
    _fake_spawner(monkeypatch)
    app, state = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/train", json={
            "package": str(pkg), "resume_from": str(stopped),
            "epochs": 5, "batch_size": 2, "accelerator": "cpu",
        })
        assert r.status_code == 200, r.text
        assert r.json()["job"]["resumed_from"]["epoch"] == 2
        for _ in range(300):
            if not state.voice_train["running"]:
                break
            import time as _t
            _t.sleep(0.02)
        assert state.voice_train["status"] == "done", state.voice_train["error"]
        assert state.voice_train["resumed_from"]["run_dir"] == str(stopped)

        bad = client.post("/api/voices/port/train", json={
            "package": str(pkg), "resume_from": str(pkg), "epochs": 5,
        })
        assert bad.status_code == 400


def test_start_run_spawns_a_detached_supervisor(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=5,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"
    seen: dict = {}

    class _Popen:
        pid = 4321

    def _popen(argv, *, cwd, log_path):
        seen.update(argv=list(argv), cwd=cwd, log=Path(log_path))
        return _Popen()

    monkeypatch.setattr(pt, "_popen_supervisor", _popen)
    job = pt.new_job()
    snap = pt.start_run(job, options)

    assert snap["running"] is True
    assert snap["pid"] == 4321
    run_dir = Path(snap["run_dir"])
    assert seen["argv"][1:3] == ["-m", pt.SUPERVISOR_MODULE]
    assert seen["argv"][-1] == str(run_dir)
    assert seen["log"] == run_dir / pt.SUPERVISOR_LOG
    assert seen["cwd"] == piper_setup.REPO_ROOT

    cfg = json.loads((run_dir / pt.RUN_CONFIG_FILE).read_text(encoding="utf-8"))
    assert cfg["package"] == str(pkg)
    assert cfg["flavor"] == "piper1-gpl"
    assert cfg["epochs"] == 5

    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["status"] == "running" and disk["pid"] == 4321
    assert disk["run_dir"] == str(run_dir)
    # A started run is immediately cancellable, before the child writes anything.
    assert pt.cancel_job(job) is True
    assert (run_dir / pt.CANCEL_FILE).is_file()


def test_supervise_runs_the_config_and_writes_the_job(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=2,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    job = pt.new_job()
    run_dir = Path(pt.start_run(job, options)["run_dir"])

    _fake_spawner(monkeypatch)
    pt.supervise(run_dir)

    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["status"] == "done", disk.get("error")
    assert disk["running"] is False
    assert disk["progress"] == 100.0
    assert disk["epoch"] == 1 and disk["epochs"] == 2
    assert Path(disk["exported"]["onnx"]).is_file()
    assert Path(disk["exported"]["config"]).is_file()
    assert disk["pid"] != 4321          # the supervisor's own pid replaced the seed
    assert (run_dir / "train.log").read_text(encoding="utf-8").count("Epoch") >= 2
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "done"


def test_supervise_stops_on_the_cancel_marker(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=200,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    run_dir = Path(pt.start_run(pt.new_job(), options)["run_dir"])
    _fake_spawner(monkeypatch, silent=True)

    import threading
    import time as _t
    thread = threading.Thread(target=pt.supervise, args=(run_dir,), daemon=True)
    thread.start()
    deadline = _t.time() + 10
    while _t.time() < deadline:
        data = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
        if data.get("child_pid"):
            break
        _t.sleep(0.05)
    assert data.get("child_pid"), "the trainer process pid must reach job.json"
    (run_dir / pt.CANCEL_FILE).write_text("1", encoding="utf-8")
    thread.join(timeout=10)
    assert not thread.is_alive()

    final = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert final["status"] == "cancelled"
    assert "cancelled" in final["log"]
    assert final["running"] is False


def test_start_run_carries_the_notification_settings_into_the_run(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=2,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"
    options["notify"] = {"windows": True, "discord": False}

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    job = pt.new_job()
    job["notified"] = {"windows": "shown"}      # left over from a previous run
    run_dir = Path(pt.start_run(job, options)["run_dir"])

    cfg = json.loads((run_dir / pt.RUN_CONFIG_FILE).read_text(encoding="utf-8"))
    assert cfg["notify"] == {"windows": True, "discord": False}
    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["notified"] == {}, "a new run must not inherit the last run's notice"


def test_supervise_announces_the_result_with_the_runs_own_prefs(dirs, fast_probe, monkeypatch):
    from tts import train_notify

    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=2,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"
    options["notify"] = {"windows": False, "discord": True}

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    run_dir = Path(pt.start_run(pt.new_job(), options)["run_dir"])

    seen: list[dict] = []

    def _notify(run, job, prefs):
        seen.append({"run": Path(run), "status": job.get("status"), "prefs": dict(prefs)})
        return {"discord": "sent"}

    monkeypatch.setattr(train_notify, "notify_run", _notify)
    _fake_spawner(monkeypatch)
    pt.supervise(run_dir)

    assert len(seen) == 1
    assert seen[0]["run"] == run_dir
    assert seen[0]["status"] == "done"
    assert seen[0]["prefs"] == {"windows": False, "discord": True}
    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["notified"] == {"discord": "sent"}


def test_a_failing_notifier_never_breaks_the_run(dirs, fast_probe, monkeypatch):
    from tts import train_notify

    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=2,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"
    options["notify"] = {"windows": True}

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    run_dir = Path(pt.start_run(pt.new_job(), options)["run_dir"])

    def _boom(*a, **k):
        raise RuntimeError("notification module is broken")

    monkeypatch.setattr(train_notify, "notify_run", _boom)
    _fake_spawner(monkeypatch)
    pt.supervise(run_dir)

    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["status"] == "done"
    assert Path(disk["exported"]["onnx"]).is_file()
    assert disk["notified"] == {}


def test_supervise_installs_the_model_when_the_run_carries_an_attach_spec(
    dirs, fast_probe, monkeypatch, profile,
):
    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=2,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"
    options["attach"] = {
        "profile": "luna", "name": "", "display_name": "Luna local",
    }

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    run_dir = Path(pt.start_run(pt.new_job(), options)["run_dir"])
    assert json.loads((run_dir / pt.RUN_CONFIG_FILE).read_text(encoding="utf-8"))["attach"] == {
        "profile": "luna", "name": "", "display_name": "Luna local",
    }

    _fake_spawner(monkeypatch)
    pt.supervise(run_dir)

    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["status"] == "done", disk.get("error")
    assert disk["attach_error"] == ""
    attached = disk["attached"]
    assert attached["name"] == "Luna local"
    assert attached["profile"] == "luna"
    assert Path(attached["path"]).is_file()
    # The voice is in the library a picker reads, not just on disk.
    assert "Luna local" in [p.name for p in voice_lab.load_library("luna")]
    # …and the run remembers it, so a restarted dashboard shows the install.
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["attached"] == attached


def test_a_failed_auto_attach_leaves_the_finished_run_attachable(
    dirs, fast_probe, monkeypatch, profile,
):
    pkg = make_package()
    options = pt.validate_start(package=str(pkg), voice_name="", epochs=2,
                                batch_size=2, accelerator="cpu", python="",
                                checkpoint="")
    options["flavor"] = "piper1-gpl"
    options["attach"] = {"profile": "luna", "name": "not-a-piper-name"}

    class _Popen:
        pid = 4321

    monkeypatch.setattr(pt, "_popen_supervisor", lambda *a, **k: _Popen())
    run_dir = Path(pt.start_run(pt.new_job(), options)["run_dir"])
    _fake_spawner(monkeypatch)
    pt.supervise(run_dir)

    disk = json.loads((run_dir / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert disk["status"] == "done"
    assert Path(disk["exported"]["onnx"]).is_file()
    assert disk["attached"] == {}
    assert "not a Piper voice name" in disk["attach_error"]
    assert voice_lab.load_library("luna") == []


def test_train_route_snapshots_the_profiles_notify_toggles(dirs, fast_probe, monkeypatch):
    pkg = make_package()
    seen: dict = {}

    def _start(job, options):
        seen.update(options)
        pt.begin(job, kind="train", package=options["package"],
                 voice_name=options["voice_name"], epochs=options["epochs"],
                 python=options["python"])
        job.update(status="running", run_dir=str(pkg / "train" / "run-x"), pid=1)
        return pt.snapshot(job)

    monkeypatch.setattr(pt, "start_run", _start)
    import dashboard.server as srv
    cfg = AppConfig(profile_name="luna")
    cfg.training_notify.windows = False
    cfg.training_notify.discord = True
    monkeypatch.setattr(srv, "load_profile", lambda: cfg)

    app, state = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/train", json={
            "package": str(pkg), "epochs": 2, "batch_size": 2, "accelerator": "cpu",
        })
        assert r.status_code == 200, r.text
    assert seen["notify"] == {"windows": False, "discord": True}


def test_notify_test_route_reports_discords_answer(dirs, monkeypatch):
    """The API Keys button: one real POST, and the reason when it fails."""
    from tts import train_notify

    answers = iter([(True, "test message delivered — check the channel"),
                    (False, "Discord rejected the webhook: HTTP 404 — Unknown Webhook (code 10015)")])
    monkeypatch.setattr(train_notify, "send_test", lambda: next(answers))

    app, _ = _app(dirs, monkeypatch)
    with app as client:
        ok = client.post("/api/voices/port/train/notify/test")
        assert ok.status_code == 200, ok.text
        assert ok.json() == {"ok": True, "note": "test message delivered — check the channel"}

        bad = client.post("/api/voices/port/train/notify/test")
        assert bad.status_code == 200, bad.text
        body = bad.json()
        assert body["ok"] is False
        assert "Unknown Webhook" in body["error"]


def test_notify_test_route_refuses_without_a_webhook(dirs, monkeypatch):
    from tts import train_notify

    monkeypatch.delenv(train_notify.WEBHOOK_ENV, raising=False)
    monkeypatch.setattr(train_notify, "webhook_url", lambda: "")
    posted: list[str] = []
    monkeypatch.setattr(
        train_notify, "_post_json", lambda url, payload, **kw: posted.append(url) or "",
    )

    app, _ = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/train/notify/test")
        assert r.status_code == 200, r.text
    assert r.json()["ok"] is False
    assert train_notify.WEBHOOK_ENV in r.json()["error"]
    assert posted == []


def test_active_job_finds_a_live_run_and_syncs_state(dirs, monkeypatch):
    pkg = make_package()
    run = pkg / "train" / "run-20260301-000000"
    run.mkdir(parents=True)
    (run / "train.log").write_text("Epoch 2: 100%|##| 8/8\n", encoding="utf-8")
    (run / pt.JOB_FILE).write_text(json.dumps({
        "status": "running", "pid": os.getpid(), "child_pid": 999_999,
        "epoch": 2, "epochs": 10, "progress": 20.0,
        "package": str(pkg), "run_dir": str(run), "updated_at": 1.0,
    }), encoding="utf-8")

    data = pt.active_job()
    assert data is not None
    assert data["run_dir"] == str(run)
    assert data["package"] == str(pkg)
    assert "Epoch 2" in data["log"]

    job = pt.new_job()
    snap = pt.sync_job(job)
    assert snap["running"] is True
    assert snap["epoch"] == 2 and snap["progress"] == 20.0
    assert "Epoch 2" in snap["log"]
    assert pt.active_job()["pid"] == os.getpid()


def test_active_job_buries_a_dead_supervisor(dirs):
    pkg = make_package()
    run = pkg / "train" / "run-20260302-000000"
    run.mkdir(parents=True)
    (run / pt.JOB_FILE).write_text(json.dumps({
        "status": "running", "pid": 999_999, "updated_at": 1.0,
    }), encoding="utf-8")

    assert pt.active_job() is None
    buried = json.loads((run / pt.JOB_FILE).read_text(encoding="utf-8"))
    assert buried["status"] == "error" and buried["running"] is False
    assert "resume" in buried["error"]


def test_sync_job_picks_up_the_finished_runs_notification(dirs):
    """The last poll can miss the notification; the folder has the last word."""
    pkg = make_package()
    run = pkg / "train" / "run-20260305-000000"
    run.mkdir(parents=True)
    (run / pt.JOB_FILE).write_text(json.dumps({
        "status": "done", "pid": 4242, "running": False, "epoch": 5, "epochs": 5,
        "run_dir": str(run), "updated_at": 9.0,
        "notified": {"windows": "shown", "discord": "skipped: no webhook configured"},
    }), encoding="utf-8")

    job = pt.new_job()
    job.update(run_dir=str(run), status="done", running=False, epoch=4)
    snap = pt.sync_job(job)

    assert snap["notified"]["windows"] == "shown"
    assert "no webhook" in snap["notified"]["discord"]
    assert snap["running"] is False


def test_cancel_job_marks_and_terminates_the_child(dirs, monkeypatch):
    pkg = make_package()
    run = pkg / "train" / "run-20260303-000000"
    run.mkdir(parents=True)
    (run / pt.JOB_FILE).write_text(json.dumps({
        "status": "running", "pid": os.getpid(), "child_pid": 123456,
        "package": str(pkg), "run_dir": str(run), "updated_at": 2.0,
    }), encoding="utf-8")
    killed: list[int] = []
    monkeypatch.setattr(pt, "_terminate_pid", lambda pid: killed.append(pid) or True)

    job = {"running": True, "run_dir": str(run), "cancelled": False}
    assert pt.cancel_job(job) is True
    assert (run / pt.CANCEL_FILE).is_file()
    assert killed == [123456]
    assert job["cancelled"] is True

    # Nothing to stop once the supervisor recorded a terminal status.
    data = json.loads((run / pt.JOB_FILE).read_text(encoding="utf-8"))
    data["status"] = "done"
    (run / pt.JOB_FILE).write_text(json.dumps(data), encoding="utf-8")
    assert pt.cancel_job(job) is False


def test_train_route_refuses_a_second_run_and_a_bad_package(dirs, fast_probe, monkeypatch):
    app, state = _app(dirs, monkeypatch)
    state.voice_train["running"] = True
    with app as client:
        r = client.post("/api/voices/port/train", json={"package": "whatever"})
        assert r.status_code == 409
        r = client.post("/api/voices/port/train/setup", json={})
        assert r.status_code == 409
        r = client.post("/api/voices/port/train/cancel")
        assert r.status_code == 200

    state.voice_train["running"] = False
    with app as client:
        r = client.post("/api/voices/port/train", json={"package": "/nope"})
        assert r.status_code == 400
        r = client.post("/api/voices/port/train/cancel")
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# Markup drift — the panel has no runtime in CI
# ---------------------------------------------------------------------------

def test_voice_lab_ships_the_train_panel():
    html = (REPO / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
    app = (REPO / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")
    assert "Train the Piper voice" in html
    for hook in ('@click="trainVoice()"', "@click=\"cancelVoiceTrain()\"",
                 "@click=\"trainAttach()\"", "@click=\"trainSetup()\"",
                 "@click=\"trainResume()\"", 'x-ref="trainlog"',
                 'list="trainRefVoices"', 'id="trainRefVoices"'):
        assert hook in html, hook
    # A .onnx can seed the weights; the field must say so and offer saved voices.
    assert ".onnx reference" in html
    assert 'x-for="r in voiceTrain.references"' in html
    assert "voiceTrain.resume" in html
    for method in ("loadVoiceTrain", "trainVoice", "trainResume", "cancelVoiceTrain",
                   "trainSetup", "trainAttach", "trainStatusLine"):
        assert f"\n    {method}(" in app or f"async {method}(" in app, method
    assert "voiceTrain:" in app
    assert "resume_from: resumeFrom" in app
    assert '"/api/voices/port/train"' in app
    assert '"/api/voices/port/train/cancel"' in app
    assert '"/api/voices/port/train/attach"' in app
    assert '"/api/voices/port/train/setup"' in app


# ---------------------------------------------------------------------------
# monotonic_align — the core piper's wheel forgets to ship
# ---------------------------------------------------------------------------


def fake_piper_tree(root: Path, *, core: str | None = None) -> str:
    """The installed piper layout, with or without a compiled core file."""
    pkg = root / "site-packages" / "piper"
    outer = pkg / "train" / "vits" / "monotonic_align"
    outer.mkdir(parents=True)
    (outer / "monotonic_align").mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (outer / "__init__.py").write_text(
        "from .monotonic_align.core import maximum_path_c\n", encoding="utf-8"
    )
    if core:
        (outer / "monotonic_align" / core).write_bytes(b"")
    return str(pkg)


# A raw probe payload for an env whose export dependency is satisfied.
_SANE = {"torch_v": "2.14.1+cpu", "onnxscript": True}


def test_env_payload_flags_the_missing_alignment_core(dirs):
    missing = fake_piper_tree(dirs)
    raw = {"gpl": True, "legacy": False, "torch": True, "lightning": True,
           "piper_root": missing, **_SANE}
    env = pt._env_payload("py", raw)
    assert env["can_train"] and env["monotonic"] is False
    assert "monotonic_align" in env["hint"]

    filled = fake_piper_tree(dirs / "other", core="core.cp312-win_amd64.pyd")
    raw = {"gpl": True, "legacy": False, "torch": True, "lightning": True,
           "piper_root": filled, **_SANE}
    env = pt._env_payload("py", raw)
    assert env["monotonic"] is True


def test_the_legacy_trainer_never_demands_the_core(dirs):
    raw = {"gpl": False, "legacy": True, "torch": True, "lightning": True,
           "piper_root": "", **_SANE}
    env = pt._env_payload("py", raw)
    assert env["monotonic"] is True


def test_ensure_installs_the_fallback_into_the_empty_slot(dirs):
    root = fake_piper_tree(dirs)
    healed = pt.ensure_monotonic_align("py", root)
    assert healed["state"] == "installed"
    core_py = Path(healed["target"])
    assert core_py.is_file() and core_py.name == "core.py"
    assert "maximum_path_c" in core_py.read_text(encoding="utf-8")
    # a second pass sees the slot filled and writes nothing over the file
    assert pt.ensure_monotonic_align("py", root)["state"] == "present"


def test_ensure_skips_a_locally_compiled_core(dirs):
    root = fake_piper_tree(dirs, core="core.cp312-win_amd64.pyd")
    assert pt.ensure_monotonic_align("py", root)["state"] == "present"
    slot = Path(root) / "train" / "vits" / "monotonic_align" / "monotonic_align"
    assert not (slot / "core.py").exists()


def test_ensure_without_a_gpl_install_is_a_noop(dirs, monkeypatch):
    assert pt.ensure_monotonic_align("py", str(dirs / "nowhere"))["state"] == "absent"
    # An empty root falls back to probing the interpreter — stub "no piper"
    # so the check can never reach the real venv from a test.
    monkeypatch.setattr(pt, "_piper_root", lambda python: "")
    assert pt.ensure_monotonic_align("py", "")["state"] == "absent"


def test_ensure_reports_an_unwritable_target(dirs, monkeypatch):
    from tts import monotonic_core

    root = fake_piper_tree(dirs)
    monkeypatch.setattr(monotonic_core, "__file__", str(dirs / "gone.py"))
    healed = pt.ensure_monotonic_align("py", root)
    assert healed["state"] == "error"
    assert healed["detail"] and healed["target"].endswith("core.py")


@pytest.mark.asyncio
async def test_run_train_heals_the_missing_core_before_the_trainer_starts(dirs, monkeypatch):
    pkg = make_package()
    root = fake_piper_tree(dirs)

    async def probe(python="", *, force=False):
        env = fake_env(python=pt.interpreter(python))
        env.update(monotonic=False, piper_root=root)
        return env

    seen: dict = {}

    def heal(python, root_arg=""):
        seen["args"] = (python, root_arg)
        return {"state": "installed", "target": "core.py"}

    streams: list[list[str]] = []

    async def stream(job, argv, **kw):
        streams.append(list(argv))
        return 1  # fake trainer failure — late enough to prove the heal ran first

    monkeypatch.setattr(pt, "probe", probe)
    monkeypatch.setattr(pt, "ensure_monotonic_align", heal)
    monkeypatch.setattr(pt, "_stream", stream)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=1, python="py")
    await pt.run_train(job, package=str(pkg), epochs=1, batch_size=1,
                       accelerator="cpu")

    assert seen["args"][0] == pt.interpreter("")
    assert seen["args"][1] == root
    assert streams, "the heal must run before the trainer"
    assert "monotonic_align" in job["log"]


@pytest.mark.asyncio
async def test_run_train_refuses_to_start_without_an_alignment_core(dirs, monkeypatch):
    pkg = make_package()
    root = fake_piper_tree(dirs)

    async def probe(python="", *, force=False):
        env = fake_env(python=pt.interpreter(python))
        env.update(monotonic=False, piper_root=root)
        return env

    async def spawn(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("training must not start without an alignment core")

    monkeypatch.setattr(pt, "probe", probe)
    monkeypatch.setattr(pt, "ensure_monotonic_align",
                        lambda *a, **kw: {"state": "error", "detail": "denied",
                                          "target": "x"})
    monkeypatch.setattr(pt, "_spawn", spawn)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=1, python="py")
    await pt.run_train(job, package=str(pkg), epochs=1, batch_size=1,
                       accelerator="cpu")

    assert job["status"] == "error"
    assert "monotonic_align" in job["error"] and "denied" in job["error"]


@pytest.mark.asyncio
async def test_setup_fills_the_alignment_slot_even_when_the_trainer_is_there(dirs, monkeypatch):
    root = fake_piper_tree(dirs)

    async def probe(python="", *, force=False):
        env = fake_env(python=pt.interpreter(python))
        env.update(monotonic=False, piper_root=root)
        return env

    seen: dict = {}

    def heal(python, root_arg=""):
        seen["args"] = (python, root_arg)
        return {"state": "installed", "target": "core.py"}

    monkeypatch.setattr(pt, "probe", probe)
    monkeypatch.setattr(pt, "ensure_monotonic_align", heal)

    job = pt.new_job()
    pt.begin(job, kind="setup", python=pt.interpreter(""), epochs=0)
    await pt.run_setup(job, "")

    assert job["status"] == "done", job.get("error")
    assert seen["args"][1] == root
    assert "already installed" in job["log"]
    assert "monotonic_align" in job["log"]


# ---------------------------------------------------------------------------
# onnxscript — the export dependency piper's [train] extra never declares
# ---------------------------------------------------------------------------


def test_env_payload_flags_the_undeclared_onnxscript(dirs):
    raw = {"gpl": True, "legacy": False, "torch": True, "lightning": True,
           "torch_v": "2.14.1+cpu", "onnxscript": False,
           "piper_root": fake_piper_tree(dirs)}
    env = pt._env_payload("py", raw)
    assert env["export_missing"] is True
    assert "onnxscript" in env["missing"] and env["can_train"] is False
    assert env["setup_command"].endswith(" onnxscript")
    assert "onnxscript" in env["hint"]


def test_an_old_torch_needs_no_onnxscript(dirs):
    raw = {"gpl": True, "legacy": False, "torch": True, "lightning": True,
           "torch_v": "2.8.0", "onnxscript": False,
           "piper_root": fake_piper_tree(dirs)}
    env = pt._env_payload("py", raw)
    assert env["export_missing"] is False
    assert "onnxscript" not in env["missing"] and env["can_train"] is True
    assert not env["setup_command"].endswith(" onnxscript")


def test_a_fresh_install_counts_as_needing_onnxscript(dirs):
    # No torch yet: pip resolves the extra to the latest, which is past 2.9.
    raw = {"gpl": False, "legacy": False, "torch": False, "lightning": False,
           "torch_v": "", "onnxscript": False, "piper_root": ""}
    env = pt._env_payload("py", raw)
    assert env["export_missing"] is True
    assert env["setup_command"].endswith(" onnxscript")


@pytest.mark.asyncio
async def test_setup_installs_onnxscript_when_the_export_needs_it(dirs, monkeypatch):
    seen: list[list[str]] = []
    probes = {"n": 0}

    async def probe(python="", *, force=False):
        probes["n"] += 1
        env = fake_env("piper1-gpl")
        if probes["n"] == 1:
            env.update(can_train=False, missing=["onnxscript"],
                       export_missing=True, hint="onnxscript is missing")
        return env

    async def spawn(argv, *, cwd, env):
        seen.append(list(argv))
        return _Proc([b"Successfully installed onnxscript\n"])

    monkeypatch.setattr(pt, "probe", probe)
    monkeypatch.setattr(pt, "_spawn", spawn)

    job = pt.new_job()
    pt.begin(job, kind="setup", python=pt.interpreter(""), epochs=0)
    await pt.run_setup(job, "")

    assert job["status"] == "done", job.get("error")
    assert seen and seen[0][1:5] == ["-m", "pip", "install", "--upgrade"]
    assert seen[0][-1] == "onnxscript"
    assert "onnxscript" in job["log"]


# ---------------------------------------------------------------------------
# export_onnx.py — piper's call predates the dynamo default (torch 2.9+)
# ---------------------------------------------------------------------------

_FAKE_EXPORT = (
    "OPSET_VERSION = 15\r\n"
    "\r\n"
    "def main():\r\n"
    "    torch.onnx.export(\r\n"
    "        model=model_g,\r\n"
    "        args=dummy_input,\r\n"
    "        opset_version=OPSET_VERSION,\r\n"
    "    )\r\n"
)


def fake_export_src(root: Path) -> Path:
    src = root / "site-packages" / "piper" / "train" / "export_onnx.py"
    src.parent.mkdir(parents=True)
    src.write_bytes(_FAKE_EXPORT.encode("utf-8"))
    return src


def test_export_compat_patches_dynamo_false_and_is_idempotent(dirs):
    src = fake_export_src(dirs)
    root = str(dirs / "site-packages" / "piper")
    healed = pt.ensure_export_compat("", root)
    assert healed["state"] == "installed"

    text = src.read_bytes().decode("utf-8")
    assert "dynamo=False," in text
    compile(text, "export_onnx.py", "exec")        # still valid Python
    assert text.count("torch.onnx.export(") == 1
    # the patched line follows the file's own CRLF style, the rest untouched
    assert "\r\n        dynamo=False," in text

    before = src.read_bytes()
    again = pt.ensure_export_compat("", root)
    assert again["state"] == "present"
    assert src.read_bytes() == before, "a second pass must rewrite nothing"


def test_export_compat_leaves_a_fixed_exporter_alone(dirs):
    fixed = _FAKE_EXPORT.replace("torch.onnx.export(", "torch.onnx.export(dynamo=False,")
    src = fake_export_src(dirs)
    src.write_bytes(fixed.encode("utf-8"))
    healed = pt.ensure_export_compat("", str(dirs / "site-packages" / "piper"))
    assert healed["state"] == "present"
    assert src.read_bytes().decode("utf-8") == fixed


def test_export_compat_without_a_known_install_is_a_noop(dirs, monkeypatch):
    assert pt.ensure_export_compat("", str(dirs / "nowhere"))["state"] == "absent"
    # An empty root falls back to probing the test interpreter, which must
    # never reach the real venv: stub "no piper here" in.
    monkeypatch.setattr(pt, "_piper_root", lambda python: "")
    assert pt.ensure_export_compat("", "")["state"] == "absent"


def test_export_compat_reports_an_unpatchable_file(dirs):
    train = dirs / "site-packages" / "piper" / "train"
    train.mkdir(parents=True)
    (train / "export_onnx.py").write_text("something else entirely\n", encoding="utf-8")
    healed = pt.ensure_export_compat("", str(dirs / "site-packages" / "piper"))
    assert healed["state"] == "error"
    assert "torch.onnx.export(" in healed["detail"]
    assert healed["target"].endswith("export_onnx.py")


@pytest.mark.asyncio
async def test_run_train_patches_pipers_export_before_the_training(dirs, monkeypatch):
    pkg = make_package()
    fake_export_src(dirs)
    root = str(dirs / "site-packages")

    async def probe(python="", *, force=False):
        env = fake_env(python=pt.interpreter(python))
        env.update(piper_root=root)
        return env

    seen: dict = {}

    def compat(python, root_arg=""):
        seen["root"] = root_arg
        return {"state": "installed", "target": "export_onnx.py"}

    streams: list[list[str]] = []

    async def stream(job, argv, **kw):
        streams.append(list(argv))
        return 1  # fake trainer failure — late enough to prove the patch ran first

    monkeypatch.setattr(pt, "probe", probe)
    monkeypatch.setattr(pt, "ensure_export_compat", compat)
    monkeypatch.setattr(pt, "_stream", stream)

    job = pt.new_job()
    pt.begin(job, kind="train", package=str(pkg), epochs=1, python="py")
    await pt.run_train(job, package=str(pkg), epochs=1, batch_size=1,
                       accelerator="cpu")

    assert seen["root"] == root
    assert streams, "the export patch must run before the trainer"
    assert "dynamo=False" in job["log"]


# ---------------------------------------------------------------------------
# train/__main__.py — the optional val_mos callback aborts validation
# ---------------------------------------------------------------------------

_FAKE_MAIN = (
    "from lightning.pytorch.callbacks import ModelCheckpoint\n"
    "\n"
    "_DEFAULT_CALLBACKS = [\n"
    "    ModelCheckpoint(\n"
    "        monitor=\"val_mel\",\n"
    "        mode=\"min\",\n"
    "    ),\n"
    "    ModelCheckpoint(\n"
    "        monitor=\"val_mos\",\n"
    "        mode=\"max\",\n"
    "    ),\n"
    "]\n"
)


def fake_main_src(root: Path) -> Path:
    src = root / "site-packages" / "piper" / "train" / "__main__.py"
    src.parent.mkdir(parents=True)
    src.write_text(_FAKE_MAIN, encoding="utf-8")
    return src


def test_checkpoint_compat_removes_val_mos_and_is_idempotent(dirs):
    src = fake_main_src(dirs)
    root = str(dirs / "site-packages" / "piper")

    healed = pt.ensure_checkpoint_compat("", root)
    assert healed["state"] == "installed"

    text = src.read_text(encoding="utf-8")
    assert 'monitor="val_mos"' not in text
    assert 'monitor="val_mel"' in text
    compile(text, "train/__main__.py", "exec")       # still valid Python
    assert text.count("ModelCheckpoint(") == 1       # only the val_mel one

    before = src.read_bytes()
    again = pt.ensure_checkpoint_compat("", root)
    assert again["state"] == "present"
    assert src.read_bytes() == before, "a second pass must rewrite nothing"


def test_checkpoint_compat_leaves_a_fixed_cli_alone(dirs):
    fixed = _FAKE_MAIN.replace('    ModelCheckpoint(\n        monitor="val_mos",',
                               '    ModelCheckpoint(\n        monitor="other",')
    src = fake_main_src(dirs)
    src.write_text(fixed, encoding="utf-8")

    healed = pt.ensure_checkpoint_compat("", str(dirs / "site-packages" / "piper"))
    assert healed["state"] == "present"
    assert src.read_text(encoding="utf-8") == fixed


def test_checkpoint_compat_without_a_known_install_is_a_noop(dirs, monkeypatch):
    assert pt.ensure_checkpoint_compat("", str(dirs / "nowhere"))["state"] == "absent"
    monkeypatch.setattr(pt, "_piper_root", lambda python: "")
    assert pt.ensure_checkpoint_compat("", "")["state"] == "absent"


def test_checkpoint_compat_reports_an_unpatchable_file(dirs):
    train = dirs / "site-packages" / "piper" / "train"
    train.mkdir(parents=True)
    (train / "__main__.py").write_text(
        '# monitor="val_mos" but the callback is gone\n', encoding="utf-8"
    )

    healed = pt.ensure_checkpoint_compat("", str(dirs / "site-packages" / "piper"))
    assert healed["state"] == "error"
    assert "val_mos" in healed["detail"]
    assert healed["target"].endswith("__main__.py")

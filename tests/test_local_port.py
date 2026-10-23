"""Local voice port — a saved cloud voice becomes a local dataset, fully offline.

The contract this file pins down:

* no provider is ever contacted and no key is read — a local build is audio in,
  package out;
* the package has exactly the shape the capture path produces, so the Train
  panel and the attach step need no special case;
* a missing transcript is refused (Piper trains on the text), and the local
  Whisper is the only thing allowed to fill the gap;
* path traversal and silent/non-audio files never reach the package.
"""
from __future__ import annotations

import base64
import io
import json
import math
import struct
import wave
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets
from tts import kokoro_setup, local_port as lp, piper_setup, piper_train as pt, voice_lab
from tts import voice_port as vp

REPO = Path(__file__).resolve().parents[1]
NO_KEYS = Secrets(fish_api_key="", elevenlabs_api_key="")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    """Isolated profile dir + voices tree, shared with the engines' loaders."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "profiles").mkdir()
    monkeypatch.setattr(piper_setup, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(kokoro_setup, "DEFAULT_VOICES_DIR", tmp_path / "voices" / "kokoro")
    return tmp_path


def _wav(seconds: float = 1.0, rate: int = 22050, chunk: int = 12000,
         channels: int = 1, freq: float = 220.0) -> bytes:
    """A real PCM WAV — a tone is enough for decoding/trimming."""
    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = int(chunk * math.sin(2 * math.pi * freq * i / rate))
        for _ in range(channels):
            frames += struct.pack("<h", value)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _silence(seconds: float = 1.0, rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _clips(n: int = 3) -> list[tuple[str, bytes]]:
    return [(f"take{i}.wav", _wav()) for i in range(1, n + 1)]


def _texts(n: int = 3) -> list[str]:
    return [f"linha numero {i}" for i in range(1, n + 1)]


def _metadata(package: str | Path) -> list[tuple[str, str]]:
    rows = (Path(package) / "dataset" / "metadata.csv").read_text(encoding="utf-8")
    return [tuple(row.split("|", 1)) for row in rows.splitlines() if row]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def test_wav_is_normalized_to_mono_pcm_at_the_capture_rate(dirs):
    pcm = lp.decode_audio(_wav(seconds=1.0, rate=44100, channels=2), "stereo.wav")
    assert len(pcm) // 2 == pytest.approx(vp.CAPTURE_SR, abs=vp.CAPTURE_SR // 50)
    assert len(pcm) % 2 == 0


def test_silence_decodes_to_nothing_so_it_is_never_a_dataset_row(dirs):
    assert lp.decode_audio(_silence(), "quiet.wav") == b""


def test_uploaded_names_cannot_escape_the_voice_folder(dirs):
    assert lp.audio_name("../../evil.wav") == "evil.wav"
    assert lp.audio_name("C:\\Users\\x\\take 1.WAV") == "take-1.wav"
    with pytest.raises(lp.LocalPortError):
        lp.audio_name("notes.exe")


# ---------------------------------------------------------------------------
# Preparing the package
# ---------------------------------------------------------------------------

def test_local_audio_becomes_the_same_piper_package(dirs):
    report = lp.prepare(
        name="Luna", provider="fish", voice_id="v1",
        target="piper", lang="pt", files=_clips(), texts=_texts(),
    )
    assert report["clips"] == 3
    assert report["origin"] == "local-audio"
    assert report["network"] is False and report["keys_used"] is False
    assert report["transcribed"] == 0

    rows = _metadata(report["package"])
    assert [text for _name, text in rows] == _texts()
    assert all(name.startswith("wav/") for name, _text in rows)

    manifest = json.loads(
        (Path(report["package"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["origin_mode"] == "local-audio"
    assert manifest["network"] is False
    assert manifest["source"]["source"] == "local:fish"

    readme = (Path(report["package"]) / "README.md").read_text(encoding="utf-8")
    assert "Imported from" in readme
    assert "Captured with Wallie's Voice Lab" not in readme


def test_the_dataset_keeps_the_engines_rate(dirs):
    report = lp.prepare(name="Luna", target="piper", files=_clips(), texts=_texts())
    with wave.open(str(Path(report["package"]) / "dataset" / "wav" / "0001.wav"), "rb") as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 22050)


def test_silent_clips_are_skipped_and_reported(dirs):
    files = _clips() + [("dead.wav", _silence())]
    report = lp.prepare(name="Luna", target="piper", files=files, texts=_texts() + [""])
    assert report["clips"] == 3
    assert report["skipped"][0]["filename"] == "dead.wav"
    assert "silence" in report["skipped"][0]["reason"]


def test_a_missing_transcript_is_refused_when_whisper_cannot_run(dirs, monkeypatch):
    monkeypatch.setattr(lp, "whisper_status", lambda size="": {
        "available": True, "cached": False, "model": "small",
        "engine": "faster-whisper", "reason": "the Whisper 'small' model is not cached",
    })
    with pytest.raises(lp.LocalPortError) as e:
        lp.prepare(name="Luna", target="piper", files=_clips())
    assert "take1.wav" in str(e.value)
    assert "not cached" in str(e.value)


def test_the_local_whisper_fills_the_missing_lines(dirs, monkeypatch):
    monkeypatch.setattr(lp, "whisper_status", lambda size="": {
        "available": True, "cached": True, "model": "small",
        "engine": "faster-whisper", "reason": "",
    })
    monkeypatch.setattr(
        lp, "transcribe_pcm",
        lambda pcm, lang="pt", size="": "o rato roeu a roupa do rei",
    )
    report = lp.prepare(name="Luna", target="piper", files=_clips())
    assert report["transcribed"] == 3
    assert report["transcriber"] == "faster-whisper small"
    assert {text for _name, text in _metadata(report["package"])} == {
        "o rato roeu a roupa do rei"
    }
    manifest = json.loads(
        (Path(report["package"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["transcriber"] == "faster-whisper small"
    assert manifest["clips_meta"][0]["transcript"] == "local Whisper"


def test_typed_lines_and_a_sidecar_map_by_name(dirs):
    sidecar = b"take2|frase do arquivo\ntake3|outra frase\n"
    report = lp.prepare(
        name="Luna", target="piper", files=_clips(3) + [("take.txt", sidecar)],
        texts=["frase digitada", "", ""],
    )
    rows = dict(_metadata(report["package"]))
    assert rows["wav/0001.wav"] == "frase digitada"
    assert rows["wav/0002.wav"] == "frase do arquivo"
    assert rows["wav/0003.wav"] == "outra frase"


def test_imported_clips_are_reused_on_the_next_build(dirs):
    first = lp.prepare(name="Luna", target="piper", files=_clips(), texts=_texts())
    assert lp.scan("Luna")["clips"] == 3

    second = lp.prepare(name="Luna", target="kokoro", texts=_texts())
    assert second["clips"] == 3
    assert second["saved_clips"] == 3
    assert second["target"] == "kokoro"
    assert (Path(second["package"]) / "corpus" / "0001.wav").is_file()
    assert first["clips"] == 3


def test_kokoro_does_not_require_a_transcript(dirs):
    report = lp.prepare(name="Luna", target="kokoro", files=_clips())
    assert report["clips"] == 3
    assert report["target"] == "kokoro"


def test_no_audio_and_too_few_clips_are_refused_with_a_next_step(dirs):
    with pytest.raises(lp.LocalPortError) as e:
        lp.prepare(name="Luna", target="piper")
    assert "no audio" in str(e.value)

    with pytest.raises(lp.LocalPortError) as e:
        lp.prepare(name="Luna", target="piper", files=_clips(2), texts=_texts(2))
    assert "at least" in str(e.value)


def test_a_non_audio_upload_is_refused_before_anything_is_written(dirs):
    with pytest.raises(lp.LocalPortError) as e:
        lp.prepare(name="Luna", files=[("virus.exe", b"MZ")], texts=["x"])
    assert "not audio" in str(e.value)


def test_prepare_never_calls_the_capture_path(dirs, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("a local build must not speak with a provider")

    monkeypatch.setattr(vp, "capture", _boom)
    monkeypatch.setattr(vp, "_tts_for", _boom)
    report = lp.prepare(name="Luna", provider="elevenlabs", files=_clips(), texts=_texts())
    assert report["keys_used"] is False


# ---------------------------------------------------------------------------
# Panel + routes
# ---------------------------------------------------------------------------

def test_panel_reports_cached_whisper_without_loading_a_model(dirs, monkeypatch):
    monkeypatch.setattr(lp, "_whisper_cache_dir", lambda: dirs / "empty-cache")
    data = lp.panel_payload("Luna", "small")
    assert data["saved"]["clips"] == 0
    assert data["transcriber"]["cached"] is False
    assert data["transcriber"]["reason"]
    assert ".wav" in data["audio_exts"] and ".txt" in data["transcript_exts"]


def _app(tmp_path, monkeypatch, secrets: Secrets | None = None):
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
        lambda: Runtime(config=cfg, secrets=NO_KEYS if secrets is None else secrets),
    )
    from dashboard.server import DashboardState, _build_app
    from fastapi.testclient import TestClient
    return TestClient(_build_app(DashboardState(), None))


def test_local_panel_route_reads_the_saved_clips(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "Luna", "provider": "fish", "voice_id": "v1"})
    lp.prepare(name="Luna", target="piper", files=_clips(), texts=_texts())

    app = _app(dirs, monkeypatch)
    with app as client:
        data = client.get("/api/voices/port/local", params={"name": "Luna"}).json()
    assert data["ok"] is True
    assert data["saved"]["clips"] == 3
    assert data["saved"]["seconds"] > 0
    assert data["transcriber"]["engine"] == "faster-whisper"


def test_local_route_builds_a_dataset_without_keys(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "Luna", "provider": "fish", "voice_id": "v1"})
    payload = [
        {"filename": name, "data": base64.b64encode(blob).decode("ascii")}
        for name, blob in _clips()
    ]
    app = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/local", json={
            "name": "Luna", "provider": "fish", "voice_id": "v1",
            "target": "piper", "lang": "pt", "files": payload, "texts": _texts(),
        })
        assert r.status_code == 200, r.text
        data = r.json()
    assert data["clips"] == 3
    assert data["network"] is False
    assert (Path(data["package"]) / "dataset" / "metadata.csv").is_file()


def test_local_route_reports_a_missing_transcript_instead_of_a_500(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "Luna", "provider": "fish", "voice_id": "v1"})
    monkeypatch.setattr(lp, "whisper_status", lambda size="": {
        "available": False, "cached": False, "model": "small",
        "engine": "faster-whisper", "reason": "faster-whisper is not installed",
    })
    payload = [
        {"filename": name, "data": base64.b64encode(blob).decode("ascii")}
        for name, blob in _clips()
    ]
    app = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/local", json={
            "name": "Luna", "target": "piper", "files": payload,
        })
        assert r.status_code == 400
        assert "not installed" in r.json()["detail"]


def _stub_trainer(monkeypatch, seen: dict, *, can_train: bool = True, hint: str = ""):
    async def _probe(python="", *, force=False):
        return {
            "python": python, "can_train": can_train,
            "flavor": "piper1-gpl" if can_train else "",
            "hint": hint or ("ready" if can_train else "no trainer found"),
            "setup_command": "python -m pip install piper-tts[train]",
        }

    def _start(job, options):
        seen.update(options)
        pt.begin(job, kind="train", package=options["package"],
                 voice_name=options["voice_name"], epochs=options["epochs"],
                 python=options["python"])
        job.update(status="running", running=True, pid=1,
                   run_dir=str(Path(options["package"]) / "train" / "run-x"))
        return pt.snapshot(job)

    monkeypatch.setattr(pt, "probe", _probe)
    monkeypatch.setattr(pt, "start_run", _start)


def test_a_local_build_starts_the_training_and_carries_the_attach_spec(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "Luna", "provider": "fish", "voice_id": "v1"})
    seen: dict = {}
    _stub_trainer(monkeypatch, seen)
    payload = [
        {"filename": name, "data": base64.b64encode(blob).decode("ascii")}
        for name, blob in _clips()
    ]
    app = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/local", json={
            "name": "Luna", "provider": "fish", "voice_id": "v1",
            "target": "piper", "lang": "pt", "files": payload,
            "texts": _texts(), "auto_train": True,
        })
        assert r.status_code == 200, r.text
        data = r.json()
    assert data["train"]["started"] is True
    assert data["train"]["epochs"] == pt.DEFAULT_EPOCHS
    assert data["train"]["job"]["running"] is True
    # Half of the auto-port lives in the run config: the supervisor must know
    # where to install the model with no dashboard watching.
    assert seen["attach"] == {"profile": "luna", "name": "", "display_name": "Luna"}
    assert seen["notify"] == {"windows": True, "discord": False}
    assert seen["package"] == data["package"]
    assert seen["voice_name"].startswith("pt_BR-")


def test_a_missing_trainer_leaves_a_successful_dataset_and_a_reason(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "Luna", "provider": "fish", "voice_id": "v1"})
    _stub_trainer(
        monkeypatch, {}, can_train=False,
        hint="piper1-gpl is importable but lightning is missing — install piper-tts[train]",
    )
    payload = [
        {"filename": name, "data": base64.b64encode(blob).decode("ascii")}
        for name, blob in _clips()
    ]
    app = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/local", json={
            "name": "Luna", "target": "piper", "files": payload, "texts": _texts(),
        })
        assert r.status_code == 200, r.text
        data = r.json()
    assert data["clips"] == 3
    assert data["train"]["started"] is False
    assert "lightning" in data["train"]["reason"]
    assert "piper-tts[train]" in data["train"]["setup_command"]
    assert (Path(data["package"]) / "dataset" / "metadata.csv").is_file()


def test_auto_training_can_be_turned_off(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "Luna", "provider": "fish", "voice_id": "v1"})
    calls: list = []
    monkeypatch.setattr(pt, "start_run", lambda *a, **k: calls.append(a) or {})
    payload = [
        {"filename": name, "data": base64.b64encode(blob).decode("ascii")}
        for name, blob in _clips()
    ]
    app = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port/local", json={
            "name": "Luna", "target": "piper", "files": payload,
            "texts": _texts(), "auto_train": False,
        })
        assert r.status_code == 200, r.text
        data = r.json()
    assert data["train"] == {"started": False, "reason": ""}
    assert calls == []


# ---------------------------------------------------------------------------
# Markup drift — the local block has no runtime in CI
# ---------------------------------------------------------------------------

def test_voice_lab_ships_the_local_audio_block():
    html = (REPO / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
    app = (REPO / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")
    for hook in ("portLocalFiles($event)", "portLocalBuild()", "portLocalTranscriber()",
                 "voicePort.local.autoTrain", "trainAttachLine()"):
        assert hook in html, hook
    for method in ("portLocalFiles", "portLocalBuild", "portLoadLocal", "trainAttachLine"):
        assert f"\n    {method}(" in app or f"async {method}(" in app, method
    assert '"/api/voices/port/local' in app
    assert "voicePort.local" in html
    assert "auto_train" in app

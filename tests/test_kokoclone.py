"""KokoClone — the optional clone backend: detection, clips, worker, provider.

The heavy half of KokoClone (torch, kokoro-onnx, kanade) is never imported
here. What IS exercised for real is the part this project wrote: the JSON-lines
protocol against a stand-in ``core/cloner.py`` in a temp checkout, so a worker
that hangs, prints into the stream, or dies mid-job fails in CI instead of in
someone's stream.
"""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from config import Secrets, TTSConfig
from tts import kokoclone as kk
from tts.base import TTSError
from tts.voice_lab import pcm16_bytes_to_wav

# A KokoClone stand-in: same class name and generate() signature, stdlib only.
FAKE_CLONER = '''
import os
import wave


class KokoClone:
    sample_rate = 24000

    def __init__(self):
        print("fake KokoClone ready")     # chatty on purpose: must not corrupt the protocol
        print("env api_key=" + os.environ.get("WALLIE_TEST_API_KEY", "<none>"))
        print("env pin=" + os.environ.get("DASHBOARD_PIN", "<none>"))

    def generate(self, text, lang, reference_audio, output_path):
        if text == "boom":
            raise RuntimeError("kanade exploded")
        print(f"generating {text!r} in {lang}")
        frames = b"\\x01\\x00" * 12000      # 0.5 s of PCM16 at 24 kHz
        with wave.open(output_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(frames)
'''


def _wav(seconds: float = 4.0, rate: int = 24000, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    arr = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return pcm16_bytes_to_wav((arr * 32767).astype("<i2").tobytes(), rate)


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    """A checkout that looks like kokoclone's and answers like it."""
    root = tmp_path / "kokoclone"
    (root / "core").mkdir(parents=True)
    (root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (root / "core" / "cloner.py").write_text(FAKE_CLONER, encoding="utf-8")
    monkeypatch.setenv("WALLIE_KOKOCLONE_DIR", str(root))
    # Same layout as production: voices/kokoclone/<slug>/ref.wav.
    monkeypatch.setattr(kk, "DEFAULT_VOICES_DIR", tmp_path / "voices" / "kokoclone")
    yield root
    kk.stop_worker()


@pytest.fixture()
def no_checkout(tmp_path, monkeypatch):
    root = tmp_path / "missing"
    monkeypatch.setenv("WALLIE_KOKOCLONE_DIR", str(root))
    monkeypatch.setattr(kk, "DEFAULT_VOICES_DIR", tmp_path / "voices" / "kokoclone")
    return root


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_a_missing_checkout_says_how_to_get_one(no_checkout):
    status = kk.detect()
    assert status["installed"] is False
    assert status["checkout"] is False
    assert "git clone" in status["hint"]
    assert "kokoclone.git" in status["hint"]
    assert status["root"] == str(no_checkout)


def test_a_checkout_without_the_packages_is_not_installed(fake_root, monkeypatch):
    monkeypatch.setattr(kk, "REQUIRED_PACKAGES", ("definitely_not_installed_pkg",))
    status = kk.detect()
    assert status["checkout"] is True
    assert status["installed"] is False
    assert status["missing"] == ["definitely_not_installed_pkg"]
    # The hint must name the way out: the button (same script) — not "reinstall pip".
    assert "install KokoClone" in status["hint"]
    assert "install_kokoclone.py --install" in status["hint"]


def test_a_complete_checkout_is_installed(fake_root, monkeypatch):
    monkeypatch.setattr(kk, "REQUIRED_PACKAGES", ())
    status = kk.detect()
    assert status["installed"] is True
    assert status["missing"] == []
    assert [lang["id"] for lang in status["languages"]][:2] == ["pt", "en"]
    assert status["sample_rate"] == 24000


# ---------------------------------------------------------------------------
# Reference clips
# ---------------------------------------------------------------------------

def test_a_short_clip_is_refused_with_the_measured_length(fake_root):
    with pytest.raises(kk.KokoCloneError, match="needs at least"):
        kk.store_voice("Luna", _wav(1.0), lang="pt")


def test_a_long_clip_is_refused(fake_root):
    with pytest.raises(kk.KokoCloneError, match="trim it"):
        kk.store_voice("Luna", _wav(40.0), lang="pt")


def test_a_clip_is_kept_as_24k_mono_wav_with_its_recipe(fake_root):
    clip = kk.store_voice("Luna Clone", _wav(4.0), lang="pt", filename="take.wav")
    assert clip["name"] == "Luna Clone"
    assert clip["slug"] == "luna_clone"
    assert clip["lang"] == "pt"
    assert clip["seconds"] == pytest.approx(4.0, abs=0.2)
    assert clip["ref"].replace("\\", "/").endswith("luna_clone/ref.wav")

    path = fake_root.parent / "voices" / "kokoclone" / "luna_clone" / "ref.wav"
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000

    voices = kk.list_voices()
    assert [v["name"] for v in voices] == ["Luna Clone"]
    assert voices[0]["lang"] == "pt"


def test_an_unknown_language_is_refused(fake_root):
    with pytest.raises(kk.KokoCloneError, match="unknown KokoClone language"):
        kk.store_voice("Luna", _wav(4.0), lang="xx")


def test_an_empty_clip_is_refused(fake_root):
    with pytest.raises(kk.KokoCloneError, match="attach a reference clip"):
        kk.store_voice("Luna", b"", lang="pt")


def test_a_huge_clip_is_refused_before_it_is_decoded(fake_root):
    # The size check runs first, so decoding never sees a multi-gigabyte blob.
    with pytest.raises(kk.KokoCloneError, match="keep it under"):
        kk.store_voice("Luna", b"0" * (kk.MAX_REF_BYTES + 1), lang="pt")
    assert kk.list_voices() == []


def test_a_corrupt_recipe_does_not_break_the_clip_list(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    meta = kk.voices_dir() / "luna" / "meta.json"
    meta.write_text('{"name": "Luna", "seconds": "fast", "lang": "pt"}',
                    encoding="utf-8")
    assert kk.list_voices()[0]["seconds"] == 0.0


def test_detect_offers_the_one_click_install(no_checkout):
    status = kk.detect()
    assert status["installed"] is False
    assert status["installer_present"] is True      # scripts/install_kokoclone.py
    assert status["can_install"] is True
    assert isinstance(status["git"], bool)
    assert "install KokoClone" in status["hint"]


def test_a_supported_interpreter_is_required_to_offer_the_button(no_checkout, monkeypatch):
    monkeypatch.setattr(kk, "python_ok", lambda *a, **k: False)
    status = kk.detect()
    assert status["can_install"] is False
    assert "needs Python" in status["hint"]


def test_install_argv_runs_the_installer_for_the_probed_root(no_checkout):
    import sys

    argv = kk.install_argv()
    assert argv[0] == sys.executable
    assert argv[1] == str(kk.INSTALL_SCRIPT)
    assert "--install" in argv
    # The install must land where detect() looks next time, even with an override.
    assert argv[argv.index("--root") + 1] == str(kk.root_dir())


def _installer_module():
    """scripts/install_kokoclone.py as a module (it is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("install_kokoclone_script",
                                                kk.INSTALL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_installer_ships_the_subset_the_worker_imports(tmp_path, monkeypatch):
    """requirements.txt builds pyopenjtalk with CMake and pulls gradio: not ours."""
    mod = _installer_module()
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "call",
                        lambda cmd, **kw: calls.append(list(cmd)) or 0)
    (tmp_path / "requirements.txt").write_text("torch\n", encoding="utf-8")

    assert mod._pip_install(tmp_path) == 0
    specs = " ".join(calls[0])
    assert "-r" not in specs and "requirements.txt" not in specs
    assert "misaki[ja,zh]" not in specs and "gradio" not in specs
    assert "kanade-tokenizer" in specs and "kokoro-onnx" in specs

    calls.clear()
    assert mod._pip_install(tmp_path, upstream=True) == 0
    assert calls[0][-2:] == ["-r", str(tmp_path / "requirements.txt")]


def test_the_installer_script_checks_without_changing_anything(tmp_path):
    import subprocess
    import sys

    script = kk.INSTALL_SCRIPT
    r = subprocess.run(
        [sys.executable, str(script), "--check", "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 4                 # no checkout at that root
    assert "checkout" in r.stdout and "missing" in r.stdout


def test_the_installer_refuses_a_folder_that_is_not_a_checkout(tmp_path):
    import subprocess
    import sys

    occupied = tmp_path / "kokoclone"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("mine", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(kk.INSTALL_SCRIPT), "--install", "--root", str(occupied)],
        capture_output=True, text=True, timeout=120,
    )
    # It never clones into, or deletes, a folder it did not create.
    assert r.returncode == 8
    assert "not a KokoClone checkout" in r.stdout
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_resolve_ref_accepts_a_name_and_an_absolute_path(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    by_name = kk.resolve_ref("Luna")
    by_absolute = kk.resolve_ref(str(by_name))
    assert by_name == by_absolute
    assert by_name.is_file()
    with pytest.raises(kk.KokoCloneError, match="pick a KokoClone voice"):
        kk.resolve_ref("")


def test_deleting_a_voice_removes_the_clip_folder(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    assert kk.remove_voice("Luna") is True
    assert kk.list_voices() == []
    assert kk.remove_voice("Luna") is False
    # A name can never walk out of the clip tree: it is slugged first, so a
    # "../outside" voice lands INSIDE voices/kokoclone and deletes nothing else.
    outside = fake_root.parent / "voices" / "outside"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    assert kk.voice_dir("../outside").parent == kk.voices_dir()
    assert kk.remove_voice("../outside") is False
    assert (outside / "keep.txt").is_file()


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------

def test_the_worker_synthesises_through_the_json_protocol(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    pcm = kk.synthesize_pcm("hello there", reference="Luna", lang="pt", timeout=60)
    assert len(pcm) == 12000 * 2            # 0.5 s of PCM16, straight from the fake

    status = kk.worker_status()
    assert status["running"] is True
    assert status["ready"] is True
    # The library's own prints are on stderr, never in the protocol stream.
    assert "generating" in status["log"]
    assert kk.stop_worker() is True
    assert kk.worker_status()["running"] is False


def test_a_failing_job_reports_the_library_error(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    with pytest.raises(kk.KokoCloneError, match="kanade exploded"):
        kk.synthesize_pcm("boom", reference="Luna", lang="pt", timeout=60)


def test_a_missing_reference_clip_is_reported(fake_root):
    with pytest.raises(kk.KokoCloneError, match="reference clip is missing"):
        kk.synthesize_pcm("hi", reference="Luna", lang="pt", timeout=60)


def test_an_unknown_language_never_reaches_the_worker(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    with pytest.raises(kk.KokoCloneError, match="unknown KokoClone language"):
        kk.synthesize_pcm("hi", reference="Luna", lang="xx", timeout=60)


def test_a_reference_file_over_the_cap_is_refused(fake_root, monkeypatch):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    monkeypatch.setattr(kk, "MAX_REF_BYTES", 16)      # the clip on disk is bigger
    with pytest.raises(kk.KokoCloneError, match="MB — keep it under"):
        kk.synthesize_pcm("hi", reference="Luna", lang="pt", timeout=60)


def test_the_worker_never_inherits_credentials(fake_root, monkeypatch):
    """The checkout is third-party code: it must not see Wallie's keys."""
    monkeypatch.setenv("WALLIE_TEST_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("DASHBOARD_PIN", "4242")
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    kk.synthesize_pcm("hi", reference="Luna", lang="pt", timeout=60)
    log = kk.worker_status()["log"]
    assert "env api_key=<none>" in log
    assert "env pin=<none>" in log
    assert "sk-must-not-leak" not in log


def test_the_worker_env_keeps_paths_and_drops_every_secret(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("TWITCH_OAUTH_TOKEN", "t")
    monkeypatch.setenv("LIVEPIX_CLIENT_SECRET", "s")
    monkeypatch.setenv("SOME_PASSWORD", "p")
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.setenv("DASHBOARD_PIN", "1234")
    env = kk._worker_env()
    assert not {"ELEVENLABS_API_KEY", "TWITCH_OAUTH_TOKEN", "LIVEPIX_CLIENT_SECRET",
                "SOME_PASSWORD", "HF_TOKEN", "DASHBOARD_PIN"} & set(env)
    assert env.get("PATH") == __import__("os").environ.get("PATH")
    assert env["PYTHONIOENCODING"] == "utf-8"


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------

def test_the_provider_streams_pcm(fake_root):
    kk.store_voice("Luna", _wav(4.0), lang="pt")
    provider = kk.KokoCloneTTS(reference="Luna", lang="pt")

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in provider.synthesize("hello")])

    assert asyncio.run(collect()) == b"\x01\x00" * 12000
    assert provider.sample_rate == 24000 and provider.channels == 1


def test_the_provider_refuses_a_missing_reference(fake_root):
    with pytest.raises(TTSError, match="kokoclone"):
        kk.KokoCloneTTS(reference="", lang="pt")
    with pytest.raises(TTSError, match="unknown language"):
        kk.KokoCloneTTS(reference="Luna", lang="xx")


def test_the_factory_builds_the_provider(fake_root):
    from tts.factory import build_tts

    kk.store_voice("Luna", _wav(4.0), lang="pt")
    tts = build_tts(
        TTSConfig(provider="kokoclone", kokoclone_ref="Luna", kokoclone_lang="pt"),
        Secrets(),
    )
    assert isinstance(tts, kk.KokoCloneTTS)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _client(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "profiles" / "state.json")
    from config import AppConfig, Runtime, activate_profile, load_profile, save_profile
    (tmp_path / "profiles").mkdir(exist_ok=True)
    save_profile(AppConfig(profile_name="koko-p"), "koko-p")
    activate_profile("koko-p")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile("koko-p"), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    app = _build_app(DashboardState(), None)
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_dashboard_routes_create_and_delete_a_voice(tmp_path, monkeypatch, fake_root):
    import base64

    monkeypatch.setattr(kk, "REQUIRED_PACKAGES", ())
    sample = base64.b64encode(_wav(4.0)).decode("ascii")

    with _client(tmp_path, monkeypatch) as client:
        panel = client.get("/api/voices/kokoclone").json()
        assert panel["ok"] and panel["koko_presets"] == [] and panel["clips"] == []
        assert panel["installed"] is True

        r = client.post("/api/voices/kokoclone", json={
            "name": "Luna Clone", "lang": "pt",
            "sample": {"filename": "take.wav", "data": sample},
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["clip"]["slug"] == "luna_clone"
        assert body["preset"]["provider"] == "kokoclone"
        assert body["preset"]["tts"]["kokoclone_lang"] == "pt"
        assert body["preset"]["tts"]["kokoclone_ref"].endswith("ref.wav")
        # The WHOLE library comes back as presets — the panel reads koko_presets.
        assert [p["name"] for p in body["presets"]] == ["Luna Clone"]
        assert [p["name"] for p in body["koko_presets"]] == ["Luna Clone"]

        r = client.delete("/api/voices/kokoclone/Luna Clone")
        assert r.status_code == 200, r.text
        assert r.json()["koko_presets"] == []
        assert r.json()["clips"] == []
        assert r.json()["clip_removed"] is True
        assert client.delete("/api/voices/kokoclone/Luna Clone").status_code == 404


def _installable(monkeypatch, argv):
    """The probe says "a button may run" and the installer is a fast fake."""
    monkeypatch.setattr(kk, "detect", lambda: {
        "installed": False, "can_install": True, "checkout": True, "git": True,
        "hint": "press install", "root": str(kk.root_dir()),
    })
    monkeypatch.setattr(kk, "install_argv", lambda root=None: argv)


def test_dashboard_installs_kokoclone_in_the_background(tmp_path, monkeypatch, fake_root):
    import sys
    import time as _t

    argv = [sys.executable, "-c", "print('cloning… (50%)'); print('done')"]
    _installable(monkeypatch, argv)

    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/api/voices/kokoclone/install")
        assert r.status_code == 200, r.text
        assert r.json()["started"] is True

        status = {}
        deadline = _t.time() + 30
        while _t.time() < deadline:
            status = client.get("/api/voices/kokoclone/install").json()["install"]
            if not status["running"]:
                break
            _t.sleep(0.1)
        assert status["running"] is False
        assert status["returncode"] == 0
        assert "done" in status["log"]


def test_dashboard_refuses_a_second_install_or_an_impossible_one(
    tmp_path, monkeypatch, fake_root,
):
    import sys

    _installable(monkeypatch, [sys.executable, "-c", "print('slow')"])
    with _client(tmp_path, monkeypatch) as client:
        # Nothing to install: the same call is a conflict, not a second run.
        monkeypatch.setattr(kk, "detect", lambda: {
            "installed": True, "can_install": True, "checkout": True, "git": True,
            "hint": "", "root": str(kk.root_dir()),
        })
        assert client.post("/api/voices/kokoclone/install").status_code == 409

        monkeypatch.setattr(kk, "detect", lambda: {
            "installed": False, "can_install": False, "checkout": False, "git": False,
            "hint": "needs Python 3.10–3.12", "root": str(kk.root_dir()),
        })
        r = client.post("/api/voices/kokoclone/install")
        assert r.status_code == 400
        assert "needs Python" in r.json()["detail"]


def test_dashboard_rejects_an_oversized_reference(tmp_path, monkeypatch, fake_root):
    monkeypatch.setattr(kk, "REQUIRED_PACKAGES", ())
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/api/voices/kokoclone", json={
            "name": "Huge", "lang": "pt",
            "sample": {"filename": "x.wav", "data": "A" * (kk.MAX_REF_BYTES * 2)},
        })
        assert r.status_code == 400
        assert "too large" in r.json()["detail"]


def test_dashboard_rejects_a_short_reference(tmp_path, monkeypatch, fake_root):
    import base64

    monkeypatch.setattr(kk, "REQUIRED_PACKAGES", ())
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/api/voices/kokoclone", json={
            "name": "Too Short", "lang": "pt",
            "sample": {"filename": "x.wav",
                       "data": base64.b64encode(_wav(0.8)).decode("ascii")},
        })
        assert r.status_code == 400
        assert "needs at least" in r.json()["detail"]
        assert client.get("/api/voices/kokoclone").json()["koko_presets"] == []

"""Hearing-test endpoint (POST /api/test/hearing).

The Test hearing buttons record 5 seconds from an audio input (system loopback
or the default microphone) and transcribe them with the CONFIGURED STT engine —
same provider-block resolution as the real build — and report the transcript,
level and duration, or the real reason on failure.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from config import AppConfig, Runtime, Secrets


def _profiled_app(tmp_path, monkeypatch, cfg, orch=None):
    """Isolated profiles dir + app. get_runtime is patched for the hearing route."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(cfg, "p-htest")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile("p-htest"), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    return _build_app(state, orch)


def _patch_mic(monkeypatch, frames) -> list[float]:
    """Install a fake `soundcard` module (the route imports it lazily inside
    _record_mic); returns the list of recorder open/close events."""
    import sys
    import types
    events: list[str] = []

    class _Rec:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            events.append("open")
            return self

        def __exit__(self, *a):
            events.append("close")
            return False

        def record(self, numframes):
            if frames:
                return frames.pop(0)
            return np.zeros((numframes, 1), dtype="float32")

    class _Mic:
        def recorder(self, **kw):
            return _Rec()

    fake = types.ModuleType("soundcard")
    fake.default_microphone = lambda: _Mic()
    monkeypatch.setitem(sys.modules, "soundcard", fake)
    return events


def _patch_capture(monkeypatch, latest_audio):
    """Patch SystemAudioCapture at its source module (the route imports it
    lazily from hearing.capture)."""
    import hearing.capture as cap_mod

    created: list = []

    class _FakeCap:
        def __init__(self, samplerate=16000):
            self.samplerate = samplerate
            self.opened = False
            self.closed = False
            created.append(self)

        def open(self):
            self.opened = True

        def close(self):
            self.closed = True

        def latest(self, seconds):
            return latest_audio

    monkeypatch.setattr(cap_mod, "SystemAudioCapture", _FakeCap)
    return created


def _patch_engine(monkeypatch, result_text="hello world", fail: Exception | None = None):
    """Patch RemoteWhisperModel at its source module (the route imports it
    lazily from hearing.openai_stt)."""
    import hearing.openai_stt as stt_mod

    created: dict = {}

    class _FakeEngine:
        def __init__(self, **kw):
            created.update(kw)
            created["api_key"] = kw.get("api_key")

        def transcribe(self, audio):
            if fail is not None:
                raise fail
            class _Seg:
                text = result_text

            return [_Seg()], {"language": "en"}

    monkeypatch.setattr(stt_mod, "RemoteWhisperModel", _FakeEngine)
    return created


def _stt_block_cfg() -> AppConfig:
    """STT via a REMOTE provider block. The key comes from the block env
    (PROVIDER_<ID>_API_KEY — same env the build uses), so a helper sets it."""
    return AppConfig(
        profile_name="p-htest",
        providers=[{"id": "sttvoice", "name": "STT GW", "category": "stt",
                    "base_url": "https://stt.example/v1", "model": "whisper-large-v3"}],
        hearing={"enabled": True, "engine": "openai_compatible", "provider_ref": "sttvoice"},
    )


def _block_key(monkeypatch, value="stt-block-key") -> None:
    """Give the stt block its API key via the env the build actually reads."""
    monkeypatch.setenv("PROVIDER_STTVOICE_API_KEY", value)


# ---------------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------------

def test_hearing_route_mic_transcribes_and_reports(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())
    _block_key(monkeypatch)
    created = _patch_engine(monkeypatch, result_text="test transcript")
    # ~1s of non-silent frames keeps the recorder realistic; the route checks RMS.
    frames = [np.full((16000, 1), 0.1, dtype="float32")] * 4
    events = _patch_mic(monkeypatch, frames)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 5, "source": "mic"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["ok"] is True
        assert data["text"] == "test transcript"
        assert data["source"] == "mic"
        assert data["rms"] > 0
        assert "5 chars" in data["note"]
    assert events == ["open", "close"]
    # The block's endpoint/model/key reached the engine exactly as the build wires it.
    assert    created["base_url"] == "https://stt.example/v1"
    assert created["model"] == "whisper-large-v3"
    # Also check the engine saw the key from the block env.
    assert created.get("api_key") == "stt-block-key"


def test_hearing_route_system_source_uses_loopback_capture(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())
    _block_key(monkeypatch)
    _patch_engine(monkeypatch, result_text="loopback line")
    # 5s at 16 kHz of quiet-but-not-silent audio (RMS 0.01 ≥ 1e-4).
    audio = np.full(80000, 0.01, dtype="float32")
    caps = _patch_capture(monkeypatch, audio)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 2, "source": "system"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["ok"] is True
        assert data["text"] == "loopback line"
        assert data["source"] == "system"
        assert caps and caps[0].opened and caps[0].closed


def test_hearing_route_local_engine_path(tmp_path, monkeypatch):
    cfg = AppConfig(profile_name="p-htest", hearing={"enabled": True, "engine": ""})
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    frames = [np.full((16000, 1), 0.2, dtype="float32")] * 4
    _patch_mic(monkeypatch, frames)

    # The route builds a HearingLoop for the local engine — patch it at the
    # source module (avoiding the real faster-whisper + soundcard deps).
    import hearing.hearing_loop as hl

    class _FakeModel:
        def transcribe(self, audio, **kw):
            class _Seg:
                text = " local engine line "

            return [_Seg()], None

    class _FakeLoop:
        def __init__(self, *a, **kw):
            pass

        def _load_model(self):
            return _FakeModel()

        def _transcribe(self, audio):
            return " local engine line "

    monkeypatch.setattr(hl, "HearingLoop", _FakeLoop)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 3, "source": "mic"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["ok"] is True
        assert data["text"] == "local engine line"


# ---------------------------------------------------------------------------
# failure paths — actionable reasons
# ---------------------------------------------------------------------------

def test_hearing_route_engine_error_surfaces_reason(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())
    _block_key(monkeypatch)
    _patch_engine(monkeypatch, fail=RuntimeError("openai_compatible (stt) 401: bad key"))
    frames = [np.full((16000, 1), 0.1, dtype="float32")] * 4
    _patch_mic(monkeypatch, frames)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 5, "source": "mic"})
        assert r.status_code == 400
        assert "401: bad key" in r.json()["detail"]


def test_hearing_route_remote_without_key_is_actionable(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())
    _patch_engine(monkeypatch)  # must never be reached
    monkeypatch.delenv("PROVIDER_STTVOICE_API_KEY", raising=False)
    frames = [np.full((16000, 1), 0.1, dtype="float32")] * 4
    _patch_mic(monkeypatch, frames)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 5, "source": "mic"})
        assert r.status_code == 400
        # _compat_api_key's actionable phrasing, surfaced via the route.
        assert "need an API key" in r.json()["detail"]


def test_hearing_route_silence_is_reported_not_transcribed(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())
    _patch_engine(monkeypatch)  # must never be reached
    frames = [np.zeros((16000, 1), dtype="float32")] * 4
    _patch_mic(monkeypatch, frames)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 5, "source": "mic"})
        assert r.status_code == 400
        assert "silence" in r.json()["detail"]


def test_hearing_route_device_error_is_actionable(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())
    _patch_engine(monkeypatch)
    # No fake soundcard module installed → the lazy import fails with an
    # actionable message (this venv has no soundcard; message names it).

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 5, "source": "mic"})
        assert r.status_code == 400
        assert "soundcard" in r.json()["detail"]


def test_hearing_route_unknown_source_rejected(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _stt_block_cfg())

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/hearing", json={"seconds": 5, "source": "radio"})
        assert r.status_code == 400
        assert "system" in r.json()["detail"] and "mic" in r.json()["detail"]


# ---------------------------------------------------------------------------
# resolution helper
# ---------------------------------------------------------------------------

def test_effective_stt_applies_block(tmp_path, monkeypatch):
    from config import Runtime, Secrets
    from wallie import _effective_stt

    monkeypatch.setenv("PROVIDER_STTVOICE_API_KEY", "sk-block")
    cfg = _stt_block_cfg()
    rt = Runtime(config=cfg, secrets=Secrets())
    hearing, sec = _effective_stt(rt)
    assert hearing.openai_compatible_base_url == "https://stt.example/v1"
    assert hearing.openai_compatible_model == "whisper-large-v3"
    assert sec.openai_compatible_stt_api_key == "sk-block"


def test_effective_stt_passthrough_for_local_engine(tmp_path, monkeypatch):
    from config import Runtime, Secrets
    from wallie import _effective_stt

    cfg = AppConfig(profile_name="p-htest", hearing={"enabled": True, "engine": ""})
    rt = Runtime(config=cfg, secrets=Secrets())
    hearing, sec = _effective_stt(rt)
    assert hearing.engine == ""
    assert sec.openai_compatible_stt_api_key == ""


def test_effective_stt_localhost_block_gets_placeholder_key(tmp_path, monkeypatch):
    from config import Runtime, Secrets
    from wallie import _effective_stt

    cfg = AppConfig(
        profile_name="p-htest",
        providers=[{"id": "sttloc", "name": "Local STT", "category": "stt",
                    "base_url": "http://localhost:8000/v1", "model": "whisper-1"}],
        hearing={"enabled": True, "engine": "openai_compatible", "provider_ref": "sttloc"},
    )
    rt = Runtime(config=cfg, secrets=Secrets())
    hearing, sec = _effective_stt(rt)
    assert sec.openai_compatible_stt_api_key == "local-no-key"

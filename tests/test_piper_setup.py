"""Piper local TTS — the voice-file scanner, the installer/downloader argv
builders, and the status / install / download endpoints.

Piper is the optional CPU engine; these tests pin the pieces the Voice-tab
badge depends on: a cheap detector that never imports the native runtime, a
strict voice-name validator (so a crafted name can never become a subprocess
flag), and the background job bookkeeping.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets


def _app(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(AppConfig(profile_name="p-piper"), "p-piper")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile("p-piper"), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    return _build_app(state, None), state


# ─────────────────────────────────────────────────────────────────────
# tts.piper_setup
# ─────────────────────────────────────────────────────────────────────

def test_detect_when_piper_absent(monkeypatch):
    from tts import piper_setup as k
    monkeypatch.setattr(k, "_importable", lambda module: False)
    monkeypatch.setattr(k, "list_voices", lambda directory=None: [])
    d = k.detect()
    assert d["installed"] is False
    assert d["voice_count"] == 0
    assert d["installer_present"] is True
    assert d["can_install"] is True


def test_detect_never_imports_the_native_runtime(monkeypatch):
    """Detection must stay import-free — loading piper is slow and can fail."""
    import builtins
    from tts import piper_setup as k
    real_import = builtins.__import__

    def _guard(name, *a, **kw):
        assert name != "piper", "detect imported the piper runtime"
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _guard)
    k.detect()  # must not raise


def test_list_voices_reads_config_metadata(tmp_path):
    from tts import piper_setup as k

    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"\x00" * 100)
    (tmp_path / "en_US-amy-medium.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}, "language": {"code": "en-US"}}),
        encoding="utf-8",
    )
    # A bare .onnx with no config is listed but flagged.
    (tmp_path / "pt_BR-faber-medium.onnx").write_bytes(b"\x00" * 50)

    voices = k.list_voices(tmp_path)
    assert [v["name"] for v in voices] == ["en_US-amy-medium", "pt_BR-faber-medium"]
    amy = voices[0]
    assert amy["has_config"] is True
    assert amy["sample_rate"] == 22050
    assert amy["language"] == "en-US"
    assert amy["quality"] == "medium"
    assert voices[1]["has_config"] is False
    assert voices[1]["sample_rate"] is None


def test_list_voices_missing_dir_is_empty(tmp_path):
    from tts import piper_setup as k
    assert k.list_voices(tmp_path / "nope") == []


def test_install_argv_installs_piper_tts():
    from tts import piper_setup as k
    argv = k.install_argv()
    assert argv[1:3] == ["-m", "pip"]
    assert argv[3] == "install"
    assert argv[4] == "piper-tts"


def test_download_argv_validates_and_targets_the_script():
    from tts import piper_setup as k
    argv = k.download_argv("en_US-amy-medium")
    assert argv[1].endswith("download_piper_voice.py")
    assert argv[2:] == ["en_US-amy-medium"]
    assert k.download_argv("pt_BR-faber-medium", "voices")[2:] == [
        "pt_BR-faber-medium", "--dest", "voices",
    ]


def test_download_argv_rejects_malformed_and_flag_like_names():
    from tts import piper_setup as k
    for bad in ("", "amy", "en_US-amy", "--dest", "-x-y-z", "en_US-amy-medium extra"):
        assert k.is_valid_voice_name(bad) is False
        with pytest.raises(ValueError):
            k.download_argv(bad)


# ─────────────────────────────────────────────────────────────────────
# factory wiring — the tuning knobs reach the provider
# ─────────────────────────────────────────────────────────────────────

def test_factory_passes_piper_tuning(monkeypatch):
    import tts.piper as piper_mod

    seen: dict = {}

    class _StubPiper:
        name = "piper"
        sample_rate = 22050
        channels = 1

        def __init__(self, *, model_path, length_scale, noise_scale, noise_w):
            seen.update(model_path=model_path, length_scale=length_scale,
                        noise_scale=noise_scale, noise_w=noise_w)

    monkeypatch.setattr(piper_mod, "PiperTTS", _StubPiper)
    from config import TTSConfig
    from tts import build_tts

    build_tts(TTSConfig(
        provider="piper", piper_model_path="voices/x.onnx",
        piper_length_scale=1.2, piper_noise_scale=0.8, piper_noise_w=0.9,
    ), Secrets())
    assert seen == {
        "model_path": "voices/x.onnx", "length_scale": 1.2,
        "noise_scale": 0.8, "noise_w": 0.9,
    }
    # Defaults stay Piper's own so untouched profiles sound identical.
    cfg = TTSConfig()
    assert cfg.piper_noise_scale == 0.667 and cfg.piper_noise_w == 0.8


# ─────────────────────────────────────────────────────────────────────
# endpoints
# ─────────────────────────────────────────────────────────────────────

def test_status_endpoint(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.get("/api/tts/piper/status")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert "installed" in d and isinstance(d["voices"], list)
        assert d["job"]["running"] is False


def _fake_proc(output: bytes, returncode: int = 0):
    """A subprocess stand-in whose stdout is a real StreamReader (the job now
    streams it, so ``communicate`` alone is not enough)."""
    async def _exec(*argv, **kw):
        reader = asyncio.StreamReader()
        reader.feed_data(output)
        reader.feed_eof()

        class _Proc:
            stdout = reader

            async def wait(self):
                return returncode

            def kill(self):
                pass

        _Proc.returncode = returncode
        return _Proc()
    return _exec


def test_install_endpoint_runs_and_records(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    seen: dict = {}
    base = _fake_proc(b"ok: installed piper")

    async def _fake_exec(*argv, **kw):
        seen["argv"] = list(argv)
        seen["cwd"] = kw.get("cwd")
        return await base()

    monkeypatch.setattr("dashboard.server.asyncio.create_subprocess_exec", _fake_exec)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/piper/install")
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "install"
        for _ in range(200):
            if not state.piper_job["running"]:
                break
            time.sleep(0.02)

    assert state.piper_job["running"] is False
    assert state.piper_job["returncode"] == 0, state.piper_job["log"]
    assert "installed piper" in state.piper_job["log"]
    assert seen["argv"][1:3] == ["-m", "pip"]
    assert seen["argv"][3:] == ["install", "piper-tts"]


def test_download_endpoint_starts_the_helper(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    seen: dict = {}
    base = _fake_proc(b"saved voices/en_US-amy-medium.onnx")

    async def _fake_exec(*argv, **kw):
        seen["argv"] = list(argv)
        return await base()

    monkeypatch.setattr("dashboard.server.asyncio.create_subprocess_exec", _fake_exec)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/piper/download", json={"voice": "en_US-amy-medium"})
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "download"
        for _ in range(200):
            if not state.piper_job["running"]:
                break
            time.sleep(0.02)

    assert seen["argv"][1].endswith("download_piper_voice.py")
    assert seen["argv"][2] == "en_US-amy-medium"
    assert state.piper_job["kind"] == "download"


def test_download_endpoint_rejects_bad_voice_name(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/piper/download", json={"voice": "--dest"})
        assert r.status_code == 400
        assert "voice name" in r.json()["detail"]


def test_jobs_refuse_to_overlap(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    state.piper_job["running"] = True
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        assert client.post("/api/tts/piper/install").status_code == 409
        assert client.post(
            "/api/tts/piper/download", json={"voice": "en_US-amy-medium"}
        ).status_code == 409


def test_download_endpoint_reports_missing_script(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    monkeypatch.setattr("tts.piper_setup.INSTALL_SCRIPT", Path("/nope/download_piper_voice.py"))
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/piper/download", json={"voice": "en_US-amy-medium"})
        assert r.status_code == 400
        assert "helper script missing" in r.json()["detail"]


# ─────────────────────────────────────────────────────────────────────
# Online catalogue + live download progress
# ─────────────────────────────────────────────────────────────────────

def test_normalize_catalog_sorts_and_sums_model_files():
    from tts import piper_setup as k
    raw = {
        "en_US-amy-medium": {
            "key": "en_US-amy-medium",
            "language": {"code": "en_US", "name_english": "English"},
            "quality": "medium", "num_speakers": 1,
            "files": {"a.onnx": {"size_bytes": 100}, "a.onnx.json": {"size_bytes": 5},
                      "MODEL_CARD": {"size_bytes": 999}},
        },
        "ar_JO-kareem-low": {
            "key": "ar_JO-kareem-low",
            "language": {"code": "ar_JO", "name_english": "Arabic"},
            "quality": "low", "num_speakers": 1,
            "files": {"b.onnx": {"size_bytes": 50}},
        },
        "junk": "not a dict",
    }
    voices = k.normalize_catalog(raw)
    assert [v["name"] for v in voices] == ["ar_JO-kareem-low", "en_US-amy-medium"]
    assert voices[1]["size_bytes"] == 105          # MODEL_CARD excluded
    assert voices[1]["language_name"] == "English"
    assert voices[1]["quality"] == "medium"


def test_fetch_catalog_parses_and_caches(monkeypatch):
    import httpx
    from tts import piper_setup as k

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={
            "en_US-amy-medium": {
                "key": "en_US-amy-medium", "language": {"code": "en_US"},
                "quality": "medium", "files": {"a.onnx": {"size_bytes": 7}},
            },
        })

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler)))
    k._catalog_cache.update(at=0.0, data=None)

    first = asyncio.run(k.fetch_catalog(force=True))
    assert first[0]["name"] == "en_US-amy-medium" and first[0]["size_bytes"] == 7
    # A second call inside the TTL is served from memory — no new request.
    second = asyncio.run(k.fetch_catalog())
    assert second == first and len(calls) == 1
    assert k.CATALOG_URL in calls[0]


def test_fetch_catalog_network_error_is_actionable(monkeypatch):
    import httpx
    from tts import piper_setup as k

    real = httpx.AsyncClient

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(boom)))
    k._catalog_cache.update(at=0.0, data=None)
    with pytest.raises(k.PiperCatalogError):
        asyncio.run(k.fetch_catalog(force=True))


def test_last_percent_parser():
    from dashboard.server import _last_percent
    assert _last_percent("  3.1 / 63.2 MB  (5%)") == 5
    assert _last_percent("(10%)\n(42%)") == 42      # last wins
    assert _last_percent("no percent here") is None
    assert _last_percent("(999%)") == 100           # clamped


def test_catalog_endpoint_lists_and_reports_failure(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    import tts.piper_setup as k

    async def _stub(force: bool = False, **kw):
        return [{"name": "pt_BR-faber-medium", "language": "pt_BR",
                 "language_name": "Portuguese", "quality": "medium",
                 "speakers": 1, "size_bytes": 12}]

    monkeypatch.setattr(k, "fetch_catalog", _stub)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.get("/api/tts/piper/catalog")
        assert r.status_code == 200, r.text
        assert r.json()["count"] == 1
        assert r.json()["voices"][0]["name"] == "pt_BR-faber-medium"

        async def _boom(force: bool = False, **kw):
            raise k.PiperCatalogError("the catalogue is offline")

        monkeypatch.setattr(k, "fetch_catalog", _boom)
        r = client.get("/api/tts/piper/catalog")
        assert r.status_code == 400
        assert "offline" in r.json()["detail"]


def test_download_job_streams_progress(tmp_path, monkeypatch):
    """The job must surface the downloader's last ``(NN%)`` while it runs, not
    only at the end. A non-zero exit keeps that last value (success forces 100)."""
    app, state = _app(tmp_path, monkeypatch)

    async def _fake_exec(*argv, **kw):
        reader = asyncio.StreamReader()
        reader.feed_data(b"downloading\n  1.0 / 63.2 MB (12%)\n")
        reader.feed_data(b"  8.0 / 63.2 MB (37%)\n")
        reader.feed_eof()

        class _Proc:
            stdout = reader
            returncode = 1

            async def wait(self):
                return 1

            def kill(self):
                pass

        return _Proc()

    monkeypatch.setattr("dashboard.server.asyncio.create_subprocess_exec", _fake_exec)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/piper/download", json={"voice": "en_US-amy-medium"})
        assert r.status_code == 200, r.text
        for _ in range(200):
            if not state.piper_job["running"]:
                break
            time.sleep(0.02)

    assert state.piper_job["running"] is False
    assert state.piper_job["progress"] == 37
    assert state.piper_job["returncode"] == 1

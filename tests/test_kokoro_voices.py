"""Kokoro ready-made voices: catalogue, download command and the local store.

Nothing is bundled with the project — the Voice tab lists the voices Kokoro
publishes and downloads only the ones the user picks into ``voices/kokoro/``.
These tests cover the cheap parts (id validation, catalogue normalisation, argv
building, the endpoints) plus the provider's pipeline cache and streaming, all
without importing torch or touching the network.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from config import AppConfig, Runtime, Secrets
from tts import kokoro as kp
from tts import kokoro_setup as k


# ---------------------------------------------------------------------------
# Voice id validation + download argv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["af_heart", "pf_dora", "zf_xiaobei", "am_adam", "bf_alice"])
def test_valid_voice_names(name):
    assert k.is_valid_voice_name(name)


@pytest.mark.parametrize("name", ["", "--install", "af", "af-heart", "AF_Heart", "-x_y"])
def test_invalid_voice_names(name):
    assert not k.is_valid_voice_name(name)
    with pytest.raises(ValueError):
        k.download_argv(name)


def test_download_argv_targets_the_helper_and_dest():
    argv = k.download_argv("af_heart")
    assert argv[1].endswith("download_kokoro_voice.py")
    assert argv[2] == "af_heart"
    assert k.download_argv("pf_dora", "/tmp/v")[2:] == ["pf_dora", "--dest", "/tmp/v"]


# ---------------------------------------------------------------------------
# Catalogue normalisation
# ---------------------------------------------------------------------------

def test_catalog_voices_normalizes_and_sorts():
    raw = [
        {"path": "voices/pf_dora.pt", "size": 100},
        {"path": "voices/af_heart.pt", "size": 200},
        {"path": "voices/README.md", "size": 1},
        {"path": "voices/BAD NAME.pt", "size": 3},
        {"not": "a path"},
        "nope",
    ]
    out = k.catalog_voices(raw)
    assert [v["name"] for v in out] == ["af_heart", "pf_dora"]
    assert out[0]["size_bytes"] == 200
    assert k.catalog_voices(None) == []


def test_fetch_catalog_uses_the_client_and_caches(monkeypatch):
    monkeypatch.setattr(k, "_catalog_cache", {"at": 0.0, "data": None})
    calls = {"n": 0}

    class _Resp:
        status_code = 200

        def json(self):
            return [{"path": "voices/af_heart.pt", "size": 10}]

    class _Client:
        async def get(self, url):
            calls["n"] += 1
            return _Resp()

        async def aclose(self):
            pass

    voices = asyncio.run(k.fetch_catalog(client=_Client()))
    assert [v["name"] for v in voices] == ["af_heart"]
    again = asyncio.run(k.fetch_catalog(client=_Client()))
    assert again == voices
    assert calls["n"] == 1   # second call served from the in-process cache


def test_fetch_catalog_reports_network_errors(monkeypatch):
    monkeypatch.setattr(k, "_catalog_cache", {"at": 0.0, "data": None})
    import httpx

    class _Client:
        async def get(self, url):
            raise httpx.ConnectError("boom")

        async def aclose(self):
            pass

    with pytest.raises(k.KokoroCatalogError):
        asyncio.run(k.fetch_catalog(client=_Client()))


# ---------------------------------------------------------------------------
# Local store — voices/kokoro/
# ---------------------------------------------------------------------------

def test_list_local_voices_reads_only_pt_files(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path)
    assert k.list_local_voices() == []
    (tmp_path / "af_heart.pt").write_bytes(b"x" * 7)
    (tmp_path / "notes.txt").write_text("ignore me")
    out = k.list_local_voices()
    assert [v["name"] for v in out] == ["af_heart"]
    assert out[0]["size_bytes"] == 7
    assert k.voice_path("af_heart") == tmp_path / "af_heart.pt"


# ---------------------------------------------------------------------------
# KokoroTTS: cached pipeline + streaming
# ---------------------------------------------------------------------------

def _fake_pipeline(chunks):
    def _call(text, voice=None, speed=None):
        for c in chunks:
            yield ("graphemes", "phonemes", c)
    return _call


def test_pipeline_is_cached_per_language(monkeypatch):
    kp.clear_pipeline_cache()
    made: list[dict] = []
    fake = types.ModuleType("kokoro")

    class KPipeline:
        def __init__(self, **kw):
            made.append(kw)

    fake.KPipeline = KPipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake)
    a = kp._get_pipeline("a", "")
    b = kp._get_pipeline("a", "")
    assert a is b
    assert len(made) == 1   # the 300 MB model is loaded once, not per instance
    kp.clear_pipeline_cache()


def test_synthesize_streams_pcm_chunks(monkeypatch):
    kp.clear_pipeline_cache()
    monkeypatch.setattr(kp, "_get_pipeline", lambda lang, device="": _fake_pipeline([
        np.array([0.0, 0.5], dtype="float32"),
        np.array([-0.5], dtype="float32"),
    ]))
    monkeypatch.setattr(kp, "_local_voice_path", lambda voice: "")

    async def _collect():
        tts = kp.KokoroTTS(voice="af_heart")
        return [c async for c in tts.synthesize("hello")]

    chunks = asyncio.run(_collect())
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) for c in chunks)
    assert len(chunks[0]) == 4   # 2 samples * 2 bytes
    kp.clear_pipeline_cache()


def test_synthesize_surfaces_generator_errors(monkeypatch):
    kp.clear_pipeline_cache()

    def _boom(text, voice=None, speed=None):
        yield ("g", "p", np.array([0.1], dtype="float32"))
        raise RuntimeError("bad voice")

    monkeypatch.setattr(kp, "_get_pipeline", lambda lang, device="": _boom)
    monkeypatch.setattr(kp, "_local_voice_path", lambda voice: "")

    async def _drive():
        tts = kp.KokoroTTS()
        got = []
        with pytest.raises(kp.TTSError):
            async for c in tts.synthesize("hi"):
                got.append(c)
        return got

    got = asyncio.run(_drive())
    # The first chunk still streamed before the failure surfaced.
    assert len(got) == 1
    kp.clear_pipeline_cache()


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------

def _app(tmp_path: Path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(AppConfig(profile_name="p-kv"), "p-kv")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile("p-kv"), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    return _build_app(state, None), state


def test_status_reports_downloaded_voices(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    app, _ = _app(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        d = client.get("/api/tts/kokoro/status").json()
    assert d["voice_count"] == 0
    assert d["voices"] == []
    assert d["job"]["running"] is False
    assert d["voices_dir"].endswith("voices")


class _FakeProc:
    returncode = 0

    def __init__(self):
        self.stdout = self

    async def read(self, n):
        return b""

    async def wait(self):
        return 0

    def kill(self):
        pass


def test_download_endpoint_records_argv_and_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    app, state = _app(tmp_path, monkeypatch)
    seen: dict = {}

    async def _fake_exec(*argv, **kw):
        seen["argv"] = list(argv)
        return _FakeProc()

    monkeypatch.setattr("dashboard.server.asyncio.create_subprocess_exec", _fake_exec)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/kokoro/download", json={"voice": "pf_dora"})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "started": True, "voice": "pf_dora"}
        for _ in range(200):
            if not state.kokoro_job["running"]:
                break
            time.sleep(0.02)

    assert state.kokoro_job["running"] is False
    assert state.kokoro_job["voice"] == "pf_dora"
    assert state.kokoro_job["returncode"] == 0
    assert state.kokoro_job["progress"] == 100
    assert seen["argv"][1].endswith("download_kokoro_voice.py")
    assert seen["argv"][2] == "pf_dora"
    assert seen["argv"][3] == "--dest"


def test_download_endpoint_rejects_bad_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    app, _ = _app(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/kokoro/download", json={"voice": "--install"})
    assert r.status_code == 400


def test_download_endpoint_conflicts_when_running(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    app, state = _app(tmp_path, monkeypatch)
    state.kokoro_job["running"] = True
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        assert client.post("/api/tts/kokoro/download", json={"voice": "af_heart"}).status_code == 409


def test_catalog_endpoint_marks_local_voices(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    app, _ = _app(tmp_path, monkeypatch)
    (tmp_path / "voices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "voices" / "af_heart.pt").write_bytes(b"x")

    async def _fake_fetch(*, force=False):
        return [
            {"name": "af_heart", "size_bytes": 10},
            {"name": "pf_dora", "size_bytes": 20},
        ]

    monkeypatch.setattr(k, "fetch_catalog", _fake_fetch)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        d = client.get("/api/tts/kokoro/catalog").json()
    assert d["count"] == 2
    by = {v["name"]: v for v in d["voices"]}
    assert by["af_heart"]["local"] is True
    assert by["pf_dora"]["local"] is False


def test_download_helper_survives_a_cp1252_pipe():
    """The dashboard runs the downloader with stdout as a PIPE; on Windows that
    is cp1252, so the '↓'/'✓' progress glyphs raised UnicodeEncodeError and the
    voice never landed in voices/kokoro/ (the UI looked like it downloaded).
    The helper must force UTF-8 on its own streams."""
    import os
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "download_kokoro_voice.py"
    code = (
        "import importlib.util as u\n"
        f"spec = u.spec_from_file_location('dkv', r'{script}')\n"
        "m = u.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print('\\u2193 \\u2713 \\u00b7')  # the glyphs the progress prints\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "UnicodeEncodeError" not in r.stderr


def test_catalog_endpoint_reports_catalog_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    app, _ = _app(tmp_path, monkeypatch)

    async def _boom(*, force=False):
        raise k.KokoroCatalogError("no network")

    monkeypatch.setattr(k, "fetch_catalog", _boom)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.get("/api/tts/kokoro/catalog")
    assert r.status_code == 400

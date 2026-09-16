"""TTS voice discovery (wallie.fetch_tts_voices + POST /api/tts/voices).

The Voice page's openai_compatible panel asks the configured endpoint for its
available voices so the user picks from a dropdown instead of typing blind.
The fetch must mirror the build's provider resolution exactly (block →
category → legacy fields, localhost-needs-no-key).
"""
from __future__ import annotations

import httpx
import pytest

from config import AppConfig, Runtime


# ---------------------------------------------------------------------------
# Extractor — whatever an endpoint calls "voices" inside /models
# ---------------------------------------------------------------------------

def test_extract_voices_empty_and_non_dict():
    from wallie import _tts_voices_from_models
    assert _tts_voices_from_models({}) == []
    assert _tts_voices_from_models(None) == []
    assert _tts_voices_from_models("nope") == []


def test_extract_voices_dedupes_and_strips():
    from wallie import _tts_voices_from_models
    out = _tts_voices_from_models({"voices": ["af_heart", " af_heart ", "", {"id": "alloy"}, {"name": "bella"}]})
    assert out == ["af_heart", "alloy", "bella"]


def test_extract_voices_filters_non_tts_models():
    from wallie import _tts_voices_from_models
    data = {"data": [
        {"id": "tts-1"},            # a speech MODEL, not a voice
        {"id": "whisper-1"},        # STT
        {"id": "text-embedding-3"}, # embeddings
        {"id": "gpt-4o-mini"},      # chat
        {"id": "af_heart"},         # a voice exposed as a model ✓
        {"id": "kokoro"},           # gateway-style voice model ✓
    ]}
    assert _tts_voices_from_models(data) == ["af_heart", "kokoro"]


# ---------------------------------------------------------------------------
# fetch_tts_voices — resolution + HTTP against a mocked endpoint
# ---------------------------------------------------------------------------

@pytest.fixture()
def tts_profile(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    cfg = AppConfig(profile_name="p-voices", providers=[
        {"id": "ttsvoice", "name": "Voice GW", "category": "tts",
         "base_url": "https://tts.example/v1", "model": "speech-1.5"},
    ])
    save_profile(cfg, "p-voices")
    return load_profile("p-voices")


def _patch_httpx(monkeypatch, handler):
    """Route every AsyncClient through a MockTransport (fetch builds its own client)."""
    real = httpx.AsyncClient

    def factory(**kwargs):
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.mark.anyio
async def test_fetch_voices_via_block(tts_profile, monkeypatch):
    import wallie
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")
    from config import Secrets
    cfg = tts_profile

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer key-tts"
        return httpx.Response(200, json={"data": [{"id": "tts-1"}, {"id": "af_heart"}, {"id": "am_adam"}]})

    _patch_httpx(monkeypatch, handler)
    out = await wallie.fetch_tts_voices(cfg, Secrets())
    assert out["endpoint"] == "https://tts.example/v1"
    assert out["source"] == "provider:ttsvoice"
    assert out["voices"] == ["af_heart", "am_adam"]


@pytest.mark.anyio
async def test_fetch_voices_without_endpoint_raises(monkeypatch):
    import wallie
    from config import Secrets
    monkeypatch.delenv("OPENAI_COMPATIBLE_TTS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no TTS endpoint"):
        await wallie.fetch_tts_voices(AppConfig(), Secrets())


@pytest.mark.anyio
async def test_fetch_voices_remote_without_key_raises(tts_profile, monkeypatch):
    import wallie
    from config import Secrets
    monkeypatch.delenv("PROVIDER_TTSVOICE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="need an API key"):
        await wallie.fetch_tts_voices(tts_profile, Secrets())


@pytest.mark.anyio
async def test_fetch_voices_non_json_endpoint(tts_profile, monkeypatch):
    import wallie
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")
    from config import Secrets
    _patch_httpx(monkeypatch, lambda req: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(RuntimeError, match="non-JSON"):
        await wallie.fetch_tts_voices(tts_profile, Secrets())


# ---------------------------------------------------------------------------
# Dashboard route wiring
# ---------------------------------------------------------------------------

def _profiled_app(tmp_path, monkeypatch, cfg):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(cfg, "p-voices")
    runtime = Runtime(config=load_profile("p-voices"), secrets=__import__("config").Secrets())
    monkeypatch.setattr(config, "get_runtime", lambda: runtime)
    from dashboard.server import DashboardState, _build_app
    return _build_app(DashboardState(), None, pin="")


def test_api_tts_voices_route_ok(tmp_path, monkeypatch):
    import wallie
    cfg = AppConfig(profile_name="p-voices", providers=[
        {"id": "ttsvoice", "name": "Voice GW", "category": "tts",
         "base_url": "https://tts.example/v1", "model": "speech-1.5"},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"voices": [{"id": "alloy"}, {"id": "nova"}]})

    _patch_httpx(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    with TestClient(app) as client:
        r = client.post("/api/tts/voices", json={"provider_ref": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["voices"] == ["alloy", "nova"]
        assert data["source"] == "provider:ttsvoice"


def test_api_tts_voices_route_config_error_is_400(tmp_path, monkeypatch):
    cfg = AppConfig(profile_name="p-voices")  # no blocks, no legacy URL
    monkeypatch.delenv("OPENAI_COMPATIBLE_TTS_API_KEY", raising=False)

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    with TestClient(app) as client:
        r = client.post("/api/tts/voices", json={})
        assert r.status_code == 400
        assert "no TTS endpoint" in r.json()["detail"]

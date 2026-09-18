"""Vision model discovery (wallie.fetch_vision_models + POST /api/vision/models).

The Vision page's dedicated-provider panel asks the configured endpoint for
its available models so the user picks from a dropdown instead of typing
blind. The fetch must mirror the build's vision resolution exactly
(dedicated VISION block → dedicated legacy field → engine's generic URL,
localhost-needs-no-key).
"""
from __future__ import annotations

import httpx
import pytest

from config import AppConfig, Runtime


# ---------------------------------------------------------------------------
# Extractor — drop other subsystems' models, VL-looking ones first
# ---------------------------------------------------------------------------

def test_extract_models_empty_and_non_dict():
    from wallie import _vl_models_from_models
    assert _vl_models_from_models({}) == []
    assert _vl_models_from_models(None) == []
    assert _vl_models_from_models("nope") == []


def test_extract_models_from_bare_list():
    from wallie import _vl_models_from_models
    assert _vl_models_from_models([{"id": "qwen2.5-vl-7b"}]) == ["qwen2.5-vl-7b"]


def test_extract_models_dedupes_and_strips():
    from wallie import _vl_models_from_models
    out = _vl_models_from_models({"data": [" qwen-vl-max ", "qwen-vl-max", "", {"id": "qwen-vl-max"}]})
    assert out == ["qwen-vl-max"]


def test_extract_models_filters_non_vision_models():
    from wallie import _vl_models_from_models
    data = {"data": [
        {"id": "tts-1"},
        {"id": "whisper-1"},
        {"id": "text-embedding-3"},
        {"id": "dall-e-3"},
        {"id": "omni-moderation-latest"},
        {"id": "qwen2.5-vl-72b-instruct"},  # VL ✓
        {"id": "llama-4-scout"},            # VL ✓
        {"id": "gpt-4o"},                   # vision-capable chat ✓
        {"id": "claude-sonnet-4-5"},        # vision-capable chat ✓
    ]}
    assert _vl_models_from_models(data) == [
        "qwen2.5-vl-72b-instruct", "llama-4-scout", "gpt-4o", "claude-sonnet-4-5",
    ]


# ---------------------------------------------------------------------------
# fetch_vision_models — resolution + HTTP against a mocked endpoint
# ---------------------------------------------------------------------------

@pytest.fixture()
def vision_profile(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    cfg = AppConfig(profile_name="p-vlmodels", providers=[
        {"id": "vlblock", "name": "VL GW", "category": "vision",
         "base_url": "https://vl.example/v1", "model": "qwen2.5-vl-7b"},
    ])
    save_profile(cfg, "p-vlmodels")
    return load_profile("p-vlmodels")


def _patch_httpx(monkeypatch, handler):
    """Route every AsyncClient through a MockTransport (fetch builds its own client)."""
    real = httpx.AsyncClient

    def factory(**kwargs):
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.mark.anyio
async def test_fetch_models_via_block(vision_profile, monkeypatch):
    import wallie
    monkeypatch.setenv("PROVIDER_VLBLOCK_API_KEY", "key-vl")
    from config import Secrets
    cfg = vision_profile

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer key-vl"
        return httpx.Response(200, json={"data": [{"id": "qwen2.5-vl-7b"}, {"id": "whisper-1"}]})

    _patch_httpx(monkeypatch, handler)
    out = await wallie.fetch_vision_models(cfg, Secrets())
    assert out["endpoint"] == "https://vl.example/v1"
    assert out["source"] == "provider:vlblock"
    assert out["models"] == ["qwen2.5-vl-7b"]


@pytest.mark.anyio
async def test_fetch_models_fallback_to_legacy_vision_url(monkeypatch):
    import wallie
    from config import AppConfig, Secrets
    monkeypatch.delenv("OPENAI_COMPATIBLE_VISION_API_KEY", raising=False)
    # No VISION block at all → the DEDICATED legacy field wins; localhost needs
    # no key. (A block matching the ref is covered by test_fetch_models_via_block;
    # a non-matching ref falls through to the first category block, exactly as
    # _resolve_provider/the build resolve it.)
    cfg = AppConfig(profile_name="p-vlmodels", llm={
        "vision_provider": "openai_compatible",
        "vision_openai_compatible_base_url": "http://127.0.0.1:9991/v1",
    })

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        return httpx.Response(200, json={"data": [{"id": "qwen-vl-max"}]})

    _patch_httpx(monkeypatch, handler)
    out = await wallie.fetch_vision_models(cfg, Secrets(), ref="missing-block")
    assert out["endpoint"] == "http://127.0.0.1:9991/v1"
    assert out["source"] == "legacy"
    assert out["models"] == ["qwen-vl-max"]


@pytest.mark.anyio
async def test_fetch_models_without_endpoint_raises(monkeypatch):
    import wallie
    from config import Secrets
    monkeypatch.delenv("OPENAI_COMPATIBLE_VISION_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no vision endpoint"):
        await wallie.fetch_vision_models(AppConfig(), Secrets())


@pytest.mark.anyio
async def test_fetch_models_remote_without_key_raises(vision_profile, monkeypatch):
    import wallie
    from config import Secrets
    monkeypatch.delenv("PROVIDER_VLBLOCK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="need an API key"):
        await wallie.fetch_vision_models(vision_profile, Secrets())


@pytest.mark.anyio
async def test_fetch_models_non_json_endpoint(vision_profile, monkeypatch):
    import wallie
    monkeypatch.setenv("PROVIDER_VLBLOCK_API_KEY", "key-vl")
    from config import Secrets
    _patch_httpx(monkeypatch, lambda req: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(RuntimeError, match="non-JSON"):
        await wallie.fetch_vision_models(vision_profile, Secrets())


# ---------------------------------------------------------------------------
# Dashboard route wiring
# ---------------------------------------------------------------------------

def _profiled_app(tmp_path, monkeypatch, cfg):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(cfg, "p-vlmodels")
    runtime = Runtime(config=load_profile("p-vlmodels"), secrets=__import__("config").Secrets())
    monkeypatch.setattr(config, "get_runtime", lambda: runtime)
    from dashboard.server import DashboardState, _build_app
    return _build_app(DashboardState(), None, pin="")


def test_api_vision_models_route_ok(tmp_path, monkeypatch):
    cfg = AppConfig(profile_name="p-vlmodels", providers=[
        {"id": "vlblock", "name": "VL GW", "category": "vision",
         "base_url": "https://vl.example/v1", "model": "qwen2.5-vl-7b"},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "qwen-vl-max"}, {"id": "qwen2.5-vl-72b"}]})

    _patch_httpx(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_VLBLOCK_API_KEY", "key-vl")

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    with TestClient(app) as client:
        r = client.post("/api/vision/models", json={"provider_ref": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["models"] == ["qwen-vl-max", "qwen2.5-vl-72b"]
        assert data["source"] == "provider:vlblock"


def test_api_vision_models_route_config_error_is_400(tmp_path, monkeypatch):
    cfg = AppConfig(profile_name="p-vlmodels")  # no blocks, no legacy URL
    monkeypatch.delenv("OPENAI_COMPATIBLE_VISION_API_KEY", raising=False)

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    with TestClient(app) as client:
        r = client.post("/api/vision/models", json={})
        assert r.status_code == 400
        assert "no vision endpoint" in r.json()["detail"]

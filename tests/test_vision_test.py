"""Vision-test endpoint (POST /api/test/vision).

The test button must exercise the SAME vision provider a live session uses:
a resolved dedicated vision block (API Keys page) when one exists, otherwise
the main engine — and the no-provider gate must say exactly what to fix.
"""
from __future__ import annotations

import numpy as np  # noqa: F401 — keeps parity with sibling route-test modules
import pytest

from config import AppConfig, Runtime, Secrets


def _profiled_app(tmp_path, monkeypatch, cfg):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(cfg, "p-vtest")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile("p-vtest"), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    return _build_app(state, None)


class _Frame:
    jpeg = b"\xff\xd8\xff\xe9fakejpeg"
    width = 64
    height = 32


class _FakeCapture:
    def __init__(self, *a, **kw):
        pass

    def grab(self):
        return _Frame()

    def close(self):
        pass


class _FakeProvider:
    name = "openai_compatible_vision"
    model = "qwen2.5-vl-7b"

    def __init__(self):
        self.streamed_msgs = None
        self.closed = False

    async def stream(self, msgs, **kw):
        self.streamed_msgs = msgs
        yield "screen says hi"

    async def aclose(self):
        self.closed = True


def _vision_block_cfg() -> AppConfig:
    return AppConfig(
        profile_name="p-vtest",
        providers=[{
            "id": "visx", "name": "Local VL", "category": "vision",
            "base_url": "http://127.0.0.1:9991/v1", "model": "qwen2.5-vl-7b",
        }],
        llm={"provider": "groq", "vision_provider": "openai_compatible",
             "vision_provider_ref": "visx"},
        vision={"enabled": True},
    )


def _patch_screen_capture(monkeypatch) -> None:
    import vision
    monkeypatch.setattr(vision, "ScreenCapture", _FakeCapture)


def test_vision_test_route_uses_dedicated_block(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _vision_block_cfg())
    _patch_screen_capture(monkeypatch)

    created: dict = {}

    def _fake_build(vis_cfg, vis_sec):
        created["base_url"] = vis_cfg.vision_openai_compatible_base_url
        created["model"] = vis_cfg.vision_model
        return _FakeProvider()

    monkeypatch.setattr("wallie._build_vision_llm", _fake_build)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision")
        assert r.status_code == 200, r.text
        data = r.json()
        # The DEDICATED provider answered — not the main engine.
        assert data["dedicated"] is True
        assert data["provider"] == "openai_compatible_vision"
        assert data["model"] == "qwen2.5-vl-7b"
        assert data["text"] == "screen says hi"
        assert data["frame_size"] == [64, 32]
    # The block's endpoint/model reached the builder exactly as the build wires it.
    assert created["base_url"] == "http://127.0.0.1:9991/v1"
    assert created["model"] == "qwen2.5-vl-7b"


def test_vision_test_route_without_any_provider_is_actionable(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(
        profile_name="p-vtest",
        llm={"provider": "groq", "vision_capable": False},
        vision={"enabled": True},
    ))
    _patch_screen_capture(monkeypatch)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision")
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "vision_capable" in detail or "VISION block" in detail


def test_vision_test_route_gate_passes_for_capable_engine(tmp_path, monkeypatch):
    """No dedicated block, but the engine is vision-capable → the main LLM is
    used (dedicated=False) instead of a hard 400."""
    cfg = AppConfig(
        profile_name="p-vtest",
        llm={"provider": "groq", "vision_capable": True, "model": "m"},
        vision={"enabled": True},
    )
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    _patch_screen_capture(monkeypatch)

    used: dict = {}

    class _MainProvider(_FakeProvider):
        name = "groq"
        model = "m"

        async def stream(self, msgs, **kw):
            used["main"] = True
            yield "main-llm-answer"

    # The route imports build_provider from llm.factory at call time — patch
    # the SOURCE module (and keep the user's real Groq key out of the test).
    monkeypatch.setattr("llm.factory.build_provider", lambda c, s: _MainProvider())

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["dedicated"] is False
        assert data["provider"] == "groq"
        assert data["text"] == "main-llm-answer"
    assert used.get("main") is True

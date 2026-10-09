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


def _non_capable_cfg() -> AppConfig:
    """Engine marked NOT vision-capable and no dedicated block — the exact shape
    that used to make the test button answer 400 no matter what."""
    return AppConfig(
        profile_name="p-vtest",
        llm={"provider": "gemini", "model": "gemini-2.5-flash", "vision_capable": False},
        vision={"enabled": True},
    )


def test_vision_test_override_probes_another_model_without_saving(tmp_path, monkeypatch):
    """The escape hatch for a retired model: name a provider/model and the test
    builds a throwaway provider — bypassing the vision_capable gate — while the
    saved profile keeps the old model so nothing is committed by accident."""
    app = _profiled_app(tmp_path, monkeypatch, _non_capable_cfg())
    _patch_screen_capture(monkeypatch)

    built: dict = {}

    class _Probe(_FakeProvider):
        name = "gemini"
        model = "gemini-3.8-flash"

        async def stream(self, msgs, **kw):
            built["got_image"] = any(
                isinstance(b, dict) and b.get("type") == "image"
                for m in msgs if isinstance(m.get("content"), list)
                for b in m["content"]
            )
            yield "the replacement works"

    def _fake_build(llm_cfg, secrets):
        built["provider"] = llm_cfg.provider
        built["model"] = llm_cfg.model
        built["vision_capable"] = llm_cfg.vision_capable
        return _Probe()

    monkeypatch.setattr("llm.factory.build_provider", _fake_build)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision", json={
            "provider": "gemini", "model": "gemini-3.8-flash",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["override"] is True
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-3.8-flash"
        assert data["text"] == "the replacement works"

    assert built == {
        "provider": "gemini", "model": "gemini-3.8-flash",
        "vision_capable": True, "got_image": True,
    }

    # The profile still holds the OLD model — a probe never persists itself.
    from config import load_profile
    assert load_profile("p-vtest").llm.model == "gemini-2.5-flash"


def test_vision_test_retired_model_surfaces_as_400(tmp_path, monkeypatch):
    """A model the provider retired is a CONFIG problem: the route must return
    400 with the provider's own message (the one naming the replacement), not a
    bare 500, and must not leak key-shaped strings."""
    app = _profiled_app(tmp_path, monkeypatch, _non_capable_cfg())
    _patch_screen_capture(monkeypatch)

    class _Gone(_FakeProvider):
        name = "gemini"
        model = "gemini-2.5-flash"

        async def stream(self, msgs, **kw):
            raise RuntimeError(
                "404 This model models/gemini-2.5-flash is no longer available to "
                "new users. Please update your code to use models/gemini-3.8-flash "
                "(key sk-abcdefghijklmnopqrstuvwxyz was rejected)"
            )
            yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("llm.factory.build_provider", lambda c, s: _Gone())

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision", json={"provider": "gemini", "model": "gemini-2.5-flash"})
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert detail.startswith("LLM error:")
        assert "gemini-3.8-flash" in detail      # the actionable part survives truncation
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in detail
        assert "[redacted]" in detail


def test_vision_test_override_needs_a_model(tmp_path, monkeypatch):
    cfg = _non_capable_cfg()
    cfg.llm.model = ""
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    _patch_screen_capture(monkeypatch)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision", json={"provider": "gemini"})
        assert r.status_code == 400
        assert "pick a model" in r.json()["detail"]


def test_vision_test_override_unknown_provider_is_actionable(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, _non_capable_cfg())
    _patch_screen_capture(monkeypatch)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision", json={"provider": "nope", "model": "m"})
        assert r.status_code == 400
        assert "could not build nope" in r.json()["detail"]


def test_vision_test_override_openai_compatible_uses_legacy_fields(tmp_path, monkeypatch):
    """openai_compatible probes take the vision legacy endpoint/model — the
    dedicated fields the live vision path reads."""
    cfg = _non_capable_cfg()
    cfg.llm.model = "text-model"
    app = _profiled_app(tmp_path, monkeypatch, cfg)
    _patch_screen_capture(monkeypatch)

    built: dict = {}

    def _fake_build(vis_cfg, vis_sec):
        built["base_url"] = vis_cfg.vision_openai_compatible_base_url
        built["model"] = vis_cfg.vision_model
        return _FakeProvider()

    monkeypatch.setattr("wallie._build_vision_llm", _fake_build)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/test/vision", json={
            "provider": "openai_compatible",
            "model": "qwen2.5-vl-7b",
            "base_url": "http://127.0.0.1:9991/v1",
        })
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "screen says hi"

    assert built == {"base_url": "http://127.0.0.1:9991/v1", "model": "qwen2.5-vl-7b"}


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

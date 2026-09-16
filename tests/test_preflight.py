"""Pre-start checklist (wallie.preflight + GET /api/preflight).

The checklist must mirror what build_orchestrator() will do — same provider
resolution order, same localhost-doesn't-need-a-key rule — so the dashboard
can warn BEFORE the user clicks Start instead of surfacing a traceback after.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets


def _runtime(cfg: AppConfig, tmp_path: Path | None = None) -> Runtime:
    return Runtime(config=cfg, secrets=Secrets(), base_dir=tmp_path or Path("."))


def _issues(cfg: AppConfig, tmp_path: Path | None = None) -> list[dict]:
    import wallie
    return wallie.preflight(_runtime(cfg, tmp_path))


def _sections(issues: list[dict]) -> list[str]:
    return [i["section"] for i in issues]


# ---------------------------------------------------------------------------
# Engine (LLM)
# ---------------------------------------------------------------------------

def test_preflight_clean_config_reports_nothing(tmp_path):
    """A fully-configured config produces zero issues."""
    cfg = AppConfig(
        llm={"provider": "openai_compatible", "provider_ref": "brain",
             "model": "llama3.1"},
        tts={"provider": "openai_compatible", "provider_ref": "voice"},
        providers=[
            {"name": "brain", "category": "llm",
             "base_url": "http://localhost:11434/v1", "model": "llama3.1"},
            {"name": "voice", "category": "tts",
             "base_url": "http://localhost:5000/v1", "model": "tts-1"},
        ],
    )
    assert _issues(cfg, tmp_path) == []


def test_preflight_flags_unconfigured_tts(tmp_path, monkeypatch):
    """The default profile (groq brain, fish TTS, no keys) is two errors."""
    # Hermetic: a real GROQ_API_KEY in the machine's .env would mask the error.
    for env in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    issues = _issues(AppConfig(), tmp_path)
    sections = _sections(issues)
    assert sorted(sections) == ["Engine (LLM)", "Voice (TTS)"]
    assert all(i["level"] == "error" for i in issues)
    assert any("GROQ_API_KEY" in i["message"] for i in issues)
    assert any("FISH_API_KEY" in i["message"] for i in issues)


def test_preflight_piper_missing_vs_bogus_model_path(tmp_path):
    missing = _issues(AppConfig(tts={"provider": "piper", "piper_model_path": ""}), tmp_path)
    tts = [i for i in missing if i["section"] == "Voice (TTS)"]
    assert tts and "no model" in tts[0]["message"]

    bogus = _issues(AppConfig(tts={"provider": "piper", "piper_model_path": "Z:/nope.onnx"}), tmp_path)
    tts = [i for i in bogus if i["section"] == "Voice (TTS)"]
    assert tts and "not found" in tts[0]["message"]

    # A real file passes.
    real = tmp_path / "voice.onnx"
    real.write_bytes(b"x")
    ok = _issues(AppConfig(tts={"provider": "piper", "piper_model_path": str(real)}), tmp_path)
    assert "Voice (TTS)" not in _sections(ok)


def test_preflight_llm_compat_without_endpoint(tmp_path):
    cfg = AppConfig(llm={"provider": "openai_compatible"})  # no blocks, no legacy URL
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Engine (LLM)" and i["level"] == "error" for i in issues)


def test_preflight_llm_compat_legacy_fallback_used(tmp_path):
    """No blocks → legacy openai_compatible_base_url feeds the check (localhost = ok)."""
    cfg = AppConfig(llm={
        "provider": "openai_compatible",
        "openai_compatible_base_url": "http://127.0.0.1:11434/v1",
    })
    assert not any(i["section"] == "Engine (LLM)" for i in _issues(cfg, tmp_path))


def test_preflight_llm_compat_remote_block_needs_key(tmp_path, monkeypatch):
    cfg = AppConfig(
        llm={"provider": "openai_compatible", "provider_ref": "brain", "model": "x"},
        providers=[{"name": "brain", "category": "llm",
                    "base_url": "https://api.remote.example/v1", "model": "big"}],
    )
    # No key env → error.
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Engine (LLM)" and "no API key" in i["message"] for i in issues)
    # With the dynamic env key set → clean.
    monkeypatch.setenv("PROVIDER_BRAIN_API_KEY", "sk-test")
    assert not any(i["section"] == "Engine (LLM)" for i in _issues(cfg, tmp_path))


def test_preflight_cloud_llm_without_key(tmp_path, monkeypatch):
    # The dev machine may carry real keys in .env — the test needs them absent.
    for env in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    for prov, env in [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                      ("anthropic", "ANTHROPIC_API_KEY")]:
        issues = _issues(AppConfig(llm={"provider": prov}), tmp_path)
        assert any(i["section"] == "Engine (LLM)" and env in i["message"] for i in issues), prov


# ---------------------------------------------------------------------------
# Hearing (STT)
# ---------------------------------------------------------------------------

def test_preflight_hearing_off_is_silent(tmp_path):
    assert not any(i["section"] == "Hearing (STT)" for i in _issues(AppConfig(), tmp_path))


def test_preflight_hearing_remote_stt_without_endpoint(tmp_path):
    cfg = AppConfig(hearing={"enabled": True, "engine": "openai_compatible"})
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Hearing (STT)" and i["level"] == "error" for i in issues)


def test_preflight_hearing_remote_stt_remote_without_key_is_warn(tmp_path):
    """Remote STT with no key degrades (401s) instead of refusing the build — warn."""
    cfg = AppConfig(
        hearing={"enabled": True, "engine": "openai_compatible", "provider_ref": "ears"},
        providers=[{"name": "ears", "category": "stt",
                    "base_url": "https://stt.remote.example/v1"}],
    )
    issues = _issues(cfg, tmp_path)
    stt = [i for i in issues if i["section"] == "Hearing (STT)"]
    assert stt and all(i["level"] == "warn" for i in stt)


def test_preflight_hearing_local_missing_dep_is_warn(tmp_path, monkeypatch):
    import wallie
    monkeypatch.setattr(wallie, "_pf_dep_missing", lambda pkg: pkg == "faster_whisper")
    cfg = AppConfig(hearing={"enabled": True})
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Hearing (STT)" and "faster-whisper" in i["message"]
               and i["level"] == "warn" for i in issues)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

def test_preflight_vision_dedicated_without_model(tmp_path):
    cfg = AppConfig(
        llm={"vision_capable": True, "vision_provider": "openai_compatible",
             "vision_openai_compatible_base_url": "http://localhost:9000/v1"},
        vision={"enabled": True},
    )
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Vision" and "cannot be inferred" in i["message"] for i in issues)


def test_preflight_vision_not_capable_is_warn(tmp_path):
    cfg = AppConfig(llm={"vision_capable": False}, vision={"enabled": True})
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Vision" and i["level"] == "warn" for i in issues)


# ---------------------------------------------------------------------------
# Memory / Thoughts
# ---------------------------------------------------------------------------

def test_preflight_memory_extractor_without_endpoint(tmp_path):
    cfg = AppConfig(memory={"enabled": True, "extractor": "openai_compatible"})
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Memory" and i["level"] == "error" for i in issues)


def test_preflight_memory_block_endpoint_ok(tmp_path):
    cfg = AppConfig(
        memory={"enabled": True, "extractor": "openai_compatible", "provider_ref": "mem"},
        providers=[{"name": "mem", "category": "memory",
                    "base_url": "http://localhost:11434/v1"}],
    )
    assert not any(i["section"] == "Memory" for i in _issues(cfg, tmp_path))


def test_preflight_thoughts_generator_without_endpoint(tmp_path):
    cfg = AppConfig(random_thoughts={"enabled": True, "generator": "openai_compatible"})
    issues = _issues(cfg, tmp_path)
    assert any(i["section"] == "Thoughts" and i["level"] == "error" for i in issues)


# ---------------------------------------------------------------------------
# Dashboard route wiring
# ---------------------------------------------------------------------------

def test_api_preflight_route(monkeypatch):
    from fastapi.testclient import TestClient
    import wallie
    from dashboard.server import DashboardState, _build_app

    canned = [{"level": "error", "section": "Voice (TTS)", "message": "boom"}]
    monkeypatch.setattr(wallie, "preflight", lambda runtime=None: canned)

    app = _build_app(DashboardState(), None, pin="")
    with TestClient(app) as client:
        r = client.get("/api/preflight")
        assert r.status_code == 200
        assert r.json() == canned

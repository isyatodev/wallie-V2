"""Voice-test endpoint (POST /api/test/voice).

The Play button synthesizes a sample line with the CONFIGURED TTS and plays
it — using the same provider-block resolution as the real build — and reports
what happened: bytes/duration, ffmpeg decoding, or the real reason on failure.
"""
from __future__ import annotations

import pytest

from config import AppConfig, Runtime, Secrets


def _profiled_app(tmp_path, monkeypatch, cfg, orch=None):
    """Isolated profiles dir + app. get_runtime is patched for the voice route."""
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
    return _build_app(state, orch)


class _FakePlayer:
    def __init__(self):
        self.written: list[bytes] = []

    async def write(self, pcm: bytes) -> None:
        self.written.append(pcm)


def _tts_block_cfg(**overrides) -> AppConfig:
    base = {
        "profile_name": "p-vtest",
        "providers": [{"id": "ttsvoice", "name": "GW", "category": "tts",
                       "base_url": "https://tts.example/v1", "model": "speech-1.5"}],
        "tts": {"provider": "openai_compatible", "provider_ref": "ttsvoice",
                "openai_compatible_voice": "alloy", "openai_compatible_pcm_sample_rate": 24000},
    }
    base["tts"].update(overrides)
    return AppConfig(**base)


def _patch_speech(monkeypatch, handler) -> None:
    """Route the REAL OpenAICompatibleTTS client through a MockTransport."""
    import httpx
    real = httpx.AsyncClient

    def factory(**kwargs):
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def test_voice_route_plays_and_reports(tmp_path, monkeypatch):
    """Happy path with a remote block: plays through the fake preview player."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        return httpx.Response(200, content=b"\x01\x00\x02\x00" * 500)  # raw PCM

    _patch_speech(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())

    class _PlayerProbe:
        pass

    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hello world"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["ok"] is True
        assert data["routed"] == "preview-player"
        assert data["bytes"] == 2000
        assert "decoded" in data and data["decoded"] is False


def test_voice_route_reports_real_reason_on_error(tmp_path, monkeypatch):
    """A bad endpoint name surfaces the gateway's error, not a silent 500."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "voice 'nope' not found"})

    _patch_speech(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hello"})
        assert r.status_code == 400
        assert "404" in r.json()["detail"]


def test_voice_route_empty_synth_is_actionable(tmp_path, monkeypatch):
    """200-with-no-audio (bad model name on a lenient gateway) → actionable."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    _patch_speech(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hello"})
        assert r.status_code == 400
        assert "no audio" in r.json()["detail"]


def test_voice_route_empty_text_rejected(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "   "})
        assert r.status_code == 400
        assert "empty" in r.json()["detail"]


def test_voice_route_mp3_decoded_by_client_layer(tmp_path, monkeypatch):
    """openai_compatible + MP3 answer + ffmpeg: the CLIENT decodes transparently
    (auto-decode lives in the provider), the route just plays the PCM."""
    import httpx
    import tts.decode

    mp3 = b"ID3\x04\x00\x00\x00" + b"\x00" * 300

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mp3)

    _patch_speech(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")
    # The client imported find_ffmpeg/decode_to_pcm16 into ITS namespace —
    # patch there (tts.decode patches only affect the route's fallback branch).
    monkeypatch.setattr("tts.openai_tts.find_ffmpeg", lambda: "fake-ffmpeg")
    monkeypatch.setattr("tts.openai_tts.decode_to_pcm16", _fake_decode)

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hello"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["ok"] is True
        assert data["decoded"] is False        # decode happened in the client
        assert data["bytes"] == 1200           # the DECODED pcm reached playback


def test_voice_route_decode_branch_for_other_providers(tmp_path, monkeypatch):
    """Defense-in-depth: a provider that yields compressed bytes raw (no client
    decode) is still decoded by the route before playback."""
    import dashboard.server as ds
    import tts.decode
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")  # remote block needs a key

    mp3 = b"ID3\x04\x00\x00\x00" + b"\x00" * 300

    class _StubTTS:
        name = "stub"
        sample_rate = 24000
        channels = 1

        async def synthesize(self, text):
            yield mp3

        async def aclose(self):
            pass

    monkeypatch.setattr(ds, "build_tts", lambda cfg, sec: _StubTTS())
    monkeypatch.setattr(tts.decode, "find_ffmpeg", lambda: "fake-ffmpeg")
    monkeypatch.setattr(tts.decode, "decode_to_pcm16", _fake_decode)

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hello"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["decoded"] is True
        assert data["bytes"] == 1200


def test_voice_route_compressed_without_ffmpeg_is_actionable(tmp_path, monkeypatch):
    """Provider yields MP3 raw + no ffmpeg anywhere → actionable 400."""
    import dashboard.server as ds
    import tts.decode
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")  # remote block needs a key

    mp3 = b"ID3\x04\x00\x00\x00" + b"\x00" * 300

    class _StubTTS:
        name = "stub"
        sample_rate = 24000
        channels = 1

        async def synthesize(self, text):
            yield mp3

        async def aclose(self):
            pass

    monkeypatch.setattr(ds, "build_tts", lambda cfg, sec: _StubTTS())
    monkeypatch.setattr(tts.decode, "find_ffmpeg", lambda: None)

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg())
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hello"})
        assert r.status_code == 400
        assert "ffmpeg" in r.json()["detail"]


async def _fake_decode(data: bytes, ff: str, sr: int) -> bytes:
    assert ff == "fake-ffmpeg"
    assert sr == 24000
    return b"\x01\x00" * 600  # plausible PCM


def test_voice_route_live_session_routes_to_orchestrator_player(tmp_path, monkeypatch):
    """When a session is running, playback goes to the orchestrator's player."""
    import httpx

    written: list[bytes] = []

    class _Orch:
        _player = _FakePlayer()

        def status(self):
            return {"running": True}

    orch = _Orch()
    orch._player.written = written  # share the list

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x01\x00\x02\x00" * 500)

    _patch_speech(monkeypatch, handler)
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")

    from fastapi.testclient import TestClient
    app = _profiled_app(tmp_path, monkeypatch, _tts_block_cfg(), orch=orch)
    with TestClient(app) as client:
        r = client.post("/api/test/voice", json={"text": "hi"})
        assert r.status_code == 200, r.text
        assert r.json()["routed"] == "live-player"
        assert len(written[0]) == 2000


def test_effective_tts_applies_block(tmp_path, monkeypatch):
    """wallie._effective_tts mirrors the build: block endpoint/model/key win."""
    import wallie
    monkeypatch.setenv("PROVIDER_TTSVOICE_API_KEY", "key-tts")
    cfg = _tts_block_cfg()
    runtime = Runtime(config=cfg, secrets=Secrets())
    tts_cfg, secrets_eff = wallie._effective_tts(runtime)
    assert tts_cfg.openai_compatible_base_url == "https://tts.example/v1"
    assert tts_cfg.openai_compatible_model == "speech-1.5"
    assert secrets_eff.openai_compatible_tts_api_key == "key-tts"


def test_effective_tts_passthrough_for_non_compat(tmp_path, monkeypatch):
    import wallie
    cfg = AppConfig(profile_name="p-vtest", tts={"provider": "piper", "piper_model_path": "x.onnx"})
    runtime = Runtime(config=cfg, secrets=Secrets())
    tts_cfg, secrets_eff = wallie._effective_tts(runtime)
    assert tts_cfg is runtime.config.tts
    assert secrets_eff is runtime.secrets

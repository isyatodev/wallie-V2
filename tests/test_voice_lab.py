"""Voice Lab — saved-voice library (GET/POST/DELETE /api/voices/library),
provider voice cloning (POST /api/voices/clone) and reference recording
(POST /api/voices/record).

The library is per profile and must never smuggle non-TTS keys into the config;
cloning must hit the providers' documented endpoints with the right auth header
and multipart field names, and must explain a missing key instead of a traceback.
"""
from __future__ import annotations

import base64
import json

import httpx

from config import AppConfig, Runtime, Secrets


def _profiled_app(tmp_path, monkeypatch, cfg, orch=None):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import activate_profile, load_profile, save_profile
    save_profile(cfg, cfg.profile_name)
    activate_profile(cfg.profile_name)   # the routes key the library off the ACTIVE profile
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile(cfg.profile_name), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    return _build_app(DashboardState(), orch)


def _patch_httpx(monkeypatch, handler) -> None:
    """Route every AsyncClient through a MockTransport (clone builds its own)."""
    real = httpx.AsyncClient

    def factory(**kwargs):
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


# ---------------------------------------------------------------------------
# Library CRUD
# ---------------------------------------------------------------------------

def test_library_roundtrip_and_is_per_profile(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        assert client.get("/api/voices/library").json()["presets"] == []

        r = client.post("/api/voices/library", json={
            "name": "Wallie BR",
            "provider": "kokoro",
            "voice_id": "pf_dora",
            "tts": {"kokoro_lang_code": "p", "kokoro_speed": 1.1},
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["preset"]["name"] == "Wallie BR"
        assert data["preset"]["created_at"]
        assert [p["name"] for p in data["presets"]] == ["Wallie BR"]

        # A second profile must not see it.
        from config import save_profile
        save_profile(AppConfig(profile_name="p-other"), "p-other")
        r = client.put("/api/profiles/p-other/activate")
        assert r.status_code == 200
        assert client.get("/api/voices/library").json()["presets"] == []

        client.put("/api/profiles/p-lab/activate")
        assert len(client.get("/api/voices/library").json()["presets"]) == 1

        # Upsert by name (case-insensitive) instead of duplicating.
        client.post("/api/voices/library", json={
            "name": "wallie br", "provider": "elevenlabs", "voice_id": "v-2",
        })
        presets = client.get("/api/voices/library").json()["presets"]
        assert len(presets) == 1
        assert presets[0]["provider"] == "elevenlabs"

        r = client.delete("/api/voices/library/Wallie%20BR")
        assert r.status_code == 200
        assert r.json()["presets"] == []
        assert client.delete("/api/voices/library/nope").status_code == 404


def test_library_sanitizes_tts_fields(tmp_path, monkeypatch):
    """Hand-edited keys (and the routing device) never reach the config."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        r = client.post("/api/voices/library", json={
            "name": "clean",
            "provider": "piper",
            "voice_id": "x.onnx",
            "tts": {
                "piper_length_scale": 1.2,
                "output_device": "Speakers (should be dropped)",
                "totally_not_a_field": "nope",
            },
        })
        assert r.status_code == 200
        tts = r.json()["preset"]["tts"]
        assert tts == {"piper_length_scale": 1.2}

        # Unnamed presets are rejected with something actionable.
        bad = client.post("/api/voices/library", json={"name": "   "})
        assert bad.status_code == 400
        assert "name" in bad.json()["detail"]


def test_library_reports_provider_readiness(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        providers = client.get("/api/voices/library").json()["providers"]
    by_id = {p["id"]: p for p in providers}
    assert set(by_id) == {"elevenlabs", "fish"}
    assert all(p["ready"] is False for p in providers)
    assert by_id["elevenlabs"]["key_env"] == "ELEVENLABS_API_KEY"


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def test_clone_elevenlabs_uploads_multipart_and_saves_preset(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("xi-api-key")
        seen["content_type"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(200, json={"voice_id": "voice-123"})

    _patch_httpx(monkeypatch, handler)
    sample = base64.b64encode(b"RIFFfakewav").decode()

    with _client(app) as client:
        r = client.post("/api/voices/clone", json={
            "provider": "elevenlabs",
            "name": "Clone Me",
            "description": "own voice",
            "samples": [{"filename": "take1.wav", "data": sample}],
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["voice_id"] == "voice-123"
        assert data["preset"]["source"] == "clone:elevenlabs"
        assert data["preset"]["voice_id"] == "voice-123"
        # Saved into the library, ready to apply.
        assert [p["name"] for p in data["presets"]] == ["Clone Me"]

    assert seen["url"] == "https://api.elevenlabs.io/v1/voices/add"
    assert seen["key"] == "el-key"
    assert seen["content_type"].startswith("multipart/form-data")
    assert b'name="files"' in seen["body"]
    assert b"take1.wav" in seen["body"]
    assert b"Clone Me" in seen["body"]


def test_clone_fish_uses_model_endpoint_and_voices_field(tmp_path, monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "fish-key")
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"_id": "model-abc", "state": "created"})

    _patch_httpx(monkeypatch, handler)

    with _client(app) as client:
        r = client.post("/api/voices/clone", json={
            "provider": "fish",
            "name": "Fishy",
            "samples": [{"filename": "a.wav", "data": base64.b64encode(b"wavdata").decode()}],
        })
        assert r.status_code == 200, r.text
        assert r.json()["voice_id"] == "model-abc"

    assert seen["url"] == "https://api.fish.audio/model"
    assert seen["auth"] == "Bearer fish-key"
    assert b'name="voices"' in seen["body"]
    assert b'name="train_mode"' in seen["body"]
    assert b"fast" in seen["body"]


def test_clone_without_key_is_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        r = client.post("/api/voices/clone", json={
            "provider": "elevenlabs",
            "name": "No Key",
            "samples": [{"filename": "a.wav", "data": base64.b64encode(b"x").decode()}],
        })
    assert r.status_code == 400
    assert "ELEVENLABS_API_KEY" in r.json()["detail"]


def test_clone_rejects_bad_input_and_provider(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        # No samples at all.
        r = client.post("/api/voices/clone", json={"provider": "fish", "name": "x"})
        assert r.status_code == 400
        assert "sample" in r.json()["detail"]

        # A provider that has no create-a-voice API.
        r = client.post("/api/voices/clone", json={
            "provider": "piper", "name": "x",
            "samples": [{"filename": "a.wav", "data": base64.b64encode(b"x").decode()}],
        })
        assert r.status_code == 400
        assert "elevenlabs" in r.json()["detail"]

        # Provider rejected the upload → its message reaches the user.
        monkeypatch.setenv("FISH_API_KEY", "fish-key")
        _patch_httpx(monkeypatch, lambda req: httpx.Response(
            422, json={"message": "audio too short", "status": 422}))
        r = client.post("/api/voices/clone", json={
            "provider": "fish", "name": "x",
            "samples": [{"filename": "a.wav", "data": base64.b64encode(b"x").decode()}],
        })
        assert r.status_code == 400
        assert "422" in r.json()["detail"] and "audio too short" in r.json()["detail"]


def test_clone_survives_preset_save_failure(tmp_path, monkeypatch):
    """The provider already created (and billed for) the voice — the response
    must still report its id when saving the preset fails."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    _patch_httpx(monkeypatch, lambda req: httpx.Response(200, json={"voice_id": "v-9"}))

    import tts.voice_lab as vl
    monkeypatch.setattr(vl, "add_preset", lambda *a, **kw: (_ for _ in ()).throw(
        vl.VoiceLabError("library is full")))

    with _client(app) as client:
        r = client.post("/api/voices/clone", json={
            "provider": "elevenlabs", "name": "x",
            "samples": [{"filename": "a.wav", "data": base64.b64encode(b"x").decode()}],
        })
    assert r.status_code == 200, r.text
    assert r.json()["voice_id"] == "v-9"
    assert r.json()["preset"] is None


# ---------------------------------------------------------------------------
# Recording a reference take
# ---------------------------------------------------------------------------

def test_record_route_returns_playable_wav(tmp_path, monkeypatch):
    import numpy as np
    import dashboard.server as ds

    seconds = 2.0
    tone = (0.2 * np.sin(np.linspace(0, 60, int(16000 * seconds)))).astype("float32")
    monkeypatch.setattr(ds, "_record_mic", lambda s: tone)

    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        r = client.post("/api/voices/record", json={"seconds": seconds})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["seconds"] == seconds
    wav = base64.b64decode(data["wav_b64"])
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    # 16 kHz mono PCM16 → header 44 bytes + 2 bytes per sample
    assert len(wav) == 44 + int(16000 * seconds) * 2


def test_record_route_reports_empty_capture(tmp_path, monkeypatch):
    import numpy as np
    import dashboard.server as ds

    monkeypatch.setattr(ds, "_record_mic", lambda s: np.zeros(0, dtype="float32"))
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lab"))
    with _client(app) as client:
        r = client.post("/api/voices/record", json={"seconds": 3})
    assert r.status_code == 400
    assert "nothing was recorded" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_sanitize_tts_keeps_only_real_fields():
    from tts.voice_lab import sanitize_tts, tts_updates, VoicePreset

    assert sanitize_tts({"kokoro_speed": 1.2, "nope": 1}) == {"kokoro_speed": 1.2}
    assert sanitize_tts(None) == {}
    assert sanitize_tts("nope") == {}

    preset = VoicePreset(name="x", provider="kokoro", voice_id="pf_dora",
                         tts={"kokoro_lang_code": "p", "kokoro_speed": 1.2})
    assert tts_updates(preset) == {
        "provider": "kokoro", "voice_id": "pf_dora",
        "kokoro_lang_code": "p", "kokoro_speed": 1.2,
    }


# ---------------------------------------------------------------------------
# A/B preview — same line, two voices, audio back instead of playback
# ---------------------------------------------------------------------------

class _StubTTS:
    """Captures the config it was built from and yields fixed PCM16."""

    name = "stub"
    sample_rate = 22050
    channels = 1

    def __init__(self, cfg=None, pcm: bytes = b"\x01\x00" * 400, error=None):
        self.cfg = cfg
        self._pcm = pcm
        self._error = error
        self.text = None
        self.closed = False

    async def synthesize(self, text):
        self.text = text
        if self._error:
            raise self._error
        yield self._pcm

    async def aclose(self):
        self.closed = True


def _patch_tts(monkeypatch, captured: list, **kw) -> None:
    import dashboard.server as ds

    def _build(cfg, secrets):
        stub = _StubTTS(cfg=cfg, **kw)
        captured.append(stub)
        return stub

    monkeypatch.setattr(ds, "build_tts", _build)


class _RecordingPlayer:
    def __init__(self):
        self.written: list[bytes] = []

    async def write(self, data):
        self.written.append(data)


class _LiveOrch:
    """A running session — the A/B preview must NOT route audio into it."""

    def __init__(self):
        self._player = _RecordingPlayer()

    def status(self):
        return {"running": True}


def test_preview_route_returns_playable_wav_for_a_saved_voice(tmp_path, monkeypatch):
    captured: list = []
    _patch_tts(monkeypatch, captured)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(
        profile_name="p-ab", tts={"provider": "piper", "voice_id": "base.onnx"},
    ))

    with _client(app) as client:
        client.post("/api/voices/library", json={
            "name": "BR", "provider": "kokoro", "voice_id": "pf_dora",
            "tts": {"kokoro_lang_code": "p", "kokoro_speed": 1.15},
        })
        r = client.post("/api/voices/preview", json={"text": "same line", "name": "BR"})
        assert r.status_code == 200, r.text
        data = r.json()

    # The saved voice's provider/voice/knobs drove the synthesis…
    assert captured[0].cfg.provider == "kokoro"
    assert captured[0].cfg.voice_id == "pf_dora"
    assert captured[0].cfg.kokoro_lang_code == "p"
    assert captured[0].cfg.kokoro_speed == 1.15
    assert captured[0].text == "same line"
    assert captured[0].closed is True

    # …and the caller got audio back, not playback.
    assert data["provider"] == "kokoro" and data["voice_id"] == "pf_dora"
    assert data["source"].startswith("saved voice")
    assert data["sample_rate"] == 22050 and data["bytes"] == 800
    wav = base64.b64decode(data["wav_b64"])
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) == 44 + 800          # header + PCM16 payload


def test_preview_route_inline_overrides_do_not_touch_the_profile(tmp_path, monkeypatch):
    """Voice A = a preset, Voice B = whatever the Voice page shows right now.
    The inline half must apply (and be sanitized) without being saved."""
    captured: list = []
    _patch_tts(monkeypatch, captured)
    cfg = AppConfig(profile_name="p-ab", tts={"provider": "piper", "voice_id": "base.onnx"})
    app = _profiled_app(tmp_path, monkeypatch, cfg)

    from config import load_profile
    with _client(app) as client:
        r = client.post("/api/voices/preview", json={
            "text": "hi",
            "provider": "elevenlabs",
            "voice_id": "v-unsaved",
            "tts": {"el_stability": 0.2, "output_device": "dropped", "bogus": 1},
        })
        assert r.status_code == 200, r.text
        assert r.json()["source"] == "unsaved Voice page settings"

    assert captured[0].cfg.provider == "elevenlabs"
    assert captured[0].cfg.voice_id == "v-unsaved"
    assert captured[0].cfg.el_stability == 0.2
    # The profile itself is untouched, and output_device never travelled.
    saved = load_profile("p-ab")
    assert saved.tts.provider == "piper" and saved.tts.voice_id == "base.onnx"


def test_preview_route_without_preset_or_overrides_uses_saved_config(tmp_path, monkeypatch):
    captured: list = []
    _patch_tts(monkeypatch, captured)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(
        profile_name="p-ab", tts={"provider": "piper", "voice_id": "base.onnx"},
    ))
    with _client(app) as client:
        r = client.post("/api/voices/preview", json={"text": "hi"})
        assert r.status_code == 200, r.text
        assert r.json()["source"] == "saved config"
    assert captured[0].cfg.provider == "piper"


def test_preview_route_never_plays_into_a_live_session(tmp_path, monkeypatch):
    """The whole point of returning audio: comparing must not talk over the
    stream. Unlike /api/test/voice, a running orchestrator gets nothing."""
    captured: list = []
    _patch_tts(monkeypatch, captured)
    orch = _LiveOrch()
    app = _profiled_app(tmp_path, monkeypatch,
                        AppConfig(profile_name="p-ab", tts={"provider": "piper"}), orch)
    with _client(app) as client:
        r = client.post("/api/voices/preview", json={"text": "hi"})
        assert r.status_code == 200, r.text
    assert orch._player.written == []


def test_preview_route_errors_are_actionable(tmp_path, monkeypatch):
    captured: list = []
    _patch_tts(monkeypatch, captured)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-ab"))

    with _client(app) as client:
        # Empty / oversized text never reaches the provider.
        assert client.post("/api/voices/preview", json={"text": "  "}).status_code == 400
        r = client.post("/api/voices/preview", json={"text": "x" * 601})
        assert r.status_code == 400 and "too long" in r.json()["detail"]

        # Unknown saved voice.
        r = client.post("/api/voices/preview", json={"text": "hi", "name": "nope"})
        assert r.status_code == 404 and "nope" in r.json()["detail"]

        # Provider failure surfaces its own message.
        from tts.base import TTSError
        _patch_tts(monkeypatch, [], error=TTSError("elevenlabs 401: bad key"))
        r = client.post("/api/voices/preview", json={"text": "hi"})
        assert r.status_code == 400 and "401" in r.json()["detail"]

        # 200-with-no-audio is called out instead of returning a silent clip.
        _patch_tts(monkeypatch, [], pcm=b"")
        r = client.post("/api/voices/preview", json={"text": "hi"})
        assert r.status_code == 400 and "no audio" in r.json()["detail"]


def test_preview_keeps_pcm_that_looks_like_an_mp3_frame(tmp_path, monkeypatch):
    """Regression: raw PCM16 whose first sample is -1 begins with FF FF — exactly
    the MP3 sync word. ffmpeg then fails on real PCM, and before this the whole
    clip was reported as an error. The bytes must survive as audio.
    """
    import tts.decode

    pcm = b"\xff\xff" + b"\x00\x00" * 200          # first sample == -1
    _patch_tts(monkeypatch, [], pcm=pcm)
    monkeypatch.setattr(tts.decode, "find_ffmpeg", lambda: "fake-ffmpeg")

    async def _boom(data, ff, sr):
        raise RuntimeError("ffmpeg decode failed: Invalid data found when processing input")

    monkeypatch.setattr(tts.decode, "decode_to_pcm16", _boom)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-ab"))

    with _client(app) as client:
        r = client.post("/api/voices/preview", json={"text": "hi"})
        assert r.status_code == 200, r.text
        data = r.json()
    assert data["decoded"] is False                 # nothing was decoded
    assert data["bytes"] == len(pcm)               # and nothing was thrown away
    assert base64.b64decode(data["wav_b64"])[44:] == pcm


def test_preview_still_rejects_genuinely_undecodable_containers(tmp_path, monkeypatch):
    """The fallback is narrow: a real container signature that ffmpeg cannot read
    is still an error, not silently played as static."""
    import tts.decode

    _patch_tts(monkeypatch, [], pcm=b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 200)
    monkeypatch.setattr(tts.decode, "find_ffmpeg", lambda: "fake-ffmpeg")

    async def _boom(data, ff, sr):
        raise RuntimeError("ffmpeg decode failed: Invalid data found")

    monkeypatch.setattr(tts.decode, "decode_to_pcm16", _boom)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-ab"))
    with _client(app) as client:
        r = client.post("/api/voices/preview", json={"text": "hi"})
    assert r.status_code == 400
    assert "Invalid data found" in r.json()["detail"]


def test_preview_surfaces_a_provider_error_hidden_in_http_200(tmp_path, monkeypatch):
    """Some gateways answer 200 with a JSON error body. Sending that to ffmpeg
    produced a confusing 'ffmpeg decode failed'; the endpoint's own message is
    the useful thing to show."""
    body = b'{"detail": {"message": "quota exhausted for this key"}}'
    _patch_tts(monkeypatch, [], pcm=body)
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-ab"))
    with _client(app) as client:
        r = client.post("/api/voices/preview", json={"text": "hi"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "error body" in detail and "quota exhausted" in detail


def test_mp3_sync_detector_flags_only_the_ambiguous_signature():
    from tts.decode import looks_like_mp3_frame_sync, sniff_compressed

    assert looks_like_mp3_frame_sync(b"\xff\xff") is True      # PCM sample -1
    assert looks_like_mp3_frame_sync(b"\xff\xfb") is True      # a real frame
    assert looks_like_mp3_frame_sync(b"\xfe\xff") is False
    assert looks_like_mp3_frame_sync(b"\xff\xe0") is False     # reserved layer bits
    assert looks_like_mp3_frame_sync(b"\xff") is False
    # Unambiguous containers are NOT covered by the soft fallback.
    assert looks_like_mp3_frame_sync(b"RIFF") is False
    assert sniff_compressed(b"ID3\x04\x00") == "mp3"
    assert looks_like_mp3_frame_sync(b"ID3\x04\x00") is False


def test_pcm16_bytes_to_wav_lengths_and_odd_tail():
    from tts.voice_lab import pcm16_bytes_to_wav

    wav = pcm16_bytes_to_wav(b"\x00\x01" * 10, 24000, 1)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and len(wav) == 44 + 20
    # An odd trailing byte cannot be half a sample — it is dropped.
    assert len(pcm16_bytes_to_wav(b"\x00" * 21, 16000)) == 44 + 20
    # Mono float32 helper still wraps through the same path.
    import numpy as np
    from tts.voice_lab import pcm16_wav_bytes
    assert len(pcm16_wav_bytes(np.zeros(50, dtype="float32"), 16000)) == 44 + 100


def test_kokoro_pt_br_language_reaches_the_provider(monkeypatch):
    """pt-BR is Kokoro's lang_code 'p' (voice prefix pf_/pm_) — the config value
    must arrive at the provider constructor untouched. The model weights
    themselves are downloaded by kokoro at runtime."""
    import tts.kokoro as kok

    seen: dict = {}

    class _StubKokoro:
        name = "kokoro"
        sample_rate = 24000
        channels = 1

        def __init__(self, *, voice, lang_code, speed):
            seen.update(voice=voice, lang_code=lang_code, speed=speed)

    monkeypatch.setattr(kok, "KokoroTTS", _StubKokoro)
    from config import TTSConfig
    from tts import build_tts

    build_tts(TTSConfig(
        provider="kokoro", kokoro_voice="pf_dora",
        kokoro_lang_code="p", kokoro_speed=1.05,
    ), Secrets())
    assert seen == {"voice": "pf_dora", "lang_code": "p", "speed": 1.05}

    # And the default stays US English so existing setups are untouched.
    assert TTSConfig().kokoro_lang_code == "a"


def test_library_file_corruption_is_not_fatal(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts.voice_lab import load_library, add_preset

    (tmp_path / "p-bad.voices.json").write_text("{ not json", encoding="utf-8")
    assert load_library("p-bad") == []

    add_preset("p-bad", {"name": "recovered", "provider": "piper"})
    assert [p.name for p in load_library("p-bad")] == ["recovered"]
    on_disk = json.loads((tmp_path / "p-bad.voices.json").read_text(encoding="utf-8"))
    assert on_disk["profile"] == "p-bad"

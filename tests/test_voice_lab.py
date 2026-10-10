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
from dataclasses import replace

import httpx
import pytest

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
# Backup file — the saved voices as a downloadable / uploadable .json
# ---------------------------------------------------------------------------

def test_backup_bundle_roundtrips_and_keeps_the_voice_intact(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    vl.add_preset("p-src", {
        "name": "Wallie BR", "provider": "kokoro", "voice_id": "pf_dora",
        "notes": "main host", "source": "local:kokoro", "created_at": "2026-01-02T03:04:05+00:00",
        "tts": {"kokoro_lang_code": "p", "output_device": "dropped"},
    })
    bundle = vl.export_bundle("p-src")
    assert bundle["format"] == vl.EXPORT_FORMAT
    assert bundle["version"] == vl.EXPORT_VERSION
    assert bundle["source_profile"] == "p-src" and bundle["exported_at"]
    assert bundle["count"] == 1 and [p["name"] for p in bundle["presets"]] == ["Wallie BR"]

    # The file is JSON text; parse it back the way an upload does.
    result = vl.import_bundle("p-src2", vl.parse_bundle_text(json.dumps(bundle)))
    assert result == {"imported": 1, "added": 1, "skipped": 0,
                     "replaced": 0, "unchanged": 0, "total": 1}
    got = vl.load_library("p-src2")[0]
    assert (got.name, got.provider, got.voice_id, got.source) == (
        "Wallie BR", "kokoro", "pf_dora", "local:kokoro")
    assert got.notes == "main host" and got.created_at == "2026-01-02T03:04:05+00:00"
    assert got.tts == {"kokoro_lang_code": "p"}          # routing field dropped

    # A single named voice can be exported on its own.
    vl.add_preset("p-src", {"name": "Other", "provider": "piper"})
    one = vl.export_bundle("p-src", ["Other"])
    assert [p["name"] for p in one["presets"]] == ["Other"]
    with pytest.raises(vl.VoiceLabError):
        vl.export_bundle("p-src", ["not saved here"])


def test_import_merges_by_name_and_skips_junk(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    vl.add_preset("p", {"name": "one", "provider": "piper", "voice_id": "a.onnx"})
    payload = {"presets": [
        {"name": "ONE", "provider": "elevenlabs", "voice_id": "v-9"},   # replaces
        {"provider": "piper"},                                          # no name
        "junk",
        {"name": "   "},
        {"name": "Two", "provider": "fish", "voice_id": "f-1"},          # new
    ]}
    # Replacing a differently tuned same-named voice asks first...
    with pytest.raises(vl.VoiceConflictError) as err:
        vl.import_bundle("p", payload)
    assert err.value.plan["replaced"] == ["ONE"] and err.value.plan["added"] == ["Two"]
    assert [p.name for p in vl.load_library("p")] == ["one"]      # nothing written

    result = vl.import_bundle("p", payload, overwrite=True)
    assert result == {"imported": 2, "added": 1, "skipped": 2,
                     "replaced": 1, "unchanged": 0, "total": 2}
    got = {p.name: p for p in vl.load_library("p")}
    assert set(got) == {"ONE", "Two"} and got["ONE"].provider == "elevenlabs"

    # A bare list, a library file and a single saved voice are all accepted.
    assert vl.import_bundle("p", [{"name": "From list"}])["imported"] == 1
    assert vl.import_bundle("p", {"profile": "p", "presets": [{"name": "Lib"}]})["imported"] == 1
    assert vl.import_bundle("p", {"name": "Solo", "provider": "piper"})["imported"] == 1


def test_merge_plan_flags_only_real_differences(tmp_path, monkeypatch):
    """The warning must fire on a different voice — and stay quiet on a re-save
    (``created_at``/``source`` change on every save, so they are ignored)."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    base = vl.VoicePreset(name="V", provider="piper", voice_id="x.onnx", notes="hi",
                          tts={"piper_noise_w": 0.9},
                          created_at="2020-01-01T00:00:00+00:00")
    resaved = replace(base, created_at="2026-01-01T00:00:00+00:00", source="local:piper")
    plan = vl.merge_plan([base], [resaved])
    assert plan == {"added": [], "replaced": [], "unchanged": ["V"], "conflicts": []}

    for field, value in [("provider", "kokoro"), ("voice_id", "y.onnx"),
                         ("notes", "bye"), ("tts", {"piper_noise_w": 0.1})]:
        plan = vl.merge_plan([base], [replace(base, **{field: value})])
        assert plan["replaced"] == ["V"], field
        assert plan["conflicts"][0]["name"] == "V"
        assert plan["unchanged"] == [] and plan["added"] == []

    # A brand-new name is an add, and the message names the voices.
    plan = vl.merge_plan([base], [base, vl.VoicePreset(name="New")])
    assert plan["added"] == ["New"] and plan["unchanged"] == ["V"]
    assert "V" in vl.conflict_message("dst", vl.merge_plan([base], [replace(base, provider="x")]))


def test_copying_between_profiles_asks_before_replacing(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    vl.add_preset("dst", {"name": "Wallie BR", "provider": "kokoro", "voice_id": "pf_dora",
                          "tts": {"kokoro_lang_code": "p"}})
    vl.add_preset("src", {"name": "Wallie BR", "provider": "elevenlabs", "voice_id": "v-9"})

    with pytest.raises(vl.VoiceConflictError) as err:
        vl.export_presets("src", "dst")
    conflict = err.value.plan["conflicts"][0]
    assert conflict["name"] == "Wallie BR"
    assert conflict["current"] == "kokoro : pf_dora (kokoro_lang_code=p)"
    assert conflict["incoming"] == "elevenlabs : v-9"
    assert "Wallie BR" in str(err.value)
    assert vl.load_library("dst")[0].provider == "kokoro"      # untouched until confirmed
    assert vl.plan_export("src", "dst")["replaced"] == ["Wallie BR"]

    assert vl.export_presets("src", "dst", overwrite=True) == 1
    assert vl.load_library("dst")[0].provider == "elevenlabs"


def test_import_refuses_unusable_files_without_writing(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    for bad in ["", "   ", "{not json", "[1, 2, 3]", '{"hello": 1}', '{"presets": []}']:
        with pytest.raises(vl.VoiceLabError):
            vl.import_bundle("p", vl.parse_bundle_text(bad))
    with pytest.raises(vl.VoiceLabError):
        vl.parse_bundle_text("x" * (vl.MAX_IMPORT_CHARS + 1))
    # A backup from a newer format is refused instead of half-read.
    with pytest.raises(vl.VoiceLabError):
        vl.import_bundle("p", {"format": vl.EXPORT_FORMAT,
                                "version": vl.EXPORT_VERSION + 1,
                                "presets": [{"name": "Future"}]})
    assert not vl.library_path("p").exists()      # nothing was written

    # Overflowing the library fails loudly — and again writes nothing.
    vl.add_preset("full", {"name": "One"})
    from tts.voice_lab import MAX_PRESETS
    big = {"presets": [{"name": f"v{i}"} for i in range(MAX_PRESETS)]}
    with pytest.raises(vl.VoiceLabError):
        vl.import_bundle("full", big)
    assert [p.name for p in vl.load_library("full")] == ["One"]


def test_backup_endpoint_downloads_and_restores(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-bk"))
    with _client(app) as client:
        assert client.get("/api/voices/backup").status_code == 400   # nothing saved yet
        client.post("/api/voices/library", json={
            "name": "A", "provider": "piper", "voice_id": "voices/a.onnx"})
        client.post("/api/voices/library", json={
            "name": "B", "provider": "elevenlabs", "voice_id": "v-1"})

        bundle = client.get("/api/voices/backup").json()
        assert [p["name"] for p in bundle["presets"]] == ["B", "A"]
        one = client.get("/api/voices/backup", params={"names": "A"}).json()
        assert [p["name"] for p in one["presets"]] == ["A"]
        assert client.get("/api/voices/backup", params={"names": "nope"}).status_code == 400

        # Wipe the library, then restore it from the exported file's text.
        client.delete("/api/voices/library/A")
        client.delete("/api/voices/library/B")
        assert client.get("/api/voices/library").json()["presets"] == []
        r = client.post("/api/voices/import", json={"content": json.dumps(bundle)})
        assert r.status_code == 200, r.text
        assert r.json()["imported"] == 2 and r.json()["total"] == 2
        assert [p["name"] for p in r.json()["presets"]] == ["B", "A"]
        assert client.get("/api/voices/library").json()["presets"][1]["voice_id"] == "voices/a.onnx"

        # Re-importing the same file upserts instead of duplicating.
        r = client.post("/api/voices/import", json={"content": json.dumps(bundle)})
        assert r.json()["imported"] == 2 and r.json()["total"] == 2

        # Bad uploads are actionable 400s, not tracebacks.
        for content in ["", "{not json", '{"hello": 1}']:
            bad = client.post("/api/voices/import", json={"content": content})
            assert bad.status_code == 400, bad.text
            assert bad.json()["detail"]


def test_import_endpoint_asks_before_replacing_a_different_voice(tmp_path, monkeypatch):
    """Restoring a backup must not silently overwrite a voice this profile has
    tuned differently: 409 with the diff first, then apply with overwrite."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-imp"))
    with _client(app) as client:
        client.post("/api/voices/library", json={
            "name": "A", "provider": "piper", "voice_id": "x.onnx"})
        bundle = client.get("/api/voices/backup").json()
        bundle["presets"][0]["voice_id"] = "y.onnx"        # same name, different voice

        r = client.post("/api/voices/import", json={"content": json.dumps(bundle)})
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["conflicts"] == [{"name": "A", "current": "piper : x.onnx",
                                        "incoming": "piper : y.onnx"}]
        assert "A" in detail["message"]
        assert client.get("/api/voices/library").json()["presets"][0]["voice_id"] == "x.onnx"

        r = client.post("/api/voices/import",
                        json={"content": json.dumps(bundle), "overwrite": True})
        assert r.status_code == 200, r.text
        assert r.json()["replaced"] == 1 and r.json()["added"] == 0
        assert client.get("/api/voices/library").json()["presets"][0]["voice_id"] == "y.onnx"

        # Re-importing the very same file is not a conflict — it just re-saves.
        r = client.post("/api/voices/import", json={"content": json.dumps(bundle)})
        assert r.status_code == 200, r.text
        assert r.json()["replaced"] == 0 and r.json()["unchanged"] == 1


# ---------------------------------------------------------------------------
# Mirroring two libraries
# ---------------------------------------------------------------------------

def test_mirror_plan_is_a_two_way_diff(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    vl.add_preset("a", {"name": "Mine only", "provider": "piper"})
    vl.add_preset("a", {"name": "Shared", "provider": "piper", "voice_id": "x.onnx"})
    vl.add_preset("a", {"name": "Clash", "provider": "kokoro", "voice_id": "pf_dora"})
    vl.add_preset("b", {"name": "Theirs only", "provider": "fish", "voice_id": "f-1"})
    vl.add_preset("b", {"name": "Shared", "provider": "piper", "voice_id": "x.onnx"})
    vl.add_preset("b", {"name": "Clash", "provider": "elevenlabs", "voice_id": "v-9"})

    plan = vl.mirror_plan("a", "b")
    assert plan["this_profile"] == "a" and plan["other_profile"] == "b"
    assert plan["to_other"] == ["Mine only"]
    assert plan["to_this"] == ["Theirs only"]
    assert plan["identical"] == ["Shared"]
    assert plan["conflicts"] == [{"name": "Clash", "this": "kokoro : pf_dora",
                                 "other": "elevenlabs : v-9"}]
    assert plan["this_count"] == 3 and plan["other_count"] == 3
    # Planning writes nothing.
    assert len(vl.load_library("a")) == 3 and len(vl.load_library("b")) == 3


def test_mirror_copies_both_ways_and_skips_clashes_by_default(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    vl.add_preset("a", {"name": "Mine only"})
    vl.add_preset("a", {"name": "Clash", "provider": "piper", "voice_id": "x.onnx"})
    vl.add_preset("b", {"name": "Theirs only"})
    vl.add_preset("b", {"name": "Clash", "provider": "elevenlabs", "voice_id": "v-9"})

    result = vl.mirror_libraries("a", "b")
    assert result["policy"] == "skip"
    assert result["to_other"] == 1 and result["to_this"] == 1 and result["skipped"] == 1
    assert result["this_total"] == 3 and result["other_total"] == 3
    assert sorted(p.name for p in vl.load_library("a")) == ["Clash", "Mine only", "Theirs only"]
    assert sorted(p.name for p in vl.load_library("b")) == ["Clash", "Mine only", "Theirs only"]
    # The clashing voice kept each side's own settings.
    assert {p.name: p.provider for p in vl.load_library("a")}["Clash"] == "piper"
    assert {p.name: p.provider for p in vl.load_library("b")}["Clash"] == "elevenlabs"
    # Applying again is a no-op (already mirrored).
    again = vl.mirror_libraries("a", "b")
    assert again["to_other"] == 0 and again["to_this"] == 0 and again["skipped"] == 1
    assert again["plan"]["conflicts"] == result["plan"]["conflicts"]


def test_mirror_conflict_policies_pick_a_winner(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    def seed():
        for profile in ("a", "b"):
            for preset in vl.load_library(profile):
                vl.remove_preset(profile, preset.name)
        vl.add_preset("a", {"name": "Clash", "provider": "piper", "voice_id": "x.onnx",
                            "tts": {"piper_noise_w": 0.9}})
        vl.add_preset("b", {"name": "Clash", "provider": "elevenlabs", "voice_id": "v-9"})

    seed()
    result = vl.mirror_libraries("a", "b", conflicts="this")
    assert result["policy"] == "this" and result["skipped"] == 0
    assert vl.load_library("b")[0].provider == "piper"        # mine won
    assert vl.load_library("a")[0].provider == "piper"        # and mine was not touched

    seed()
    result = vl.mirror_libraries("a", "b", conflicts="other")
    assert result["policy"] == "other"
    assert vl.load_library("a")[0].provider == "elevenlabs"   # theirs won
    assert vl.load_library("b")[0].provider == "elevenlabs"
    assert vl.load_library("a")[0].tts == {}                  # mine's tuning is gone

    with pytest.raises(vl.VoiceLabError):
        vl.mirror_libraries("a", "b", conflicts="nonsense")


def test_mirror_refuses_when_a_library_would_overflow(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl
    from tts.voice_lab import MAX_PRESETS

    for i in range(MAX_PRESETS):
        vl.add_preset("a", {"name": f"a{i}"})
    for i in range(MAX_PRESETS):
        vl.add_preset("b", {"name": f"b{i}"})

    with pytest.raises(vl.VoiceLabError):
        vl.mirror_libraries("a", "b")
    # Nothing was written on either side.
    assert len(vl.load_library("a")) == MAX_PRESETS
    assert len(vl.load_library("b")) == MAX_PRESETS
    assert [p.name for p in vl.load_library("a")][:1] == [f"a{MAX_PRESETS - 1}"]


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


# ---------------------------------------------------------------------------
# Voice partner — is this profile still in sync with its partner?
# ---------------------------------------------------------------------------

def test_sync_state_measures_a_profile_against_its_partner(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import voice_lab as vl

    vl.add_preset("a", {"name": "Mine only", "provider": "piper"})
    vl.add_preset("a", {"name": "Clash", "provider": "piper", "voice_id": "x.onnx"})
    vl.add_preset("b", {"name": "Theirs only"})
    vl.add_preset("b", {"name": "Clash", "provider": "elevenlabs", "voice_id": "v-9"})

    state = vl.sync_state("a", "b")
    assert state["partner"] == "b" and state["out_of_sync"] is True
    assert (state["to_other"], state["to_this"], state["conflicts"]) == (1, 1, 1)
    # Measuring writes nothing to either library.
    assert len(vl.load_library("a")) == 2 and len(vl.load_library("b")) == 2

    # After a default (skip) mirror the missing voices match but the clashing
    # one does not — that is still divergence, which is exactly what the badge
    # is for ("skip" alone would never resolve it).
    vl.mirror_libraries("a", "b")
    state = vl.sync_state("a", "b")
    assert state["out_of_sync"] is True
    assert state["conflicts"] == 1 and state["identical"] == 2

    # Picking a winner leaves nothing to report.
    vl.mirror_libraries("a", "b", conflicts="this")
    state = vl.sync_state("a", "b")
    assert state["out_of_sync"] is False
    assert state["conflicts"] == 0 and state["identical"] == 3
    assert state["this_count"] == 3 and state["other_count"] == 3

    # Two profiles with no saved voices at all are trivially in sync.
    assert vl.sync_state("empty-a", "empty-b")["out_of_sync"] is False

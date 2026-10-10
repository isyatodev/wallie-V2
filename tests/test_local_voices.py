"""Per-model local voice library — tts.local_voices + the /api/voices/local
endpoints.

The point of the store is that Piper voices and Kokoro voices live in SEPARATE
lists, and that a saved entry may only carry the knobs belonging to its own
engine (a Piper voice can never smuggle in Kokoro settings).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets


def _profiled_app(tmp_path, monkeypatch, cfg):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import activate_profile, load_profile, save_profile
    save_profile(cfg, cfg.profile_name)
    activate_profile(cfg.profile_name)
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile(cfg.profile_name), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    return _build_app(DashboardState(), None)


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_buckets_are_separate_per_provider(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv

    lv.add_local("p1", "piper", {"name": "Amy", "voice_id": "voices/amy.onnx",
                                 "tts": {"piper_length_scale": 1.1, "kokoro_speed": 9}})
    lv.add_local("p1", "kokoro", {"name": "Dora", "voice_id": "pf_dora",
                                  "tts": {"kokoro_lang_code": "p"}})

    assert [v.name for v in lv.list_local("p1", "piper")] == ["Amy"]
    assert [v.name for v in lv.list_local("p1", "kokoro")] == ["Dora"]
    # The Piper entry kept only Piper knobs (kokoro_speed dropped, length kept).
    assert lv.list_local("p1", "piper")[0].tts == {"piper_length_scale": 1.1}


def test_add_is_upsert_by_name_and_per_profile(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv

    lv.add_local("p1", "piper", {"name": "Main", "voice_id": "a.onnx"})
    lv.add_local("p1", "piper", {"name": "main", "voice_id": "b.onnx"})
    voices = lv.list_local("p1", "piper")
    assert len(voices) == 1 and voices[0].voice_id == "b.onnx"
    # A different profile starts empty.
    assert lv.list_local("p2", "piper") == []


def test_remove_and_unknown_provider(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv

    lv.add_local("p1", "kokoro", {"name": "BR", "voice_id": "pf_dora"})
    assert lv.remove_local("p1", "kokoro", "BR") is True
    assert lv.remove_local("p1", "kokoro", "BR") is False
    assert lv.list_local("p1", "kokoro") == []

    with pytest.raises(lv.LocalVoiceError):
        lv.list_local("p1", "elevenlabs")


def test_tts_updates_switch_provider_and_set_the_voice_field():
    from tts.local_voices import LocalVoice, tts_updates

    p = tts_updates("piper", LocalVoice(
        name="x", voice_id="voices/x.onnx", tts={"piper_noise_w": 0.9}))
    assert p == {"provider": "piper", "piper_model_path": "voices/x.onnx",
                 "piper_noise_w": 0.9}

    k = tts_updates("kokoro", LocalVoice(
        name="y", voice_id="pf_dora", tts={"kokoro_lang_code": "p"}))
    assert k == {"provider": "kokoro", "kokoro_voice": "pf_dora",
                 "kokoro_lang_code": "p"}


def test_tiny_entry_needs_a_name(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv
    with pytest.raises(lv.LocalVoiceError):
        lv.add_local("p1", "piper", {"name": "   "})


def test_corrupt_library_file_is_not_fatal(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv

    (tmp_path / "p-bad.voices.json").write_text("{ not json", encoding="utf-8")
    assert lv.list_local("p-bad", "piper") == []
    lv.add_local("p-bad", "piper", {"name": "recovered"})
    assert [v.name for v in lv.list_local("p-bad", "piper")] == ["recovered"]
    on_disk = json.loads((tmp_path / "p-bad.voices.json").read_text(encoding="utf-8"))
    assert on_disk["profile"] == "p-bad"


# ---------------------------------------------------------------------------
# Unification with the Voice Lab library
# ---------------------------------------------------------------------------

def test_local_voice_is_a_library_preset(tmp_path, monkeypatch):
    """A local voice must BE a Voice Lab preset — that is what makes it show up
    in the A/B dropdowns and travel with an export."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv, voice_lab as vl

    lv.add_local("p1", "piper", {"name": "Amy", "voice_id": "voices/amy.onnx",
                                 "tts": {"piper_length_scale": 1.1, "kokoro_speed": 9}})
    presets = vl.load_library("p1")
    assert [p.name for p in presets] == ["Amy"]
    assert presets[0].provider == "piper"
    assert presets[0].source == "local:piper"
    # Only the engine's own knob reached the shared library.
    assert presets[0].tts == {"piper_length_scale": 1.1}


def test_remove_local_only_touches_its_own_provider(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv, voice_lab as vl

    vl.add_preset("p1", {"name": "Piper One", "provider": "piper"})
    vl.add_preset("p1", {"name": "Cloud One", "provider": "elevenlabs"})
    assert lv.remove_local("p1", "piper", "Cloud One") is False   # wrong provider
    assert lv.remove_local("p1", "piper", "Piper One") is True
    assert [p.name for p in vl.load_library("p1")] == ["Cloud One"]


def test_export_presets_copies_between_profiles(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    from tts import local_voices as lv, voice_lab as vl

    lv.add_local("src", "kokoro", {"name": "Dora", "voice_id": "pf_dora",
                                   "tts": {"kokoro_lang_code": "p"}})
    vl.add_preset("src", {"name": "Cloud", "provider": "elevenlabs", "voice_id": "v9"})

    assert vl.export_presets("src", "dst") == 2
    assert sorted(p.name for p in vl.load_library("dst")) == ["Cloud", "Dora"]
    # Re-export upserts instead of duplicating.
    assert vl.export_presets("src", "dst") == 2
    assert len(vl.load_library("dst")) == 2
    # A named subset copies just that voice.
    assert vl.export_presets("src", "dst2", ["Dora"]) == 1
    assert [p.name for p in vl.load_library("dst2")] == ["Dora"]


def test_legacy_localvoices_file_is_migrated(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    import tts.local_voices as lv

    lv._migrated.clear()
    legacy = tmp_path / "p-legacy.localvoices.json"
    legacy.write_text(json.dumps({"profile": "p-legacy", "providers": {
        "piper": [{"name": "Old Amy", "voice_id": "voices/old.onnx",
                    "tts": {"piper_noise_w": 0.5, "kokoro_speed": 9}}],
        "kokoro": [{"name": "Old Dora", "voice_id": "pf_dora"}],
    }}), encoding="utf-8")

    buckets = lv.load_local("p-legacy")
    assert [v.name for v in buckets["piper"]] == ["Old Amy"]
    assert buckets["piper"][0].tts == {"piper_noise_w": 0.5}
    assert [v.name for v in buckets["kokoro"]] == ["Old Dora"]
    assert not legacy.exists()                       # consumed
    assert (tmp_path / "p-legacy.voices.json").is_file()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_local_voices_endpoints_roundtrip(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lv"))
    with _client(app) as client:
        base = client.get("/api/voices/local").json()
        assert base["ok"] and base["supported"] == ["piper", "kokoro"]
        assert base["providers"] == {"piper": [], "kokoro": []}

        r = client.post("/api/voices/local/piper", json={
            "name": "Amy", "voice_id": "voices/amy.onnx",
            "tts": {"piper_length_scale": 1.05, "output_device": "dropped"},
        })
        assert r.status_code == 200, r.text
        assert [v["name"] for v in r.json()["providers"]["piper"]] == ["Amy"]
        # Routing fields never travel into a saved voice.
        assert r.json()["voice"]["tts"] == {"piper_length_scale": 1.05}

        r = client.delete("/api/voices/local/piper/Amy")
        assert r.status_code == 200
        assert r.json()["providers"]["piper"] == []
        assert client.delete("/api/voices/local/piper/Amy").status_code == 404


def test_local_voices_rejects_unknown_provider(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lv"))
    with _client(app) as client:
        r = client.post("/api/voices/local/elevenlabs", json={"name": "x"})
        assert r.status_code == 400


def test_export_endpoint_copies_to_another_profile(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lv"))
    from config import save_profile
    save_profile(AppConfig(profile_name="p-dst"), "p-dst")
    with _client(app) as client:
        client.post("/api/voices/local/kokoro", json={"name": "Dora", "voice_id": "pf_dora"})
        r = client.post("/api/voices/export", json={"target_profile": "p-dst"})
        assert r.status_code == 200, r.text
        assert r.json()["copied"] == 1 and r.json()["total"] == 1
        # The same profile is not a valid target, and an unknown one is a 404.
        assert client.post(
            "/api/voices/export", json={"target_profile": "p-lv"}
        ).status_code == 400
        assert client.post(
            "/api/voices/export", json={"target_profile": "nope"}
        ).status_code == 404


def test_export_endpoint_copies_a_single_named_voice(tmp_path, monkeypatch):
    """The per-row ⇪ button exports one voice via ``names``."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-one"))
    from config import save_profile
    save_profile(AppConfig(profile_name="p-only"), "p-only")
    with _client(app) as client:
        client.post("/api/voices/local/kokoro", json={"name": "Dora", "voice_id": "pf_dora"})
        client.post("/api/voices/local/kokoro", json={"name": "Alex", "voice_id": "pm_alex"})
        r = client.post("/api/voices/export",
                        json={"target_profile": "p-only", "names": ["Dora"]})
        assert r.status_code == 200, r.text
        assert r.json()["copied"] == 1 and r.json()["total"] == 1

    from tts.voice_lab import load_library
    assert [p.name for p in load_library("p-only")] == ["Dora"]


def test_mirror_endpoints_diff_then_equalise_both_ways(tmp_path, monkeypatch):
    """⇄ mirror: the GET only reports the diff, the POST makes both libraries equal."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-mir"))
    from config import save_profile
    from tts import local_voices as lv
    save_profile(AppConfig(profile_name="p-other"), "p-other")
    lv.add_local("p-mir", "kokoro", {"name": "Mine only", "voice_id": "pf_dora"})
    lv.add_local("p-other", "piper", {"name": "Theirs only", "voice_id": "voices/a.onnx"})
    lv.add_local("p-other", "piper", {"name": "Shared", "voice_id": "voices/s.onnx"})

    def names(client, profile=None):
        params = {"profile": profile} if profile else None
        data = client.get("/api/voices/library", params=params).json()
        return sorted(p["name"] for p in data["presets"])

    with _client(app) as client:
        r = client.get("/api/voices/mirror", params={"other": "p-other"})
        assert r.status_code == 200, r.text
        diff = r.json()
        assert diff["this_profile"] == "p-mir" and diff["other_profile"] == "p-other"
        assert diff["to_other"] == ["Mine only"]
        assert sorted(diff["to_this"]) == ["Shared", "Theirs only"]
        assert diff["this_count"] == 1 and diff["other_count"] == 2
        # Reading the diff changes nothing on either side.
        assert names(client) == ["Mine only"]
        assert names(client, "p-other") == ["Shared", "Theirs only"]

        for bad in [{"other": "p-mir"}, {"other": "nope"}, {}]:
            assert client.get("/api/voices/mirror", params=bad).status_code in (400, 404)
        assert client.post("/api/voices/mirror", json={"other_profile": "p-mir"}).status_code == 400
        assert client.post("/api/voices/mirror", json={"other_profile": "nope"}).status_code == 404
        assert client.post("/api/voices/mirror", json={"other_profile": ""}).status_code == 400

        r = client.post("/api/voices/mirror", json={"other_profile": "p-other"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["to_other"] == 1 and d["to_this"] == 2
        assert d["this_total"] == 3 and d["other_total"] == 3
        assert names(client) == ["Mine only", "Shared", "Theirs only"]
        assert names(client, "p-other") == ["Mine only", "Shared", "Theirs only"]
        # The response carries the diff as it stands AFTER applying: nothing left.
        assert d["plan"]["to_other"] == [] and d["plan"]["to_this"] == []
        assert sorted(p["name"] for p in d["presets"]) == ["Mine only", "Shared", "Theirs only"]

        # An unknown policy is refused instead of guessing.
        r = client.post("/api/voices/mirror",
                        json={"other_profile": "p-other", "conflicts": "nope"})
        assert r.status_code == 400 and "policy" in r.json()["detail"]

        # A second mirror with no differences writes nothing.
        d = client.post("/api/voices/mirror", json={"other_profile": "p-other"}).json()
        assert d["to_other"] == 0 and d["to_this"] == 0
        assert d["this_total"] == 3 and d["other_total"] == 3


def test_mirror_endpoint_conflict_policy_picks_a_winner(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-win"))
    from config import save_profile
    from tts import local_voices as lv, voice_lab as vl
    save_profile(AppConfig(profile_name="p-lose"), "p-lose")
    lv.add_local("p-win", "piper", {"name": "Clash", "voice_id": "voices/mine.onnx"})
    lv.add_local("p-lose", "kokoro", {"name": "Clash", "voice_id": "pf_dora"})

    with _client(app) as client:
        # Default policy: differing voices are left exactly as they are.
        d = client.post("/api/voices/mirror", json={"other_profile": "p-lose"}).json()
        assert d["skipped"] == 1 and d["to_other"] == 0 and d["to_this"] == 0
        assert vl.load_library("p-lose")[0].voice_id == "pf_dora"

        # "this": my version overwrites theirs.
        d = client.post("/api/voices/mirror",
                        json={"other_profile": "p-lose", "conflicts": "this"}).json()
        assert d["policy"] == "this" and d["to_other"] == 1 and d["skipped"] == 0
        assert vl.load_library("p-lose")[0].voice_id == "voices/mine.onnx"

        # "other": their version overwrites mine.
        vl.add_preset("p-lose", {"name": "Clash", "provider": "kokoro", "voice_id": "pf_other"})
        d = client.post("/api/voices/mirror",
                        json={"other_profile": "p-lose", "conflicts": "other"}).json()
        assert d["to_this"] == 1
        assert vl.load_library("p-win")[0].voice_id == "pf_other"


def test_voice_partner_badge_tracks_divergence(tmp_path, monkeypatch):
    """Voice partner + its badge: GET /api/voices/partners reports each profile
    against its partner, POST /api/voices/partner only *records* the choice."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-sync"))
    from config import load_profile, save_profile
    from tts import local_voices as lv
    save_profile(AppConfig(profile_name="p-buddy"), "p-buddy")
    lv.add_local("p-sync", "kokoro", {"name": "Mine", "voice_id": "pf_dora"})

    blank = {"partner": "", "stale": False, "out_of_sync": False,
             "to_other": 0, "to_this": 0, "conflicts": 0}

    with _client(app) as client:
        # Nothing chosen yet → no badge anywhere.
        d = client.get("/api/voices/partners").json()
        assert d["active"] == "p-sync"
        assert d["profiles"]["p-sync"] == blank
        assert d["profiles"]["p-buddy"] == blank

        # Choosing a partner records it and copies nothing.
        r = client.post("/api/voices/partner", json={"partner": "p-buddy"})
        assert r.status_code == 200, r.text
        assert r.json()["out_of_sync"] is True and r.json()["to_other"] == 1
        assert load_profile("p-sync").voice_partner == "p-buddy"
        assert lv.list_local("p-buddy", "kokoro") == []
        d = client.get("/api/voices/partners").json()["profiles"]
        assert d["p-sync"]["out_of_sync"] is True and d["p-sync"]["to_other"] == 1
        assert d["p-buddy"] == blank          # the partner chose nothing itself

        # Refusals: a profile cannot partner itself, and the name must exist.
        assert client.post("/api/voices/partner", json={"partner": "p-sync"}).status_code == 400
        assert client.post("/api/voices/partner", json={"partner": "nope"}).status_code == 404

        # Mirroring evens them out → the badge clears.
        client.post("/api/voices/mirror", json={"other_profile": "p-buddy"})
        d = client.get("/api/voices/partners").json()["profiles"]["p-sync"]
        assert d["out_of_sync"] is False and d["identical"] == 1

        # A shared name whose settings DIFFER is divergence too ("skip" would
        # never resolve it), which is the case the badge exists for.
        lv.add_local("p-buddy", "kokoro", {"name": "Mine", "voice_id": "pf_other"})
        d = client.get("/api/voices/partners").json()["profiles"]["p-sync"]
        assert d["out_of_sync"] is True and d["conflicts"] == 1

        # Clearing the partnership removes the badge.
        assert client.post("/api/voices/partner", json={}).status_code == 200
        assert load_profile("p-sync").voice_partner == ""
        d = client.get("/api/voices/partners").json()["profiles"]["p-sync"]
        assert d == blank


def test_switching_to_a_partnered_profile_syncs_its_voices(tmp_path, monkeypatch):
    """Activating a profile with a voice partner brings the two libraries in
    step on the spot — and reports what it copied."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-main"))
    from config import load_profile, save_profile
    from tts import local_voices as lv, voice_lab as vl
    save_profile(AppConfig(profile_name="p-friend"), "p-friend")
    save_profile(AppConfig(profile_name="p-lone"), "p-lone")
    lv.add_local("p-main", "kokoro", {"name": "Mine", "voice_id": "pf_dora"})
    lv.add_local("p-friend", "piper", {"name": "Theirs", "voice_id": "voices/a.onnx"})

    with _client(app) as client:
        # No partner on the target → the switch only switches, and writes nothing.
        d = client.put("/api/profiles/p-lone/activate").json()
        assert d["partner_sync"] == {}
        assert lv.list_local("p-lone", "piper") == []

        client.put("/api/profiles/p-main/activate")
        cfg = load_profile("p-friend")
        cfg.voice_partner = "p-main"
        save_profile(cfg, "p-friend")

        d = client.put("/api/profiles/p-friend/activate").json()
        assert d["active"] == "p-friend"
        assert d["partner_sync"] == {
            "partner": "p-main", "synced": True,
            "to_other": 1, "to_this": 1, "skipped": 0,
        }
        assert sorted(p.name for p in vl.load_library("p-main")) == ["Mine", "Theirs"]
        assert sorted(p.name for p in vl.load_library("p-friend")) == ["Mine", "Theirs"]

        # Switched back, p-main now points at p-friend — already in step, no report.
        cfg = load_profile("p-main")
        cfg.voice_partner = "p-friend"
        save_profile(cfg, "p-main")
        assert client.put("/api/profiles/p-main/activate").json()["partner_sync"] == {}

        # A partner that was deleted is ignored (the badge reports it, not this).
        cfg = load_profile("p-main")
        cfg.voice_partner = "p-vanished"
        save_profile(cfg, "p-main")
        assert client.put("/api/profiles/p-main/activate").json()["partner_sync"] == {}


def test_auto_sync_leaves_differing_voices_alone_and_reports_them(tmp_path, monkeypatch):
    """The switch sync uses the "skip" policy: a shared name with different
    settings is never overwritten, it is reported as left alone."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-one"))
    from config import save_profile
    from tts import local_voices as lv, voice_lab as vl
    save_profile(AppConfig(profile_name="p-two", voice_partner="p-one"), "p-two")
    lv.add_local("p-one", "piper", {"name": "Clash", "voice_id": "voices/mine.onnx"})
    lv.add_local("p-two", "kokoro", {"name": "Clash", "voice_id": "pf_dora"})

    with _client(app) as client:
        d = client.put("/api/profiles/p-two/activate").json()["partner_sync"]
        assert d == {"partner": "p-one", "synced": True,
                     "to_other": 0, "to_this": 0, "skipped": 1}
        # Neither version was touched.
        assert vl.load_library("p-one")[0].voice_id == "voices/mine.onnx"
        assert vl.load_library("p-two")[0].voice_id == "pf_dora"


def test_auto_sync_failure_never_breaks_the_switch(tmp_path, monkeypatch):
    """A library too full to take the partner's voices reports the refusal — the
    profile switch itself must still succeed."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-big"))
    from config import save_profile
    from tts import voice_lab as vl
    save_profile(AppConfig(profile_name="p-small", voice_partner="p-big"), "p-small")
    for i in range(vl.MAX_PRESETS):
        vl.add_preset("p-big", {"name": f"v{i:02d}", "provider": "piper"})
    vl.add_preset("p-small", {"name": "one-more", "provider": "piper"})

    with _client(app) as client:
        d = client.put("/api/profiles/p-small/activate").json()
        assert d["ok"] is True and d["active"] == "p-small"
        sync = d["partner_sync"]
        assert sync["partner"] == "p-big" and sync["synced"] is False
        assert str(vl.MAX_PRESETS) in sync["error"]
        # Nothing was written on either side.
        assert len(vl.load_library("p-big")) == vl.MAX_PRESETS
        assert [p.name for p in vl.load_library("p-small")] == ["one-more"]


def test_voice_partner_that_disappeared_is_reported_stale(tmp_path, monkeypatch):
    """A partner that was renamed/deleted (or a hand-edited YAML pointing at
    itself) is flagged stale — never silently rewritten."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-a"))
    from config import delete_profile, load_profile, save_profile
    save_profile(AppConfig(profile_name="p-gone"), "p-gone")
    save_profile(AppConfig(profile_name="p-self", voice_partner="p-self"), "p-self")

    with _client(app) as client:
        client.post("/api/voices/partner", json={"partner": "p-gone"})
        delete_profile("p-gone")
        d = client.get("/api/voices/partners").json()["profiles"]
        assert d["p-a"]["partner"] == "p-gone" and d["p-a"]["stale"] is True
        assert d["p-a"]["out_of_sync"] is False
        assert d["p-self"]["stale"] is True
        # The stale choice stays on disk — reading must not rewrite a profile.
        assert load_profile("p-a").voice_partner == "p-gone"


def test_copy_and_pull_ask_before_replacing_a_tuned_voice(tmp_path, monkeypatch):
    """Both directions answer 409 with the exact diff, then apply with overwrite."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-copy"))
    from config import save_profile
    from tts import local_voices as lv, voice_lab as vl
    save_profile(AppConfig(profile_name="p-other"), "p-other")
    lv.add_local("p-copy", "piper", {"name": "Amy", "voice_id": "voices/new.onnx"})
    lv.add_local("p-other", "piper", {"name": "Amy", "voice_id": "voices/old.onnx"})

    with _client(app) as client:
        # ⇪ copy: the destination would lose its differently tuned “Amy”.
        r = client.post("/api/voices/export",
                        json={"target_profile": "p-other", "names": ["Amy"]})
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["conflicts"] == [{
            "name": "Amy", "current": "piper : voices/old.onnx",
            "incoming": "piper : voices/new.onnx"}]
        assert "Amy" in r.json()["detail"]["message"]
        assert vl.load_library("p-other")[0].voice_id == "voices/old.onnx"   # nothing written

        r = client.post("/api/voices/export", json={
            "target_profile": "p-other", "names": ["Amy"], "overwrite": True})
        assert r.status_code == 200, r.text
        assert r.json()["replaced"] == 1 and r.json()["added"] == 0
        assert vl.load_library("p-other")[0].voice_id == "voices/new.onnx"

        # ⇩ pull: the guard fires in the other direction too.
        vl.add_preset("p-other", {"name": "Amy", "provider": "elevenlabs", "voice_id": "v-1"})
        r = client.post("/api/voices/pull",
                        json={"source_profile": "p-other", "names": ["Amy"]})
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["conflicts"][0]["incoming"] == "elevenlabs : v-1"
        r = client.post("/api/voices/pull", json={
            "source_profile": "p-other", "names": ["Amy"], "overwrite": True})
        assert r.status_code == 200, r.text
        assert r.json()["replaced"] == 1
        assert vl.load_library("p-copy")[0].provider == "elevenlabs"


def test_pull_endpoint_brings_voices_from_another_profile(tmp_path, monkeypatch):
    """⇩ pull is the inverse of ⇪: it reads ANOTHER profile's library and copies
    the ticked voices into the active one."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-dst"))
    from config import save_profile
    from tts import local_voices as lv, voice_lab as vl
    save_profile(AppConfig(profile_name="p-src"), "p-src")
    lv.add_local("p-src", "kokoro", {"name": "Dora", "voice_id": "pf_dora",
                                      "tts": {"kokoro_lang_code": "p"}})
    lv.add_local("p-src", "piper", {"name": "Amy", "voice_id": "voices/amy.onnx"})

    with _client(app) as client:
        # A peek at another profile's library is read-only...
        peek = client.get("/api/voices/library", params={"profile": "p-src"})
        assert peek.status_code == 200, peek.text
        assert peek.json()["profile"] == "p-src"
        assert sorted(p["name"] for p in peek.json()["presets"]) == ["Amy", "Dora"]
        assert client.get("/api/voices/library", params={"profile": "nope"}).status_code == 404
        # ...and the active profile is still the default target of the plain call.
        assert client.get("/api/voices/library").json()["profile"] == "p-dst"

        r = client.post("/api/voices/pull", json={"source_profile": "p-src"})
        assert r.status_code == 200, r.text
        assert r.json()["copied"] == 2 and r.json()["total"] == 2
        assert sorted(p["name"] for p in r.json()["presets"]) == ["Amy", "Dora"]

        # A subset pulls just those (and the export direction still works).
        client.delete("/api/voices/library/Amy")
        r = client.post("/api/voices/pull",
                        json={"source_profile": "p-src", "names": ["Amy"]})
        assert r.status_code == 200, r.text
        assert r.json()["copied"] == 1 and r.json()["total"] == 2

        # Refusals: same profile, unknown profile, empty source, unknown names.
        assert client.post("/api/voices/pull", json={"source_profile": "p-dst"}).status_code == 400
        assert client.post("/api/voices/pull", json={"source_profile": "nope"}).status_code == 404
        assert client.post("/api/voices/pull", json={"source_profile": ""}).status_code == 400
        assert client.post("/api/voices/pull",
                           json={"source_profile": "p-src",
                                 "names": ["ghost"]}).status_code == 400

    # Pulling never mutates the source library.
    assert sorted(p.name for p in vl.load_library("p-src")) == ["Amy", "Dora"]


def test_pull_endpoint_refuses_an_empty_source_profile(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-dst2"))
    from config import save_profile
    save_profile(AppConfig(profile_name="p-bare"), "p-bare")
    with _client(app) as client:
        r = client.post("/api/voices/pull", json={"source_profile": "p-bare"})
        assert r.status_code == 400, r.text
        assert "no saved voices" in r.json()["detail"]


def test_export_endpoint_copies_only_the_ticked_voices(tmp_path, monkeypatch):
    """The checkbox selection sends a LIST of names; only those are copied."""
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-pick"))
    from config import save_profile
    save_profile(AppConfig(profile_name="p-bulk"), "p-bulk")
    with _client(app) as client:
        for name, voice in [("Dora", "pf_dora"), ("Alex", "pm_alex"), ("Sam", "am_sam")]:
            client.post("/api/voices/local/kokoro", json={"name": name, "voice_id": voice})
        r = client.post("/api/voices/export",
                        json={"target_profile": "p-bulk", "names": ["Dora", "Sam"]})
        assert r.status_code == 200, r.text
        assert r.json()["copied"] == 2 and r.json()["total"] == 2

    from tts.voice_lab import load_library
    assert sorted(p.name for p in load_library("p-bulk")) == ["Dora", "Sam"]


def test_local_voices_is_per_profile(tmp_path, monkeypatch):
    app = _profiled_app(tmp_path, monkeypatch, AppConfig(profile_name="p-lv"))
    from config import save_profile
    with _client(app) as client:
        client.post("/api/voices/local/kokoro", json={"name": "A", "voice_id": "af_heart"})
        save_profile(AppConfig(profile_name="p-other"), "p-other")
        assert client.put("/api/profiles/p-other/activate").status_code == 200
        assert client.get("/api/voices/local").json()["providers"]["kokoro"] == []
        client.put("/api/profiles/p-lv/activate")
        assert len(client.get("/api/voices/local").json()["providers"]["kokoro"]) == 1

"""Kokoro voice blending — averaging packs into a new local voice.

A blend is only arithmetic over style tensors plus a recipe file, which is
exactly what these tests pin: the weights are normalized, the packs must share
a language and a shape, missing packs are named instead of skipped, and saving
registers the voice so the pickers (and the A/B) see it like any other.
Torch is stubbed out — the mixing itself is numpy, and loading a real model is
not what is being tested here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from config import AppConfig, Runtime, Secrets
from tts import kokoro_blend as kb
from tts import kokoro_setup as ks


class _FakeTorch:
    """The only torch surface the blender touches: from_numpy + save."""

    float32 = "float32"

    @staticmethod
    def from_numpy(arr):          # noqa: D401 - a shim, not an API
        return arr

    @staticmethod
    def save(obj, path):
        Path(path).write_bytes(b"blend-pack")


@pytest.fixture()
def voices(tmp_path, monkeypatch):
    directory = tmp_path / "kokoro"
    directory.mkdir()
    monkeypatch.setattr(ks, "DEFAULT_VOICES_DIR", directory)
    monkeypatch.setattr(kb, "_torch", lambda: _FakeTorch)
    return directory


def _pack(directory: Path, name: str, per_row: float, rows: int = 4, dim: int = 3):
    """A downloaded pack on disk + the tensor a loader would return."""
    (directory / f"{name}.pt").write_bytes(b"x")
    return np.repeat(
        np.full((1, 1, dim), per_row, dtype=np.float32), rows, axis=0
    )


def _loader(mapping):
    return lambda path: mapping[Path(path).stem]


# ---------------------------------------------------------------------------
# Names and languages
# ---------------------------------------------------------------------------

def test_blend_id_is_a_legal_kokoro_voice_name():
    assert kb.blend_id("Warm Dora") == "kb_warm_dora"
    assert kb.blend_id("Voz Quente!") == "kb_voz_quente"
    assert ks.is_valid_voice_name(kb.blend_id("Voz Quente!"))
    assert kb.is_blend_id(kb.blend_id("a b"))


@pytest.mark.parametrize("name", ["", "   ", "!!!"])
def test_blend_id_refuses_a_nameless_blend(name):
    with pytest.raises(kb.KokoroBlendError):
        kb.blend_id(name)


def test_lang_code_comes_from_the_id_or_the_recipe(voices):
    assert kb.lang_code_for("pf_dora") == "p"
    assert kb.lang_code_for("af_heart") == "a"
    _pack(voices, "pf_dora", 0.2)
    _pack(voices, "pm_alex", 0.8)
    entry = kb.save_blend("Dora Alex", [
        {"voice": "pf_dora", "weight": 1}, {"voice": "pm_alex", "weight": 1},
    ], loader=_loader({"pf_dora": _pack(voices, "pf_dora", 0.2),
                       "pm_alex": _pack(voices, "pm_alex", 0.8)}))
    # The blend id says nothing about the phoneme language; the recipe does.
    assert kb.lang_code_for(entry["id"]) == "p"


# ---------------------------------------------------------------------------
# The mix
# ---------------------------------------------------------------------------

def test_weights_are_normalized_to_a_share(voices):
    loader = _loader({"a_lo": _pack(voices, "a_lo", 0.0),
                      "a_hi": _pack(voices, "a_hi", 1.0)})
    mixed = kb.blend_sources(
        [{"voice": "a_lo", "weight": 25}, {"voice": "a_hi", "weight": 75}],
        loader=loader,
    )
    assert mixed["weights"] == [0.25, 0.75]
    assert np.allclose(np.asarray(mixed["tensor"]), 0.75)


def test_absolute_weight_scale_does_not_change_the_result(voices):
    loader = _loader({"a_lo": _pack(voices, "a_lo", 0.0),
                      "a_hi": _pack(voices, "a_hi", 1.0)})
    small = kb.blend_sources(
        [{"voice": "a_lo", "weight": 1}, {"voice": "a_hi", "weight": 3}], loader=loader)
    big = kb.blend_sources(
        [{"voice": "a_lo", "weight": 10}, {"voice": "a_hi", "weight": 30}], loader=loader)
    assert np.allclose(np.asarray(small["tensor"]), np.asarray(big["tensor"]))


def test_a_real_midpoint_sits_between_the_two_packs(voices):
    loader = _loader({"a_lo": _pack(voices, "a_lo", -1.0),
                      "a_hi": _pack(voices, "a_hi", 1.0)})
    mixed = kb.blend_sources(
        [{"voice": "a_lo", "weight": 1}, {"voice": "a_hi", "weight": 1}], loader=loader)
    assert np.allclose(np.asarray(mixed["tensor"]), 0.0)


def test_different_row_counts_are_padded_with_the_last_row(voices):
    short = _pack(voices, "a_short", 1.0, rows=2)
    long = _pack(voices, "a_long", 3.0, rows=4)
    mixed = kb.blend_sources(
        [{"voice": "a_short", "weight": 1}, {"voice": "a_long", "weight": 1}],
        loader=_loader({"a_short": short, "a_long": long}),
    )
    tensor = np.asarray(mixed["tensor"])
    assert tensor.shape[0] == 4
    assert np.allclose(tensor, 2.0)


def test_a_mismatched_shape_is_refused(voices):
    loader = _loader({"a_one": _pack(voices, "a_one", 1.0, dim=3),
                      "a_two": _pack(voices, "a_two", 1.0, dim=5)})
    with pytest.raises(kb.KokoroBlendError, match="different style shape"):
        kb.blend_sources(
            [{"voice": "a_one", "weight": 1}, {"voice": "a_two", "weight": 1}],
            loader=loader,
        )


@pytest.mark.parametrize("sources, message", [
    ([{"voice": "af_heart"}], "at least 2"),
    ([{"voice": "af_heart"}, {"voice": "af_heart"}], "already in the blend"),
    ([{"voice": "af_heart", "weight": 0}, {"voice": "am_adam"}], "above zero"),
    ([{"voice": "af_heart", "weight": "x"}, {"voice": "am_adam"}], "not a number"),
    ([{"voice": "not a voice"}, {"voice": "am_adam"}], "does not look like"),
    ([{"voice": f"a_v{i}"} for i in range(5)], "at most 4"),
])
def test_bad_source_lists_are_refused(voices, sources, message):
    for source in sources:
        voice = source.get("voice", "")
        if ks.is_valid_voice_name(voice):
            _pack(voices, voice, 0.5)
    with pytest.raises(kb.KokoroBlendError, match=message):
        kb.blend_sources(sources)


def test_mixed_languages_are_refused(voices):
    loader = _loader({"af_heart": _pack(voices, "af_heart", 0.5),
                      "pf_dora": _pack(voices, "pf_dora", 0.5)})
    with pytest.raises(kb.KokoroBlendError, match="different languages"):
        kb.blend_sources(
            [{"voice": "af_heart", "weight": 1}, {"voice": "pf_dora", "weight": 1}],
            loader=loader,
        )


def test_a_pack_that_was_never_downloaded_is_named(voices):
    loader = _loader({"af_heart": _pack(voices, "af_heart", 0.5)})
    with pytest.raises(kb.KokoroBlendError, match="am_adam is not downloaded"):
        kb.blend_sources(
            [{"voice": "af_heart", "weight": 1}, {"voice": "am_adam", "weight": 1}],
            loader=loader,
        )


# ---------------------------------------------------------------------------
# The registry — what went into each blend
# ---------------------------------------------------------------------------

def test_saving_writes_the_pack_and_the_recipe(voices):
    loader = _loader({"pf_dora": _pack(voices, "pf_dora", 0.2),
                      "pm_alex": _pack(voices, "pm_alex", 0.8)})
    entry = kb.save_blend("Dora Alex", [
        {"voice": "pf_dora", "weight": 60}, {"voice": "pm_alex", "weight": 40},
    ], loader=loader, notes="bright + calm")

    assert (voices / "kb_dora_alex.pt").is_file()
    assert entry["id"] == "kb_dora_alex"
    assert entry["lang_code"] == "p"
    assert entry["sources"] == [
        {"voice": "pf_dora", "weight": 0.6},
        {"voice": "pm_alex", "weight": 0.4},
    ]
    assert kb.describe(entry) == "pf_dora 60% + pm_alex 40% (p)"

    listed = kb.list_blends()
    assert [b["id"] for b in listed] == ["kb_dora_alex"]
    assert listed[0]["notes"] == "bright + calm"
    assert kb.get_blend("kb_dora_alex")["name"] == "Dora Alex"


def test_saving_the_same_name_replaces_the_recipe(voices):
    loader = _loader({"af_heart": _pack(voices, "af_heart", 0.5),
                      "am_adam": _pack(voices, "am_adam", 0.5)})
    first = kb.save_blend("Main", [
        {"voice": "af_heart", "weight": 1}, {"voice": "am_adam", "weight": 1},
    ], loader=loader)
    second = kb.save_blend("Main", [
        {"voice": "af_heart", "weight": 3}, {"voice": "am_adam", "weight": 1},
    ], loader=loader)
    assert first["id"] == second["id"]
    assert len(kb.list_blends()) == 1
    assert second["sources"][0]["weight"] == 0.75


def test_deleting_a_blend_removes_pack_and_recipe(voices):
    loader = _loader({"af_heart": _pack(voices, "af_heart", 0.5),
                      "am_adam": _pack(voices, "am_adam", 0.5)})
    entry = kb.save_blend("Gone", [
        {"voice": "af_heart", "weight": 1}, {"voice": "am_adam", "weight": 1},
    ], loader=loader)
    assert kb.remove_blend(entry["id"]) is True
    assert not (voices / f"{entry['id']}.pt").exists()
    assert kb.list_blends() == []
    assert kb.remove_blend(entry["id"]) is False


def test_a_plain_voice_cannot_be_deleted_through_the_blend_route(voices):
    _pack(voices, "af_heart", 0.5)
    with pytest.raises(kb.KokoroBlendError, match="not a blend id"):
        kb.remove_blend("af_heart")
    assert (voices / "af_heart.pt").is_file()


def test_a_corrupt_weight_in_the_registry_does_not_break_the_list(voices):
    """The registry is a plain file: a bad number must not 500 every blend route."""
    loader = _loader({"af_heart": _pack(voices, "af_heart", 0.5),
                      "am_adam": _pack(voices, "am_adam", 0.5)})
    kb.save_blend("Mixed", [
        {"voice": "af_heart", "weight": 1}, {"voice": "am_adam", "weight": 1},
    ], loader=loader)
    registry = voices / "blends.json"
    registry.write_text(
        registry.read_text(encoding="utf-8").replace('"weight": 0.5', '"weight": "half"'),
        encoding="utf-8",
    )
    listed = kb.list_blends()
    assert [b["id"] for b in listed] == ["kb_mixed"]
    assert [s["weight"] for s in listed[0]["sources"]] == [0.0, 0.0]
    assert kb.describe(listed[0]) == "af_heart 0% + am_adam 0% (a)"


def test_available_voices_marks_blends_and_keeps_plain_packs(voices):
    loader = _loader({"af_heart": _pack(voices, "af_heart", 0.5),
                      "am_adam": _pack(voices, "am_adam", 0.5)})
    entry = kb.save_blend("Mixed", [
        {"voice": "af_heart", "weight": 1}, {"voice": "am_adam", "weight": 1},
    ], loader=loader)
    names = {v["name"]: v for v in kb.available_voices()}
    assert names["af_heart"]["blend"] is False
    assert names["af_heart"]["lang_code"] == "a"
    assert names[entry["id"]]["blend"] is True
    assert names[entry["id"]]["lang_code"] == "a"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _client(tmp_path, monkeypatch, cfg=None):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import activate_profile, load_profile, save_profile
    cfg = cfg or AppConfig(profile_name="blend-p")
    save_profile(cfg, cfg.profile_name)
    activate_profile(cfg.profile_name)
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile(cfg.profile_name), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    app = _build_app(DashboardState(), None)
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_blend_endpoints_create_list_and_delete(tmp_path, monkeypatch, voices):
    packs = {"pf_dora": _pack(voices, "pf_dora", 0.2),
             "pm_alex": _pack(voices, "pm_alex", 0.8)}
    # The route mixes through blend_sources with no test loader, so the pack
    # reader is the seam to fake — no torch and no 300 MB download in CI.
    monkeypatch.setattr(kb, "load_style_pack", lambda voice, loader=None: packs[voice])

    with _client(tmp_path, monkeypatch) as client:
        base = client.get("/api/tts/kokoro/blends").json()
        assert base["ok"] and base["blends"] == []
        assert {v["name"] for v in base["voices"]} == {"pf_dora", "pm_alex"}

        r = client.post("/api/tts/kokoro/blends", json={
            "name": "Dora Alex",
            "sources": [{"voice": "pf_dora", "weight": 60},
                        {"voice": "pm_alex", "weight": 40}],
            "tts": {"kokoro_speed": 1.1, "output_device": "dropped"},
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["blend"]["id"] == "kb_dora_alex"
        # The saved voice is a normal library preset, with the language attached
        # (the response's own "preset" is the Kokoro-bucket projection of it).
        assert body["preset"]["voice_id"] == "kb_dora_alex"
        assert body["preset"]["tts"] == {"kokoro_speed": 1.1, "kokoro_lang_code": "p"}
        library = {p["name"]: p for p in body["presets"]}
        assert library["Dora Alex"]["provider"] == "kokoro"
        assert library["Dora Alex"]["voice_id"] == "kb_dora_alex"
        assert library["Dora Alex"]["source"] == "blend:kokoro"

        r = client.delete("/api/tts/kokoro/blends/kb_dora_alex")
        assert r.status_code == 200, r.text
        assert r.json()["blends"] == []
        assert r.json()["presets"] == []          # the stale saved voice went with it
        assert r.json()["voices_removed"] == 1
        assert client.delete("/api/tts/kokoro/blends/kb_dora_alex").status_code == 404


def test_blend_preview_returns_a_playable_wav(tmp_path, monkeypatch, voices):
    packs = {"af_heart": _pack(voices, "af_heart", 0.2),
             "am_adam": _pack(voices, "am_adam", 0.8)}
    monkeypatch.setattr(kb, "load_style_pack", lambda voice, loader=None: packs[voice])
    from tts import kokoro as kp
    monkeypatch.setattr(
        kp, "iter_style_pcm",
        lambda text, style, **kw: iter([(np.zeros(2400, dtype="<i2")).tobytes()]),
    )

    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/api/tts/kokoro/blends/preview", json={
            "sources": [{"voice": "af_heart", "weight": 1},
                        {"voice": "am_adam", "weight": 1}],
            "text": "one line",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["lang_code"] == "a"
        assert body["seconds"] == 0.1
        assert body["wav_b64"]          # a real RIFF/WAVE payload, base64'd
        assert client.post("/api/tts/kokoro/blends/preview",
                           json={"sources": [], "text": ""}).status_code == 400


def test_blend_preview_refuses_a_nonsense_speed(tmp_path, monkeypatch, voices):
    packs = {"af_heart": _pack(voices, "af_heart", 0.2),
             "am_adam": _pack(voices, "am_adam", 0.8)}
    monkeypatch.setattr(kb, "load_style_pack", lambda voice, loader=None: packs[voice])
    body = {"sources": [{"voice": "af_heart", "weight": 1},
                         {"voice": "am_adam", "weight": 1}], "text": "one line"}

    with _client(tmp_path, monkeypatch) as client:
        for bad in (0, 0.01, 50):
            r = client.post("/api/tts/kokoro/blends/preview", json={**body, "speed": bad})
            assert r.status_code == 400, (bad, r.text)
            assert "speed" in r.json()["detail"]
        # NaN/inf cannot go through ``json=`` (the client refuses to serialize
        # them), so they are posted as raw JSON — which the server does parse.
        for bad in ("NaN", "Infinity"):
            raw = json.dumps(body)[:-1] + f', "speed": {bad}}}'
            r = client.post("/api/tts/kokoro/blends/preview", content=raw,
                            headers={"content-type": "application/json"})
            assert r.status_code == 400, (bad, r.text)
        assert client.post("/api/tts/kokoro/blends/preview",
                           json={**body, "pitch": 120}).status_code == 400


def test_an_oversized_registry_is_not_read(voices):
    (voices / "blends.json").write_bytes(b" " * (kb.REGISTRY_MAX_BYTES + 1))
    assert kb.list_blends() == []
    assert kb.available_voices() is not None      # still a usable list, not a 500

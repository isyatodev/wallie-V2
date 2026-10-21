"""Voice port — cloned cloud voice → local Piper/Kokoro package.

Two things must never silently break here:

* the ``.json`` shapes users actually have on disk (Wallie backup, Fish model
  dump, ElevenLabs voice dump) have to be understood — a wrong guess builds a
  package for the wrong voice, or for no voice at all;
* the package layout has to match what each local engine's loader expects
  (Piper 22.05 kHz + ``metadata.csv``, Kokoro 24 kHz, engine-sane file names at
  attach time), because the failure only shows up much later, inside the TTS
  provider.

No network: the capture provider is stubbed, and every path is a tmp dir.
"""
from __future__ import annotations

import base64
import json
import wave
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets
from tts import kokoro_setup, local_voices, piper_setup, voice_lab
from tts import voice_port as vp
from tts.base import TTSError

REPO = Path(__file__).resolve().parents[1]
LUNA_JSON = REPO / "voices" / "wallie-voice-luna.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    """Isolated profile dir + voices tree; the engines' loaders read the same."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "profiles").mkdir()
    monkeypatch.setattr(piper_setup, "DEFAULT_VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(kokoro_setup, "DEFAULT_VOICES_DIR", tmp_path / "voices" / "kokoro")
    return tmp_path


class _FakeTTS:
    """Streams one second of tone wrapped in silence, per line."""

    def __init__(self) -> None:
        self.sample_rate = vp.CAPTURE_SR
        self.texts: list[str] = []
        self.closed = False

    async def synthesize(self, text: str):
        self.texts.append(text)
        silence = b"\x00\x00" * 2400          # 100 ms at 24 kHz
        tone = b"\x00\x64" * 12000            # 25600 in little-endian int16
        yield silence + tone + silence

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_tts(monkeypatch):
    fake = _FakeTTS()
    monkeypatch.setattr(vp, "_tts_for", lambda provider, voice_id, secrets: fake)
    return fake


def _wav_info(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


# ---------------------------------------------------------------------------
# Reading the .json
# ---------------------------------------------------------------------------

def test_real_wallie_backup_is_understood():
    payload = json.loads(LUNA_JSON.read_text(encoding="utf-8"))
    sources = vp.parse_sources(payload, text=LUNA_JSON.read_text(encoding="utf-8"))
    assert len(sources) == 1
    src = sources[0]
    assert src.name == "luna"
    assert src.provider == "fish"
    assert src.voice_id == "1d76aed89caa49e28d9c7ff6d35a9f3c"
    assert src.supported


def test_fish_and_elevenlabs_dumps_are_understood():
    fish = vp.parse_sources({
        "_id": "abc123", "title": "Luna Fish", "train_mode": "fast",
        "visibility": "private", "state": "trained",
    })[0]
    assert (fish.provider, fish.voice_id, fish.name) == ("fish", "abc123", "Luna Fish")

    eleven = vp.parse_sources({
        "voice_id": "el_9", "name": "Luna EL", "labels": {"accent": "brazilian"},
        "category": "cloned",
    })[0]
    assert (eleven.provider, eleven.voice_id) == ("elevenlabs", "el_9")


def test_library_file_and_bare_list_are_understood():
    raw = {"profile": "luna", "presets": [
        {"name": "a", "provider": "fish", "voice_id": "fa"},
        {"name": "b", "provider": "piper", "voice_id": "pt_BR-x-medium.onnx"},
    ]}
    sources = vp.parse_sources(raw)
    assert [s.name for s in sources] == ["a", "b"]
    # A local voice needs no port — and says why instead of failing later.
    assert sources[1].reason and "already local" in sources[1].reason

    single = vp.parse_sources({"name": "one", "provider": "fish", "voice_id": "v1"})
    assert len(single) == 1


def test_unknown_file_shape_names_the_formats():
    with pytest.raises(vp.VoicePortError) as e:
        vp.parse_sources({"unrelated": 1})
    assert "Wallie backup" in str(e.value)

    with pytest.raises(vp.VoicePortError):
        vp.parse_sources([{"nothing": "here"}])


def test_a_voice_without_an_id_is_reported_not_ported():
    src = vp.parse_sources({"name": "ghost", "provider": "fish"})[0]
    assert not src.supported
    assert "voice id" in src.reason


def test_duplicates_inside_one_file_collapse():
    sources = vp.parse_sources({"presets": [
        {"name": "a", "provider": "fish", "voice_id": "same"},
        {"name": "a again", "provider": "fish", "voice_id": "same"},
    ]})
    assert len(sources) == 1


# ---------------------------------------------------------------------------
# Preflight / corpus sizing
# ---------------------------------------------------------------------------

def test_clips_cover_the_requested_minutes_and_stay_bounded():
    assert vp.clips_for(1, "pt") == 8
    assert vp.clips_for(999, "pt") == vp.MAX_CLIPS
    # Below the floor means the floor, not a negative clip count.
    assert vp.clips_for(-4, "pt") == vp.clips_for(vp.MIN_MINUTES, "pt")
    assert vp.clips_for(-4, "pt") >= vp.MIN_CLIPS
    with pytest.raises(vp.VoicePortError):
        vp.clips_for(1, "zz")


def test_corpus_cycles_when_more_clips_than_lines_are_asked():
    lines = vp.corpus_lines("en", len(vp.CORPUS["en"]["lines"]) + 3)
    assert len(lines) == len(vp.CORPUS["en"]["lines"]) + 3
    assert lines[-1] == vp.CORPUS["en"]["lines"][2]


# An empty key, not "whatever the repo's .env holds" — Secrets reads .env.
NO_KEYS = Secrets(fish_api_key="", elevenlabs_api_key="")


def test_preflight_needs_the_provider_key_and_says_what_is_missing():
    with pytest.raises(vp.VoicePortError) as e:
        vp.preflight(provider="fish", voice_id="v1", target="piper",
                     seconds=60, lang="pt", secrets=NO_KEYS)
    assert "FISH_API_KEY" in str(e.value)

    plan = vp.preflight(provider="fish", voice_id="v1", target="piper",
                        seconds=60, lang="pt", secrets=Secrets(fish_api_key="k"))
    assert plan["clips"] == 8
    assert plan["target"]["sample_rate"] == 22050
    # The honest half: a port captures audio, it does not fetch the model.
    assert any("not its weights" in w for w in plan["warnings"])
    assert any("training dataset" in w for w in plan["warnings"])


def test_preflight_refuses_a_local_voice_as_a_source():
    with pytest.raises(vp.VoicePortError) as e:
        vp.preflight(provider="piper", voice_id="x.model", target="kokoro",
                     seconds=60, lang="pt", secrets=Secrets(fish_api_key="k"))
    assert "already local" in str(e.value)


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def test_trim_silence_keeps_the_speech_and_a_margin():
    pcm = b"\x00\x00" * 4800 + b"\x10\x27" * 2400 + b"\x00\x00" * 4800
    trimmed = vp.trim_silence(pcm, vp.CAPTURE_SR)
    assert 0 < len(trimmed) < len(pcm)
    # The tone survives.
    assert b"\x10\x27" in trimmed
    # …and it is bounded: margin is 50 ms each side, not the whole room tone.
    assert len(trimmed) <= 2400 * 2 + 2 * int(vp.CAPTURE_SR * vp.TRIM_MARGIN_SEC) * 2


def test_trim_silence_returns_nothing_for_silence():
    assert vp.trim_silence(b"\x00\x00" * 1000) == b""
    assert vp.trim_silence(b"") == b""


def test_resample_changes_the_frame_count_and_keeps_the_duration():
    pcm = b"\x00\x64" * 24000               # 1 s at 24 kHz
    out = vp.resample_pcm16(pcm, 24000, 22050)
    frames = len(out) // 2
    assert abs(frames - 22050) <= 1
    assert vp.resample_pcm16(pcm, 24000, 24000) is pcm   # same rate is a no-op


# ---------------------------------------------------------------------------
# Convert → package
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_convert_builds_a_piper_dataset(dirs, fake_tts):
    report = await vp.convert(
        profile="luna", provider="fish", voice_id="v1", name="Luna BR",
        target="piper", minutes=1, lang="pt",
        secrets=Secrets(fish_api_key="k"),
    )
    pkg = Path(report["package"])
    assert pkg.is_dir() and pkg.parent == vp.ported_dir()
    assert report["clips"] == 8
    assert fake_tts.closed, "the provider must be closed even on the happy path"
    assert len(fake_tts.texts) == 8

    wavs = sorted((pkg / "dataset" / "wav").glob("*.wav"))
    assert len(wavs) == 8
    ch, width, rate, frames = _wav_info(wavs[0])
    assert (ch, width, rate) == (1, 2, 22050)
    assert frames > 0

    rows = (pkg / "dataset" / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 8
    assert rows[0].startswith("wav/0001.wav|")
    assert rows[0].split("|", 1)[1] == vp.corpus_lines("pt", 8)[0]

    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "wallie.port"
    assert manifest["source"]["provider"] == "fish"
    assert manifest["target"]["sample_rate"] == 22050
    assert manifest["clips"] == 8
    # The originals are kept at the capture rate for a second engine.
    assert _wav_info(pkg / "corpus" / "0001.wav")[2] == vp.CAPTURE_SR
    assert report["audition"]["url"].startswith("data:audio/wav;base64,")
    assert "attach" in report["next"]


@pytest.mark.asyncio
async def test_convert_kokoro_package_uses_the_engines_rate(dirs, fake_tts):
    report = await vp.convert(
        profile="luna", provider="elevenlabs", voice_id="el1", name="Luna EL",
        target="kokoro", minutes=1, lang="en",
        secrets=Secrets(elevenlabs_api_key="k"),
    )
    pkg = Path(report["package"])
    assert pkg.name == "Luna-EL-kokoro"
    assert _wav_info(pkg / "corpus" / "0001.wav")[2] == 24000
    assert not (pkg / "dataset").exists()
    assert "fine-tune" in report["next"]
    assert report["attach_kind"] == ".pt voicepack"


@pytest.mark.asyncio
async def test_a_silent_clip_fails_the_port_instead_of_building_a_bad_dataset(dirs, monkeypatch):
    class _Mute(_FakeTTS):
        async def synthesize(self, text: str):
            yield b"\x00\x00" * 4800

    monkeypatch.setattr(vp, "_tts_for", lambda *a: _Mute())
    with pytest.raises(vp.VoicePortError) as e:
        await vp.convert(
            profile="luna", provider="fish", voice_id="v1", name="Silent",
            target="piper", minutes=1, lang="pt", secrets=Secrets(fish_api_key="k"),
        )
    assert "silent" in str(e.value)


def test_package_dir_never_escapes_the_ported_folder(dirs):
    """A crafted source name is a folder name — it must not climb out."""
    report = vp.build_package(
        target="piper", name="../../etc", takes=[], source=vp.PortSource(name="x"),
        lang="pt",
    )
    pkg = Path(report["package"]).resolve()
    assert pkg.parent == vp.ported_dir().resolve()
    assert pkg.name == "etc-piper"


# ---------------------------------------------------------------------------
# Attach — the model the user produced from the package
# ---------------------------------------------------------------------------

def test_attach_piper_copies_both_files_and_registers_the_voice(dirs):
    res = vp.attach(
        profile="luna", target="piper", name="pt_BR-luna-medium",
        display_name="Luna local", source_provider="fish",
        files=[("luna.onnx", b"ONNX"), ("luna.onnx.json", b"{}")],
    )
    assert res["ok"] and res["target"] == "piper"
    # Named after the id the engine validates, not after the upload.
    assert Path(res["path"]).name == "pt_BR-luna-medium.onnx"
    assert (piper_setup.voices_dir() / "pt_BR-luna-medium.onnx.json").is_file()

    saved = local_voices.list_local("luna", "piper")
    assert [v.name for v in saved] == ["Luna local"]
    assert saved[0].voice_id.endswith("pt_BR-luna-medium.onnx")


def test_attach_piper_refuses_a_bad_name_or_a_missing_config(dirs):
    with pytest.raises(vp.VoicePortError) as e:
        vp.attach(profile="luna", target="piper", name="luna",
                  files=[("luna.onnx", b"ONNX"), ("luna.onnx.json", b"{}")])
    assert "pt_BR-luna-medium" in str(e.value)

    with pytest.raises(vp.VoicePortError) as e:
        vp.attach(profile="luna", target="piper", name="pt_BR-luna-medium",
                  files=[("pt_BR-luna-medium.onnx", b"ONNX")])
    assert "config" in str(e.value)


def test_attach_kokoro_stores_the_pack_and_registers_by_id(dirs):
    res = vp.attach(
        profile="luna", target="kokoro", name="pf_luna",
        display_name="Luna Kokoro", source_provider="fish",
        files=[("luna.pt", b"PACK")],
    )
    assert Path(res["path"]).name == "pf_luna.pt"
    # Kokoro resolves a voice BY ID (voices/kokoro/<id>.pt) — a path there
    # would fail the engine's own name validation on the Voice page.
    saved = local_voices.list_local("luna", "kokoro")
    assert saved[0].voice_id == "pf_luna"


def test_attach_kokoro_refuses_a_bad_name(dirs):
    with pytest.raises(vp.VoicePortError) as e:
        vp.attach(profile="luna", target="kokoro", name="luna",
                  files=[("luna.pt", b"PACK")])
    assert "pf_luna" in str(e.value)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _app(tmp_path, monkeypatch, secrets: Secrets | None = None):  # noqa: D401
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "profiles").mkdir(exist_ok=True)
    from config import activate_profile, save_profile
    cfg = AppConfig(profile_name="luna")
    save_profile(cfg, "luna")
    activate_profile("luna")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=cfg, secrets=NO_KEYS if secrets is None else secrets),
    )
    from dashboard.server import DashboardState, _build_app
    from fastapi.testclient import TestClient
    return TestClient(_build_app(DashboardState(), None))


def test_panel_lists_targets_languages_and_only_cloud_voices(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "luna", "provider": "fish", "voice_id": "v1"})
    voice_lab.add_preset("luna", {"name": "local", "provider": "piper", "voice_id": "/x.onnx"})
    app = _app(dirs, monkeypatch)
    with app as client:
        data = client.get("/api/voices/port").json()
    assert {t["id"] for t in data["targets"]} == {"piper", "kokoro"}
    assert {lang["id"] for lang in data["languages"]} == {"pt", "en"}
    by_name = {s["name"]: s for s in data["library"]}
    assert by_name["luna"]["supported"]
    assert not by_name["local"]["supported"]


def test_inspect_reads_an_uploaded_json_and_a_saved_voice(dirs, monkeypatch):
    voice_lab.add_preset("luna", {"name": "luna", "provider": "fish", "voice_id": "v1"})
    app = _app(dirs, monkeypatch)
    content = LUNA_JSON.read_text(encoding="utf-8")
    with app as client:
        r = client.post("/api/voices/port/inspect", json={"content": content})
        assert r.status_code == 200, r.text
        assert r.json()["sources"][0]["provider"] == "fish"

        r = client.post("/api/voices/port/inspect", json={"name": "luna"})
        assert r.status_code == 200, r.text
        assert r.json()["sources"][0]["voice_id"] == "v1"

        assert client.post("/api/voices/port/inspect", json={"name": "nope"}).status_code == 404
        bad = client.post("/api/voices/port/inspect", json={"content": "{}"})
        assert bad.status_code == 400


def test_port_route_captures_and_packages(dirs, monkeypatch, fake_tts):
    app = _app(dirs, monkeypatch, Secrets(fish_api_key="k"))
    with app as client:
        r = client.post("/api/voices/port", json={
            "provider": "fish", "voice_id": "v1", "target": "piper",
            "name": "Luna", "minutes": 1, "lang": "pt",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["clips"] == 8
        assert Path(data["package"], "dataset", "metadata.csv").is_file()


def test_port_route_without_a_key_spends_nothing(dirs, monkeypatch, fake_tts):
    app = _app(dirs, monkeypatch)
    with app as client:
        r = client.post("/api/voices/port", json={
            "provider": "fish", "voice_id": "v1", "target": "piper", "minutes": 1,
        })
        assert r.status_code == 400
        assert "FISH_API_KEY" in r.json()["detail"]
    assert fake_tts.texts == [], "no clip may be captured without a key"


def test_attach_route_registers_the_voice(dirs, monkeypatch):
    app = _app(dirs, monkeypatch)
    files = [
        {"filename": "pt_BR-luna-medium.onnx", "data": base64.b64encode(b"ONNX").decode()},
        {"filename": "pt_BR-luna-medium.onnx.json", "data": base64.b64encode(b"{}").decode()},
    ]
    with app as client:
        r = client.post("/api/voices/port/attach", json={
            "target": "piper", "name": "pt_BR-luna-medium",
            "display_name": "Luna local", "source_provider": "fish", "files": files,
        })
        assert r.status_code == 200, r.text
        names = [p["name"] for p in r.json()["presets"]]
        assert "Luna local" in names

        bad = client.post("/api/voices/port/attach", json={
            "target": "piper", "name": "luna", "files": files,
        })
        assert bad.status_code == 400


# ---------------------------------------------------------------------------
# Markup drift — the panel has no runtime in CI
# ---------------------------------------------------------------------------

def test_voice_lab_ships_the_port_panel():
    html = (REPO / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
    app = (REPO / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")
    assert "Convert to a local engine" in html
    for hook in ('@click="portConvert()"', 'portPickFile($event)', 'portAttach()'):
        assert hook in html, hook
    for method in ("portPickFile", "portPickSaved", "portConvert", "portAttachFiles", "portAttach"):
        assert f"\n    {method}(" in app or f"async {method}(" in app, method
    assert "voicePort:" in app
    assert '"/api/voices/port"' in app
    assert '"/api/voices/port/inspect"' in app
    assert '"/api/voices/port/attach"' in app

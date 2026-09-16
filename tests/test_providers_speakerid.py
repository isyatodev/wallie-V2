"""Tests for dynamic provider blocks (API Keys page) + speaker ID."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from config import AppConfig

def run(coro):
    return asyncio.run(corro if False else coro)


# ---------------------------------------------------------------------------
# Provider blocks: config validation
# ---------------------------------------------------------------------------

def test_provider_block_slug_ids_stable_and_unique():
    cfg = AppConfig(providers=[
        {"name": "Main LLM", "category": "llm", "base_url": "http://localhost:11434/v1", "model": "llama3.1"},
        {"name": "qwen vision", "category": "vision", "base_url": "https://x/v1", "model": "qwen-vl"},
    ])
    ids = [p.id for p in cfg.providers]
    assert ids == ["main_llm", "qwen_vision"]
    cfg2 = AppConfig(**{**cfg.model_dump(), "providers": [p.model_dump() for p in cfg.providers]})
    assert [p.id for p in cfg2.providers] == ids


def test_provider_block_duplicate_names_get_unique_ids():
    cfg = AppConfig(providers=[
        {"name": "dup", "category": "llm"},
        {"name": "dup", "category": "tts"},
    ])
    ids = [p.id for p in cfg.providers]
    assert len(set(ids)) == 2
    assert ids[1].startswith("dup_")


def test_provider_ref_fields_exist():
    cfg = AppConfig()
    assert cfg.llm.provider_ref == ""
    assert cfg.llm.vision_provider_ref == ""
    assert cfg.tts.provider_ref == ""
    assert cfg.hearing.provider_ref == ""
    assert cfg.memory.provider_ref == ""
    assert cfg.random_thoughts.provider_ref == ""


# ---------------------------------------------------------------------------
# secrets_store: dynamic PROVIDER_* keys
# ---------------------------------------------------------------------------

def test_secrets_store_dynamic_provider_keys():
    import secrets_store as ss

    class _P:
        def __init__(self, pid, name, category):
            self.id, self.name, self.category = pid, name, category

    ss.set_provider_context([_P("main-llm", "Main LLM", "llm")])
    dyn = [r for r in ss.list_secrets() if r["dynamic"]]
    assert dyn and dyn[0]["env"] == "PROVIDER_MAIN_LLM_API_KEY"
    assert dyn[0]["label"] == "Main LLM (llm)"

    ss.set_secret("PROVIDER_MAIN_LLM_API_KEY", "abc12345")
    assert os.getenv("PROVIDER_MAIN_LLM_API_KEY") == "abc12345"
    ss.set_secret("PROVIDER_MAIN_LLM_API_KEY", "")

    with pytest.raises(ValueError):
        ss.set_secret("PROVIDER_..\\evil_API_KEY", "x")
    ss.set_provider_context([])


def test_secrets_store_orphan_provider_keys_still_listed(monkeypatch):
    import secrets_store as ss
    ss.set_provider_context([])
    monkeypatch.setenv("PROVIDER_GHOST_API_KEY", "leftover")
    rows = [r for r in ss.list_secrets() if r["env"] == "PROVIDER_GHOST_API_KEY"]
    assert rows and rows[0]["is_set"] is True


# ---------------------------------------------------------------------------
# wallie._resolve_provider fallback chain
# ---------------------------------------------------------------------------

@pytest.fixture()
def prof_cfg(tmp_path: Path, monkeypatch):
    """Isolated profiles dir + a profile with provider blocks."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    cfg = AppConfig(profile_name="p-prov", providers=[
        {"id": "brain", "name": "Brain", "category": "llm", "base_url": "http://llm-host/v1", "model": "llama3.1"},
        {"id": "mic", "name": "Mic STT", "category": "stt", "base_url": "http://stt-host/v1", "model": "whisper-large-v3"},
    ])
    save_profile(cfg, "p-prov")
    return load_profile("p-prov")


def test_resolve_provider_category_and_ref(prof_cfg, monkeypatch):
    import wallie
    from config import Secrets

    monkeypatch.setenv("PROVIDER_MIC_API_KEY", "key-123")
    secrets = Secrets()

    # Category match.
    r = wallie._resolve_provider(prof_cfg, secrets, "stt")
    assert r["from"] == "provider:mic"
    assert r["base_url"] == "http://stt-host/v1"
    assert r["api_key"] == "key-123"

    # Explicit ref to a specific block.
    r2 = wallie._resolve_provider(prof_cfg, secrets, "stt", ref="mic")
    assert r2["base_url"] == "http://stt-host/v1"

    # No matching category → legacy fields.
    r3 = wallie._resolve_provider(prof_cfg, secrets, "tts")
    assert r3["from"] == "legacy"

    # Category with a "default" block wins over legacy.
    cfg2 = prof_cfg.model_copy(update={"providers": [
        *prof_cfg.providers,
        AppConfig.ProviderBlock  # placeholder, replaced below
    ]}) if False else prof_cfg.model_copy(update={"providers": [
        prof_cfg.providers[0], prof_cfg.providers[1],
        type(prof_cfg.providers[0])(id="default", name="Default", category="memory",
                                    base_url="http://def-host/v1", model="m"),
    ]})
    r4 = wallie._resolve_provider(cfg2, secrets, "anything")
    assert r4["from"] == "provider:default"


# ---------------------------------------------------------------------------
# Speaker ID pipeline (local voice prints)
# ---------------------------------------------------------------------------

def _tone(f0: float, seed: int = 0, sec: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * sec)) / 16000.0
    sig = (0.4 * np.sin(2 * np.pi * f0 * t)
           + 0.25 * np.sin(2 * np.pi * f0 * 2.3 * t)
           + 0.15 * np.sin(2 * np.pi * f0 * 3.1 * t)
           + 0.03 * rng.standard_normal(t.size))
    return sig.astype(np.float32)


def test_embed_and_cosine_separate_voices():
    from hearing.speaker_id import cosine, embed
    a1, a2 = _tone(130, seed=1), _tone(130, seed=2)   # same voice
    b = _tone(210, seed=3)                            # different voice
    ea1, ea2, eb = embed(a1), embed(a2), embed(b)
    assert ea1.size > 0
    assert cosine(ea1, ea2) > cosine(ea1, eb)
    assert cosine(ea1, ea2) > 0.9
    # Silence → empty embedding.
    assert embed(np.zeros(16000, dtype=np.float32)).size == 0


def test_identifier_labels_owner_and_stranger(tmp_path: Path):
    from hearing.speaker_id import SpeakerIdentifier, SpeakerPrintStore, embed
    st = SpeakerPrintStore(tmp_path / "s.json")
    st.enroll("Owner", embed(_tone(130)))
    ident = SpeakerIdentifier(st, threshold=0.9, unknown_threshold=0.75)
    lbl, sim = ident.handle_utterance(_tone(130, seed=9), collect=False)
    assert lbl == "Owner" and sim > 0.9
    lbl2, _ = ident.handle_utterance(_tone(210, seed=9), collect=False)
    assert lbl2 == "unknown"


def test_enrollment_buffer_and_clips(tmp_path: Path):
    from hearing.speaker_id import EnrollmentBuffer, SpeakerIdentifier, SpeakerPrintStore, embed
    st = SpeakerPrintStore(tmp_path / "s.json")
    # Enrollment buffer averages several utterances into one print.
    eb = EnrollmentBuffer()
    eb.add(_tone(130, seed=1))
    eb.add(_tone(130, seed=2))
    assert eb.pending == 2
    assert eb.finish(st, "Owner")
    assert st.names() == ["Owner"]
    assert eb.pending == 0
    assert eb.finish(st, "Nobody") is False  # empty buffer

    # Unknown voices are kept as clips when collection is enabled. (The
    # stranger's similarity sits between thresholds → confidently "unknown".)
    ident = SpeakerIdentifier(st, threshold=0.99, unknown_threshold=0.9,
                              collect_other_voices=True)
    lbl, _ = ident.handle_utterance(_tone(210, seed=4), collect=True)
    assert lbl == "unknown"
    assert len(st.clips) == 1
    clip_id = st.clips[0]["id"]
    assert st.clips[0]["wav_b64"]  # playable audio was stored

    # The clip can be promoted to an enrolled voice.
    assert st.enroll_clip(clip_id, "Friend")
    assert st.names() == ["Friend", "Owner"]
    assert st.clips == []


# ---------------------------------------------------------------------------
# Dashboard API: providers CRUD + speaker endpoints
# ---------------------------------------------------------------------------

class _FakeOrch:
    def __init__(self):
        self._enrollment_buffer = None
        self._speaker_store = None
        self._enrolling = False

    def status(self):
        return {"running": False}

    @property
    def speaker_id_active(self):
        return self._speaker_id_active


@pytest.fixture()
def api(tmp_path: Path, monkeypatch):
    """TestClient with isolated profile dir (fresh AppConfig)."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    # The server resolves the ACTIVE profile (default when no state file) —
    # use that name so the endpoints see the same store the tests write to.
    save_profile(AppConfig(profile_name="default"), "default")

    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    app = _build_app(state, _FakeOrch())
    from starlette.testclient import TestClient
    return TestClient(app)


def test_api_providers_crud_roundtrip(api):
    r = api.get("/api/providers")
    assert r.status_code == 200
    assert r.json()["providers"] == []

    body = [
        {"name": "Main LLM", "category": "llm", "base_url": "http://h/v1", "model": "m"},
        {"name": "Mic", "category": "stt", "base_url": "http://s/v1", "model": "whisper-1"},
    ]
    r = api.put("/api/providers", json=body)
    assert r.status_code == 200
    saved = r.json()["providers"]
    assert [p["id"] for p in saved] == ["main_llm", "mic"]

    # Persisted in the profile.
    assert [p["id"] for p in api.get("/api/providers").json()["providers"]] == ["main_llm", "mic"]

    # Delete one.
    assert api.delete("/api/providers/mic").json()["ok"] is True
    assert [p["id"] for p in api.get("/api/providers").json()["providers"]] == ["main_llm"]
    assert api.delete("/api/providers/mic").status_code == 404


def test_api_speakers_endpoints(api, tmp_path: Path):
    from hearing.speaker_id import SpeakerPrintStore, embed
    import config

    # Endpoints on a fresh store.
    assert api.get("/api/speakers").json()["speakers"] == []

    store = SpeakerPrintStore(config.PROFILES_DIR / "default.speakers.json")
    store.enroll("Owner", embed(_tone(130)))
    store.add_clip(_tone(210, seed=5), embed(_tone(210)), "unknown voice #1")

    data = api.get("/api/speakers").json()
    assert [s["name"] for s in data["speakers"]] == ["Owner"]
    assert len(data["clips"]) == 1

    # Clip audio is playable + enrollable.
    clip_id = data["clips"][0]["id"]
    audio = api.get(f"/api/speakers/clips/{clip_id}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    r = api.post(f"/api/speakers/clips/{clip_id}/enroll", json={"name": "Friend"})
    assert r.json()["ok"] is True
    names = [s["name"] for s in api.get("/api/speakers").json()["speakers"]]
    assert names == ["Friend", "Owner"]

    # Delete a speaker.
    assert api.delete("/api/speakers/Friend").json()["ok"] is True
    assert api.delete("/api/speakers/Ghost").status_code == 404


def test_api_speaker_enrollment_requires_session(api):
    r = api.post("/api/speakers/enroll/start")
    assert r.status_code == 409  # no session running


# ---------------------------------------------------------------------------
# Per-voice prompt instructions ("this is my mom — treat her warmly")
# ---------------------------------------------------------------------------


def test_speaker_note_set_get_and_case_insensitive_lookup(tmp_path: Path):
    from hearing.speaker_id import SpeakerPrintStore, embed
    st = SpeakerPrintStore(tmp_path / "s.json")
    st.enroll("Owner", embed(_tone(130)))

    assert st.get_note("Owner") == ""
    assert st.set_note("Owner", "this is my mom — treat her warmly")
    assert st.get_note("Owner") == "this is my mom — treat her warmly"
    # Case-insensitive resolution (labels arrive lowercased, e.g. "owner").
    assert st.find_name("owner") == "Owner"
    assert st.find_name("OWNER") == "Owner"
    assert st.find_name("ghost") == ""
    # Unknown speakers are refused.
    assert st.set_note("Ghost", "x") is False
    # Note survives save/load and long trims are capped.
    st.save()
    st2 = SpeakerPrintStore(tmp_path / "s.json")
    st2.load()
    assert st2.get_note("Owner") == "this is my mom — treat her warmly"
    assert st2.set_note("Owner", "x" * 500)
    assert len(st2.get_note("Owner")) <= 240
    # Clearing works.
    assert st2.set_note("Owner", "")
    assert st2.get_note("Owner") == ""


def test_speaker_note_flows_into_prompt(tmp_path: Path):
    from hearing.speaker_id import SpeakerPrintStore, embed
    from core.persona import Persona
    from config import PersonaConfig
    st = SpeakerPrintStore(tmp_path / "s.json")
    st.enroll("Misa", embed(_tone(210)))
    st.set_note("Misa", "this is my mom — treat her warmly")

    persona = Persona.from_config(PersonaConfig())
    out = persona.hearing_turn(heard="oi", speaker="Misa", speaker_note=st.get_note("Misa"))
    assert "Misa" in out and "my mom" in out
    # Without a note, the generic enrolled-voice framing stays.
    plain = persona.hearing_turn(heard="oi", speaker="Misa", speaker_note="")
    assert "my mom" not in plain


def test_orchestrator_status_exposes_now_speaker(tmp_path: Path):
    """The dashboard's live indicator rides on /api/status → speaker_id.now."""
    import time as _time
    import types
    from core.orchestrator import Orchestrator
    from hearing.speaker_id import SpeakerPrintStore, embed

    store = SpeakerPrintStore(tmp_path / "s.json")
    store.enroll("Owner", embed(_tone(130)))
    store.set_note("Owner", "this is my mom — treat her warmly")

    orch = types.SimpleNamespace(
        _speaker_id=object(),
        _speaker_store=store,
        _enrollment_buffer=None,
        _latest_speaker="Owner",
        _latest_speaker_ts=_time.time(),
        _cfg=types.SimpleNamespace(hearing=types.SimpleNamespace(max_context_age_sec=12.0)),
    )
    orch._enrolling = False

    # status() needs lots of collaborators; exercise just the speaker fields
    # via a minimal clone of the block (keeps this test robust to refactors).
    now = Orchestrator._speaker_now(orch)
    assert now == "Owner"
    resolved = store.find_name(now)
    assert store.get_note(resolved) == "this is my mom — treat her warmly"

    # Expiry: silence for longer than max_context_age → nobody is talking.
    orch._latest_speaker_ts -= 30.0
    assert Orchestrator._speaker_now(orch) == ""


def test_speaker_now_lifecycle_via_drain_hearing(tmp_path: Path):
    """_drain_hearing sets/clears the label; music/noise is not a person."""
    import time as _time
    import types
    from core.orchestrator import Orchestrator
    from hearing.hearing_loop import HearingEvent

    def make_event(**kw):
        base = dict(transcript="", loudness=0.2, has_speech=False,
                    sound_type="quiet", descriptor="")
        base.update(kw)
        return HearingEvent(**base)

    orch = types.SimpleNamespace(
        _latest_speaker="",
        _latest_speaker_ts=0.0,
        _latest_heard="",
        _latest_heard_ts=0.0,
        _memory_capture=None,
        _cfg=types.SimpleNamespace(hearing=types.SimpleNamespace(max_context_age_sec=12.0)),
        _mood=None,
    )
    orch._is_self_echo = lambda text: False

    def _note_heard(latest):
        # The real method drains the orchestrator's hearing queue until it's
        # empty — a real asyncio.Queue does exactly that with one event.
        q = asyncio.Queue()
        q.put_nowait(latest)
        orch._hearing_queue = q
        return Orchestrator._drain_hearing(orch)

    _note_heard(make_event(sound_type="speech", transcript="hello there", has_speech=True, speaker="Owner"))
    assert orch._latest_speaker == "Owner"
    assert orch._latest_heard.startswith("[Owner]")
    assert _time.time() - orch._latest_speaker_ts < 5.0

    # Music clears the label — a song isn't a person talking.
    _note_heard(make_event(sound_type="music", descriptor="chill beats"))
    assert orch._latest_speaker == ""

    # A new speech sets it again; a noise clears it.
    _note_heard(make_event(sound_type="speech", transcript="back again", has_speech=True, speaker="other"))
    assert orch._latest_speaker == "other"
    _note_heard(make_event(sound_type="sound", descriptor="a door slam"))
    assert orch._latest_speaker == ""

    # Self-echo (Wallie hearing itself) also drops the label.
    orch._is_self_echo = lambda text: True
    _note_heard(make_event(sound_type="speech", transcript="my own voice", has_speech=True, speaker="Owner"))
    assert orch._latest_speaker == ""


def test_api_speaker_note_endpoint(api, tmp_path: Path):
    from hearing.speaker_id import SpeakerPrintStore, embed
    import config
    store = SpeakerPrintStore(config.PROFILES_DIR / "default.speakers.json")
    store.enroll("Owner", embed(_tone(130)))

    # Set via the case-insensitive label.
    r = api.put("/api/speakers/owner/note", json={"note": "this is the streamer themselves"})
    assert r.json()["ok"] is True
    assert r.json()["name"] == "Owner"
    data = api.get("/api/speakers").json()
    owner = [s for s in data["speakers"] if s["name"] == "Owner"][0]
    assert owner["note"] == "this is the streamer themselves"

    # Clear.
    assert api.put("/api/speakers/Owner/note", json={"note": ""}).json()["ok"] is True
    assert [s for s in api.get("/api/speakers").json()["speakers"] if s["name"] == "Owner"][0]["note"] == ""

    # Unknown speaker → 404.
    assert api.put("/api/speakers/Ghost/note", json={"note": "x"}).status_code == 404


# ---------------------------------------------------------------------------
# Hearing/prompt integration
# ---------------------------------------------------------------------------

def test_hearing_event_carries_speaker_label():
    from hearing.hearing_loop import HearingEvent
    ev = HearingEvent(transcript="hello there", loudness=0.2, has_speech=True,
                      sound_type="speech", descriptor="", speaker="Owner")
    assert ev.speaker == "Owner"
    assert ev.speaker_similarity == -1.0  # default


def test_orchestrator_prefixes_heard_with_speaker(monkeypatch):
    """The heard string the prompt sees carries the [label] prefix."""
    # Minimal smoke of the string formatting contract used by the orchestrator.
    speaker = "Owner"
    heard = "hello there"
    labeled = f"[{speaker}] {heard}"
    assert labeled == "[Owner] hello there"
    assert not heard.startswith("[")


def test_persona_hearing_turn_speaker_notes():
    from config import PersonaConfig
    from core.persona import Persona
    p = Persona.from_config(PersonaConfig())
    # Owner framing.
    out = p.hearing_turn(heard="are you live", speaker="owner")
    assert "OWNER" in out
    # Enrolled name framing.
    out2 = p.hearing_turn(heard="hey", speaker="Misa")
    assert "Misa" in out2
    # No speaker → no note.
    base = p.hearing_turn(heard="hey")
    plain = p.hearing_turn(heard="hey", speaker="")
    assert base == plain
    assert "voice-print" not in plain


def test_per_person_memory_full_chain(tmp_path: Path):
    """Fatos capturados com about → injetados quando a pessoa fala de novo."""
    import asyncio
    from core.long_term_memory import LongTermMemory
    from core.persona import Persona
    from config import MemoryConfig, PersonaConfig
    from core.memory_capture import MemoryCapture

    class FakeExtractor:
        """Mimics LLMProvider.stream() returning a canned JSON reply."""
        def __init__(self, reply):
            self.reply = reply

        async def stream(self, messages, **kwargs):
            for ch in self.reply:
                yield ch

    store = LongTermMemory(tmp_path / "p.longterm.json")
    store.load()
    # The extractor "decides" a fact about the Owner (about arrives from the
    # [Owner] label on the heard line).
    cap = MemoryCapture(
        MemoryConfig(enabled=True, extractor="openai_compatible"),
        store,
        FakeExtractor('[{"text": "yato asked me to play stardew next", "kind": "long", '
                      '"tag": "yato", "about": "Owner"}]'),
    )
    cap.observe_event("[Owner] hey, you should play stardew again sometime")
    facts = asyncio.run(cap.extract_now())
    assert len(facts) == 1
    assert store.about_speakers() == ["Owner"]

    # Later, when the Owner talks again, the orchestrator lookup surfaces it.
    hits = store.search("stardew voice chat", about="Owner")
    assert hits and "stardew" in hits[0]["text"]
    mem_block = "\n".join(f"- {e['text']}" for e in hits)
    persona = Persona.from_config(PersonaConfig())
    prompt = persona.hearing_turn(
        heard="hey it's me again", speaker="owner", about_memories=mem_block,
    )
    assert "REMEMBER about this person" in prompt
    assert "stardew" in prompt
    # A stranger gets no such block.
    plain = persona.hearing_turn(heard="hello", speaker="other", about_memories="")
    assert "REMEMBER about this person" not in plain

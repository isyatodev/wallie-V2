"""Engagement gate (persona.require_engagement) — the credit guard.

Every user-initiated turn (chat reply, hearing turn) costs an LLM call. With
require_engagement on, inputs that don't show the user is engaging WALLIE are
dropped before the LLM: gate decides cheaply (no LLM, by design), the
orchestrator re-queues gated-out chat (a mention a moment later still lands)
and drops pure-reaction chatter. Optionally (persona.acknowledge_rate) a SHORT
canned acknowledgement is spoken instead of full silence — no LLM involved.
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets

from core.engagement import EngagementGate, is_low_value_chat


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# EngagementGate — pure decision layer
# ---------------------------------------------------------------------------

def _gate(**kw) -> EngagementGate:
    kw.setdefault("name", "Wallie")
    kw.setdefault("aliases", ["@wallie"])
    return EngagementGate(**kw)


def test_named_message_passes_with_handle_and_accents():
    g = _gate()
    for text in ("Wallie what is this game", "wallie, explain", "@wallie olha isso",
                 "Wallié você viu isso?"):
        ok, reason = g.should_reply(text, username="viewer1")
        assert ok, text
        assert reason == "named"


def test_plain_statement_from_stranger_is_gated_out():
    g = _gate()
    ok, reason = g.should_reply("this game is way too hard honestly", username="randomviewer")
    assert ok is False and reason == "no-engagement"


def test_thread_continues_after_wallie_addresses_user():
    g = _gate()
    g.note_reply_to("yato")
    ok, reason = g.should_reply("and what about the boss fight", username="yato")
    assert ok and reason == "thread-you-addressed"


def test_thread_expires_after_window():
    g = _gate()
    g.note_reply_to("yato")
    g.state.last_reply_user_ts = time.time() - EngagementGate.WINDOW_SEC - 1
    ok, reason = g.should_reply("still there?", username="yato")
    assert ok is False and reason == "no-engagement"


def test_replying_to_what_wallie_just_said():
    g = _gate()
    g.note_reply("I think this boss is the hardest one in the whole game")
    ok, reason = g.should_reply("that boss is really insane yes", username="viewer2")
    assert ok and reason == "replying-to-her"


def test_topic_words_keep_stranger_on_her_topic():
    g = _gate()
    g.note_topic("this speedrun record uses a really unusual strategy")
    ok, reason = g.should_reply("what strategy is that", username="viewer3")
    assert ok and reason == "on-her-topic"


def test_streamer_always_passes():
    g = _gate()
    ok, reason = g.should_reply("completely unrelated words here", username="owner", streamer=True)
    assert ok and reason == "streamer"


def test_low_value_chatter_never_worth_a_reply():
    assert is_low_value_chat("kkkk")
    assert is_low_value_chat("😂😂😂")
    assert is_low_value_chat(":wave: gg")
    assert is_low_value_chat("lol")
    assert not is_low_value_chat("Wallie look at this")
    assert not is_low_value_chat("this boss fight is absolutely insane")


def test_question_about_wallie_passes():
    ok, reason = _gate().should_reply("what do you think about the update", username="v")
    assert ok and reason == "question-about-her"


def test_answer_marker_only_inside_live_window():
    g = _gate()
    ok, _ = g.should_reply("sim", username="v")
    assert ok is False                       # no live conversation → no reply
    g.note_reply("should we try the harder route then")
    ok, reason = g.should_reply("sim", username="v")
    assert ok and reason == "answer-marker"


def test_accents_are_folded_for_topic_overlap():
    g = _gate()
    g.note_topic("a música dessa fase é incrível")
    ok, reason = g.should_reply("voce gosta dessa musica", username="v")
    assert ok and reason == "on-her-topic"


# ---------------------------------------------------------------------------
# Orchestrator wiring — real Orchestrator, real ChatManager, fake LLM/TTS
# ---------------------------------------------------------------------------

class _FakeLLM:
    name = "fake"
    model = "fake-model"

    async def stream(self, messages, **kw):
        raise AssertionError("LLM must not be called in gate tests")


class _FakeTTS:
    name = "fake"
    sample_rate = 24000

    async def synthesize(self, text, **kw):
        raise AssertionError("TTS.synthesize must not be called in gate tests")


class _FakePlayer:
    def __init__(self, speaking: bool = False):
        self.speaking = speaking

    def speaking_recently(self, window):
        return self.speaking

    def seconds_queued(self):
        return 0.0


def _orchestrator(tmp_path: Path, cfg: AppConfig):
    from core.orchestrator import Orchestrator
    from core.persona import Persona

    cfg = cfg.model_copy(deep=True)
    cfg.chat.reply_probability = 1.0   # determinism: the reply-prob roll must
    cfg.chat.min_reply_interval_sec = 0.0  # not mask what the gate decides
    rt = Runtime(config=cfg, secrets=Secrets(), base_dir=tmp_path)
    return Orchestrator(
        runtime=rt,
        persona=Persona.from_config(cfg.persona),
        llm=_FakeLLM(),
        tts=_FakeTTS(),
        player=_FakePlayer(),
    )


def _chat_with(msgs):
    from chat import ChatManager
    from config import ChatConfig

    mgr = ChatManager(ChatConfig(), Secrets())
    for m in msgs:
        mgr.queue.put_nowait(m)
    return mgr


def _msg(text, username="viewer1", is_streamer=False):
    from chat import ChatMessage

    return ChatMessage(platform="twitch", username=username, text=text,
                       is_streamer=is_streamer)


def test_orchestrator_gate_blocks_unaddressed_chat(tmp_path):
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("this game is way too hard honestly", "viewer1")])
    assert orch._pop_ordinary_chat() is None          # gated out
    assert not orch._chat.queue.empty()               # ...and kept for a later mention
    assert orch._engagement.state.last_input_text == ""  # no reply recorded


def test_orchestrator_gate_passes_mention(tmp_path):
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("hey Wallie what is this", "viewer1")])
    got = orch._pop_ordinary_chat()
    assert got is not None and got.text.startswith("hey Wallie")
    assert orch._engagement.state.last_input_text == ""  # recorded at SEGMENT time


def test_orchestrator_gate_off_by_default(tmp_path):
    cfg = AppConfig()
    assert cfg.persona.require_engagement is False     # toggle ships disabled
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("just chatting along here", "viewer1")])
    assert orch._pop_ordinary_chat() is not None       # no gate → behaves as before


def test_orchestrator_gate_requeues_then_mention_lands(tmp_path):
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([
        _msg("completely unrelated statement", "viewer1"),
        _msg("Wallie pick that up", "viewer1"),       # same user: mention bypasses
    ])
    first = orch._pop_ordinary_chat()
    assert first is None                              # gated, re-queued
    got = orch._pop_ordinary_chat()
    assert got is not None and "Wallie pick" in got.text


def test_orchestrator_gate_streamer_bypass(tmp_path):
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("chat about anything at all", "theowner", is_streamer=True)])
    assert orch._pop_ordinary_chat() is not None


def test_orchestrator_gate_thread_via_note_input(tmp_path):
    """After Wallie replies to viewer1 (note_input at segment time), their next
    message is part of a live thread — no name needed."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._engagement.note_input("what do you think of this boss", "viewer1")
    orch._engagement.note_reply("honestly this boss design is brilliant")
    orch._chat = _chat_with([_msg("agreed, brilliant design", "viewer1")])
    assert orch._pop_ordinary_chat() is not None


def test_orchestrator_low_value_chat_dropped_not_requeued(tmp_path):
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("kkkk", "viewer1")])
    assert orch._pop_ordinary_chat() is None
    assert orch._chat.queue.empty()                   # dropped, not re-queued


def test_orchestrator_gate_never_touched_hearing_when_off(tmp_path):
    """Gate off → hearing turns behave exactly as before (no changes)."""
    cfg = AppConfig(hearing={"enabled": True}, persona={"require_engagement": False})
    orch = _orchestrator(tmp_path, cfg)
    assert orch._hearing_engaged("[yato] hello there") is True


def test_config_toggle_roundtrip():
    cfg = AppConfig(persona={"require_engagement": True})
    assert cfg.persona.require_engagement is True
    d = cfg.model_dump()
    cfg2 = AppConfig(**d)
    assert cfg2.persona.require_engagement is True


# ---------------------------------------------------------------------------
# Canned acknowledgements (persona.acknowledge_rate) — no-LLM proof-of-life
# ---------------------------------------------------------------------------

def test_pick_ack_varies_and_stays_in_pool():
    from core.engagement import _ACKS, pick_ack

    picks = {pick_ack() for _ in range(60)}
    assert picks and picks.issubset(set(_ACKS))
    assert len(picks) >= 2                      # no lockstep repeats


def _patch_ack_speech(monkeypatch, orch, called: list) -> None:
    """Replace ONLY the TTS/play path: the ack must reach speech without the
    LLM, and the fake TTS records the line that was spoken."""

    async def fake_buffer(self, sentence):
        called.append(("tts", sentence))
        return b"\x00\x00"

    async def fake_play(self, sentence, audio):
        called.append(("play", sentence))

    monkeypatch.setattr(type(orch), "_buffer_tts", fake_buffer)
    monkeypatch.setattr(type(orch), "_play_buffered", fake_play)


def test_ack_fires_on_rate_and_never_calls_llm(tmp_path, monkeypatch):
    async def scenario():
        orch = _orchestrator(tmp_path, AppConfig(persona={"require_engagement": True,
                                                          "acknowledge_rate": 1.0}))
        called: list = []
        _patch_ack_speech(monkeypatch, orch, called)
        monkeypatch.setattr(random, "random", lambda: 0.0)  # force the rate roll
        task = orch._maybe_acknowledge("viewer1: some unheard take")
        assert task is not None
        await task
        return orch, called

    orch, called = _run(scenario())
    assert [k for k, _ in called] == ["tts", "play"]          # spoken, no LLM
    line = called[0][1]
    msgs = orch._conv.to_provider_messages("sys")
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == line
    assert orch._ack_pending is False


def test_ack_silenced_at_rate_zero(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, AppConfig(persona={"require_engagement": True,
                                                      "acknowledge_rate": 0.0}))
    monkeypatch.setattr(random, "random", lambda: 0.0)
    assert orch._maybe_acknowledge("viewer1: anything") is None
    assert orch._ack_pending is False


def test_ack_throttle_and_speaking_guard(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, AppConfig(persona={"require_engagement": True,
                                                      "acknowledge_rate": 1.0}))
    monkeypatch.setattr(random, "random", lambda: 0.0)

    # Under throttle: last ack just fired → silent.
    orch._last_ack_ts = time.time()
    assert orch._maybe_acknowledge("x") is None
    assert orch._ack_pending is False

    # While she is already speaking → silent.
    orch._last_ack_ts = 0.0
    orch._player = _FakePlayer(speaking=True)
    assert orch._maybe_acknowledge("x") is None
    assert orch._ack_pending is False


def test_ack_not_fired_for_low_value_chatter(tmp_path, monkeypatch):
    """Low-value chatter is DROPPED before the ack hook — no 'hmm' for 'kkkk'."""
    cfg = AppConfig(persona={"require_engagement": True, "acknowledge_rate": 1.0})
    orch = _orchestrator(tmp_path, cfg)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    orch._chat = _chat_with([_msg("kkkk", "viewer1")])
    assert orch._pop_ordinary_chat() is None
    assert orch._ack_pending is False


# ---------------------------------------------------------------------------
# Gate session stats (status()['engagement_gate']) — the dashboard panel feed
# ---------------------------------------------------------------------------

def test_gate_stats_counters_track_all_paths(tmp_path, monkeypatch):
    """Replies, skips (with examples), spam and hearing all count separately."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    monkeypatch.setattr(random, "random", lambda: 0.99)  # ack roll loses (0.99 ≥ 0.25)

    # One reply (mention) + one skip + one spam drop.
    orch._chat = _chat_with([
        _msg("Wallie hi", "a1"),
        _msg("talking about something random", "b2"),
        _msg("kkkk", "c3"),
    ])
    first = orch._pop_ordinary_chat()
    assert first is not None and first.username == "a1"
    assert orch._pop_ordinary_chat() is None           # skipped → requeued+example
    assert orch._pop_ordinary_chat() is None           # spam → dropped
    # The requeued message is still in the queue; drain it out without counting twice.
    orch._chat.next_nowait()

    # Hearing skip counts too.
    assert orch._hearing_engaged("[stranger] random talk no direction") is False

    stats = orch._gate_stats
    assert stats["chat_replied"] == 1
    assert stats["chat_skipped"] == 1
    assert stats["chat_low_value"] == 1
    assert stats["hearing_skipped"] == 1
    assert stats["acks_sent"] == 0                    # rate default 0.25 + no forced roll
    assert any("talking about something random" in e for e in orch._gate_examples)
    assert any(e.startswith("[voice]") for e in orch._gate_examples)


def test_llm_calls_saved_counts_and_unwinds(tmp_path):
    """Savings = avoided turns, NET: a re-queued message that later lands
    unwinds its +1 so the panel never overstates."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)

    # Spam drop + hearing skip = two inputs that will never cost a call.
    orch._chat = _chat_with([_msg("kkkk", "c3")])
    orch._pop_ordinary_chat()
    assert orch._hearing_engaged("[stranger] noise") is False
    assert orch._gate_stats["llm_calls_saved"] == 2

    # A skipped message is RE-QUEUED: +1 saved now...
    orch._chat = _chat_with([_msg("random unrelated words", "b2")])
    assert orch._pop_ordinary_chat() is None
    assert orch._gate_stats["llm_calls_saved"] == 3

    # ...but once a thread exists it lands and WILL cost a call → unwind.
    orch._engagement.note_input("earlier question", "b2")
    got = orch._pop_ordinary_chat()
    assert got is not None
    assert orch._gate_stats["chat_replied"] == 1
    assert orch._gate_stats["llm_calls_saved"] == 2


def test_gate_stats_surface_in_status(tmp_path):
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("just noise words here ok", "b2")])
    orch._pop_ordinary_chat()
    s = orch.status()["engagement_gate"]
    assert s["enabled"] is True
    assert s["chat_skipped"] == 1
    assert s["llm_calls_saved"] == 1
    assert s["examples"] and "b2" in s["examples"][0]


def test_gate_stats_absent_when_gate_never_installed(tmp_path):
    """Gate off → counters still exist (zeroed) and status reports enabled=False.
    The UI hides the panel on this flag, so old dashboards stay truthful."""
    cfg = AppConfig()                                  # gate disabled
    orch = _orchestrator(tmp_path, cfg)
    s = orch.status()["engagement_gate"]
    assert s["enabled"] is False
    assert s["chat_skipped"] == 0 and s["chat_replied"] == 0
    assert s["llm_calls_saved"] == 0
    assert s["examples"] == []
    assert s["skipped"] == []


# ---------------------------------------------------------------------------
# Gate inspector + force-reply — what the gate ignored, and answering it anyway
# ---------------------------------------------------------------------------

def test_gate_skipped_registry_records_and_version_bumps(tmp_path):
    """Every gated-out chat is registered for inspection (id/kind/platform set);
    low-value chatter is NOT (it was never answerable); version bumps only when
    the registry actually changes, so the open modal knows when to refresh."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([
        _msg("totally unrelated words here", "b2"),
        _msg("kkkk", "c3"),
    ])
    assert orch._pop_ordinary_chat() is None           # skipped → registered
    assert orch._pop_ordinary_chat() is None           # spam → dropped, NOT registered
    reg = orch.gate_skipped()
    assert len(reg) == 1 and reg[0]["username"] == "b2"
    assert reg[0]["kind"] == "skipped"
    assert reg[0]["platform"] == "twitch"
    assert reg[0]["text"] == "totally unrelated words here"
    assert reg[0]["id"].startswith("chat-")
    v0 = orch._gate_skipped_version
    orch._pop_ordinary_chat()                          # nothing new to register
    assert orch._gate_skipped_version == v0            # no change → no bump


def test_gate_force_reply_reconstructs_and_unwinds(tmp_path):
    """Forcing rebuilds the message with its ORIGINAL platform as an urgent next
    turn, unwinds the savings credit (+1 → −1), and a second force on the same
    id is rejected — never two identical replies."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("also the chest by the stairs is fake", "b2")])
    assert orch._pop_ordinary_chat() is None           # gated out → in registry
    entry = orch.gate_skipped()[0]
    saved_before = orch._gate_stats["llm_calls_saved"]

    info = orch.gate_force_reply(entry["id"])
    assert info["ok"] is True and info["username"] == "b2"
    assert orch._gate_stats["llm_calls_saved"] == saved_before - 1

    intent = _run(orch._choose_intent())
    assert intent is not None and intent.kind == "chat" and intent.urgent is True
    assert intent.chat.username == "b2"
    assert "chest by the stairs" in intent.chat.text
    assert intent.chat.platform == "twitch"            # original platform kept

    with pytest.raises(ValueError):
        orch.gate_force_reply(entry["id"])             # already forced → no-op guard


def test_gate_force_reply_consumed_once_by_choose_intent(tmp_path):
    """The forced reply is consumed on the next _choose_intent call and a second
    call falls through to the normal flow (no double execution)."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("some unrelated random words", "b2")])
    assert orch._pop_ordinary_chat() is None
    entry = orch.gate_skipped()[0]
    orch.gate_force_reply(entry["id"])

    first = _run(orch._choose_intent())
    assert first.kind == "chat" and "unrelated random words" in first.chat.text
    second = _run(orch._choose_intent())
    assert second is None or second.kind != "chat"     # nothing left pending


# ---------------------------------------------------------------------------
# Dashboard routes — GET /api/gate/skipped, POST /api/gate/force-reply/{id}
# ---------------------------------------------------------------------------

class _StubGateOrch:
    """Just the gate surface the two routes touch."""

    def __init__(self, items=None, fail_on=None):
        self._items = items or []
        self._fail_on = fail_on
        self.forced: list[str] = []

    def gate_skipped(self):
        return self._items

    def gate_force_reply(self, msg_id):
        if msg_id == self._fail_on:
            raise ValueError("skipped message not found (expired or already forced)")
        self.forced.append(msg_id)
        return {"ok": True, "username": "u", "text": "t"}


def _gate_app(tmp_path, monkeypatch, orch):
    from fastapi.testclient import TestClient

    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from dashboard.server import DashboardState, _build_app
    return TestClient(_build_app(DashboardState(), orch, pin="")), orch


def test_gate_routes_list_and_force(tmp_path, monkeypatch):
    stub = _StubGateOrch(items=[{
        "id": "chat-abc", "username": "b2", "text": "hello?", "ts": 1.0,
        "kind": "skipped", "platform": "twitch",
    }])
    client, _ = _gate_app(tmp_path, monkeypatch, stub)

    r = client.get("/api/gate/skipped")
    assert r.status_code == 200 and r.json()[0]["id"] == "chat-abc"

    r = client.post("/api/gate/force-reply/chat-abc")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert stub.forced == ["chat-abc"]


def test_gate_route_404_on_unknown_id_and_empty_when_no_session(tmp_path, monkeypatch):
    stub = _StubGateOrch(items=[], fail_on="chat-nope")
    client, _ = _gate_app(tmp_path, monkeypatch, stub)

    r = client.post("/api/gate/force-reply/chat-nope")
    assert r.status_code == 404 and "not found" in r.json()["detail"]
    assert stub.forced == []                      # never force-called on miss

    client2, _ = _gate_app(tmp_path, monkeypatch, None)
    assert client2.get("/api/gate/skipped").json() == []     # no session → empty list
    assert client2.post("/api/gate/force-reply/x").status_code == 409  # no session


def test_gate_force_reply_removes_queued_twin_and_does_not_double_count(tmp_path):
    """Skips are RE-QUEUED. Forcing must (a) remove the gated original still
    waiting in the queue so the input can't be answered twice, and (b) not
    re-register identical rows when the queue re-passes the message."""
    cfg = AppConfig(persona={"require_engagement": True})
    orch = _orchestrator(tmp_path, cfg)
    orch._chat = _chat_with([_msg("a random statement nobody asked about", "b2")])
    assert orch._pop_ordinary_chat() is None               # gated, re-queued, registered once
    reg0 = orch.gate_skipped()
    assert len(reg0) == 1
    orch._pop_ordinary_chat()                              # re-pass of the re-queued twin
    assert len(orch.gate_skipped()) == 1                   # dedup — no second row
    assert orch._gate_skipped_version == 1

    entry = reg0[0]
    saved_before = orch._gate_stats["llm_calls_saved"]
    orch.gate_force_reply(entry["id"])
    assert orch._gate_stats["llm_calls_saved"] == saved_before - 1
    assert orch._chat.queue.empty()                        # twin removed → answered ONCE

    intent = _run(orch._choose_intent())
    assert intent.kind == "chat" and intent.urgent and intent.chat.username == "b2"
    second = _run(orch._choose_intent())
    assert second is None or second.kind != "chat"         # no queued twin behind it

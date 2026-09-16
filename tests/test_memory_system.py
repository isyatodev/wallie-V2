"""Tests for the durable memory system + spontaneous thought scheduler."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from config import MemoryConfig, RandomThoughtsConfig
from core.long_term_memory import LongTermMemory, MemoryError
from core.memory_capture import (
    MemoryCapture,
    MemoryConsolidator,
    ThoughtScheduler,
    _parse_json_array,
    _parse_merges,
)


def run(coro):
    """Run a coroutine to completion on a fresh loop (3.14-safe)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# LongTermMemory store
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> LongTermMemory:
    s = LongTermMemory(tmp_path / "test.longterm.json")
    s.load()
    return s


def test_add_and_list_tiers(store: LongTermMemory):
    store.add("yato's cat is called Misa", kind="long_term", tag="yato")
    store.add("viewer asked about stardew", kind="short_term", ttl_sec=60.0)
    d = store.to_dashboard()
    assert len(d["long_term"]) == 1
    assert len(d["short_term"]) == 1
    assert d["stats"]["long_term"] == 1
    assert "yato" in d["tags"]


def test_dedupe_reinforces_hits(store: LongTermMemory):
    e1 = store.add("same fact", kind="long_term")
    e2 = store.add("same fact", kind="long_term")
    assert e2["id"] == e1["id"]            # deduped, no new id
    assert e2["hits"] == 2                 # reinforced


def test_short_term_expires_via_janitor(store: LongTermMemory):
    store.add("fleeting", kind="short_term", ttl_sec=0.05)
    time.sleep(0.06)
    res = store.janitor_pass(promote_hits=0)
    assert res["expired"] == 1
    assert store.stats()["short_term"] == 0


def test_promotion_on_repeated_hits(store: LongTermMemory):
    for _ in range(3):
        store.add("recurring fact", kind="short_term", ttl_sec=3600.0)
    res = store.janitor_pass(promote_hits=3)
    assert res["promoted"] == 1
    assert store.stats()["long_term"] == 1


def test_update_move_between_tiers(store: LongTermMemory):
    e = store.add("temp fact", kind="short_term", ttl_sec=60.0)
    store.update(e["id"], kind="long_term")
    assert store.stats() == {"long_term": 1, "short_term": 0}


def test_remove_and_clear(store: LongTermMemory):
    e = store.add("bye", kind="long_term")
    assert store.remove(e["id"]) is True
    assert store.remove(e["id"]) is False
    store.add("x", kind="long_term")
    store.add("y", kind="short_term", ttl_sec=60.0)
    assert store.clear() == 2


def test_persistence_roundtrip(tmp_path: Path):
    p = tmp_path / "rt.longterm.json"
    s1 = LongTermMemory(p)
    s1.load()
    s1.add("persisted fact", kind="long_term", tag="tag1")
    s1.add("session fact", kind="short_term", ttl_sec=120.0)
    s1.save()
    s2 = LongTermMemory(p)
    s2.load()
    assert s2.stats()["long_term"] == 1
    assert s2.stats()["short_term"] == 1
    assert s2.list("long_term")[0]["text"] == "persisted fact"


def test_prompt_block_budget(store: LongTermMemory):
    for i in range(20):
        store.add(f"fact number {i} with some words", kind="long_term")
    block = store.prompt_block(max_chars=200)
    assert len(block) <= 200
    assert block.count("- ") >= 2


def test_bad_kind_raises(store: LongTermMemory):
    with pytest.raises(MemoryError):
        store.add("x", kind="bogus")
    with pytest.raises(MemoryError):
        store.update(999, text="nope")


# ---------------------------------------------------------------------------
# JSON extraction parsing
# ---------------------------------------------------------------------------

def test_parse_json_array_plain():
    assert _parse_json_array('[{"text":"a","kind":"long","tag":"t"}]') == [
        {"text": "a", "kind": "long", "tag": "t"}
    ]


def test_parse_json_array_fenced_and_chatty():
    raw = 'Sure! Here you go:\n```json\n[{"text":"a","kind":"short"}]\n```\nDone.'
    assert _parse_json_array(raw)[0]["text"] == "a"


def test_parse_json_array_garbage():
    assert _parse_json_array("no json here") == []
    assert _parse_json_array("") == []


# ---------------------------------------------------------------------------
# MemoryCapture (fake extractor)
# ---------------------------------------------------------------------------

class FakeExtractor:
    """Mimics LLMProvider.stream() returning a canned JSON reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def stream(self, messages, **kwargs):  # noqa: ANN001, ANN003
        for ch in self.reply:
            yield ch


def _capture(store: LongTermMemory, reply: str) -> MemoryCapture:
    cfg = MemoryConfig(enabled=True, extractor="openai_compatible")
    return MemoryCapture(cfg, store, FakeExtractor(reply))


def test_capture_disabled_without_extractor(store: LongTermMemory):
    cfg = MemoryConfig(enabled=True)
    cap = MemoryCapture(cfg, store, None)
    assert cap.enabled is False
    cap.observe_spoken("hello")
    cap.schedule_extraction()          # must not raise nor schedule
    assert cap._recent_texts == []


def test_capture_files_facts(store: LongTermMemory):
    reply = json.dumps([
        {"text": "yato plays stardew", "kind": "long", "tag": "yato"},
        {"text": "chat likes the cooking bit", "kind": "short", "tag": "bit"},
    ])
    cap = _capture(store, reply)
    assert cap.enabled
    cap.observe_spoken("so yato plays stardew huh")
    cap.observe_event("[viewer] do you like stardew?")
    facts = run(cap.extract_now())
    assert len(facts) == 2
    assert store.stats()["long_term"] == 1
    assert store.stats()["short_term"] == 1
    # extract_now is a dry test probe: live buffers stay intact for the real
    # schedule_extraction() path.
    assert cap._recent_texts == ["so yato plays stardew huh"]


def test_capture_bad_json_is_silent(store: LongTermMemory):
    cap = _capture(store, "blah blah no json")
    facts = run(cap.extract_now())
    assert facts == []
    assert store.stats()["long_term"] == 0


# ---------------------------------------------------------------------------
# ThoughtScheduler
# ---------------------------------------------------------------------------

def test_thought_disabled_never_due():
    s = ThoughtScheduler(RandomThoughtsConfig(enabled=False))
    assert s.due(last_spoken_ts=0.0) is False
    assert s.seconds_until_next() == -1.0


def test_thought_fixed_cadence():
    cfg = RandomThoughtsConfig(enabled=True, schedule="fixed", interval_sec=60.0,
                               only_when_quiet_sec=0.0, min_segments_between=0,
                               jitter=0.0)
    s = ThoughtScheduler(cfg)
    s.reset()
    assert s.due(last_spoken_ts=time.time()) is False      # not yet time
    # Simulate the timer having elapsed and fire.
    s._next_at = time.time() - 0.1
    assert s.due(last_spoken_ts=time.time()) is True
    run(s.generate_seed())
    # After firing, the next slot is pushed one full interval into the future.
    assert s._next_at >= time.time() + 55


def test_thought_respects_quiet_window():
    cfg = RandomThoughtsConfig(enabled=True, only_when_quiet_sec=30.0,
                               min_segments_between=0)
    s = ThoughtScheduler(cfg)
    s._next_at = time.time() - 0.1
    # Spoke 5s ago → not quiet enough; rescheduled into the future.
    assert s.due(last_spoken_ts=time.time() - 5) is False
    assert s._next_at > time.time()


def test_thought_rate_limit_per_hour():
    cfg = RandomThoughtsConfig(enabled=True, max_per_hour=2,
                               only_when_quiet_sec=0.0, min_segments_between=0)
    s = ThoughtScheduler(cfg)
    s._next_at = time.time() - 0.1
    s._fired_this_hour = [time.time(), time.time() - 10]
    assert s.due(last_spoken_ts=0.0) is False


def test_seed_pool_beats_generator():
    cfg = RandomThoughtsConfig(enabled=True, seed_topics=["ask about cats"])
    s = ThoughtScheduler(cfg)
    seed = run(s.generate_seed())
    assert seed == "ask about cats"


def test_seed_without_generator_falls_back_to_topic():
    cfg = RandomThoughtsConfig(enabled=True, generator="off", style="context")
    s = ThoughtScheduler(cfg)
    seed = run(s.generate_seed(current_topic="speedruns"))
    assert "speedruns" in seed.lower()


# ---------------------------------------------------------------------------
# Memory callbacks (thoughts that surface old memories)
# ---------------------------------------------------------------------------

def _callback_scheduler(store, chance=1.0):
    cfg = RandomThoughtsConfig(
        enabled=True,
        memory_callback_chance=chance,
        generator="off",
    )
    return ThoughtScheduler(cfg, ltm=store)


def test_callback_always_fires_at_chance_1(store):
    store.add("yato owns a red guitar", kind="long_term", tag="yato")
    s = _callback_scheduler(store, chance=1.0)
    seed = run(s.generate_seed(current_topic="music gear"))
    assert "red guitar" in seed
    assert s._callback_ids  # recorded for anti-repeat


def test_callback_never_fires_at_chance_0(store):
    store.add("yato owns a red guitar", kind="long_term")
    s = _callback_scheduler(store, chance=0.0)
    seed = run(s.generate_seed(current_topic="music gear"))
    assert "red guitar" not in seed


def test_callback_prefers_contextually_relevant(store):
    store.add("chat loves the cooking bit", kind="long_term", tag="bit")
    store.add("yato's cat is called Misa", kind="long_term", tag="yato")
    s = _callback_scheduler(store, chance=1.0)
    # Talk about the cat → cat memory must win over the unrelated cooking bit.
    seed = run(s.generate_seed(current_topic="my cat Misa at home"))
    assert "Misa" in seed


def test_callback_fallback_to_high_hits_when_no_match(store):
    e1 = store.add("obscure fact nobody touches", kind="long_term")
    e2 = store.add("famous recurring fact", kind="long_term")
    store.add("famous recurring fact", kind="long_term")  # hits=2
    s = _callback_scheduler(store, chance=1.0)
    seed = run(s.generate_seed(current_topic="zzzunrelatedzzz"))
    assert "famous recurring fact" in seed
    assert e1["id"] != e2["id"]


def test_callback_respects_kind_pool(store):
    store.add("long memory", kind="long_term")
    store.add("short memory", kind="short_term", ttl_sec=3600.0)
    cfg = RandomThoughtsConfig(enabled=True, memory_callback_chance=1.0,
                               memory_callback_kinds=["long_term"])
    s = ThoughtScheduler(cfg, ltm=store)
    seed = run(s.generate_seed())
    assert "long memory" in seed
    assert "short memory" not in seed


def test_callback_no_repeat_until_pool_exhausted(store):
    for i in range(5):
        store.add(f"fact number {i}", kind="long_term")
    s = _callback_scheduler(store, chance=1.0)
    seeds = [run(s.generate_seed()) for _ in range(5)]
    # All 5 facts surfaced before any repeat (deque cap is 8 > 5).
    texts = sorted(seeds)
    assert len(set(texts)) == 5


def test_callback_empty_store_returns_empty_seed(store):
    s = _callback_scheduler(store, chance=1.0)
    seed = run(s.generate_seed())
    assert seed == ""


def test_callback_skips_expired_short_term(store):
    store.add("stale short-term", kind="short_term", ttl_sec=0.05)
    time.sleep(0.06)
    store.add("fresh long-term", kind="long_term")
    cfg = RandomThoughtsConfig(enabled=True, memory_callback_chance=1.0,
                               memory_callback_kinds=["long_term", "short_term"])
    s = ThoughtScheduler(cfg, ltm=store)
    seed = run(s.generate_seed())
    assert "fresh long-term" in seed
    assert "stale" not in seed


def test_search_kind_filter(store):
    store.add("guitar fact", kind="long_term")
    store.add("guitar fact but short", kind="short_term", ttl_sec=600.0)
    long_only = store.search("guitar", kind="long_term")
    assert len(long_only) == 1 and "but short" not in long_only[0]["text"]
    both = store.search("guitar")
    assert len(both) == 2


# ---------------------------------------------------------------------------
# Dashboard live feed (on_captured hook → WS queue → undo endpoint)
# ---------------------------------------------------------------------------

def test_on_captured_hook_fires_per_fact(store):
    cap = _capture(store, json.dumps([
        {"text": "fact one", "kind": "long", "tag": "a"},
        {"text": "fact two", "kind": "short", "tag": ""},
    ]))
    seen: list[dict] = []
    cap.on_captured = seen.append
    cap.observe_spoken("blah")
    run(cap.extract_now())
    assert [s["text"] for s in seen] == ["fact one", "fact two"]
    assert seen[0]["id"] > 0 and seen[0]["kind"] == "long_term"
    assert isinstance(seen[0]["ts"], float)


def test_on_captured_hook_exception_is_swallowed(store):
    cap = _capture(store, json.dumps([{"text": "fact", "kind": "long"}]))
    def boom(payload):
        raise RuntimeError("sink broke")
    cap.on_captured = boom
    cap.observe_spoken("blah")
    facts = run(cap.extract_now())          # must not raise
    assert len(facts) == 1


def test_dashboard_memory_feed_end_to_end(tmp_path, monkeypatch):
    """Capture → emit_memory_event queue → undo via DELETE /api/longterm.

    Uses the real profile-store filename so the server's ``_ltm_store()``
    (which loads ``PROFILES_DIR/<profile>.longterm.json``) sees the same data
    the running capture pipeline holds in memory — exactly like production.
    """
    import config as _cfgmod
    monkeypatch.setattr(_cfgmod, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(_cfgmod, "STATE_FILE", tmp_path / "state.json")

    store = LongTermMemory(tmp_path / "default.longterm.json")
    store.load()
    cap = _capture(store, json.dumps([{"text": "live fact", "kind": "long"}]))

    class _FakeOrch:
        def __init__(self, c):
            self._memory_capture = c
            self._ltm = c._store

        def status(self):
            return {"running": False}

    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    app = _build_app(state, _FakeOrch(cap))

    # _wire_memory_feed ran at build time → hook attached (bound methods
    # compare equal via __self__/__func__, never via `is`).
    assert cap.on_captured == state.emit_memory_event

    cap.observe_spoken("talk talk")
    run(cap.extract_now())
    store.save()   # orchestrator does this in its capture tick
    entry = state._queue.get_nowait()
    assert entry["type"] == "memory"
    assert entry["data"]["text"] == "live fact"
    fid = entry["data"]["id"]

    from starlette.testclient import TestClient
    c = TestClient(app)
    assert c.get("/api/longterm").json()["stats"]["long_term"] == 1
    assert c.delete(f"/api/longterm/{fid}").json()["ok"] is True
    assert c.get("/api/longterm").json()["stats"]["long_term"] == 0


# ---------------------------------------------------------------------------
# Auto-consolidation (LLM merges related old entries into summaries)
# ---------------------------------------------------------------------------


def _merge_reply(*groups: tuple[tuple[int, ...], str]) -> str:
    return json.dumps(
        [
            {"ids": list(ids), "text": text, "tag": "games"}
            for ids, text in groups
        ]
    )


def test_candidates_skip_summaries_and_high_hits(store: LongTermMemory):
    for i in range(4):
        store.add(f"old fact number {i}", source="ai")
    store.add("summary of games talk", source="summary")
    hi = store.add("very reinforced fact", source="ai")
    for _ in range(4):
        store.add("very reinforced fact", source="ai")   # dedupe path bumps hits
    cands = store.candidates_for_consolidation(limit=10)
    ids = [e["id"] for e in cands]
    assert "summary of games talk" not in [e["text"] for e in cands]
    assert "very reinforced fact" not in [e["text"] for e in cands]
    assert len(cands) == 4
    # oldest first among equal hits
    assert ids == sorted(ids, key=lambda i: store.get(i)["created_at"])


def test_parse_merges_drops_hallucinated_ids():
    raw = json.dumps([
        {"ids": [1, 2], "text": "ok merge"},
        {"ids": [1, 99], "text": "foreign id dropped"},
        {"ids": [3], "text": "solo id dropped"},
        {"ids": "nope", "text": "bad shape"},
        {"ids": [1, 2], "text": ""},
    ])
    out = _parse_merges(raw, allowed_ids={1, 2, 3})
    assert len(out) == 1
    assert out[0]["ids"] == [1, 2]
    assert out[0]["text"] == "ok merge"


def test_parse_merges_strips_code_fence():
    raw = '```json\n[{"ids": [4, 5], "text": "fenced merge"}]\n```'
    out = _parse_merges(raw, allowed_ids={4, 5})
    assert out == [{"ids": [4, 5], "text": "fenced merge", "tag": ""}]


def test_consolidator_merges_and_marks_source(store: LongTermMemory):
    a = store.add("yato beat elden ring boss 1", source="ai")
    b = store.add("yato beat elden ring boss 2", source="ai")
    c = store.add("yato beat elden ring final boss", source="ai")
    reply = _merge_reply(
        ((a["id"], b["id"], c["id"]), "yato eventually beat all of elden ring")
    )
    cons = MemoryConsolidator(
        MemoryConfig(enabled=True, consolidate_threshold=300),
        store,
        FakeExtractor(reply),
    )
    assert cons.enabled
    result = run(cons.consolidate_now())
    assert result["removed"] == 3
    assert result["added"] == 1
    d = store.to_dashboard()
    assert len(d["long_term"]) == 1
    merged = d["long_term"][0]
    assert merged["text"] == "yato eventually beat all of elden ring"
    assert merged["source"] == "summary"
    # importance is inherited: hits of the group sum up
    assert merged["hits"] == 3
    # age is inherited from the oldest member, not from consolidation time
    assert merged["created_at"] == min(a["created_at"], b["created_at"], c["created_at"])


def test_consolidator_respects_threshold(store: LongTermMemory):
    store.add("fact one", source="ai")
    cons = MemoryConsolidator(
        MemoryConfig(enabled=True, consolidate_threshold=300),
        store,
        FakeExtractor("[]"),
    )
    assert cons.maybe_consolidate() is False      # 1 < 300
    assert cons.last_result == {"removed": 0, "added": 0}


def test_consolidator_disabled_without_extractor_or_flag(store: LongTermMemory):
    cfg = MemoryConfig(enabled=True, consolidate_threshold=0)
    assert MemoryConsolidator(cfg, store, FakeExtractor("[]")).enabled is False
    cfg2 = MemoryConfig(enabled=True, consolidate_threshold=300)
    assert MemoryConsolidator(cfg2, store, None).enabled is False


def test_consolidator_handles_bad_model_output(store: LongTermMemory):
    a = store.add("fact alpha", source="ai")
    b = store.add("fact beta", source="ai")
    cons = MemoryConsolidator(
        MemoryConfig(enabled=True, consolidate_threshold=300),
        store,
        FakeExtractor("I cannot do that, sorry — no JSON here"),
    )
    result = run(cons.consolidate_now())
    assert result["removed"] == 0
    assert result["added"] == 0
    # nothing was lost
    assert len(store.list("long_term")) == 2


def test_apply_consolidation_skips_partial_groups(store: LongTermMemory):
    a = store.add("entry a", source="ai")
    b = store.add("entry b", source="ai")
    # one id already deleted → group has < 2 valid members → skipped
    store.remove(b["id"])
    merges = [{"ids": [a["id"], b["id"]], "text": "should not apply"}]
    result = store.apply_consolidation(merges)
    assert result == {"removed": 0, "added": 0}
    assert len(store.list("long_term")) == 1

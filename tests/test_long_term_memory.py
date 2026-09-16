from core.long_term_memory import LongTermMemory, MemoryError

import tempfile
import time
from pathlib import Path


def _store(tmp_path: Path) -> LongTermMemory:
    s = LongTermMemory(tmp_path / "m.json")
    s.load()
    return s


def test_add_list_update_remove(tmp_path):
    s = _store(tmp_path)
    e1 = s.add("I own a corgi", tag="pets")
    e2 = s.add("chat hates my cooking takes", kind="short_term", ttl_sec=3600)
    assert e1["id"] != e2["id"]
    assert len(s.list("long_term")) == 1
    assert len(s.list("short_term")) == 1

    s.update(e1["id"], text="I own two corgis", tag="pets")
    assert s.get(e1["id"])["text"] == "I own two corgis"

    assert s.remove(e2["id"]) is True
    assert s.remove(e2["id"]) is False
    assert s.list("short_term") == []


def test_dedupe_bumps_hits(tmp_path):
    s = _store(tmp_path)
    a = s.add("favorite game is stardew", tag="games")
    b = s.add("Favorite game is Stardew")
    assert a["id"] == b["id"]
    assert b["hits"] == 2


def test_ttl_expiry_and_janitor(tmp_path):
    s = _store(tmp_path)
    s.add("temporary take", kind="short_term", ttl_sec=0.05)
    s.add("durable fact")
    time.sleep(0.06)
    stats = s.janitor_pass(promote_hits=3)
    assert stats["expired"] == 1
    assert len(s.list("short_term")) == 0
    assert len(s.list("long_term")) == 1


def test_promotion_on_recurrence(tmp_path):
    s = _store(tmp_path)
    e = s.add("streamer name is Yato", kind="short_term", ttl_sec=3600)
    for _ in range(2):
        s.add("streamer name is Yato", kind="short_term", ttl_sec=3600)
    assert e["hits"] >= 3
    s.janitor_pass(promote_hits=3)
    kinds = [m["id"] for m in s.list("long_term")]
    assert e["id"] in kinds


def test_persistence_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.add("fact one", tag="t1")
    s.add("fact two", kind="short_term", ttl_sec=9999)
    s.save()
    s2 = LongTermMemory(tmp_path / "m.json")
    s2.load()
    assert len(s2.list("long_term")) == 1
    assert len(s2.list("short_term")) == 1
    assert s2.all_tags() == ["t1"]


def test_search_and_prompt_block(tmp_path):
    s = _store(tmp_path)
    s.add("I own a corgi named Biscoito", tag="pets")
    s.add("I stream on twitch", tag="stream")
    hits = s.search("corgi")
    assert hits and "Biscoito" in hits[0]["text"]
    block = s.prompt_block()
    assert "- I own a corgi named Biscoito [pets]" in block


def test_clear(tmp_path):
    s = _store(tmp_path)
    s.add("a")
    s.add("b", kind="short_term")
    assert s.clear() == 2
    assert s.list() == []


def test_prompt_block_respects_budget(tmp_path):
    s = _store(tmp_path)
    for i in range(50):
        s.add(f"memory entry number {i} with some padding text to grow", tag="bulk")
    block = s.prompt_block(max_chars=400)
    assert len(block) <= 400

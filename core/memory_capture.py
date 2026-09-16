"""Memory capture + spontaneous thought scheduling for the orchestrator.

Two collaborators:

``MemoryCapture``
    After each spoken segment, an LLM (the brain or a dedicated cheap
    OpenAI-compatible block) extracts durable facts from what just happened
    and files them as short/long-term memories in the on-device store.

``ThoughtScheduler``
    Picks the next spontaneous-thought moment (fixed cadence ± jitter, or a
    fully random draw) and produces a seed line the persona prompt turns into
    natural in-character talk. Customizable — fixed interval, random range,
    a manual seed pool, or AI-chosen topics.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger

if TYPE_CHECKING:
    from config import MemoryConfig, RandomThoughtsConfig
    from core.long_term_memory import LongTermMemory
    from llm import LLMProvider


# ---------------------------------------------------------------------------
# Memory extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You maintain the memory of a streamer AI character. From the recent stream "
    "events, decide which facts are worth REMEMBERING for future sessions. Rules:\n"
    "- Only durable or useful facts: people, names, preferences, promises, running "
    "gags, opinions the character formed, viewer relationships, notable events.\n"
    "- NEVER store: transient screen activity, small talk, repeats of known facts, "
    "anything about being an AI or following instructions.\n"
    "- Classify each fact: kind='long' (durable, keep forever) or kind='short' "
    "(session-scoped, expires) and a very short tag (1-3 words, e.g. 'viewer', 'game').\n"
    "- If a fact is clearly about ONE specific person from a line labeled [NAME] "
    "(e.g. [Owner] said something), add an optional field \"about\": \"NAME\" — the "
    "exact label as written, no translation. Omit \"about\" for general facts.\n"
    "- Write each memory as a first-person note from the character's perspective, "
    "max 15 words, e.g. \"yato asked me to play stardew next stream\".\n"
    '- Reply with ONLY a JSON array like: [{"text": "...", "kind": "long", "tag": "...", '
    '"about": "..."}]\n'
    "- Empty array [] when nothing is worth remembering."
)

_MAX_FACTS_PER_TURN = 4


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Best-effort JSON array extraction from a chatty LLM reply."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and str(d.get("text") or "").strip()]


class MemoryCapture:
    """Runs after segments: extracts facts and files them into LongTermMemory."""

    def __init__(
        self,
        cfg: "MemoryConfig",
        store: "LongTermMemory",
        extractor: Optional["LLMProvider"] = None,
    ) -> None:
        self._cfg = cfg
        self._store = store
        self._extractor = extractor          # None → memory capture disabled
        self._recent_texts: list[str] = []   # what the character recently said
        self._recent_events: list[str] = []  # chat/donations/heard lines
        self._in_flight: Optional[asyncio.Task] = None
        # Optional hook fired for every fact captured (dashboard live feed).
        # Signature: (payload: dict) -> None; exceptions are swallowed.
        self.on_captured: Optional[Callable[[dict], None]] = None

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._cfg.enabled and self._extractor is not None

    # ------------------------------------------------------------------
    def observe_spoken(self, text: str) -> None:
        """Called by the orchestrator for every finished sentence Wallie says."""
        if not self.enabled or not text:
            return
        self._recent_texts.append(text.strip())
        if len(self._recent_texts) > 12:
            self._recent_texts.pop(0)

    def observe_event(self, text: str) -> None:
        """Chat messages / donations / heard speech worth remembering context for."""
        if not self.enabled or not text:
            return
        self._recent_events.append(text.strip()[:300])
        if len(self._recent_events) > 20:
            self._recent_events.pop(0)

    # ------------------------------------------------------------------
    def schedule_extraction(self) -> None:
        """Fire-and-forget extraction after a segment finished. One at a time."""
        if not self.enabled:
            return
        if self._in_flight and not self._in_flight.done():
            return
        texts = list(self._recent_texts)
        events = list(self._recent_events)
        if not texts and not events:
            return
        self._recent_texts = []
        self._recent_events = []
        self._in_flight = asyncio.get_running_loop().create_task(self._extract(texts, events))

    async def extract_now(self) -> list[dict[str, Any]]:
        """Dashboard test endpoint: extract from recent context immediately."""
        texts = list(self._recent_texts)
        events = list(self._recent_events)
        if not texts and not events:
            return []
        return await self._extract(texts, events)

    # ------------------------------------------------------------------
    async def _extract(self, texts: list[str], events: list[str]) -> list[dict[str, Any]]:
        try:
            known = self._store.prompt_block(max_chars=800)
            lines = [
                "RECENT THINGS THE CHARACTER SAID:",
                *(f"- {t}" for t in texts[-6:]),
            ]
            if events:
                lines += [
                    "RECENT EVENTS (chat / donations / heard audio):",
                    *(f"- {e}" for e in events[-8:]),
                ]
            if known:
                lines += ["ALREADY-KNOWN MEMORIES (do not repeat these):", known]
            out: list[str] = []
            async for token in self._extractor.stream(
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                temperature=0.1,
                max_tokens=220,
            ):
                out.append(token)
            facts = _parse_json_array("".join(out))[:_MAX_FACTS_PER_TURN]
            for fact in facts:
                kind = (
                    "long_term"
                    if str(fact.get("kind", "")).lower().startswith("long")
                    else "short_term"
                )
                try:
                    # "about" binds the fact to a voice-print speaker label
                    # (e.g. the [Owner] heard line) when the model set one.
                    about = str(fact.get("about") or "").strip()[:40]
                    entry = self._store.add(
                        str(fact.get("text") or "")[:400],
                        kind=kind,
                        tag=str(fact.get("tag") or "")[:40],
                        ttl_sec=self._cfg.short_term_ttl_sec,
                        source="ai",
                        about=about,
                    )
                    if self.on_captured:
                        try:
                            self.on_captured({
                                "id": entry["id"],
                                "text": entry["text"],
                                "kind": kind,
                                "tag": entry.get("tag", ""),
                                "about": entry.get("about", ""),
                                "ts": time.time(),
                            })
                        except Exception:
                            pass
                except Exception:
                    continue
            if facts:
                logger.info(f"memory: captured {len(facts)} fact(s) from the last segment")
            return facts
        except Exception as e:
            logger.debug(f"memory: extraction failed (non-fatal): {e}")
            return []

    async def aclose(self) -> None:
        if self._in_flight and not self._in_flight.done():
            self._in_flight.cancel()
            try:
                await self._in_flight
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Spontaneous thoughts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auto-consolidation (many entries → fewer general summaries)
# ---------------------------------------------------------------------------

_CONSOLIDATE_SYSTEM = (
    "You maintain the long-term memory of a streamer AI character. The store is "
    "getting large, so you MERGE groups of related old entries into single, more "
    "general ones. Rules:\n"
    "- Only merge entries that clearly belong together (same person, same topic, "
    "same running theme, or later facts that supersede earlier ones).\n"
    "- Write each merged memory as ONE first-person note from the character's "
    "perspective, max 20 words, preserving the most important details.\n"
    "- Never invent facts that are not implied by the group.\n"
    '- Reply with ONLY a JSON array like: [{"ids": [1, 2, 5], "text": "...", "tag": "..."}]\n'
    "- Each group must have at least 2 ids, and every id must come from the input list.\n"
    "- It is fine to leave entries unmerged; do not force groups."
    "- Empty array [] when nothing merges cleanly."
)

_MAX_MERGES_PER_PASS = 12


def _parse_merges(raw: str, allowed_ids: set[int]) -> list[dict[str, Any]]:
    """Best-effort parse of the merge proposal JSON, validated against ids
    actually offered to the model (hallucinated or foreign ids dropped)."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        try:
            ids = {int(i) for i in (m.get("ids") or [])}
        except (TypeError, ValueError):
            continue
        ids &= allowed_ids                       # only ids we actually offered
        text = str(m.get("text") or "").strip()
        if len(ids) >= 2 and text:
            out.append({
                "ids": sorted(ids),
                "text": text,
                "tag": str(m.get("tag") or "")[:40],
            })
    return out[:_MAX_MERGES_PER_PASS]


class MemoryConsolidator:
    """When the long-term tier gets big, merges related old entries into
    general summaries using the memory model. Non-fatal end to end."""

    def __init__(
        self,
        cfg: "MemoryConfig",
        store: "LongTermMemory",
        extractor: Optional["LLMProvider"] = None,
    ) -> None:
        self._cfg = cfg
        self._store = store
        self._extractor = extractor
        self._in_flight: Optional[asyncio.Task] = None
        self.last_result: dict[str, int] = {"removed": 0, "added": 0}
        self.last_run_ts: float = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self._cfg.enabled
            and self._cfg.consolidate_threshold > 0
            and self._extractor is not None
        )

    def maybe_consolidate(self) -> bool:
        """Called periodically (janitor tick). Returns True when a pass started."""
        if not self.enabled:
            return False
        if self._in_flight and not self._in_flight.done():
            return False
        stats = self._store.stats()
        if stats["long_term"] < self._cfg.consolidate_threshold:
            return False
        candidates = self._store.candidates_for_consolidation(
            limit=max(10, self._cfg.consolidate_batch)
        )
        if len(candidates) < 2:
            return False
        self._in_flight = asyncio.get_running_loop().create_task(
            self._run(candidates)
        )
        return True

    async def consolidate_now(self) -> dict[str, int]:
        """Dashboard test endpoint: force a pass regardless of threshold."""
        candidates = self._store.candidates_for_consolidation(
            limit=max(10, self._cfg.consolidate_batch)
        )
        if len(candidates) < 2:
            return {"removed": 0, "added": 0, "skipped": 1}
        return await self._run(candidates)

    async def _run(self, candidates: list[dict[str, Any]]) -> dict[str, int]:
        try:
            lines = [
                f"[{e['id']}] {e['text']}"
                + (f" (tag: {e['tag']})" if e.get("tag") else "")
                for e in candidates
            ]
            out: list[str] = []
            async for token in self._extractor.stream(
                [
                    {"role": "system", "content": _CONSOLIDATE_SYSTEM},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                temperature=0.2,
                max_tokens=600,
            ):
                out.append(token)
            allowed = {e["id"] for e in candidates}
            merges = _parse_merges("".join(out), allowed)
            result = self._store.apply_consolidation(merges)
            self.last_result = result
            self.last_run_ts = time.time()
            if result["removed"]:
                logger.info(
                    f"memory: consolidated {result['removed']} entries into "
                    f"{result['added']} summaries"
                )
            return result
        except Exception as e:
            logger.debug(f"memory: consolidation failed (non-fatal): {e}")
            return {"removed": 0, "added": 0}


# ---------------------------------------------------------------------------
# Spontaneous thoughts
# ---------------------------------------------------------------------------

THOUGHT_SYSTEM = (
    "You generate ONE spontaneous thought for a streamer AI character to bring up "
    "on stream. Reply with ONLY the thought itself — one or two short sentences, "
    "in the character's casual voice, no quotes, no preamble. It can be a topic "
    "to riff on, a question for the chat, a hot take, or a callback. Keep it under 40 words."
)


class ThoughtScheduler:
    """Decides WHEN the next spontaneous thought fires and WHAT it is."""

    def __init__(
        self,
        cfg: "RandomThoughtsConfig",
        generator: Optional["LLMProvider"] = None,
        ltm: Optional["LongTermMemory"] = None,
    ) -> None:
        self._cfg = cfg
        self._gen = generator
        self._ltm = ltm                       # memory-callback source (optional)
        self._next_at: float = float("inf")
        self._fired_this_hour: list[float] = []
        self._segments_since_thought: int = 0
        self._last_used_seed: str = ""
        self._last_topic: str = ""
        self._mix_toggle: bool = random.random() < 0.5
        self._callback_ids: deque[int] = deque(maxlen=8)   # recent callbacks (anti-repeat)

    # -- lifecycle ------------------------------------------------------
    def reset(self) -> None:
        self._next_at = self._draw_next(time.time())
        self._fired_this_hour.clear()
        self._segments_since_thought = 0

    def on_segment_spoken(self) -> None:
        self._segments_since_thought += 1

    def on_topic_changed(self, topic: str) -> None:
        self._last_topic = topic or ""

    # -- decision -------------------------------------------------------
    def _draw_next(self, now: float) -> float:
        c = self._cfg
        if c.schedule == "random":
            base = random.uniform(
                c.min_interval_sec, max(c.min_interval_sec, c.max_interval_sec)
            )
        else:
            base = c.interval_sec * (1.0 + random.uniform(-c.jitter, c.jitter))
        return now + max(15.0, base)

    def due(self, *, last_spoken_ts: float, running: bool = True) -> bool:
        """True when a spontaneous thought should fire right now."""
        if not running or not self._cfg.enabled:
            return False
        now = time.time()
        if now < self._next_at:
            return False
        c = self._cfg
        if c.max_per_hour > 0:
            self._fired_this_hour = [t for t in self._fired_this_hour if now - t < 3600]
            if len(self._fired_this_hour) >= c.max_per_hour:
                self._next_at = now + 120.0
                return False
        if now - last_spoken_ts < c.only_when_quiet_sec:
            self._next_at = now + 10.0
            return False
        if self._segments_since_thought < c.min_segments_between:
            self._next_at = now + 15.0
            return False
        return True

    def seconds_until_next(self) -> float:
        if not self._cfg.enabled:
            return -1.0
        return max(0.0, self._next_at - time.time())

    # -- seed -----------------------------------------------------------
    async def generate_seed(
        self, *, current_topic: str = "", last_spoken: str = "", memory_hints: str = ""
    ) -> str:
        """Produce the seed line injected into the persona prompt (may be empty)."""
        c = self._cfg
        self._fired_this_hour.append(time.time())
        self._segments_since_thought = 0
        self._next_at = self._draw_next(time.time())

        # MEMORY CALLBACK: with a configurable chance, surface an old memory
        # from the store instead of inventing a new topic.
        callback = self._maybe_pick_memory_callback(current_topic, last_spoken)
        if callback:
            return callback

        # A manual seed pool beats generation (deterministic, user-curated).
        if c.seed_topics:
            seed = random.choice(c.seed_topics)
            self._mix_toggle = not self._mix_toggle
            return seed

        style = c.style
        if style == "mix":
            style = "context" if self._mix_toggle else "random"
            self._mix_toggle = not self._mix_toggle

        if self._gen is None:
            # No generator: fall back to a plain topic nudge.
            topic = current_topic or self._last_topic
            return f"Bring up {topic}" if topic else ""

        if style == "context":
            user = (
                f"Current topic: {current_topic or 'freeform chat'}.\n"
                f'The character just said: "{(last_spoken or "")[-200:]}"\n'
                + (f"Memory hints: {memory_hints}" if memory_hints else "")
            )
        else:
            user = (
                "Generate an UNRELATED spontaneous curiosity, question for the chat, "
                "or hot take — something fresh that shifts the energy."
                + (f"\nAvoid repeating: {self._last_used_seed}" if self._last_used_seed else "")
            )
        try:
            out: list[str] = []
            async for token in self._gen.stream(
                [
                    {"role": "system", "content": _THOUGHT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=1.0,
                max_tokens=80,
            ):
                out.append(token)
            seed = "".join(out).strip().strip('"')
            self._last_used_seed = seed
            return seed[:300]
        except Exception as e:
            logger.debug(f"thoughts: generation failed (non-fatal): {e}")
            return ""

    # -- memory callbacks -------------------------------------------------
    def _maybe_pick_memory_callback(
        self, current_topic: str, last_spoken: str
    ) -> str:
        """Roll for a memory callback. Returns a ready-made seed line or ''."""
        c = self._cfg
        if self._ltm is None:
            return ""
        chance = max(0.0, min(1.0, c.memory_callback_chance))
        if chance <= 0.0 or random.random() >= chance:
            return ""

        # Query = what's alive right now, so callbacks feel connected to the
        # stream rather than random fact recital.
        query = " ".join(
            part for part in (current_topic, self._last_topic, last_spoken[-160:]) if part
        )
        kinds = [k for k in (c.memory_callback_kinds or ["long_term"]) if k] or [None]
        candidates: list[dict[str, Any]] = []
        if query.strip():
            for k in kinds:
                candidates.extend(self._ltm.search(query, limit=4, kind=k))
        contextual = bool(candidates)
        if not candidates:
            # Nothing contextually relevant: fall back to a high-value entry
            # (hits = the AI keeps re-learning it → it matters).
            for k in kinds:
                candidates.extend(self._ltm.list(kind=k))
            candidates.sort(key=lambda e: (-e.get("hits", 0), -e.get("updated_at", 0.0)))

        now = time.time()
        fresh = [
            e for e in candidates
            if e.get("id") not in self._callback_ids
            and (e.get("expires_at") is None or e.get("expires_at", 0) > now)
        ]
        if not fresh:
            return ""
        # Contextual matches: any of the top hits works. Fallback: take the
        # most-reinforced entry first — the anti-repeat deque rotates through
        # the rest over time.
        pick = random.choice(fresh[:6]) if contextual else fresh[0]
        self._callback_ids.append(pick["id"])
        tag = f" (about {pick['tag']})" if pick.get("tag") else ""
        logger.info(f"thoughts: memory callback → #{pick['id']} {pick['text'][:60]}")
        return (
            f"Something just reminded you of an old memory: “{pick['text']}”{tag}. "
            "Bring it up naturally, like a streamer would — maybe ask about it, "
            "maybe tell a quick story connected to it."
        )

"""Cross-session AI memory — short and long term facts stored on the local disk.

Structure (one JSON file per profile, e.g. ``profiles/default.memory.json``):
- ``long_term``: durable facts ("I own a dog", "chat hates my cooking takes").
- ``short_term``: recent-session facts with a TTL; a small janitor task drops
  expired entries and promotes hot short-term entries to long-term when they
  keep recurring (``hits >= promote_hits``).

Everything is editable from the dashboard (Memory section) and injected into
the persona prompt as compact bullet lines via :meth:`prompt_block`.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_MAX_ENTRIES = 500          # hard cap per tier


class MemoryError(Exception):
    """Raised for invalid memory CRUD operations (bad id, bad kind, …)."""


def _now() -> float:
    return time.time()


def _clip(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[:limit]


class LongTermMemory:
    """Thread-safe, file-backed fact store with short/long term tiers."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self.long_term: list[dict[str, Any]] = []
        self.short_term: list[dict[str, Any]] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self.long_term = []
            self.short_term = []
            self._next_id = 1
            if not self._path.exists():
                return
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"ltm: load failed ({e}); starting empty")
                return
            highest = 0
            for kind, bucket in (("long_term", self.long_term),
                                 ("short_term", self.short_term)):
                for raw in data.get(kind) or []:
                    if not isinstance(raw, dict):
                        continue
                    entry = {
                        "id": int(raw.get("id") or 0),
                        "text": _clip(str(raw.get("text") or ""), 400),
                        "tag": _clip(str(raw.get("tag") or ""), 40),
                        "created_at": float(raw.get("created_at") or 0.0),
                        "updated_at": float(raw.get("updated_at") or 0.0),
                        "hits": max(0, int(raw.get("hits") or 0)),
                        "last_hit_at": float(raw.get("last_hit_at") or 0.0),
                        "expires_at": raw.get("expires_at"),
                    }
                    if not entry["text"]:
                        continue
                    if kind == "short_term":
                        exp = raw.get("expires_at")
                        entry["expires_at"] = float(exp) if exp else None
                    else:
                        entry["expires_at"] = None
                    bucket.append(entry)
                    highest = max(highest, entry["id"])
            # Back-compat: profiles saved by older versions stored a single
            # string field "notes" (or nothing). Nothing to migrate — they just
            # start with an empty fact store alongside their old notes file.
            self._next_id = highest + 1
            logger.info(
                f"ltm: loaded {self._path.name} — "
                f"{len(self.long_term)} long, {len(self.short_term)} short"
            )

    def save(self) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "long_term": self._prune_locked(self.long_term),
                    "short_term": self._prune_locked(self.short_term),
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"ltm: save failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        *,
        kind: str = "long_term",
        tag: str = "",
        ttl_sec: Optional[float] = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        """Create a memory. ``ttl_sec`` only applies to short-term entries."""
        text = _clip(text)
        if not text:
            raise MemoryError("memory text is empty")
        if kind not in ("long_term", "short_term"):
            raise MemoryError("kind must be 'long_term' or 'short_term'")
        if kind == "short_term" and ttl_sec is None:
            ttl_sec = 24 * 3600.0  # one day by default
        with self._lock:
            bucket = self.long_term if kind == "long_term" else self.short_term
            # Dedupe: same (kind, text) → refresh instead of piling up.
            for entry in bucket:
                if entry["text"].lower() == text.lower():
                    entry["hits"] += 1
                    entry["last_hit_at"] = _now()
                    if kind == "short_term" and ttl_sec is not None:
                        entry["expires_at"] = _now() + ttl_sec
                    return entry
            now = _now()
            entry: dict[str, Any] = {
                "id": self._next_id,
                "text": text,
                "tag": _clip(tag, 40),
                "created_at": now,
                "updated_at": now,
                "hits": 1,
                "last_hit_at": 0.0,
                "expires_at": (now + ttl_sec) if (kind == "short_term" and ttl_sec) else None,
                "source": _clip(source, 24),
            }
            self._next_id += 1
            bucket.append(entry)
            if len(bucket) > _MAX_ENTRIES:
                # Drop the oldest low-value entries first.
                bucket.sort(key=lambda e: (e["hits"], e["created_at"]))
                del bucket[0: len(bucket) - _MAX_ENTRIES]
                bucket.sort(key=lambda e: e["created_at"])
            return entry

    def update(self, entry_id: int, *, text: Optional[str] = None,
               tag: Optional[str] = None, kind: Optional[str] = None,
               ttl_sec: Optional[float] = None) -> dict[str, Any]:
        """Edit an entry; ``kind`` change moves it between tiers."""
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                raise MemoryError(f"no memory with id {entry_id}")
            if text is not None:
                text = _clip(text)
                if not text:
                    raise MemoryError("memory text is empty")
                entry["text"] = text
            if tag is not None:
                entry["tag"] = _clip(tag, 40)
            if kind is not None:
                if kind not in ("long_term", "short_term"):
                    raise MemoryError("kind must be 'long_term' or 'short_term'")
                if kind != self._kind_of(entry):
                    self.long_term.remove(entry) if entry in self.long_term else None
                    self.short_term.remove(entry) if entry in self.short_term else None
                    target = self.long_term if kind == "long_term" else self.short_term
                    entry["expires_at"] = None if kind == "long_term" else (
                        _now() + (ttl_sec if ttl_sec else 24 * 3600.0)
                    )
                    target.append(entry)
            elif ttl_sec is not None and self._kind_of(entry) == "short_term":
                entry["expires_at"] = _now() + ttl_sec
            entry["updated_at"] = _now()
            return entry

    def remove(self, entry_id: int) -> bool:
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return False
            if entry in self.long_term:
                self.long_term.remove(entry)
            else:
                self.short_term.remove(entry)
            return True

    def clear(self, kind: Optional[str] = None) -> int:
        """Wipe entries; ``None`` = both tiers. Returns how many were removed."""
        with self._lock:
            n = 0
            if kind in (None, "long_term"):
                n += len(self.long_term)
                self.long_term = []
            if kind in (None, "short_term"):
                n += len(self.short_term)
                self.short_term = []
            return n

    # ------------------------------------------------------------------
    # Auto-consolidation
    # ------------------------------------------------------------------
    def candidates_for_consolidation(
        self, limit: int = 40, max_hits: int = 2,
    ) -> list[dict[str, Any]]:
        """Oldest low-traffic long-term entries — the best merge material.
        Newest entries and anything the AI keeps re-learning stay untouched."""
        with self._lock:
            ranked = sorted(
                (
                    e for e in self.long_term
                    if e.get("source") != "summary"
                    and e["hits"] <= max_hits      # re-learned facts stay untouched
                ),
                key=lambda e: (e["hits"], e["created_at"]),
            )
            return [dict(e) for e in ranked[:limit]]

    def apply_consolidation(self, merges: list[dict[str, Any]]) -> dict[str, int]:
        """Apply LLM-proposed merges atomically.

        ``merges`` is a list of {"ids": [...], "text": "...", "tag": "..."}.
        Each group is removed and replaced by ONE new long-term entry whose
        ``hits`` inherits the sum of the group (so consolidated memories keep
        their importance) and whose ``source`` marks it as a summary. Groups
        referencing unknown ids are skipped safely. Returns counts.
        """
        removed = added = 0
        with self._lock:
            for m in merges:
                ids = m.get("ids") or []
                text = _clip(str(m.get("text") or ""), 400)
                if len(ids) < 2 or not text:
                    continue
                group = [self._find_locked(int(i)) for i in ids]
                group = [e for e in group if e is not None]
                if len(group) < 2:
                    continue
                now = _now()
                total_hits = sum(e["hits"] for e in group)
                oldest = min(e["created_at"] for e in group)
                tag = _clip(str(m.get("tag") or group[0].get("tag") or ""), 40)
                new_entry: dict[str, Any] = {
                    "id": self._next_id,
                    "text": text,
                    "tag": tag,
                    "created_at": oldest,          # keeps the original age
                    "updated_at": now,
                    "hits": total_hits,
                    "last_hit_at": max(e.get("last_hit_at", 0.0) for e in group),
                    "expires_at": None,
                    "source": "summary",           # marks auto-consolidated notes
                }
                for e in group:
                    if e in self.long_term:
                        self.long_term.remove(e)
                        removed += 1
                    elif e in self.short_term:
                        self.short_term.remove(e)
                        removed += 1
                self.long_term.append(new_entry)
                self._next_id += 1
                added += 1
        if removed or added:
            self.save()
        return {"removed": removed, "added": added}

    def get(self, entry_id: int) -> dict[str, Any]:
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                raise MemoryError(f"no memory with id {entry_id}")
            return dict(entry)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @staticmethod
    def _kind_of(entry: dict[str, Any]) -> str:
        return "short_term" if entry.get("expires_at") is not None else "long_term"

    def _find_locked(self, entry_id: int) -> Optional[dict[str, Any]]:
        for e in self.long_term:
            if e["id"] == entry_id:
                return e
        for e in self.short_term:
            if e["id"] == entry_id:
                return e
        return None

    def list(self, kind: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            if kind == "long_term":
                return [dict(e) for e in self.long_term]
            if kind == "short_term":
                return [dict(e) for e in self.short_term]
            return [dict(e) for e in self.long_term + self.short_term]

    def search(
        self, query: str, limit: int = 8, kind: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Naive relevance: whole-word-ish token overlap on text + tag.
        ``kind`` restricts the pool ('long_term' | 'short_term')."""
        tokens = [t for t in (query or "").lower().split() if len(t) > 2]
        if not tokens:
            return []
        with self._lock:
            if kind == "long_term":
                pool = self.long_term
            elif kind == "short_term":
                pool = [e for e in self.short_term
                        if e.get("expires_at") is None or e["expires_at"] > _now()]
            else:
                pool = self.long_term + self.short_term
            scored: list[tuple[float, dict[str, Any]]] = []
            for e in pool:
                hay = f"{e['text']} {e['tag']}".lower()
                overlap = sum(1 for t in tokens if t in hay)
                if overlap:
                    # Recency + hits break ties.
                    recency = max(0.0, 1.0 - (_now() - e["updated_at"]) / (90 * 86400))
                    scored.append((overlap + recency + min(e["hits"], 5) * 0.1, e))
            scored.sort(key=lambda pair: -pair[0])
            return [dict(e) for _, e in scored[:limit]]

    def all_tags(self) -> list[str]:
        with self._lock:
            tags = {e["tag"] for e in self.long_term + self.short_term if e["tag"]}
            return sorted(tags)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def _prune_locked(self, bucket: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = _now()
        return [e for e in bucket
                if e.get("expires_at") is None or e["expires_at"] > now]

    def janitor_pass(self, promote_hits: int = 0) -> dict[str, int]:
        """Drop expired short-term entries; optionally promote recurring ones.

        Called periodically by the orchestrator. ``promote_hits > 0`` moves a
        short-term entry to long-term once it has been reinforced that many
        times (the AI keeps re-learning the same fact → it matters).
        Returns counts for logging/status.
        """
        with self._lock:
            now = _now()
            expired = 0
            kept: list[dict[str, Any]] = []
            for e in self.short_term:
                exp = e.get("expires_at")
                if exp is not None and exp <= now:
                    expired += 1
                    continue
                kept.append(e)
            self.short_term = kept
            promoted = 0
            if promote_hits > 0:
                stay: list[dict[str, Any]] = []
                for e in self.short_term:
                    if e["hits"] >= promote_hits:
                        e["expires_at"] = None
                        e["updated_at"] = now
                        self.long_term.append(e)
                        promoted += 1
                    else:
                        stay.append(e)
                self.short_term = stay
            return {"expired": expired, "promoted": promoted}

    # ------------------------------------------------------------------
    # Prompt + dashboard serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(entry: dict[str, Any], now: float) -> str:
        tag = f" [{entry['tag']}]" if entry["tag"] else ""
        return f"- {entry['text']}{tag}"

    def prompt_block(self, max_chars: int = 1200) -> str:
        """Compact bullet block for the persona system prompt (newest first)."""
        with self._lock:
            now = _now()
            lines: list[str] = []
            for e in reversed(self.long_term):
                lines.append(self._fmt(e, now))
            fresh = [e for e in self.short_term
                     if e.get("expires_at") is None or e["expires_at"] > now]
            for e in reversed(fresh):
                lines.append(self._fmt(e, now))
            if not lines:
                return ""
            out_lines: list[str] = []
            used = 0
            for line in lines:
                if used + len(line) + 1 > max_chars:
                    break
                out_lines.append(line)
                used += len(line) + 1
            return "\n".join(out_lines)

    def stats(self) -> dict[str, int]:
        with self._lock:
            now = _now()
            fresh = sum(
                1 for e in self.short_term
                if e.get("expires_at") is None or e["expires_at"] > now
            )
            return {"long_term": len(self.long_term), "short_term": fresh}

    def to_dashboard(self) -> dict[str, Any]:
        """JSON-safe snapshot for GET /api/longterm (ids stable, newest first)."""
        with self._lock:
            now = _now()

            def shape(e: dict[str, Any]) -> dict[str, Any]:
                kind = "short_term" if e.get("expires_at") is not None else "long_term"
                return {
                    "id": e["id"],
                    "kind": kind,
                    "text": e["text"],
                    "tag": e["tag"],
                    "hits": e["hits"],
                    "created_at": e["created_at"],
                    "updated_at": e["updated_at"],
                    "source": e.get("source", "manual"),
                    "expires_in_sec": (
                        max(0.0, round(e["expires_at"] - now, 1))
                        if e.get("expires_at") is not None else None
                    ),
                }

            return {
                "long_term": [shape(e) for e in sorted(self.long_term, key=lambda x: -x["id"])],
                "short_term": [shape(e) for e in sorted(self.short_term, key=lambda x: -x["id"])],
                "tags": self.all_tags_locked(),
                "stats": self.stats(),
            }

    def all_tags_locked(self) -> list[str]:
        with self._lock:
            tags = {e["tag"] for e in self.long_term + self.short_term if e["tag"]}
            return sorted(tags)

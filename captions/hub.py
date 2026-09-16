"""CaptionHub — pub/sub bridge between the TTS pipeline and caption clients.

Design:
- ``note_sentence`` is called from the orchestrator for every spoken sentence.
- ``speech_started`` / ``speech_ended`` bracket a segment; ``speech_ended``
  pushes a CLEAR event so the browser-source box empties right after Wallie
  finishes talking (no stale caption hanging on screen).
- Clients subscribe over SSE (``GET /captions/events``) served by the dashboard.

The hub is transport-agnostic: FastAPI routes live in ``dashboard.server``
and call into the hub. Events are queued per client with a small replay
buffer so a freshly connected browser source immediately shows the current
caption state instead of an empty box (until the next CLEAR).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, AsyncIterator, Optional


class _Client:
    """One connected caption client (browser source)."""

    def __init__(self, queue: "asyncio.Queue[dict[str, Any]]") -> None:
        self.queue = queue


class CaptionHub:
    """Fan-out hub for caption events.

    Event shapes (JSON over SSE):
      {"event": "update", "text": "...", "final": false|true, "ts": ...}
      {"event": "clear", "ts": ...}
      {"event": "hello", "ts": ...}          # sent once on connect
    """

    def __init__(self, *, replay: int = 4, max_clients: int = 16) -> None:
        self._clients: set[_Client] = set()
        self._replay: deque[dict[str, Any]] = deque(maxlen=replay)
        self._max_clients = max_clients
        self._last_text: str = ""
        self._cleared_at: float = 0.0

    # ------------------------------------------------------------------
    # Producer API (called by the orchestrator / test endpoints)
    # ------------------------------------------------------------------
    def note_sentence(self, text: str, *, final: bool = False) -> None:
        """A sentence was handed to TTS (or finished). Intermediate sentences
        stream as they are spoken; the last one of a segment may be flagged
        ``final=True``."""
        t = (text or "").strip()
        if not t:
            return
        self._last_text = t
        self._publish({"event": "update", "text": t, "final": bool(final),
                       "ts": round(time.time(), 3)})

    def speech_started(self) -> None:
        """Bracket start — optional; lets the page animate a 'speaking' state."""
        self._publish({"event": "speech_start", "ts": round(time.time(), 3)})

    def speech_ended(self) -> None:
        """Wallie finished the segment — CLEAR the caption box immediately so
        an old message never lingers on stream. The replay buffer is flushed of
        stale update lines: a browser source connecting right after this point
        gets hello → clear (empty box), never the old sentence."""
        self._last_text = ""
        self._cleared_at = time.time()
        self._replay.clear()
        self._publish({"event": "clear", "ts": round(time.time(), 3)})

    def clear(self) -> None:
        """Alias with explicit semantics for external callers."""
        self.speech_ended()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def current_text(self) -> str:
        return self._last_text

    def status(self) -> dict[str, Any]:
        return {
            "clients": len(self._clients),
            "text": self._last_text,
            "cleared_ago_sec": round(time.time() - self._cleared_at, 1) if self._cleared_at else None,
        }

    # ------------------------------------------------------------------
    # Client lifecycle (dashboard SSE bridge)
    # ------------------------------------------------------------------
    async def subscribe(self) -> AsyncIterator[str]:
        """Yield SSE-formatted frames for one client until it disconnects."""
        if len(self._clients) >= self._max_clients:
            yield _sse({"event": "error", "message": "too many clients"})
            return
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        client = _Client(queue)
        self._clients.add(client)
        try:
            # Hello + replay of the current caption state so a browser source
            # that (re)connects mid-sentence shows the line right away.
            hello = {"event": "hello", "ts": round(time.time(), 3)}
            yield _sse(hello)
            for ev in list(self._replay):
                yield _sse(ev)
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # comment frame keeps proxies happy
                    continue
                if ev.get("event") == "__close__":
                    return
                yield _sse(ev)
        finally:
            self._clients.discard(client)

    async def aclose(self) -> None:
        """Disconnect every client (used on orchestrator stop)."""
        for client in list(self._clients):
            try:
                client.queue.put_nowait({"event": "__close__"})
            except asyncio.QueueFull:
                pass
        self._clients.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _publish(self, ev: dict[str, Any]) -> None:
        if ev.get("event") in ("update", "clear", "speech_start"):
            self._replay.append(ev)
        for client in list(self._clients):
            try:
                client.queue.put_nowait(ev)
            except asyncio.QueueFull:
                # Slow client: drop the oldest and retry once.
                try:
                    client.queue.get_nowait()
                    client.queue.put_nowait(ev)
                except Exception:
                    pass


def _sse(ev: dict[str, Any]) -> str:
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

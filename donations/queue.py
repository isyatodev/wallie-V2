"""DonationQueue — fan-in point for every donation platform.

Reuses the orchestrator's existing queue machinery: donations become
ChatMessage objects flagged as highlights (is_highlight=True), exactly like
bits/superchats are handled today. This guarantees:
  * one single pipeline (no second AI pipeline),
  * donations jump ahead of vision/monologue (existing highlight path),
  * no overlapping speech (the orchestrator processes one intent at a time).
"""
from __future__ import annotations

import asyncio
import random
import time

from loguru import logger

from chat.base import ChatMessage
from donations.base import DonationEvent


class DonationQueue:
    """Bridges DonationEvents into the ChatManager queue used by the orchestrator."""

    def __init__(
        self,
        chat_queue: "asyncio.Queue[ChatMessage]",
        *,
        reply_probability: float = 1.0,
        cooldown_sec: float = 0.0,
    ) -> None:
        self._queue = chat_queue
        self._reply_probability = max(0.0, min(1.0, reply_probability))
        self._cooldown_sec = max(0.0, cooldown_sec)
        self._last_enqueued_ts = 0.0

    async def enqueue(self, events: list[DonationEvent]) -> None:
        now = time.time()
        if self._cooldown_sec and (now - self._last_enqueued_ts) < self._cooldown_sec:
            logger.info(
                f"[Donation] cooldown active ({self._cooldown_sec:.0f}s) — "
                f"{len(events)} donation(s) logged, no AI reply"
            )
            return
        if self._reply_probability < 1.0 and random.random() >= self._reply_probability:
            logger.info("[Donation] reply_probability roll — donation logged, no AI reply")
            return
        for ev in events:
            label = ev.source
            name = ev.donor_name
            amount = ev.formatted_amount
            text = (
                f"Donation via {label} — {name} — {amount}"
                + (f' — "{ev.message}"' if ev.message else "")
            )
            msg = ChatMessage(
                platform="livepix" if ev.source == "livepix" else "streamlabs",
                username=name,
                text=text,
                is_highlight=True,
                donation=ev,
            )
            try:
                self._queue.put_nowait(msg)
                logger.info(f"[Donation] queued ({label}): {name} {amount}")
            except asyncio.QueueFull:
                logger.warning("[Donation] queue full — dropping oldest and retrying")
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.error("[Donation] queue still full — donation dropped")
        self._last_enqueued_ts = now

"""Shared chat monitor interface."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Protocol

if TYPE_CHECKING:
    from donations.base import DonationEvent

Platform = Literal["youtube", "twitch", "kick", "livepix", "streamlabs"]


@dataclass
class ChatMessage:
    platform: Platform
    username: str
    text: str
    ts: float = field(default_factory=time.time)
    is_highlight: bool = False  # super chat / bits / donation
    # True when the message was written by the BROADCASTER (e.g. detected from
    # Twitch `badges=broadcaster/1`). Lets the LLM tell [STREAMER] from [VIEWER].
    is_streamer: bool = False
    # When this message IS a normalized donation, the raw event rides along so
    # the orchestrator can build a donation-specific turn (source/amount/message
    # stay semantically distinct from a normal viewer message for the LLM).
    donation: Optional["DonationEvent"] = None


class ChatMonitor(Protocol):
    platform: Platform

    async def start(self, out_queue: asyncio.Queue[ChatMessage]) -> None: ...
    async def stop(self) -> None: ...

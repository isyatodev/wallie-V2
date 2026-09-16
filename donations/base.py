"""Normalized donation event + idempotency store shared by every platform."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class DonationEvent:
    """A paid/confirmed donation, normalized across platforms.

    `source` distinguishes platforms; every other field has the same meaning
    no matter where the donation came from. The orchestrator consumes this
    through the existing queue machinery (as a chat-highlight-style urgent turn).
    """

    source: str                      # "livepix" | "streamlabs" | ...
    event_id: str                    # the platform's real unique id — never generated here
    donor_name: str                  # "Anônimo"/"Anonymous" when the platform hides it
    donor_id: str = ""
    amount: float = 0.0              # major units (e.g. 10.00), never cents
    currency: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_amount(self) -> str:
        cur = self.currency or ""
        value = f"{self.amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") \
            if self.currency.upper() in {"BRL", "EUR"} else f"{self.amount:,.2f}"
        return f"{cur} {value}".strip()


class DonationDedupe:
    """Bounded LRU of recently processed platform event ids.

    Guard against double responses: the same platform event id delivered twice
    (webhook retry, socket reconnect replay) is processed exactly once. Nothing
    here invents ids — callers must pass the real platform-provided id.
    """

    def __init__(self, max_entries: int = 512, ttl_sec: float = 24 * 3600.0) -> None:
        self._max = max(1, max_entries)
        self._ttl = ttl_sec
        self._seen: OrderedDict[str, float] = OrderedDict()

    def first_time(self, event_id: str) -> bool:
        """True the first time an id is seen; False for any repeat. Records the id."""
        key = (event_id or "").strip()
        if not key:
            # No usable id: cannot dedupe. Fail OPEN (process) and warn once per call —
            # dropping real donations is worse than a rare double response.
            logger.warning("donations: event without usable id — cannot dedupe, processing anyway")
            return True
        now = time.time()
        self._evict(now)
        if key in self._seen:
            logger.debug(f"donations: duplicate event suppressed ({key[:24]}…)")
            return False
        self._seen[key] = now
        if len(self._seen) > self._max:
            self._seen.popitem(last=False)
        return True

    def _evict(self, now: float) -> None:
        while self._seen:
            oldest = next(iter(self._seen))
            if now - self._seen[oldest] <= self._ttl:
                break
            self._seen.popitem(last=False)

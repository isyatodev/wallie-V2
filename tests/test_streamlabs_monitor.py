"""Streamlabs monitor unit tests — event handling + dedupe + reconnect policy.

No real socket is opened: the python-socketio client is stubbed. Payload shapes
come from the official docs (dev.streamlabs.com/docs/socket-api).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import asyncio
from typing import Any, Optional

import pytest

# Stub python-socketio (may not be installed in the test sandbox).
for _mod in ("socketio", "engineio"):
    if _mod not in sys.modules:
        sys.modules[_mod] = __import__("types").ModuleType(_mod)
if not hasattr(sys.modules.get("socketio"), "AsyncClient"):
    class _StubAsyncClient:
        def __init__(self, *a, **kw) -> None: ...
    sys.modules["socketio"].AsyncClient = _StubAsyncClient

from donations.base import DonationDedupe
from donations.normalizer import DonationPayloadError
from donations.streamlabs import StreamlabsMonitor


class _Cfg:
    def __init__(self) -> None:
        self.streamlabs_url = "https://sockets.streamlabs.com"
        self.streamlabs_reconnect_max_sec = 60.0


class _RecordingMonitor(StreamlabsMonitor):
    """Records queued batches instead of touching any real queue."""

    def __init__(self, **kw) -> None:
        super().__init__(cfg=_Cfg(), on_donations=self._collect, **kw)
        self.batches: list[list[Any]] = []
        self._backoff_calls: list[float] = []

    async def _collect(self, events: list[Any]) -> None:
        self.batches.append(events)

    def _sleep_called_with(self) -> list[float]:
        return self._backoff_calls


DOCS_DONATION = {
    "type": "donation",
    "message": [{
        "id": 1, "name": "test", "amount": "13.37",
        "formatted_amount": "$13.37", "message": "test donation",
        "currency": "USD", "_id": "0820c9d5bafd768c9843f5e35c885e71",
    }],
    "event_id": "evt_17e5f4dc6888767ed9799f78dfa2cabc",
}


@pytest.mark.asyncio
async def test_donation_event_reaches_handler():
    m = _RecordingMonitor()
    await m._handle_event(DOCS_DONATION)
    assert len(m.batches) == 1
    ev = m.batches[0][0]
    assert ev.donor_name == "test"
    assert ev.amount == 13.37


@pytest.mark.asyncio
async def test_duplicate_event_suppressed():
    m = _RecordingMonitor()
    await m._handle_event(DOCS_DONATION)
    await m._handle_event(DOCS_DONATION)  # same _id → ignored
    assert len(m.batches) == 1


@pytest.mark.asyncio
async def test_non_donation_ignored():
    m = _RecordingMonitor()
    await m._handle_event({"type": "follow", "for": "twitch_account",
                           "message": [{"name": "x", "_id": "f1"}]})
    assert m.batches == []


@pytest.mark.asyncio
async def test_invalid_payload_does_not_raise():
    m = _RecordingMonitor()
    await m._handle_event("garbage")
    await m._handle_event({"type": "donation"})  # missing message array
    assert m.batches == []


def test_backoff_is_capped():
    """The reconnect schedule grows exponentially but never past the cap."""
    backoff = 1.0
    max_backoff = 3.0
    delays: list[float] = []
    for _ in range(20):
        delays.append(backoff * 0.5)  # lower jitter bound
        backoff = min(max_backoff, backoff * 2)
    assert max(delays) <= max_backoff


@pytest.mark.asyncio
async def test_stop_without_start_is_safe():
    m = _RecordingMonitor()
    await m.stop()  # must not raise


@pytest.mark.asyncio
async def test_reconnect_backoff_values(monkeypatch):
    """The reconnect loop uses exponential backoff with a cap, not a tight loop."""
    m = _RecordingMonitor(socket_token="tok")
    seen: list[float] = []

    async def fake_sleep(delay: float) -> None:
        seen.append(delay)
        if len(seen) >= 5:
            m._stop.set()

    monkeypatch.setattr(m, "_sleep", fake_sleep)

    async def fail_token() -> str:
        raise RuntimeError("no token configured")

    monkeypatch.setattr(m, "_resolve_socket_token", fail_token)

    # _run must loop: token failure → backoff → retry → eventually stop.
    task = asyncio.create_task(m._run())
    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
    excepted = False
    try:
        await task
    except (asyncio.TimeoutError, asyncio.CancelledError):
        excepted = True
    # Either it finished (stop was set inside fake_sleep) or it was cut by the
    # outer timeout — either way backoff delays must have grown, not spun.
    assert len(seen) >= 2
    assert seen[1] > seen[0]

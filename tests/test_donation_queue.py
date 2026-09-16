"""Donation → orchestrator queue integration tests (no external APIs)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import asyncio

import pytest

from chat.base import ChatMessage
from donations.base import DonationEvent
from donations.queue import DonationQueue


def _ev(**kw) -> DonationEvent:
    base = dict(
        source="livepix",
        event_id="evt-1",
        donor_name="Maria",
        amount=10.0,
        currency="BRL",
        message="manda salve",
    )
    base.update(kw)
    return DonationEvent(**base)


def _queue(maxsize: int = 200) -> asyncio.Queue:
    return asyncio.Queue(maxsize=maxsize)


@pytest.mark.asyncio
async def test_enqueue_single_donation():
    q: asyncio.Queue[ChatMessage] = _queue()
    dq = DonationQueue(q)
    await dq.enqueue([_ev()])
    assert q.qsize() == 1
    msg = q.get_nowait()
    assert msg.is_highlight is True
    assert msg.username == "Maria"
    assert msg.donation is not None
    assert msg.donation.source == "livepix"


@pytest.mark.asyncio
async def test_enqueue_multiple_donations_all_queued():
    q: asyncio.Queue[ChatMessage] = _queue()
    dq = DonationQueue(q)
    await dq.enqueue([_ev(event_id="a"), _ev(event_id="b", source="streamlabs")])
    assert q.qsize() == 2
    first = q.get_nowait()
    second = q.get_nowait()
    assert first.donation.source == "livepix"
    assert second.donation.source == "streamlabs"


@pytest.mark.asyncio
async def test_donation_without_message_still_flows():
    q: asyncio.Queue[ChatMessage] = _queue()
    dq = DonationQueue(q)
    await dq.enqueue([_ev(message="")])
    msg = q.get_nowait()
    assert msg.donation.message == ""


@pytest.mark.asyncio
async def test_reply_probability_zero_swallows():
    q: asyncio.Queue[ChatMessage] = _queue()
    dq = DonationQueue(q, reply_probability=0.0)
    await dq.enqueue([_ev()])
    assert q.qsize() == 0


@pytest.mark.asyncio
async def test_cooldown_suppresses_rapid_second():
    q: asyncio.Queue[ChatMessage] = _queue()
    dq = DonationQueue(q, cooldown_sec=60.0)
    await dq.enqueue([_ev(event_id="a")])
    await dq.enqueue([_ev(event_id="b")])
    assert q.qsize() == 1  # second suppressed by cooldown


@pytest.mark.asyncio
async def test_queue_full_never_loses_last_donation():
    q: asyncio.Queue[ChatMessage] = _queue(maxsize=1)
    dq = DonationQueue(q)
    await dq.enqueue([_ev(event_id="a")])
    await dq.enqueue([_ev(event_id="b")])  # evicts the old one, queues the new
    msg = q.get_nowait()
    assert msg.donation.event_id == "b"

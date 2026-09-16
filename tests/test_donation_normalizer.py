"""Donation normalization tests — LivePix + Streamlabs shapes from the docs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest

from donations.base import DonationDedupe
from donations.normalizer import (
    DonationPayloadError,
    apply_livepix_details,
    normalize_livepix_webhook,
    normalize_streamlabs_event,
)


# ---------------------------------------------------------------------------
# LivePix — shapes from docs.livepix.gg
# ---------------------------------------------------------------------------

LIVEPIX_WEBHOOK = {
    "userId": "61021c7bdabe5e001225b65b",
    "clientId": "61021c7bdabe5e001225b65b",
    "event": "new",
    "resource": {"id": "res1", "reference": "ref1", "type": "message"},
}


def test_livepix_message_webhook_minimal():
    ev = normalize_livepix_webhook(LIVEPIX_WEBHOOK)
    assert ev.source == "livepix"
    assert ev.event_id == "res1"
    assert ev.metadata["reference"] == "ref1"
    assert ev.metadata["resource_type"] == "message"


def test_livepix_payment_webhook_without_type():
    payload = {"userId": "u", "clientId": "c", "event": "new",
               "resource": {"id": "p1"}}
    ev = normalize_livepix_webhook(payload)
    assert ev.event_id == "p1"
    assert ev.metadata["resource_type"] == "payment"


def test_livepix_rejects_non_object():
    with pytest.raises(DonationPayloadError):
        normalize_livepix_webhook(["nope"])
    with pytest.raises(DonationPayloadError):
        normalize_livepix_webhook("string")


def test_livepix_rejects_missing_resource():
    with pytest.raises(DonationPayloadError):
        normalize_livepix_webhook({"userId": "u", "event": "new"})


def test_livepix_rejects_resource_without_ids():
    with pytest.raises(DonationPayloadError):
        normalize_livepix_webhook({"event": "new", "resource": {"type": "message"}})


def test_livepix_rejects_unknown_event_value():
    payload = dict(LIVEPIX_WEBHOOK, event="exploded")
    with pytest.raises(DonationPayloadError):
        normalize_livepix_webhook(payload)


def test_livepix_user_id_allowlist():
    ok = normalize_livepix_webhook(LIVEPIX_WEBHOOK, expected_user_id="61021c7bdabe5e001225b65b")
    assert ok.event_id == "res1"
    with pytest.raises(DonationPayloadError):
        normalize_livepix_webhook(LIVEPIX_WEBHOOK, expected_user_id="someone-else")


def test_livepix_details_enrichment():
    ev = normalize_livepix_webhook(LIVEPIX_WEBHOOK)
    details = {"data": {"id": "res1", "username": "Maria", "message": "manda salve",
                        "amount": 10, "currency": "BRL",
                        "createdAt": "2021-01-01T00:00:00-03:00"}}
    apply_livepix_details(ev, details)
    assert ev.donor_name == "Maria"
    assert ev.message == "manda salve"
    assert ev.amount == 10.0
    assert ev.currency == "BRL"
    assert ev.formatted_amount == "BRL 10,00"


def test_livepix_details_missing_fields_keep_defaults():
    ev = normalize_livepix_webhook(LIVEPIX_WEBHOOK)
    apply_livepix_details(ev, {"data": {}})
    assert ev.donor_name == "Anônimo"
    assert ev.message == ""
    assert ev.amount == 0.0


# ---------------------------------------------------------------------------
# Streamlabs — shapes from dev.streamlabs.com/docs/socket-api
# ---------------------------------------------------------------------------

STREAMLABS_DONATION = {
    "type": "donation",
    "message": [{
        "id": 96164121,
        "name": "test",
        "amount": "13.37",
        "formatted_amount": "$13.37",
        "formattedAmount": "$13.37",
        "message": "test donation",
        "currency": "USD",
        "emotes": None,
        "iconClassName": "user",
        "to": {"name": "Streamer"},
        "from": "test",
        "from_user_id": None,
        "_id": "0820c9d5bafd768c9843f5e35c885e71",
    }],
    "event_id": "evt_17e5f4dc6888767ed9799f78dfa2cabc",
}


def test_streamlabs_docs_example():
    events = normalize_streamlabs_event(STREAMLABS_DONATION)
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "streamlabs"
    assert ev.event_id == "0820c9d5bafd768c9843f5e35c885e71"
    assert ev.donor_name == "test"
    assert ev.amount == 13.37
    assert ev.currency == "USD"
    assert ev.message == "test donation"


def test_streamlabs_message_array_normalized_all():
    payload = {
        "type": "donation",
        "message": [
            {"name": "a", "amount": 5, "currency": "BRL", "_id": "id-a"},
            {"name": "b", "amount": 7, "currency": "BRL", "_id": "id-b"},
        ],
        "event_id": "evt_x",
    }
    events = normalize_streamlabs_event(payload)
    assert [e.event_id for e in events] == ["id-a", "id-b"]
    assert [e.donor_name for e in events] == ["a", "b"]


def test_streamlabs_donation_without_message_text():
    payload = {"type": "donation",
               "message": [{"name": "quiet", "amount": 1, "_id": "id-q"}]}
    events = normalize_streamlabs_event(payload)
    assert len(events) == 1
    assert events[0].message == ""
    assert events[0].donor_name == "quiet"


def test_streamlabs_single_object_message_is_lenient():
    payload = {"type": "donation",
               "message": {"name": "solo", "amount": 2, "_id": "id-s"}}
    events = normalize_streamlabs_event(payload)
    assert len(events) == 1
    assert events[0].event_id == "id-s"


def test_streamlabs_ignores_non_donation_types():
    for t in ("follow", "subscription", "bits", "raid", "host", "superchat"):
        payload = {"type": t, "for": "twitch_account",
                   "message": [{"name": "x", "amount": 1, "_id": "z"}]}
        assert normalize_streamlabs_event(payload) == []


def test_streamlabs_invalid_amount_skipped():
    payload = {"type": "donation",
               "message": [{"name": "bad", "amount": "n/a", "_id": "id-bad"}]}
    assert normalize_streamlabs_event(payload) == []


def test_streamlabs_negative_amount_skipped():
    payload = {"type": "donation",
               "message": [{"name": "neg", "amount": -5, "_id": "id-neg"}]}
    assert normalize_streamlabs_event(payload) == []


def test_streamlabs_no_id_skipped():
    payload = {"type": "donation",
               "message": [{"name": "noid", "amount": 1}]}
    assert normalize_streamlabs_event(payload) == []


def test_streamlabs_formatted_amount_fallback_to_name_field():
    payload = {"type": "donation",
               "message": [{"from": "oldfield", "amount": 3, "_id": "id-f"}]}
    events = normalize_streamlabs_event(payload)
    assert events[0].donor_name == "oldfield"


def test_streamlabs_currency_from_formatted_when_missing():
    payload = {"type": "donation",
               "message": [{"name": "br", "amount": "10,00",
                            "formatted_amount": "R$ 10,00", "_id": "id-br"}]}
    events = normalize_streamlabs_event(payload)
    assert events[0].amount == 10.0
    assert events[0].currency == "BRL"


def test_streamlabs_invalid_payloads_raise():
    with pytest.raises(DonationPayloadError):
        normalize_streamlabs_event("not-a-dict")
    with pytest.raises(DonationPayloadError):
        normalize_streamlabs_event({"type": "donation"})  # no message field
    with pytest.raises(DonationPayloadError):
        normalize_streamlabs_event({"type": "donation", "message": 42})


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_dedupe_first_time_then_duplicate():
    d = DonationDedupe()
    assert d.first_time("evt-123") is True
    assert d.first_time("evt-123") is False
    assert d.first_time("evt-124") is True


def test_dedupe_empty_id_fails_open():
    d = DonationDedupe()
    # Without a usable id we can't dedupe — fail open (never drop a real donation).
    assert d.first_time("") is True
    assert d.first_time("") is True


def test_dedupe_multiple_donations_distinct_ids():
    d = DonationDedupe()
    ids = ["a", "b", "c", "a"]
    assert [d.first_time(i) for i in ids] == [True, True, True, False]

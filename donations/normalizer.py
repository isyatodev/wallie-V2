"""Payload normalizers — turn raw platform payloads into DonationEvent.

Only fields documented by each platform's official docs are read:
  * LivePix  — https://docs.livepix.gg  (webhook + /v2 API shapes)
  * Streamlabs — https://dev.streamlabs.com/docs/socket-api
Everything is defensive: fields may be missing, message may be an array,
amounts may be strings. Invalid payloads raise DonationPayloadError and the
caller answers the webhook with 400 without touching the AI pipeline.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .base import DonationEvent


class DonationPayloadError(ValueError):
    """Raised when a payload can't be trusted as a valid donation event."""


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float conversion; None when unusable (not zero!)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(" ", "").replace(",", ".")
        # formatted amounts like "R$ 10,00" / "$13.37" — strip currency symbols.
        cleaned = "".join(c for c in cleaned if c.isdigit() or c == ".")
        cleaned = cleaned.strip(".")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# LivePix — docs.livepix.gg
# ---------------------------------------------------------------------------

# Webhook payloads carry {userId, clientId, event, resource{id, reference, type}}.
# `event` marks the lifecycle: "new" when the payment/message is created. The
# webhook doc does not document a distinct paid-confirmation event value, so we
# accept values that indicate a new resource and reject clearly-unrelated ones.
_LIVEPIX_ACCEPTED_EVENTS = {"new", "payment", "message", "confirmed", "paid"}


def normalize_livepix_webhook(
    payload: Any,
    *,
    expected_user_id: str = "",
    now: Optional[float] = None,
) -> DonationEvent:
    if not isinstance(payload, dict):
        raise DonationPayloadError("livepix: payload must be a JSON object")
    resource = payload.get("resource")
    if not isinstance(resource, dict):
        raise DonationPayloadError("livepix: missing resource object")

    user_id = str(payload.get("userId") or "")
    if expected_user_id and user_id and user_id != expected_user_id:
        raise DonationPayloadError("livepix: userId does not match configured account")

    event_value = str(payload.get("event") or "").lower()
    if event_value and event_value not in _LIVEPIX_ACCEPTED_EVENTS:
        raise DonationPayloadError(f"livepix: unsupported event value {event_value!r}")

    resource_type = str(resource.get("type") or "").lower()
    resource_id = str(resource.get("id") or "")
    reference = str(resource.get("reference") or "")
    if not resource_id and not reference:
        raise DonationPayloadError("livepix: resource has neither id nor reference")

    # The webhook body deliberately carries only basic info (per the official
    # docs). Normalizing produces the shell; `enriched` fields, when fetched via
    # GET /v2/messages/{id} (a real documented endpoint), are merged by the caller.
    if resource_type not in ("", "payment", "message"):
        raise DonationPayloadError(f"livepix: unsupported resource type {resource_type!r}")

    return DonationEvent(
        source="livepix",
        event_id=resource_id or f"ref:{reference}",
        donor_name="Anônimo",
        timestamp=now if now is not None else time.time(),
        metadata={
            "resource_type": resource_type or "payment",
            "reference": reference,
            "user_id": user_id,
            "lifecycle": event_value or "new",
        },
    )


def apply_livepix_details(event: DonationEvent, details: Any) -> DonationEvent:
    """Merge GET /v2/messages/{id} (or /payments/{id}) `data` into an event.

    Documented fields used: username, message, amount (integer, MAJOR units in
    the samples — e.g. 1000 = 1000, not 10.00), currency, createdAt.
    """
    data = details.get("data") if isinstance(details, dict) else None
    if not isinstance(data, dict):
        return event
    name = str(data.get("username") or "").strip()
    if name:
        event.donor_name = name
    msg = data.get("message")
    if isinstance(msg, str):
        event.message = msg.strip()
    amount = _as_float(data.get("amount"))
    if amount is not None and amount >= 0:
        event.amount = amount
    currency = data.get("currency")
    if isinstance(currency, str) and currency.strip():
        event.currency = currency.strip().upper()
    created = data.get("createdAt")
    if isinstance(created, str) and created:
        event.metadata["created_at"] = created
    return event


# ---------------------------------------------------------------------------
# Streamlabs — dev.streamlabs.com/docs/socket-api
# ---------------------------------------------------------------------------

def normalize_streamlabs_event(payload: Any) -> list[DonationEvent]:
    """Normalize a Streamlabs socket `event` payload into donation events.

    Returns ALL normalized donations from the payload (the `message` field is a
    documented ARRAY — every item is normalized). Events that are not
    `type === "donation"` are ignored (follows/subs/bits/raids stay out) — the
    architecture leaves room for handling them later without rewriting this.
    """
    if not isinstance(payload, dict):
        raise DonationPayloadError("streamlabs: payload must be a JSON object")
    if str(payload.get("type") or "").lower() != "donation":
        return []  # filter: only donations, everything else is out of scope here

    raw_items = payload.get("message")
    if raw_items is None:
        raise DonationPayloadError("streamlabs: donation event without message array")
    if isinstance(raw_items, dict):
        raw_items = [raw_items]  # be lenient even though docs say array
    if not isinstance(raw_items, list):
        raise DonationPayloadError("streamlabs: message field is not an array")

    envelope_event_id = str(payload.get("event_id") or "").strip()
    out: list[DonationEvent] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        amount = _as_float(item.get("amount"))
        if amount is None or amount < 0:
            # A donation with a broken amount can't be trusted — skip the item.
            continue
        item_id = str(item.get("_id") or item.get("id") or "").strip()
        event_id = item_id or envelope_event_id
        if not event_id:
            # Docs always include _id; without ANY id we cannot dedupe — skip
            # rather than risk double responses.
            continue
        donor = str(item.get("name") or item.get("from") or "").strip() or "Anonymous"
        message = item.get("message")
        formatted = item.get("formatted_amount") or item.get("formattedAmount") or ""
        currency = str(item.get("currency") or "").strip().upper()
        if not currency and isinstance(formatted, str) and formatted:
            # Fallback: pull the currency symbol out of the formatted amount.
            for sym, code in (("R$", "BRL"), ("US$", "USD"), ("$", "USD"), ("€", "EUR")):
                if sym in formatted:
                    currency = code
                    break
        out.append(DonationEvent(
            source="streamlabs",
            event_id=event_id,
            donor_name=donor,
            donor_id=str(item.get("from_user_id") or "") or "",
            amount=amount,
            currency=currency,
            message=(message.strip() if isinstance(message, str) else ""),
            metadata={
                "envelope_event_id": envelope_event_id,
                "formatted_amount": formatted if isinstance(formatted, str) else "",
            },
        ))
    return out

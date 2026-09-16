"""LivePix integration — OAuth2 (client_credentials), webhook endpoint, and
detail enrichment. All endpoints/scopes taken from the official docs
(https://docs.livepix.gg). No invented endpoints.

LivePix docs do NOT document a webhook signature header, so the webhook is
validated by structure (normalizer) + an optional userId allowlist. This is a
documented divergence from a hypothetical HMAC scheme — nothing is invented.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger


class LivePixError(RuntimeError):
    pass


class LivePixClient:
    """OAuth2 client_credentials against oauth.livepix.gg, token auto-refreshed
    before expiry. Tokens are never logged."""

    _TOKEN_URL = "https://oauth.livepix.gg/oauth2/token"
    _API_BASE = "https://api.livepix.gg/v2"

    def __init__(self, *, client_id: str, client_secret: str) -> None:
        if not client_id or not client_secret:
            raise LivePixError("livepix: client_id/client_secret required")
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = httpx.AsyncClient(timeout=10.0)
        self._token: str = ""
        self._token_exp: float = 0.0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _ensure_token(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._token_exp - 30:
                return self._token
            resp = await self._http.post(
                self._TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    # Scopes from the docs' own example (account:read wallet:read webhooks)
                    # plus the documented read endpoints we call for enrichment.
                    "scope": "messages:read payments:read webhooks",
                },
            )
            if resp.status_code != 200:
                raise LivePixError(
                    f"livepix: token request failed with HTTP {resp.status_code}"
                )
            data = resp.json() or {}
            token = str(data.get("access_token") or "")
            if not token:
                raise LivePixError("livepix: token response missing access_token")
            expires_in = data.get("expires_in", 3600)
            try:
                expires_in = float(expires_in)
            except (TypeError, ValueError):
                expires_in = 3600.0
            self._token = token
            self._token_exp = time.time() + max(60.0, expires_in)
            logger.info("livepix: OAuth token acquired (auto-refresh before expiry)")
            return self._token

    async def get_message(self, message_id: str) -> Optional[dict[str, Any]]:
        """GET /v2/messages/{id} — documented endpoint, used for enrichment."""
        return await self._get_resource(f"{self._API_BASE}/messages/{message_id}")

    async def get_payment(self, payment_id: str) -> Optional[dict[str, Any]]:
        """GET /v2/payments/{id} — documented endpoint, used for enrichment."""
        return await self._get_resource(f"{self._API_BASE}/payments/{payment_id}")

    async def create_webhook(self, url: str) -> Optional[str]:
        """POST /v2/webhooks {url} → returns webhook id (documented endpoint)."""
        token = await self._ensure_token()
        resp = await self._http.post(
            f"{self._API_BASE}/webhooks",
            json={"url": url},
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code not in (200, 201):
            raise LivePixError(
                f"livepix: webhook creation failed with HTTP {resp.status_code}"
            )
        data = (resp.json() or {}).get("data") or {}
        webhook_id = str(data.get("id") or "")
        logger.info(
            "livepix: webhook registered (id …{})",
            webhook_id[-6:] if webhook_id else "?",
        )
        return webhook_id or None

    async def _get_resource(self, url: str) -> Optional[dict[str, Any]]:
        try:
            token = await self._ensure_token()
            resp = await self._http.get(
                url, headers={"Authorization": f"Bearer {token}"}
            )
            if resp.status_code == 429:
                logger.warning("livepix: rate limited (429) — enrichment skipped this time")
                return None
            if resp.status_code != 200:
                logger.warning(f"livepix: GET {url.rsplit('/', 1)[-1]} → HTTP {resp.status_code}")
                return None
            return resp.json()
        except LivePixError:
            raise
        except Exception as e:
            logger.warning(f"livepix: enrichment request failed (non-fatal): {e}")
            return None


def mount_livepix_webhook(
    app: Any,
    *,
    on_donation,                       # async callable(DonationEvent) -> None (queues + 200 fast)
    webhook_path: str,
    expected_user_id: str = "",
    verify_user_id: bool = True,
    enrich_client: Optional[LivePixClient] = None,
    dedupe: Any = None,
) -> Any:
    """Mount POST {webhook_path} on the given FastAPI app.

    Contract (per the master prompt): validate → queue → HTTP 200 immediately.
    The webhook NEVER waits for the LLM or TTS.
    """
    from donations.base import DonationDedupe
    from donations.normalizer import (
        DonationPayloadError,
        apply_livepix_details,
        normalize_livepix_webhook,
    )

    _dedupe = dedupe or DonationDedupe()
    check_user = verify_user_id and bool(expected_user_id)

    # Strong refs so fire-and-forget work isn't garbage-collected mid-flight.
    _background: set[asyncio.Task] = set()

    def _spawn(coro) -> None:
        task = asyncio.create_task(coro)
        _background.add(task)
        task.add_done_callback(_background.discard)

    async def livepix_webhook(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        try:
            event = normalize_livepix_webhook(
                payload,
                expected_user_id=expected_user_id if check_user else "",
            )
        except DonationPayloadError as e:
            logger.warning(f"[LivePix] webhook rejected: {e}")
            return JSONResponse({"error": str(e)}, status_code=400)

        logger.info(
            f"[LivePix] Donation received (ref={str(event.metadata.get('reference', '?'))[:12]})"
        )

        # Idempotency: the same platform event id must never produce two AI
        # responses (LivePix retries the webhook for up to 24h on failure).
        if not _dedupe.first_time(event.event_id):
            logger.info("[LivePix] duplicate webhook event suppressed")
            return JSONResponse({"ok": True, "duplicate": True}, status_code=200)

        # Everything below happens in the background: the webhook validates,
        # enqueues, and answers HTTP 200 immediately — it NEVER waits for the
        # LLM, the TTS, or the enrichment HTTP call.
        if enrich_client is not None:

            async def _enrich_and_queue() -> None:
                resource_id = event.event_id
                try:
                    if event.metadata.get("resource_type") == "message":
                        details = await enrich_client.get_message(resource_id)
                    else:
                        details = await enrich_client.get_payment(resource_id)
                    if details is not None:
                        apply_livepix_details(event, details)
                except Exception as e:
                    logger.warning(f"[LivePix] enrichment failed (queueing anyway): {e}")
                await on_donation(event)

            _spawn(_enrich_and_queue())
        else:
            _spawn(on_donation(event))

        return JSONResponse({"ok": True}, status_code=200)

    app.post(webhook_path)(livepix_webhook)
    return livepix_webhook

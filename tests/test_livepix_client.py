"""LivePix client + webhook tests (no network — httpx transport is mocked)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from donations.base import DonationDedupe
from donations.livepix import LivePixClient, LivePixError


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------

def _token_response(status_code=200, expires_in=3600):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"access_token": "tok123", "expires_in": expires_in,
                              "token_type": "bearer"}
    return resp


@pytest.mark.asyncio
async def test_token_request_and_reuse():
    client = LivePixClient(client_id="cid", client_secret="sec")
    try:
        with patch.object(client._http, "post", new=AsyncMock(return_value=_token_response())) as spy:
            tok1 = await client._ensure_token()
            tok2 = await client._ensure_token()  # cached until near expiry
        assert tok1 == tok2 == "tok123"
        assert spy.call_count == 1  # second call served from cache
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_token_http_error_raises():
    client = LivePixClient(client_id="cid", client_secret="sec")
    try:
        resp = MagicMock()
        resp.status_code = 401
        with patch.object(client._http, "post", new=AsyncMock(return_value=resp)):
            with pytest.raises(LivePixError) as ei:
                await client._ensure_token()
        assert "401" in str(ei.value)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_missing_credentials_raise():
    with pytest.raises(LivePixError):
        LivePixClient(client_id="", client_secret="")


# ---------------------------------------------------------------------------
# Webhook route (FastAPI mounted, ASGI transport)
# ---------------------------------------------------------------------------

def _build_app():
    from fastapi import FastAPI

    app = FastAPI()
    queued: list = []
    captured: dict = {}

    async def on_donation(ev) -> None:
        captured["event"] = ev
        queued.append(ev)

    from donations.livepix import mount_livepix_webhook
    mount_livepix_webhook(
        app,
        on_donation=on_donation,
        webhook_path="/webhooks/livepix",
        expected_user_id="",
        verify_user_id=False,
        dedupe=DonationDedupe(),
    )
    return app, queued, captured


def _livepix_payload(**overrides):
    payload = {
        "userId": "u1", "clientId": "c1", "event": "new",
        "resource": {"id": "res1", "reference": "ref1", "type": "message"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_webhook_accepts_and_queues_fast():
    app, queued, captured = _build_app()
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/webhooks/livepix", json=_livepix_payload())
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    await __import__("asyncio").sleep(0)  # let the queue task run
    assert len(queued) == 1
    assert captured["event"].event_id == "res1"


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_payload():
    app, queued, _ = _build_app()
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/webhooks/livepix", json={"hello": "world"})
    assert r.status_code == 400
    assert "error" in r.json()
    assert queued == []


@pytest.mark.asyncio
async def test_webhook_rejects_bad_json():
    app, queued, _ = _build_app()
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/webhooks/livepix", content=b"not json",
                         headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert queued == []


@pytest.mark.asyncio
async def test_webhook_wrong_user_id_rejected():
    app, queued, _ = _build_app()
    # Re-mount with verification on.
    from fastapi import FastAPI
    from donations.livepix import mount_livepix_webhook
    app2 = FastAPI()

    async def on_donation(ev) -> None:
        queued.append(ev)

    mount_livepix_webhook(
        app2, on_donation=on_donation, webhook_path="/webhooks/livepix",
        expected_user_id="expected-user", verify_user_id=True,
        dedupe=DonationDedupe(),
    )
    import httpx
    transport = httpx.ASGITransport(app=app2)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/webhooks/livepix", json=_livepix_payload(userId="impostor"))
    assert r.status_code == 400
    assert queued == []


@pytest.mark.asyncio
async def test_webhook_duplicate_event_processed_once():
    app, queued, _ = _build_app()
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        await c.post("/webhooks/livepix", json=_livepix_payload())
        await c.post("/webhooks/livepix", json=_livepix_payload())
    await __import__("asyncio").sleep(0)
    assert len(queued) == 1  # idempotent: same resource id → one response


@pytest.mark.asyncio
async def test_webhook_never_blocks_on_slow_handler():
    """The HTTP response must not wait for the (possibly slow) queue handler."""
    from fastapi import FastAPI
    from donations.livepix import mount_livepix_webhook
    import httpx

    app3 = FastAPI()
    release = __import__("asyncio").Event()

    async def slow_handler(ev) -> None:
        await release.wait()  # would block forever if awaited inline

    mount_livepix_webhook(
        app3, on_donation=slow_handler, webhook_path="/webhooks/livepix",
        expected_user_id="", verify_user_id=False,
        dedupe=DonationDedupe(),
    )
    transport = httpx.ASGITransport(app=app3)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await __import__("asyncio").wait_for(
            c.post("/webhooks/livepix", json=_livepix_payload()), timeout=2.0
        )
    assert r.status_code == 200

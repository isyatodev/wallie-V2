"""Streamlabs integration — Socket API (real-time donations), per the official
docs (https://dev.streamlabs.com/docs/socket-api).

Flow (documented): access token → GET /socket/token → socket token →
Socket.IO connection to sockets.streamlabs.com → `event` messages with
type === "donation". python-socketio 4.x speaks Socket.IO 2.x protocol, which
is what Streamlabs serves (their official sample uses socket.io 2.0.3).
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Optional

import httpx
from loguru import logger

from .base import DonationDedupe
from .normalizer import normalize_streamlabs_event

_STREAMLABS_API = "https://streamlabs.com/api/v2.0"


class StreamlabsMonitor:
    """Real-time donation listener over the Streamlabs Socket API.

    - Auto-connects and reconnects with exponential backoff + jitter (never a
      tight loop).
    - Guarantees a single live connection at a time.
    - Only `type === "donation"` events are normalized (follows/subs/bits/raids
      are ignored; new types can be added later without touching the pipeline).
    - Dedupe by the real Streamlabs ids (`_id` / `event_id`).
    """

    def __init__(
        self,
        *,
        cfg,                                # DonationsConfig
        on_donations,                       # async callable(list[DonationEvent]) -> None
        access_token: str = "",
        socket_token: str = "",
        dedupe: Optional[DonationDedupe] = None,
    ) -> None:
        self._cfg = cfg
        self._on_donations = on_donations
        self._access_token = access_token.strip()
        self._socket_token = socket_token.strip()
        self._dedupe = dedupe or DonationDedupe()
        self._sio: Any = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._connected_once = False

    # ----- lifecycle -----
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="streamlabs")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        await self._disconnect()

    # ----- socket token -----
    async def _resolve_socket_token(self) -> str:
        """Socket token: either configured directly (dashboard) or fetched from
        the documented GET /socket/token endpoint with the OAuth access token."""
        if self._socket_token:
            return self._socket_token
        if not self._access_token:
            raise RuntimeError("streamlabs: no STREAMLABS_SOCKET_TOKEN or STREAMLABS_ACCESS_TOKEN")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_STREAMLABS_API}/socket/token",
                params={"access_token": self._access_token},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"streamlabs: /socket/token failed with HTTP {resp.status_code}"
                    " (access token invalid or expired?)"
                )
            data = resp.json() or {}
            token = str(data.get("socket_token") or "")
            if not token:
                raise RuntimeError("streamlabs: /socket/token response missing socket_token")
            logger.info("[Streamlabs] socket token acquired from /socket/token")
            return token

    # ----- connection loop -----
    async def _run(self) -> None:
        backoff = 1.0
        max_backoff = max(5.0, float(self._cfg.streamlabs_reconnect_max_sec))
        while not self._stop.is_set():
            try:
                token = await self._resolve_socket_token()
            except Exception as e:
                logger.warning(f"[Streamlabs] token problem: {e}")
                await self._sleep(backoff * (0.5 + random.random()))
                backoff = min(max_backoff, backoff * 2)
                continue

            try:
                await self._connect(token)
                backoff = 1.0  # reset on successful connect
                await self._wait_socket()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[Streamlabs] connection error: {e}")
            if not self._stop.is_set():
                delay = backoff * (0.5 + random.random())  # jittered backoff
                await self._sleep(delay)
                backoff = min(max_backoff, backoff * 2)

    async def _sleep(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.5, delay))
        except asyncio.TimeoutError:
            pass

    async def _connect(self, socket_token: str) -> None:
        import socketio  # python-socketio (4.x → Socket.IO 2.x protocol)

        if self._sio is not None:
            await self._disconnect()
        sio = socketio.AsyncClient(
            reconnection=False,  # we own reconnection (backoff + single connection)
            logger=False, engineio_logger=False,
        )

        @sio.event
        async def connect() -> None:
            self._connected_once = True
            logger.info("[Streamlabs] Connected")

        @sio.event
        async def disconnect() -> None:
            logger.info("[Streamlabs] Disconnected")

        @sio.on("event")
        async def on_event(data: Any) -> None:
            await self._handle_event(data)

        self._sio = sio
        url = self._cfg.streamlabs_url.rstrip("/") + f"/?token={socket_token}"
        await sio.connect(url, transports=["websocket"], socketio_path="socket.io")

    async def _wait_socket(self) -> None:
        sio = self._sio
        if sio is None:
            return
        stop_task = asyncio.ensure_future(self._stop.wait())
        try:
            done, _pending = await asyncio.wait(
                {stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            stop_task.cancel()

    async def _disconnect(self) -> None:
        sio, self._sio = self._sio, None
        if sio is None:
            return
        try:
            await sio.disconnect()
        except Exception:
            pass

    # ----- events -----
    async def _handle_event(self, data: Any) -> None:
        try:
            events = normalize_streamlabs_event(data)
        except Exception as e:
            logger.warning(f"[Streamlabs] invalid payload ignored: {e}")
            return
        if not events:
            return
        fresh = [ev for ev in events if self._dedupe.first_time(ev.event_id)]
        dropped = len(events) - len(fresh)
        if dropped:
            logger.info(f"[Streamlabs] {dropped} duplicate event(s) suppressed")
        if not fresh:
            return
        for ev in fresh:
            logger.info(
                f"[Streamlabs] Donation received: {ev.donor_name} "
                f"{ev.formatted_amount or ev.amount}{(' — ' + ev.message[:40]) if ev.message else ''}"
            )
        try:
            await self._on_donations(fresh)
            for ev in fresh:
                logger.info("[Streamlabs] Donation queued")
        except Exception as e:
            logger.error(f"[Streamlabs] failed to queue donation: {e}")

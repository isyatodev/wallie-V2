"""Twitch chat monitor via the IRC gateway."""
from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from .base import ChatMessage, ChatMonitor

_IRC_WSS = "wss://irc-ws.chat.twitch.tv:443"


class TwitchChatMonitor(ChatMonitor):
    platform = "twitch"

    def __init__(self, *, channel: str, oauth_token: str = "", nick: str = "") -> None:
        self._channel = channel.lstrip("#").lower()
        self._oauth = oauth_token.strip()
        if self._oauth and not self._oauth.startswith("oauth:"):
            self._oauth = f"oauth:{self._oauth}"
        self._nick = nick.lower() if nick else f"justinfan{hash(channel) % 99999}"
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self, out_queue: asyncio.Queue[ChatMessage]) -> None:
        if not self._channel:
            logger.warning("twitch: no channel configured; monitor disabled")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(out_queue), name="twitch-chat")
        logger.info(f"twitch: chat monitor started on #{self._channel}")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run(self, out: asyncio.Queue[ChatMessage]) -> None:
        import websockets

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(_IRC_WSS, open_timeout=10) as ws:
                    await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                    if self._oauth:
                        await ws.send(f"PASS {self._oauth}")
                    await ws.send(f"NICK {self._nick}")
                    await ws.send(f"JOIN #{self._channel}")
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        for line in str(raw).split("\r\n"):
                            if not line:
                                continue
                            if line.startswith("PING"):
                                await ws.send(line.replace("PING", "PONG", 1))
                                continue
                            parsed = _parse_privmsg(line)
                            if parsed:
                                user, text, is_bits, is_broadcaster = parsed
                                try:
                                    out.put_nowait(ChatMessage(
                                        platform="twitch",
                                        username=user,
                                        text=text,
                                        is_highlight=is_bits,
                                        is_streamer=is_broadcaster,
                                    ))
                                except asyncio.QueueFull:
                                    pass
                                continue
                            notice = _parse_usernotice(line)
                            if notice:
                                try:
                                    out.put_nowait(notice)
                                except asyncio.QueueFull:
                                    pass
            except Exception as e:
                logger.warning(f"twitch: connection error: {e}, reconnecting in 5s")
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass


_HIGHLIGHT_NOTICE_IDS = {"sub", "resub", "subgift", "submysterygift", "raid"}


def _parse_tags(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in raw.lstrip("@").split(";"):
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def _parse_privmsg(line: str) -> Optional[tuple[str, str, bool, bool]]:
    """Returns (nick, text, is_bits, is_broadcaster).

    Broadcaster detection uses the documented IRC tags: `badges` containing
    broadcaster/1 marks the channel owner (the STREAMER)."""
    tags = ""
    rest = line
    if line.startswith("@"):
        tags, _, rest = line.partition(" ")
    if "PRIVMSG" not in rest:
        return None
    try:
        prefix, _, payload = rest.partition(" PRIVMSG ")
        nick = prefix.lstrip(":").split("!", 1)[0]
        _, _, text = payload.partition(" :")
        is_bits = False
        is_broadcaster = False
        if tags:
            for kv in tags.lstrip("@").split(";"):
                if kv.startswith("bits=") and kv[5:] not in {"", "0"}:
                    is_bits = True
                elif kv.startswith("badges=") and "broadcaster/1" in kv[7:]:
                    is_broadcaster = True
        return nick, text.strip(), is_bits, is_broadcaster
    except Exception:
        return None


def _parse_usernotice(line: str) -> Optional[ChatMessage]:
    if "USERNOTICE" not in line:
        return None
    tags_raw = ""
    rest = line
    if line.startswith("@"):
        tags_raw, _, rest = rest.partition(" ")
    if "USERNOTICE" not in rest:
        return None
    try:
        tags = _parse_tags(tags_raw)
        msg_id = tags.get("msg-id", "")
        if msg_id not in _HIGHLIGHT_NOTICE_IDS:
            return None
        nick = tags.get("display-name") or tags.get("login", "viewer")
        sys_msg = tags.get("system-msg", "").replace("\\s", " ")
        _, _, trailing = rest.partition(" :")
        user_text = trailing.strip() if " :" in rest else ""
        text = user_text if user_text else sys_msg
        if not text:
            text = f"{nick} just subscribed!"
        return ChatMessage(platform="twitch", username=nick, text=text, is_highlight=True)
    except Exception:
        return None

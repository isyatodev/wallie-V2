"""FastAPI dashboard — config, lifecycle, and test endpoints."""
from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import socket
import sys
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

from audio.player import list_output_devices
from config import (
    AppConfig,
    ProviderBlock,
    Secrets,
    activate_profile,
    clone_profile,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from core import Orchestrator, Persona
from llm import build_provider
from tts import build_tts

STATIC_DIR = Path(__file__).parent / "static"


# -------------------------------------------------------------------
# PIN authentication (closure-based, shared state)
# -------------------------------------------------------------------
_PUBLIC_PATHS = frozenset({"/login", "/api/auth/login"})


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


_DEFAULT_TEST_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "openrouter": "openai/gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
}


def _default_model_for(provider: str) -> str:
    return _DEFAULT_TEST_MODELS.get(provider, "")


def _provider_env_slug(pid: str) -> str:
    """Env-safe slug for a provider id (matches wallie/secrets_store)."""
    s = _re.sub(r"[^a-z0-9_]+", "_", (pid or "").lower()).strip("_")
    return s[:40] or "provider"


import re as _re

_KEY_PATTERNS = [
    _re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    _re.compile(r"sk-or-[A-Za-z0-9_\-]{10,}"),
    _re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    _re.compile(r"gsk_[A-Za-z0-9_\-]{20,}"),
    _re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),       # Google API keys
    _re.compile(r"oauth:[A-Za-z0-9_\-]{20,}"),     # Twitch
    _re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    _re.compile(r"\b[A-Fa-f0-9]{40,}\b"),          # generic hex tokens
]


def _scrub_error(msg: str, limit: int = 220) -> str:
    out = msg
    for pat in _KEY_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out[:limit]


class DashboardState:
    def __init__(self) -> None:
        self.orchestrator: Optional[Orchestrator] = None
        self.clients: set[WebSocket] = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)

    def emit_memory_event(self, payload: dict[str, Any]) -> None:
        """Sink for MemoryCapture.on_captured — pushes each AI-captured fact
        to every connected dashboard client over the events WebSocket."""
        try:
            self._queue.put_nowait({"type": "memory", "data": payload})
        except asyncio.QueueFull:
            pass

    def attach_logger(self) -> None:
        def sink(message) -> None:
            record = message.record
            entry = {
                "type": "log",
                "level": record["level"].name,
                "time": record["time"].isoformat(),
                "msg": record["message"],
            }
            try:
                self._queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass
        logger.add(sink, level="INFO", enqueue=False)

    async def broadcaster(self) -> None:
        while True:
            entry = await self._queue.get()
            dead: list[WebSocket] = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(json.dumps(entry))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


class ProfileCreateBody(BaseModel):
    name: str
    clone_from: Optional[str] = None


class LongTermCreateBody(BaseModel):
    text: str
    kind: str = "long_term"       # long_term | short_term
    tag: str = ""
    about: str = ""               # optional: bind to a voice-print speaker name
    ttl_hours: float = 24.0       # short_term only


class LongTermUpdateBody(BaseModel):
    text: Optional[str] = None
    tag: Optional[str] = None
    about: Optional[str] = None   # speaker name, or "" to clear
    kind: Optional[str] = None
    ttl_hours: Optional[float] = None


class LongTermClearBody(BaseModel):
    kind: Optional[str] = None    # long_term | short_term | None = both


class TestPersonaBody(BaseModel):
    kind: str = "monologue"  # monologue | chat | vision
    topic: Optional[str] = None
    chat_text: Optional[str] = None
    chat_user: Optional[str] = None


class TestVoiceBody(BaseModel):
    text: str


class TestExpressionBody(BaseModel):
    expression: str  # slot ("happy", "hype", ...) OR a raw hotkey id/name


class TestLookBody(BaseModel):
    x: float = 0.0
    y: float = 0.0
    hold_sec: float = 0.6


class SecretUpdateBody(BaseModel):
    env: str
    value: str  # empty string deletes


class SecretBulkBody(BaseModel):
    values: dict[str, str]


class TestProviderBody(BaseModel):
    # "openai" | "groq" | "openrouter" | "anthropic" | "gemini" | "fish" | "elevenlabs" | "piper"
    provider: str


class PinBody(BaseModel):
    pin: str


class TestDonationBody(BaseModel):
    source: str  # "livepix" | "streamlabs"
    donor: str = "TestDonor"
    amount: float = 10.0
    currency: str = "BRL"
    message: str = "test donation — ignore"


class TestCaptionBody(BaseModel):
    text: str = "this is how the caption overlay looks on stream"
    clear_after: bool = True


def _build_app(
    state: DashboardState,
    initial: Optional[Orchestrator],
    *,
    pin: str = "",
) -> FastAPI:
    state.orchestrator = initial
    app = FastAPI(title="Wallie Dashboard")

    def _wire_memory_feed() -> None:
        """Hook the live memory feed onto the CURRENT orchestrator's capture
        pipeline (re-run after every /api/start rebuild)."""
        orch = state.orchestrator
        cap = getattr(orch, "_memory_capture", None) if orch else None
        if cap is not None:
            cap.on_captured = state.emit_memory_event

    _wire_memory_feed()

    def _caption_hub():
        """The caption bridge lives on the running orchestrator (built in
        wallie.build_orchestrator when captions.enabled). None = overlay off."""
        orch = state.orchestrator
        return getattr(orch, "_captions", None) if orch else None

    _pin = pin
    _sessions: set[str] = set()

    def _is_authed(cookies: dict[str, str]) -> bool:
        s = cookies.get("wallie_session")
        return bool(s and s in _sessions)

    if _pin:
        @app.middleware("http")
        async def _pin_gate(request: Request, call_next: Any) -> Response:
            path = request.url.path
            if path in _PUBLIC_PATHS or path == "/static/style.css":
                return await call_next(request)
            if _is_authed(request.cookies):
                return await call_next(request)
            if path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse("/login", status_code=302)

    @app.on_event("startup")
    async def _on_start() -> None:
        state.attach_logger()
        asyncio.create_task(state.broadcaster(), name="dash-broadcast")
        try:
            from wallie import attach_donations_to_app
            attach_donations_to_app(app, state.orchestrator) if state.orchestrator else None
        except Exception as e:
            logger.warning(f"dashboard: donation webhook mount deferred: {e}")

    # ---------- auth ----------
    @app.get("/login")
    def login_page() -> FileResponse:
        if not _pin:
            return RedirectResponse("/", status_code=302)  # type: ignore[return-value]
        return FileResponse(str(STATIC_DIR / "login.html"))

    @app.post("/api/auth/login")
    def api_auth_login(body: PinBody) -> Any:
        if not _pin:
            return {"ok": True}
        if not secrets.compare_digest(body.pin, _pin):
            raise HTTPException(403, "wrong pin")
        token = secrets.token_urlsafe(32)
        _sessions.add(token)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            "wallie_session", token,
            httponly=True, samesite="lax", max_age=86400,
        )
        return response

    # ---------- profiles ----------
    @app.get("/api/profiles")
    def api_profiles() -> dict[str, Any]:
        cfg = load_profile()
        return {"active": cfg.profile_name, "profiles": list_profiles() or [cfg.profile_name]}

    @app.post("/api/profiles")
    def api_profiles_create(body: ProfileCreateBody) -> dict[str, Any]:
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "name required")
        if body.clone_from:
            clone_profile(body.clone_from, name)
        else:
            cfg = AppConfig(profile_name=name)
            save_profile(cfg, name)
        activate_profile(name)
        return {"ok": True, "active": name}

    @app.put("/api/profiles/{name}/activate")
    def api_profiles_activate(name: str) -> dict[str, Any]:
        cfg = activate_profile(name)
        return {"ok": True, "active": cfg.profile_name}

    @app.delete("/api/profiles/{name}")
    def api_profiles_delete(name: str) -> dict[str, Any]:
        ok = delete_profile(name)
        return {"ok": ok}

    # ---------- config ----------
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return load_profile().model_dump()

    @app.put("/api/config")
    async def put_config(payload: dict[str, Any]) -> dict[str, Any]:
        # Merge onto the existing profile so any config section the UI doesn't send
        # (e.g. hearing) is preserved instead of being silently reset to defaults.
        existing = load_profile().model_dump()
        existing.update(payload)
        cfg = AppConfig(**existing)
        save_profile(cfg, cfg.profile_name)
        return {"ok": True}

    # ---------- orchestrator lifecycle ----------
    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        orch = state.orchestrator
        snap: dict[str, Any] = orch.status() if orch else {"running": False}
        snap["profile"] = load_profile().profile_name
        return snap

    @app.get("/api/audio-devices")
    def audio_devices() -> list[dict[str, Any]]:
        return list_output_devices()

    @app.get("/api/preflight")
    async def api_preflight() -> list[dict[str, str]]:
        """Pre-start checklist: static config problems the session would hit.
        No side effects, no network calls — the Start button runs this first."""
        from wallie import preflight
        try:
            return preflight()
        except Exception as e:
            logger.exception("dashboard: preflight failed")
            raise HTTPException(status_code=500, detail=str(e) or e.__class__.__name__)

    @app.post("/api/start")
    async def api_start() -> dict[str, Any]:
        from wallie import attach_donations_to_app, build_orchestrator
        if state.orchestrator and state.orchestrator.status().get("running"):
            return {"ok": True, "already": True}
        try:
            state.orchestrator = build_orchestrator()
        except Exception as e:
            logger.exception("dashboard: build_orchestrator failed")
            raise HTTPException(status_code=500, detail=str(e) or e.__class__.__name__)
        _wire_memory_feed()
        try:
            attach_donations_to_app(app, state.orchestrator)
        except Exception as e:
            logger.warning(f"dashboard: LivePix webhook mount failed: {e}")
        try:
            await state.orchestrator.start()
        except Exception as e:
            logger.exception("dashboard: orchestrator.start failed")
            state.orchestrator = None
            raise HTTPException(status_code=500, detail=str(e) or e.__class__.__name__)
        return {"ok": True}

    @app.post("/api/stop")
    async def api_stop() -> dict[str, Any]:
        if state.orchestrator:
            await state.orchestrator.stop()
        return {"ok": True}

    @app.post("/api/test/streamlabs-connect")
    async def test_streamlabs_connect() -> dict[str, Any]:
        """Verify Streamlabs credentials reach a live socket (no donation needed)."""
        from config import Secrets as _S
        from donations.streamlabs import StreamlabsMonitor

        s = _S()
        cfg = load_profile().donations
        monitor = StreamlabsMonitor(
            cfg=cfg,
            on_donations=lambda evs: None,
            access_token=s.streamlabs_access_token,
            socket_token=s.streamlabs_socket_token,
        )
        try:
            token = await asyncio.wait_for(monitor._resolve_socket_token(), timeout=10.0)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout fetching socket token"}
        except Exception as e:
            return {"ok": False, "error": _scrub_error(str(e))}
        return {"ok": bool(token), "token_preview": (token[:6] + "…") if token else ""}

    @app.post("/api/break")
    async def api_break() -> dict[str, Any]:
        orch = state.orchestrator
        if not orch or not orch.status().get("running"):
            raise HTTPException(400, "Orchestrator not running")
        orch.trigger_break()
        return {"ok": True}

    @app.post("/api/resume")
    async def api_resume() -> dict[str, Any]:
        orch = state.orchestrator
        if not orch or not orch.status().get("running"):
            raise HTTPException(400, "Orchestrator not running")
        orch.resume_from_break()
        return {"ok": True}

    # ---------- Minecraft Play mode ----------
    _REPO = Path(__file__).resolve().parent.parent

    @app.post("/api/play/install")
    async def api_play_install() -> dict[str, Any]:
        import asyncio as _aio
        proc = await _aio.create_subprocess_exec(
            sys.executable, str(_REPO / "scripts" / "install_minecraft.py"),
            cwd=str(_REPO), stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT,
        )
        try:
            out, _ = await _aio.wait_for(proc.communicate(), timeout=300)
        except _aio.TimeoutError:
            proc.kill()
            raise HTTPException(504, "Install timed out — check your connection and try again")
        log = (out or b"").decode("utf-8", "ignore")
        return {"ok": proc.returncode == 0, "log": log[-4000:]}

    @app.post("/api/play/launch")
    async def api_play_launch() -> dict[str, Any]:
        import asyncio as _aio
        cfg = load_profile()
        if not cfg.play.enabled:
            raise HTTPException(400, "Play mode is OFF — enable it in the Play section first")
        if state.orchestrator and state.orchestrator.status().get("running"):
            await state.orchestrator.stop()        # the live runner starts its own orchestrator
            state.orchestrator = None
        await _aio.create_subprocess_exec(
            sys.executable, str(_REPO / "scripts" / "run_wallie_live.py"),
            cfg.play.goal, cfg.profile_name, cwd=str(_REPO),
        )
        return {"ok": True, "msg": "Wallie Play launching — focus the Minecraft window. F8 stops it."}

    # ---------- memory endpoints ----------
    @app.get("/api/memory")
    def api_memory_get() -> dict[str, Any]:
        from config import PROFILES_DIR
        from core import MemoryStore
        cfg = load_profile()
        profile_name = cfg.profile_name or "default"
        store = MemoryStore(PROFILES_DIR / f"{profile_name}.memory.json")
        store.load()
        return {
            "notes": store.notes,
            "viewer_log": store.recent_viewers(100),
            "profile": profile_name,
        }

    @app.delete("/api/memory")
    def api_memory_clear() -> dict[str, Any]:
        from config import PROFILES_DIR
        from core import MemoryStore
        cfg = load_profile()
        profile_name = cfg.profile_name or "default"
        path = PROFILES_DIR / f"{profile_name}.memory.json"
        if path.exists():
            path.unlink()
        return {"ok": True, "cleared": profile_name}

    # ---------- voice-print speaker ID (owner vs others) ----------
    def _speaker_store():
        from config import PROFILES_DIR
        from hearing.speaker_id import SpeakerPrintStore
        cfg = load_profile()
        store = SpeakerPrintStore(
            PROFILES_DIR / f"{cfg.profile_name or 'default'}.speakers.json"
        )
        return store

    @app.get("/api/speakers")
    def api_speakers_get() -> dict[str, Any]:
        store = _speaker_store()
        orch = state.orchestrator
        return {
            "speakers": [
                {
                    "name": n,
                    "prints": len(store.speakers.get(n, {}).get("prints", [])),
                    "note": store.get_note(n),
                }
                for n in store.names()
            ],
            "clips": [
                {k: c.get(k) for k in ("id", "created_at", "suggestion")}
                for c in store.clips
            ],
            "enrolling": (
                orch._enrollment_buffer.pending
                if orch is not None and getattr(orch, "_enrolling", False)
                   and orch._enrollment_buffer is not None else 0
            ),
            "active": bool(orch is not None and getattr(orch, "speaker_id_active", False)),
        }

    @app.get("/api/speakers/clips/{clip_id}/audio")
    def api_speaker_clip_audio(clip_id: str) -> Response:
        """Play back a collected unknown-voice clip (on-device only)."""
        import base64 as _b64
        for c in _speaker_store().clips:
            if c.get("id") == clip_id:
                return Response(
                    content=_b64.b64decode(c["wav_b64"]),
                    media_type="audio/wav",
                )
        raise HTTPException(status_code=404, detail="clip not found")

    @app.post("/api/speakers/enroll/start")
    async def api_speaker_enroll_start() -> dict[str, Any]:
        orch = state.orchestrator
        if orch is None or not orch.status().get("running"):
            raise HTTPException(status_code=409, detail="start the session first — enrollment captures your live voice")
        if getattr(orch, "_enrollment_buffer", None) is None:
            raise HTTPException(status_code=409, detail="Speaker ID is disabled in Voice settings")
        orch._enrolling = True
        n = orch.start_voice_enrollment()
        return {"ok": True, "capturing": True, "pending": max(0, n)}

    @app.post("/api/speakers/enroll/finish")
    async def api_speaker_enroll_finish(body: dict[str, Any]) -> dict[str, Any]:
        orch = state.orchestrator
        name = str((body or {}).get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if orch is None or getattr(orch, "_enrollment_buffer", None) is None:
            raise HTTPException(status_code=409, detail="no enrollment in progress")
        replace = bool((body or {}).get("replace"))
        pending = orch._enrollment_buffer.pending
        ok = orch.finish_voice_enrollment(name, replace=replace)
        orch._enrolling = False
        if not ok:
            raise HTTPException(status_code=400, detail="no voice captured yet — talk for a few seconds first")
        return {"ok": True, "name": name, "utterances": pending}

    @app.post("/api/speakers/enroll/cancel")
    async def api_speaker_enroll_cancel() -> dict[str, Any]:
        orch = state.orchestrator
        if orch is not None and getattr(orch, "_enrollment_buffer", None) is not None:
            orch.cancel_voice_enrollment()
        if orch is not None:
            orch._enrolling = False
        return {"ok": True}

    @app.put("/api/speakers/{name}/note")
    def api_speaker_note(name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Per-voice prompt instruction (e.g. "this is my mom — treat her warmly").
        Injected into the hearing prompt whenever this voice is recognized."""
        note = str((body or {}).get("note") or "")
        store = _speaker_store()
        # Resolve case-insensitively so labels like "owner" hit "Owner".
        resolved = store.find_name(name) or name
        if not store.set_note(resolved, note):
            raise HTTPException(status_code=404, detail=f"unknown speaker: {name}")
        return {"ok": True, "name": resolved, "note": store.get_note(resolved)}

    @app.delete("/api/speakers/{name}")
    def api_speaker_delete(name: str) -> dict[str, Any]:
        if not _speaker_store().remove(name):
            raise HTTPException(status_code=404, detail=f"unknown speaker: {name}")
        return {"ok": True}

    @app.post("/api/speakers/clips/{clip_id}/enroll")
    def api_speaker_clip_enroll(clip_id: str, body: dict[str, Any]) -> dict[str, Any]:
        name = str((body or {}).get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        store = _speaker_store()
        if not store.enroll_clip(clip_id, name):
            raise HTTPException(status_code=404, detail="clip not found")
        return {"ok": True, "name": name}

    @app.delete("/api/speakers/clips/{clip_id}")
    def api_speaker_clip_delete(clip_id: str) -> dict[str, Any]:
        if not _speaker_store().drop_clip(clip_id):
            raise HTTPException(status_code=404, detail="clip not found")
        return {"ok": True}

    # ---------- long-term fact memory (durable, AI + manual) ----------
    def _ltm_store():
        from config import PROFILES_DIR
        from core.long_term_memory import LongTermMemory
        cfg = load_profile()
        store = LongTermMemory(
            PROFILES_DIR / f"{cfg.profile_name or 'default'}.longterm.json"
        )
        store.load()
        return store

    @app.get("/api/longterm")
    def api_longterm_get() -> dict[str, Any]:
        return _ltm_store().to_dashboard()

    @app.post("/api/longterm")
    def api_longterm_add(body: LongTermCreateBody) -> dict[str, Any]:
        from core.long_term_memory import MemoryError
        store = _ltm_store()
        try:
            entry = store.add(
                body.text,
                kind=body.kind if body.kind in ("long_term", "short_term") else "long_term",
                tag=body.tag,
                about=body.about.strip()[:40],
                ttl_sec=max(0.1, body.ttl_hours) * 3600.0,
                source="manual",
            )
            store.save()
        except MemoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _sync_ltm_to_orchestrator()
        return {"ok": True, "entry": entry, "stats": store.stats()}

    @app.put("/api/longterm/{entry_id}")
    def api_longterm_update(entry_id: int, body: LongTermUpdateBody) -> dict[str, Any]:
        from core.long_term_memory import MemoryError
        store = _ltm_store()
        try:
            store.update(
                entry_id,
                text=body.text,
                tag=body.tag,
                about=body.about,
                kind=body.kind if body.kind in ("long_term", "short_term") else None,
                ttl_sec=(body.ttl_hours * 3600.0) if body.ttl_hours else None,
            )
            store.save()
        except MemoryError as e:
            raise HTTPException(status_code=404 if "no memory" in str(e) else 400, detail=str(e))
        _sync_ltm_to_orchestrator()
        return {"ok": True, "entry": store.get(entry_id), "stats": store.stats()}

    @app.delete("/api/longterm/{entry_id}")
    def api_longterm_remove(entry_id: int) -> dict[str, Any]:
        store = _ltm_store()
        ok = store.remove(entry_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"no memory with id {entry_id}")
        store.save()
        _sync_ltm_to_orchestrator()
        return {"ok": True, "stats": store.stats()}

    @app.post("/api/longterm/clear")
    def api_longterm_clear(body: LongTermClearBody) -> dict[str, Any]:
        kind = body.kind
        store = _ltm_store()
        removed = store.clear(kind if kind in ("long_term", "short_term", None) else None)
        store.save()
        _sync_ltm_to_orchestrator()
        return {"ok": True, "removed": removed, "stats": store.stats()}

    @app.post("/api/memory/capture-now")
    async def api_memory_capture_now() -> dict[str, Any]:
        """Run the extractor on the recent stream context immediately (test)."""
        orch = state.orchestrator
        cap = getattr(orch, "_memory_capture", None) if orch else None
        if cap is None or not cap.enabled:
            raise HTTPException(status_code=409, detail="memory capture disabled or not running")
        facts = await cap.extract_now()
        return {"ok": True, "captured": facts}

    @app.post("/api/memory/consolidate-now")
    async def api_memory_consolidate_now() -> dict[str, Any]:
        """Force a consolidation pass now (test). Falls back to a standalone
        store when no session is running, so the button works any time."""
        orch = state.orchestrator
        cons = getattr(orch, "_memory_consolidator", None) if orch else None
        if cons is None or not cons.enabled:
            cfg = load_profile()
            from config import Secrets
            from core.long_term_memory import LongTermMemory
            from core.memory_capture import MemoryConsolidator
            store = LongTermMemory(
                PROFILES_DIR / f"{cfg.profile_name or 'default'}.longterm.json"
            )
            store.load()
            mcfg = cfg.memory
            cons = MemoryConsolidator(mcfg, store, None)
            # enabled requires an extractor LLM; reuse the capture extractor's
            # configuration by building a one-off provider.
            if mcfg.extractor == "openai_compatible":
                from llm.openai_compat import OpenAICompatProvider
                base_url = (mcfg.openai_compatible_base_url or "").strip()
                if not base_url:
                    raise HTTPException(
                        status_code=409,
                        detail="Memory extractor base URL is not configured",
                    )
                cons = MemoryConsolidator(
                    mcfg, store,
                    OpenAICompatProvider(
                        name="openai_compatible_memory",
                        model=mcfg.model or "gpt-4o-mini",
                        api_key=Secrets().openai_compatible_memory_api_key or "",
                        base_url=base_url,
                        supports_vision=False,
                        timeout=mcfg.timeout,
                    ),
                )
            elif mcfg.extractor == "main":
                cons = MemoryConsolidator(
                    mcfg, store, build_provider(cfg.llm, Secrets())
                )
            else:
                raise HTTPException(status_code=409, detail="memory capture disabled or extractor=off")
        try:
            result = await cons.consolidate_now()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"consolidation failed: {e}")
        return {
            "ok": True,
            "removed": result.get("removed", 0),
            "added": result.get("added", 0),
            "skipped": result.get("skipped", 0),
        }

    def _sync_ltm_to_orchestrator() -> None:
        """Keep the running orchestrator's store handle in sync with edits made
        while stopped OR running: rebuild it from disk so prompt + janitor see
        the same data the dashboard saved."""
        orch = state.orchestrator
        if orch is None or getattr(orch, "_ltm", None) is None:
            return
        try:
            path = getattr(orch._ltm, "_path", None)
            if path is not None:
                orch._ltm.load()
        except Exception as e:
            logger.warning(f"longterm: resync failed: {e}")

    # ---------- test endpoints ----------
    @app.post("/api/test/persona")
    async def test_persona(body: TestPersonaBody) -> dict[str, Any]:
        cfg = load_profile()
        persona = Persona.from_config(cfg.persona)
        llm = build_provider(cfg.llm, Secrets())
        system = persona.system_prompt(
            topic=body.topic or (cfg.topics.topics[0] if cfg.topics.topics else None),
            vision_enabled=cfg.vision.enabled,
            topic_drift_style=cfg.topics.drift_style,
        )
        if body.kind == "chat":
            user = persona.chat_turn(
                username=body.chat_user or "regular_viewer",
                platform="twitch",
                text=body.chat_text or "hey what's up today",
                is_highlight=False,
            )
        elif body.kind == "vision":
            user = persona.vision_turn()
        else:
            oc = cfg.orchestrator
            user = persona.monologue_turn(
                topic=body.topic,
                sentences_min=oc.segment_sentences_min,
                sentences_max=oc.segment_sentences_max,
                topic_drift_style=cfg.topics.drift_style,
            )
        try:
            out: list[str] = []
            async for token in llm.stream(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                max_tokens=min(cfg.llm.max_tokens, 200),
                presence_penalty=cfg.llm.presence_penalty,
                frequency_penalty=cfg.llm.frequency_penalty,
            ):
                out.append(token)
            return {"ok": True, "text": "".join(out).strip(), "system_preview": system}
        finally:
            await llm.aclose()

    # ---------- avatar ----------
    def _live_avatar():
        orch = state.orchestrator
        avatar = getattr(orch, "_avatar", None) if orch else None
        if avatar is None:
            raise HTTPException(400, "Avatar not enabled. Start the orchestrator with avatar.enabled=true.")
        return avatar

    @app.get("/api/avatar/status")
    async def avatar_status() -> dict[str, Any]:
        orch = state.orchestrator
        avatar = getattr(orch, "_avatar", None) if orch else None
        if avatar is None:
            return {"enabled": False, "connected": False}
        return avatar.status()

    @app.get("/api/avatar/hotkeys")
    async def avatar_hotkeys() -> dict[str, Any]:
        avatar = _live_avatar()
        hotkeys = await avatar.query_hotkeys()
        return {"hotkeys": hotkeys}

    @app.get("/api/avatar/model")
    async def avatar_model() -> dict[str, Any]:
        avatar = _live_avatar()
        info = await avatar.query_model_info()
        return {"model": info}

    @app.post("/api/test/expression")
    async def test_expression(body: TestExpressionBody) -> dict[str, Any]:
        avatar = _live_avatar()
        expression = body.expression.strip()
        if not expression:
            raise HTTPException(400, "expression required")
        # Try the slot path first ("happy", "hype" etc.). If nothing matches,
        # fall back to a raw hotkey id/name.
        slot_attr = f"expr_{expression}"
        cfg = load_profile().avatar
        if hasattr(cfg, slot_attr) and getattr(cfg, slot_attr, ""):
            await avatar.trigger_emotion(expression)
        else:
            await avatar.trigger_expression(expression)
        return {"ok": True, "expression": expression}

    @app.post("/api/test/avatar_look")
    async def test_avatar_look(body: TestLookBody) -> dict[str, Any]:
        avatar = _live_avatar()
        await avatar.look_at(body.x, body.y, hold_sec=body.hold_sec)
        return {"ok": True}

    # ---------- secrets (API keys) ----------
    @app.get("/api/secrets")
    def api_secrets_list() -> dict[str, Any]:
        from secrets_store import list_secrets, set_provider_context
        # Dynamic provider blocks register their metadata so the keys page can
        # show each block's name/category next to its PROVIDER_*_API_KEY entry.
        set_provider_context(load_profile().providers)
        return {"secrets": list_secrets()}

    @app.put("/api/secrets")
    def api_secrets_update(body: SecretUpdateBody) -> dict[str, Any]:
        from secrets_store import SECRET_FIELDS, set_secret
        if body.env not in SECRET_FIELDS:
            raise HTTPException(400, "unknown secret field")
        try:
            set_secret(body.env, body.value)
        except Exception as e:
            raise HTTPException(500, f"failed to write secret: {e}")
        return {"ok": True, "env": body.env, "is_set": bool(body.value.strip())}

    @app.post("/api/secrets/bulk")
    def api_secrets_bulk(body: SecretBulkBody) -> dict[str, Any]:
        from secrets_store import update_many
        accepted = update_many(body.values)
        return {"ok": True, "updated": accepted}

    # ---------- dynamic provider blocks (API Keys page) ----------
    def _sync_provider_context() -> None:
        from secrets_store import set_provider_context
        set_provider_context(load_profile().providers)

    def _save_providers(blocks: list[ProviderBlock]) -> list[dict[str, Any]]:
        cfg = load_profile()
        try:
            cfg = cfg.model_copy(update={"providers": blocks})
            save_profile(cfg, cfg.profile_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid provider block: {e}")
        _sync_provider_context()
        return [p.model_dump() for p in blocks]

    @app.get("/api/providers")
    def api_providers_list() -> dict[str, Any]:
        cfg = load_profile()
        return {
            "providers": [p.model_dump() for p in cfg.providers],
            "categories": ["llm", "vision", "tts", "stt", "memory", "thoughts"],
        }

    @app.put("/api/providers")
    def api_providers_replace(body: list[dict[str, Any]]) -> dict[str, Any]:
        """Full-list replace: the page edits blocks locally and saves them all
        at once. New blocks get an id; existing ids are preserved."""
        try:
            blocks = [ProviderBlock(**b) for b in body]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid provider block: {e}")
        existing_ids = {p.id for p in load_profile().providers}
        seen: set[str] = set()
        for b in blocks:
            if not b.id or b.id not in existing_ids or b.id in seen:
                # Derive the slug from id, then name, so new blocks get
                # meaningful env names (PROVIDER_MAIN_LLM_API_KEY…).
                base = _provider_env_slug(b.id or b.name or "provider")
                cand, n = base, 2
                while cand in seen:
                    cand = f"{base}_{n}"
                    n += 1
                b.id = cand
            seen.add(b.id)
        saved = _save_providers(blocks)
        return {"ok": True, "providers": saved}

    @app.delete("/api/providers/{provider_id}")
    def api_providers_delete(provider_id: str) -> dict[str, Any]:
        cfg = load_profile()
        blocks = [p for p in cfg.providers if p.id != provider_id]
        if len(blocks) == len(cfg.providers):
            raise HTTPException(status_code=404, detail=f"unknown provider: {provider_id}")
        _save_providers(blocks)
        return {"ok": True}

    @app.post("/api/providers/{provider_id}/test")
    async def api_providers_test(provider_id: str) -> dict[str, Any]:
        """Live probe of a provider block. Chat categories get a tiny
        chat/completions call; tts/stt get a reachability check (models list)."""
        cfg = load_profile()
        block = next((p for p in cfg.providers if p.id == provider_id), None)
        if block is None:
            raise HTTPException(status_code=404, detail=f"unknown provider: {provider_id}")
        base_url = (block.base_url or "").strip()
        if not base_url:
            return {"ok": False, "error": "no endpoint URL set on this block"}
        from secrets_store import mask
        import httpx
        key_env = "PROVIDER_" + _provider_env_slug(provider_id) + "_API_KEY"
        api_key = os.getenv(key_env, "")
        host = base_url.split("://", 1)[-1].split("/", 1)[0].lower()
        local = host.startswith(("localhost", "127.0.0.1", "[::1]", "0.0.0.0", "::1"))
        if not api_key and not local:
            return {"ok": False, "error": "no API key set for this block (add it in the key field above)"}
        headers = {"Authorization": f"Bearer {api_key or 'local-no-key'}"}
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                if block.category in ("tts", "stt"):
                    r = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
                    if r.status_code < 400:
                        return {"ok": True, "note": f"endpoint reachable ({r.status_code})"}
                    return {"ok": False, "error": f"HTTP {r.status_code} from {base_url}"}
                r = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": block.model or "gpt-4o-mini",
                        "max_tokens": 4,
                        "messages": [{"role": "user", "content": "Say ok."}],
                    },
                )
                if r.status_code >= 400:
                    detail = r.text[:160]
                    return {"ok": False, "error": f"HTTP {r.status_code}: {detail}"}
                data = r.json()
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                return {"ok": True, "preview": (text or "").strip()[:40]}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    @app.post("/api/secrets/test")
    async def api_secrets_test(body: TestProviderBody) -> dict[str, Any]:
        from config import LLMConfig, Secrets, TTSConfig
        prov = body.provider
        try:
            if prov in ("openai", "groq", "openrouter", "anthropic", "gemini", "ollama"):
                from llm import build_provider
                cfg = LLMConfig(provider=prov, model=_default_model_for(prov), max_tokens=4)
                client = build_provider(cfg, Secrets())
                try:
                    out: list[str] = []
                    async def _drain():
                        async for tok in client.stream(
                            [
                                {"role": "system", "content": "Say only the word ok."},
                                {"role": "user", "content": "ok"},
                            ],
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=4,
                        ):
                            out.append(tok)
                    await asyncio.wait_for(_drain(), timeout=10.0)
                    return {"ok": True, "preview": "".join(out).strip()[:30]}
                finally:
                    await client.aclose()
            if prov in ("fish", "elevenlabs", "piper"):
                from tts import build_tts
                cfg = TTSConfig(provider=prov)
                if prov == "piper":
                    # Piper needs a model file path; can't test without it.
                    return {"ok": False, "error": "Piper test requires piper_model_path in config; not auto-testable here."}
                client = build_tts(cfg, Secrets())
                try:
                    received = 0
                    async def _drain():
                        nonlocal received
                        async for chunk in client.synthesize("test"):
                            received += len(chunk)
                            if received > 1024:
                                return
                    await asyncio.wait_for(_drain(), timeout=10.0)
                    if received == 0:
                        return {"ok": False, "error": "no audio bytes returned"}
                    return {"ok": True, "preview": f"{received} bytes received"}
                finally:
                    await client.aclose()
            return {"ok": False, "error": f"unknown provider: {prov}"}
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout (>10s) — server unreachable or key wrong"}
        except Exception as e:
            return {"ok": False, "error": _scrub_error(str(e))}

    @app.post("/api/audio/reset")
    async def audio_reset() -> dict[str, Any]:
        orch = state.orchestrator
        if orch is None:
            raise HTTPException(400, "Orchestrator not running")
        player = getattr(orch, "_player", None)
        if player is None:
            raise HTTPException(500, "no audio player")
        player.reset()
        return {"ok": True}

    @app.post("/api/test/vision")
    async def test_vision() -> dict[str, Any]:
        cfg = load_profile()
        if not cfg.llm.vision_capable:
            raise HTTPException(
                400,
                "Engine.vision_capable is OFF. Toggle it on for a vision-capable model.",
            )
        # Capture one frame.
        try:
            from vision import ScreenCapture
        except ModuleNotFoundError as e:
            raise HTTPException(500, f"vision deps missing: {e}")
        cap = ScreenCapture(
            monitor_index=cfg.vision.monitor_index,
            max_edge_px=cfg.vision.max_edge_px,
        )
        try:
            frame = await asyncio.to_thread(cap.grab)
        finally:
            cap.close()

        persona = Persona.from_config(cfg.persona)
        system = persona.system_prompt(
            topic=None, vision_enabled=True, session_notes=None,
            topic_drift_style=cfg.topics.drift_style,
        )
        # Force the screen-anchored prompt.
        oc = cfg.orchestrator
        user = persona.vision_turn(
        change_type="scene",
        mood_label="warm",
        target_sentences=1,   
        screen_activity="",
    )
        msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image", "data": frame.jpeg, "mime": "image/jpeg"},
                ],
            },
        ]
        llm = build_provider(cfg.llm, Secrets())
        try:
            tokens: list[str] = []
            async for tok in llm.stream(
                msgs,
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                max_tokens=min(cfg.llm.max_tokens, 80),
                presence_penalty=cfg.llm.presence_penalty,
                frequency_penalty=cfg.llm.frequency_penalty,
            ):
                tokens.append(tok)
        except Exception as e:
            await llm.aclose()
            raise HTTPException(500, f"LLM error: {e}")
        finally:
            await llm.aclose()
        text = "".join(tokens).strip()
        return {
            "ok": True,
            "frame_size": [frame.width, frame.height],
            "frame_bytes": len(frame.jpeg),
            "model": cfg.llm.model,
            "provider": cfg.llm.provider,
            "text": text,
        }

    # ---------- donations (test endpoints) ----------
    @app.post("/api/test/donation")
    async def test_donation(body: TestDonationBody) -> dict[str, Any]:
        """Inject a fake donation through the SAME queue path as real ones —
        no external API call, no money moved. Requires the orchestrator running."""
        from wallie import _queue_single_donation
        from donations.base import DonationEvent

        orch = state.orchestrator
        if not orch or not orch.status().get("running"):
            raise HTTPException(400, "Orchestrator not running")
        source = (body.source or "").strip().lower()
        if source not in ("livepix", "streamlabs"):
            raise HTTPException(400, "source must be 'livepix' or 'streamlabs'")
        import time as _time
        ev = DonationEvent(
            source=source,
            event_id=f"test-{source}-{_time.time()}",
            donor_name=body.donor or "TestDonor",
            amount=float(body.amount),
            currency=body.currency or "BRL",
            message=body.message or "",
            metadata={"test": True},
        )
        await _queue_single_donation(orch)(ev)
        return {"ok": True, "queued": source, "event_id": ev.event_id}

    @app.post("/api/donations/webhook-test")
    async def donations_webhook_test() -> JSONResponse:
        """Emulates a real LivePix webhook POST (shape per docs.livepix.gg) so the
        full validation → normalize → queue path can be exercised without payment."""
        import time as _time
        orch = state.orchestrator
        if not orch or not orch.status().get("running"):
            raise HTTPException(400, "Orchestrator not running")
        secrets = Secrets()
        payload = {
            "userId": secrets.livepix_user_id or "61021c7bdabe5e001225b65b",
            "clientId": secrets.livepix_client_id or "test-client",
            "event": "new",
            "resource": {
                "id": f"test{int(_time.time())}",
                "reference": "webhook-test",
                "type": "message",
            },
        }
        import httpx
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                getattr(orch, "_livepix_webhook_path", "/webhooks/livepix"), json=payload
            )
        return JSONResponse({"status": resp.status_code, "body": resp.json()})

    @app.post("/api/test/voice")
    async def test_voice(body: TestVoiceBody) -> dict[str, Any]:
        cfg = load_profile()
        orch = state.orchestrator
        if orch and orch.status().get("running"):
            player = orch._player  # noqa: SLF001
            tts = build_tts(cfg.tts, Secrets())
            try:
                async for pcm in tts.synthesize(body.text):
                    await player.write(pcm)
            finally:
                await tts.aclose()
            return {"ok": True, "routed": "live-player"}
        from audio import AudioPlayer
        tts = build_tts(cfg.tts, Secrets())
        player = AudioPlayer(sample_rate=tts.sample_rate, channels=tts.channels,
                             device=(cfg.tts.output_device or None))
        player.start()
        try:
            async for pcm in tts.synthesize(body.text):
                await player.write(pcm)
            await asyncio.sleep(0.3)
            await player.wait_drained()
        finally:
            player.close()
            await tts.aclose()
        return {"ok": True, "routed": "preview-player"}

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        if _pin and not _is_authed(ws.cookies):
            await ws.accept()
            await ws.close(code=4001, reason="unauthorized")
            return
        await ws.accept()
        state.clients.add(ws)
        try:
            snap = state.orchestrator.status() if state.orchestrator else {"running": False}
            await ws.send_text(json.dumps({"type": "status", "data": snap}))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            state.clients.discard(ws)

    # ---------- captions (browser-source overlay) ----------
    def _captions_page(path: str) -> FileResponse:
        cfg = load_profile()
        ccfg = getattr(cfg, "captions", None)
        if ccfg is None or not ccfg.enabled:
            return FileResponse(str(STATIC_DIR / "captions-off.html"))
        if not _caption_hub():
            return FileResponse(str(STATIC_DIR / "captions-off.html"))
        return FileResponse(str(STATIC_DIR / "captions.html"))

    @app.get("/captions")
    def captions_page() -> FileResponse:
        return _captions_page(getattr(load_profile().captions, "path", "/captions") or "/captions")

    @app.get("/captions/events")
    async def captions_events() -> StreamingResponse:
        """SSE stream of caption events for the overlay page (OBS browser source)."""
        hub = _caption_hub()
        if hub is None:
            async def _closed():
                yield "data: {\"event\": \"closed\"}\n\n"
            return StreamingResponse(_closed(), media_type="text/event-stream", status_code=503)

        async def _gen():
            async for frame in hub.subscribe():
                yield frame

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # don't buffer behind reverse proxies
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/captions/status")
    def captions_status() -> dict[str, Any]:
        hub = _caption_hub()
        cfg = load_profile().captions
        base = {
            "enabled": bool(cfg.enabled),
            "path": cfg.path,
            "url": None,
            "clients": 0,
            "text": "",
        }
        if hub is not None:
            st = hub.status()
            base.update(st)
            host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
            port = os.getenv("DASHBOARD_PORT", "8765")
            shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
            base["url"] = f"http://{shown}:{port}{cfg.path}"
        return base

    @app.post("/api/test/caption")
    async def test_caption(body: TestCaptionBody) -> dict[str, Any]:
        """Push a caption line through the real hub — verifies the SSE bridge,
        including the auto-clear, without running the full pipeline."""
        hub = _caption_hub()
        if hub is None:
            raise HTTPException(400, "Captions not enabled — toggle it on and start the orchestrator")
        hub.note_sentence(body.text, final=True)
        if body.clear_after:
            delay = max(0.0, float(load_profile().captions.clear_delay_sec))
            await asyncio.sleep(delay + 2.5)   # let the browser show it, then clear
            hub.clear()
        return {"ok": True}

    # ---------- static UI ----------
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


async def serve(orchestrator: Optional[Orchestrator]) -> None:
    state = DashboardState()

    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8765"))
    pin = os.getenv("DASHBOARD_PIN", "").strip()
    is_remote = host not in ("127.0.0.1", "localhost", "::1")

    if pin:
        logger.info("dashboard: PIN auth active (from DASHBOARD_PIN)")
    elif is_remote:
        pin = str(random.randint(1000, 9999))
        logger.info(f"dashboard: auto-generated PIN: {pin}")
        logger.info("dashboard: set DASHBOARD_PIN in .env for a permanent PIN")
    else:
        pin = ""

    if is_remote and not pin:
        logger.warning(
            f"dashboard: bound to {host}:{port} WITHOUT PIN protection. "
            "Anyone on your network can access the dashboard."
        )

    app = _build_app(state, orchestrator, pin=pin)

    if is_remote:
        local_ip = _get_local_ip()
        logger.info(f"dashboard: access from your phone/tablet: http://{local_ip}:{port}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(config)
    logger.info(f"dashboard: http://{host}:{port}")
    await server.serve()

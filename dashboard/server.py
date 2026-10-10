"""FastAPI dashboard — config, lifecycle, and test endpoints."""
from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import socket
import sys
import time
from dataclasses import dataclass
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
from starlette.types import Scope

from audio.player import check_output_device, list_output_devices, play_test_beep
from hearing.capture import list_loopback_devices, resolve_loopback_device
from config import (
    AppConfig,
    ProviderBlock,
    clear_provider_refs,
    Secrets,
    activate_profile,
    clone_profile,
    delete_profile,
    list_profiles,
    load_profile,
    heal_provider_refs,
    provider_key_env,
    save_profile,
)
from core import Orchestrator, Persona
from llm import build_provider
from tts import build_tts

STATIC_DIR = Path(__file__).parent / "static"


class _NoCacheStatic(StaticFiles):
    """Static assets, revalidated on every load.

    ``/static`` URLs are not versioned, and the browser's heuristic freshness
    (10% of the file's age) is enough to keep serving a stylesheet from cache
    for minutes after it changed — you edit the UI, reload, and the browser
    shows the old design. ``no-cache`` means "revalidate", not "don't store":
    the ETag makes each check a cheap 304, so we get fresh assets for free.
    """

    def file_response(
        self,
        full_path: os.PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,   # noqa: ARG002 — signature fixed by StaticFiles
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


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
    "gemini": "gemini-3.8-flash",
    "ollama": "llama3.2",
}


def _default_model_for(provider: str) -> str:
    return _DEFAULT_TEST_MODELS.get(provider, "")


def _provider_env_slug(pid: str) -> str:
    """Deprecated shim — canonical slug lives in config.provider_slug."""
    from config import provider_slug
    return provider_slug(pid)


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


_PIPER_PCT_RE = _re.compile(r"\((\d{1,3})%\)")


def _last_percent(text: str) -> int | None:
    """Last ``(NN%)`` a Piper helper printed, for the download progress bar.
    Returns None when the chunk has no percentage yet."""
    matches = _PIPER_PCT_RE.findall(text)
    if not matches:
        return None
    try:
        return max(0, min(100, int(matches[-1])))
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None


class DashboardState:
    def __init__(self) -> None:
        self.orchestrator: Optional[Orchestrator] = None
        self.clients: set[WebSocket] = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        # Kokoro one-click install job (Voice-tab badge). Read by
        # GET /api/tts/kokoro/status; updated by the background installer task.
        self.kokoro_install: dict[str, Any] = {
            "running": False,
            "lang": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "log": "",
        }
        # Piper one-click install / voice-download job (Voice-tab badge). Read by
        # GET /api/tts/piper/status; updated by the background task.
        self.piper_job: dict[str, Any] = {
            "running": False,
            "kind": "",
            "voice": "",
            "progress": 0,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "log": "",
        }

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


class TestAudioOutputBody(BaseModel):
    device: str = ""  # output-device name ("" = system default)


class DeviceCheckBody(BaseModel):
    tts_output: str = ""       # cfg.tts.output_device ("" = system default)
    loopback: str = ""         # cfg.hearing.loopback_device ("" = system default)


class TestHearingBody(BaseModel):
    seconds: float = 5.0
    source: str = "system"  # "system" = WASAPI loopback, "mic" = default microphone
    device: str = ""        # loopback override ("" = cfg.hearing.loopback_device or system default)


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


class TtsVoicesBody(BaseModel):
    provider_ref: str = ""


class VisionModelsBody(BaseModel):
    provider_ref: str = ""


class TestVisionBody(BaseModel):
    """Optional overrides for POST /api/test/vision.

    Blank = behave exactly like a live session (resolved dedicated vision block,
    else the main engine). Filling any field builds a THROWAWAY provider so a
    candidate model can be smoked-tested before it is saved into the profile —
    the fix for a retired model (e.g. gemini-2.5-flash returning 404)."""

    provider: str = ""      # "" = configured engine provider
    model: str = ""         # "" = configured model (vision block, else engine)
    base_url: str = ""      # "" = configured endpoint (openai_compatible / ollama)
    provider_ref: str = ""  # "" = configured vision block

    def has_override(self) -> bool:
        return bool(
            (self.provider or self.model or self.base_url).strip()
        )


class VoiceSampleBody(BaseModel):
    filename: str = ""
    data: str = ""   # base64 (a data: URL prefix is tolerated)


class VoicePresetBody(BaseModel):
    name: str
    provider: str = ""
    voice_id: str = ""
    notes: str = ""
    tts: dict[str, Any] = {}
    source: str = "manual"


class VoiceCloneBody(BaseModel):
    provider: str                  # "elevenlabs" | "fish"
    name: str
    description: str = ""
    samples: list[VoiceSampleBody] = []
    save_preset: bool = True


class VoiceRecordBody(BaseModel):
    seconds: float = 8.0


class VoicePreviewBody(BaseModel):
    """One line, one voice, audio back to the caller (no playback).

    ``name`` resolves a SAVED voice from the profile's library; when it is blank
    the inline provider/voice/tts fields win; when those are blank too the saved
    config is used."""

    text: str
    name: str = ""
    provider: str = ""
    voice_id: str = ""
    tts: dict[str, Any] = {}


class KokoroInstallBody(BaseModel):
    lang: str = "a"     # Kokoro lang_code — picks which language extra (misaki) to pull in
    voice: str = ""     # optional voice to pre-download (defaults to the language's first)


class PiperDownloadBody(BaseModel):
    voice: str          # e.g. en_US-amy-medium


class LocalVoiceBody(BaseModel):
    """One saved voice for a single local engine (Piper / Kokoro)."""

    name: str
    voice_id: str = ""
    notes: str = ""
    tts: dict[str, Any] = {}


class VoiceExportBody(BaseModel):
    """Copy saved voices from the active profile into another profile."""

    target_profile: str
    names: list[str] = []      # empty = the whole saved-voice library
    # Replacing a same-named voice with DIFFERENT settings needs this flag; the
    # server answers 409 first so the dashboard can ask.
    overwrite: bool = False


class VoiceImportBody(BaseModel):
    """A saved-voice backup file, posted as text.

    The dashboard reads the ``.json`` in the browser and sends its contents, so
    the server stays the single owner of the format: parsing, validation and the
    TTS-knob sanitizing all happen in ``tts.voice_lab``.
    """

    content: str = ""
    overwrite: bool = False    # see VoiceExportBody.overwrite


class VoicePullBody(BaseModel):
    """Bring saved voices FROM another profile's library (the inverse of an
    export, which pushes them into another profile)."""

    source_profile: str
    names: list[str] = []      # empty = every voice that profile has
    overwrite: bool = False    # see VoiceExportBody.overwrite


class VoiceMirrorBody(BaseModel):
    """Two-way sync: make this profile and another one hold the same voices."""

    other_profile: str
    # skip | this | other — what to do with same-named voices that differ.
    conflicts: str = "skip"


class VoicePartnerBody(BaseModel):
    """Record (or clear) this profile's voice partner — the profile its
    saved-voice library is meant to stay in sync with. An empty string clears
    it. This only stores the choice; nothing is copied by setting it."""

    partner: str = ""


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


def _record_mic(seconds: float) -> Any:
    """Record `seconds` of mono float32 PCM at 16 kHz from the DEFAULT MICROPHONE
    (soundcard, same stack as SystemAudioCapture). Blocking — callers run it in
    an executor. COM is initialized on the calling thread like the capture loop
    does (soundcard's WASAPI backend requires it on non-main threads)."""
    try:
        import soundcard as sc
    except Exception as e:
        raise RuntimeError(f"soundcard not available: {e}")
    import numpy as np

    # COM FIRST — before any soundcard call. The microphone lookup is itself a
    # COM operation (MMDevice enumeration), so initializing after it still
    # dies with 0x800401f0 (CO_E_NOTINITIALIZED) on worker threads. Same rule
    # the capture loop follows (hearing/capture.py).
    _co_done = False
    try:
        from ctypes import windll
        hr = windll.ole32.CoInitialize(None)
        _co_done = hr in (0, 1)  # S_OK / S_FALSE both need a matching CoUninitialize
    except Exception:  # noqa: BLE001 — non-Windows or no COM
        _co_done = False
    try:
        mic = sc.default_microphone()
        if mic is None:
            raise RuntimeError("no default microphone found")
        sr = 16000
        chunk = max(1, int(sr * 0.25))
        frames: list[Any] = []
        with mic.recorder(samplerate=sr, channels=1, blocksize=chunk) as rec:
            remaining = seconds
            while remaining > 0:
                data = rec.record(numframes=chunk)
                if data is not None and data.size:
                    frames.append(data.flatten().astype("float32"))
                    remaining -= data.shape[0] / sr
    finally:
        if _co_done:
            try:
                from ctypes import windll
                windll.ole32.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass
    return np.concatenate(frames) if frames else np.zeros(0, dtype="float32")


@dataclass
class _SynthResult:
    """One TTS synthesis plus how it was obtained — raw PCM16 on success, an
    actionable message on failure. Exactly one of ``pcm``/``error`` is set."""

    pcm: bytes = b""
    decoded: bool = False        # the payload needed the ffmpeg fallback
    sample_rate: int = 24000
    channels: int = 1
    error: str = ""


async def _synthesize_pcm(tts_cfg: Any, tts_secrets: Any, text: str) -> _SynthResult:
    """Build the TTS provider, synthesize ``text`` and normalize the answer to
    raw PCM16 (including the gateway-answered-MP3 fallback).

    Module level on purpose: the Voice page's ▶ test and the Voice Lab A/B share
    exactly one decode path, and tests can patch ``build_tts`` /
    ``tts.decode.find_ffmpeg`` and exercise either route through it.
    """
    from tts.base import TTSError
    # Imported HERE, not at module scope: tests patch tts.decode.find_ffmpeg and
    # the patch has to be visible from this call site.
    from tts.decode import find_ffmpeg, sniff_compressed

    try:
        tts = build_tts(tts_cfg, tts_secrets)
    except TTSError as e:
        return _SynthResult(error=str(e))
    res = _SynthResult(
        sample_rate=int(getattr(tts, "sample_rate", 24000) or 24000),
        channels=int(getattr(tts, "channels", 1) or 1),
    )
    chunks: list[bytes] = []
    try:
        async for pcm in tts.synthesize(text):
            chunks.append(pcm)
    except TTSError as e:
        res.error = str(e)
        return res
    except Exception as e:
        logger.exception("dashboard: tts synthesis failed")
        res.error = f"synthesis failed: {e}"
        return res
    finally:
        await tts.aclose()

    raw = b"".join(chunks)
    if not raw:
        res.error = (
            "endpoint returned no audio — check the model/voice names and "
            "that the key has TTS access"
        )
        return res

    label = sniff_compressed(raw[:12])
    if label == "json-or-xml":
        # A gateway that hid its error inside an HTTP 200. Decoding it with ffmpeg
        # would only produce "invalid data"; show the endpoint's own message.
        res.error = (
            "endpoint answered an error body instead of audio: "
            f"{_scrub_error(raw.decode('utf-8', 'replace'), 200)}"
        )
        return res
    if label:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            res.error = (
                "endpoint answered a compressed format (MP3/OGG) and ffmpeg "
                "is unavailable — install ffmpeg, `pip install imageio-ffmpeg`, "
                "or set WALLIE_FFMPEG"
            )
            return res
        from tts.decode import decode_to_pcm16, looks_like_mp3_frame_sync
        try:
            raw = await decode_to_pcm16(raw, ffmpeg, res.sample_rate)
        except RuntimeError as e:
            # Raw PCM16 whose first sample is -1 starts with FF FF, which is
            # byte-for-byte the MP3 sync word — and a leading silence ramp (how
            # TTS clips usually begin) makes that first sample common. ffmpeg
            # failing on a payload that matches ONLY that ambiguous signature
            # means it was PCM all along; failing the whole clip over it would
            # be a false alarm. Genuine containers (RIFF/OggS/ID3) still error.
            if not looks_like_mp3_frame_sync(raw[:12]):
                res.error = str(e)
                return res
            logger.info(
                "tts: payload matched the MP3 sync word but ffmpeg could not "
                "decode it — treating it as raw PCM (first sample is 0x%s)",
                raw[:2].hex(),
            )
        else:
            res.decoded = True

    if len(raw) < 64:  # plausible audio of a sentence is way longer
        res.error = f"audio too short ({len(raw)} bytes) — endpoint likely rejected the request"
        return res
    if len(raw) % 2:
        raw = raw[:-1]
    res.pcm = raw
    return res


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

    def _auto_mirror_partner(cfg: Any) -> dict[str, Any]:
        """Keep a just-activated profile's voices in step with its partner.

        Switching profiles is the moment the user expects the two libraries to
        match, so when the profile has a voice partner and the two differ, the
        voices each side is missing are copied both ways. The policy is always
        ``skip`` — the only one that can never discard a setting — so a shared
        name with different settings is left exactly as it is and merely
        reported: the ⇄ badge keeps pointing at it instead of the server picking
        a winner behind the user's back. Returns ``{}`` when there is nothing to
        do (no partner, a stale one, or already in step), and a report rather
        than an exception when the copy is refused, so a full library can never
        fail a profile switch.
        """
        partner = (getattr(cfg, "voice_partner", "") or "").strip()
        if not partner or partner == cfg.profile_name or partner not in list_profiles():
            return {}
        try:
            if not _vlab.sync_state(cfg.profile_name, partner)["out_of_sync"]:
                return {}
            result = _vlab.mirror_libraries(cfg.profile_name, partner)
        except _vlab.VoiceLabError as e:
            return {"partner": partner, "synced": False, "error": str(e)}
        return {
            "partner": partner,
            "synced": True,
            "to_other": result["to_other"],
            "to_this": result["to_this"],
            "skipped": result["skipped"],
        }

    @app.put("/api/profiles/{name}/activate")
    def api_profiles_activate(name: str) -> dict[str, Any]:
        cfg = activate_profile(name)
        return {"ok": True, "active": cfg.profile_name, "partner_sync": _auto_mirror_partner(cfg)}

    @app.delete("/api/profiles/{name}")
    def api_profiles_delete(name: str) -> dict[str, Any]:
        ok = delete_profile(name)
        return {"ok": ok}

    # ---------- config ----------
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        """Serve the active profile. Self-heals dangling provider_refs (refs
        to deleted blocks or wrong-category blocks, e.g. from hand-edited
        YAML) — blanked once on disk so the manual edit isn't re-flagged on
        every load."""
        cfg = load_profile()
        healed = heal_provider_refs(cfg)
        if healed:
            save_profile(cfg, cfg.profile_name)
        return {**cfg.model_dump(), "healed_refs": healed}

    @app.put("/api/config")
    async def put_config(payload: dict[str, Any]) -> dict[str, Any]:
        # Merge onto the existing profile so any config section the UI doesn't send
        # (e.g. hearing) is preserved instead of being silently reset to defaults.
        # Provider blocks are NOT accepted here: the UI's cfg.providers copy is
        # stale the moment /api/providers saved something, and merging it back
        # would silently wipe blocks. Blocks are owned by PUT /api/providers.
        payload.pop("providers", None)
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

    @app.get("/api/loopback-devices")
    def loopback_devices() -> list[dict[str, Any]]:
        """Speakers the hearing loopback can listen through (soundcard's view)."""
        return list_loopback_devices()

    @app.post("/api/device-check")
    def device_check(body: DeviceCheckBody) -> dict[str, Any]:
        """Do the saved audio device NAMES still exist? (post-save stale warning)

        Both fields are saved by name; a device that disappeared between saves
        (unplugged headset, Windows audio change) would otherwise fail silently
        at playback/capture time. Sync def → threadpool; the soundcard check
        COM-initializes on its own thread (see hearing.capture)."""
        try:
            return {"tts_output": check_output_device(body.tts_output),
                    "loopback": resolve_loopback_device(body.loopback)}
        except Exception as e:  # noqa: BLE001 — never block a save on the warning
            logger.warning(f"dashboard: device check failed: {e}")
            return {"tts_output": {"exists": True, "name": ""},
                    "loopback": {"exists": True, "is_default": False}}

    @app.post("/api/test/audio-output")
    def api_test_audio_output(body: TestAudioOutputBody) -> dict[str, Any]:
        """Play a short beep through the chosen output device so the user can
        confirm where the TTS voice will play. Sync def → threadpool worker,
        so the blocking sd.play() never stalls the event loop."""
        try:
            return play_test_beep(body.device or None)
        except Exception as e:
            logger.warning(f"dashboard: test output beep failed: {e}")
            raise HTTPException(status_code=400, detail=str(e)[:220])

    @app.get("/api/gate/skipped")
    def gate_skipped() -> list[dict[str, Any]]:
        """Recent inputs the engagement gate skipped, newest first."""
        orch = state.orchestrator
        return orch.gate_skipped() if orch else []

    @app.post("/api/gate/force-reply/{msg_id}")
    def gate_force_reply(msg_id: str) -> dict[str, Any]:
        """Force a reply to one skipped input — it becomes the very next turn."""
        orch = state.orchestrator
        if orch is None:
            raise HTTPException(status_code=409, detail="session not running")
        try:
            return orch.gate_force_reply(msg_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/preflight")
    def api_preflight() -> list[dict[str, str]]:
        """Pre-start checklist: static config problems the session would hit.
        No side effects, no network calls — the Start button runs this first.
        Sync def (not async) ON PURPOSE: preflight may probe saved audio
        devices, and the soundcard needs COM on the calling thread — a
        threadpool worker COM-initializes itself, the event loop must not."""
        from wallie import preflight
        try:
            return preflight()
        except Exception as e:
            logger.exception("dashboard: preflight failed")
            raise HTTPException(status_code=500, detail=str(e) or e.__class__.__name__)

    @app.post("/api/tts/voices")
    async def api_tts_voices(body: TtsVoicesBody) -> dict[str, Any]:
        """Query the configured TTS endpoint's /models for available voices,
        so the Voice page can offer a dropdown instead of a blind text input."""
        from wallie import fetch_tts_voices
        from config import get_runtime
        try:
            runtime = get_runtime()
            return await fetch_tts_voices(runtime.config, runtime.secrets, ref=body.provider_ref)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e)[:220])
        except Exception as e:
            logger.exception("dashboard: tts voices fetch failed")
            raise HTTPException(status_code=500, detail=str(e)[:220])

    # ---------- Kokoro (optional local TTS) — status + one-click install ----------
    from tts import kokoro_setup as _kokoro

    _KOKORO_INSTALL_TIMEOUT = 1800.0

    async def _run_kokoro_install(lang: str, voice: str) -> None:
        """Run `install_kokoro.py --install` off the event loop and record the
        outcome on state.kokoro_install so the Voice-tab badge can poll it."""
        job = state.kokoro_install
        try:
            proc = await asyncio.create_subprocess_exec(
                *_kokoro.install_argv(lang, voice),
                cwd=str(_kokoro.REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as e:
            job.update(running=False, returncode=-1, finished_at=time.time(),
                       log=f"could not start the installer: {_scrub_error(str(e))}")
            return
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_KOKORO_INSTALL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            job.update(running=False, returncode=-1, finished_at=time.time(),
                       log="install timed out — check your connection and try again")
            return
        job.update(
            running=False,
            returncode=proc.returncode,
            finished_at=time.time(),
            log=(out or b"").decode("utf-8", "ignore")[-4000:],
        )

    @app.get("/api/tts/kokoro/status")
    def api_kokoro_status() -> dict[str, Any]:
        """Is the local Kokoro voice installed? Cheap metadata probe + install job."""
        return {"ok": True, **_kokoro.detect(), "install": dict(state.kokoro_install)}

    @app.post("/api/tts/kokoro/install")
    async def api_kokoro_install(body: KokoroInstallBody) -> dict[str, Any]:
        """Start a background pip install of the project-compatible Kokoro packages."""
        if not _kokoro.INSTALL_SCRIPT.is_file():
            raise HTTPException(400, "installer script missing: scripts/install_kokoro.py")
        lang = (body.lang or "a").strip().lower()
        if lang not in _kokoro.LANG_CODES:
            raise HTTPException(400, f"unknown Kokoro language {lang!r}")
        if state.kokoro_install["running"]:
            raise HTTPException(409, "a Kokoro install is already running")
        voice = (body.voice or "").strip()
        state.kokoro_install.update(
            running=True, lang=lang, started_at=time.time(),
            finished_at=None, returncode=None, log="installing…",
        )
        asyncio.create_task(_run_kokoro_install(lang, voice), name="kokoro-install")
        return {"ok": True, "started": True, "lang": lang}

    # ---------- Piper (optional local TTS) — status, install, voice downloads ----------
    from tts import piper_setup as _piper

    _PIPER_JOB_TIMEOUT = 1800.0

    async def _run_piper_job(argv: list[str]) -> None:
        """Run a Piper install/download subprocess off the event loop and stream
        its output onto state.piper_job so the Voice tab can show live progress.

        The downloader prints ``… (NN%)`` on stderr/stdout; each chunk is scanned
        for the last percentage, which the UI polls. The whole job is bounded by
        ``_PIPER_JOB_TIMEOUT``."""
        job = state.piper_job
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(_piper.REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as e:
            job.update(running=False, returncode=-1, finished_at=time.time(),
                       log=f"could not start the job: {_scrub_error(str(e))}")
            return
        buf = bytearray()
        deadline = time.time() + _PIPER_JOB_TIMEOUT
        try:
            assert proc.stdout is not None
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=remaining)
                if not chunk:
                    break
                buf.extend(chunk)
                text = bytes(buf).decode("utf-8", "ignore")
                job["log"] = text[-4000:]
                pct = _last_percent(text)
                if pct is not None:
                    job["progress"] = pct
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            job.update(running=False, returncode=-1, finished_at=time.time(),
                       log="the job timed out — check your connection and try again")
            return
        job.update(
            running=False,
            returncode=proc.returncode,
            finished_at=time.time(),
            log=bytes(buf).decode("utf-8", "ignore")[-4000:],
        )
        if proc.returncode == 0:
            job["progress"] = 100

    @app.get("/api/tts/piper/status")
    def api_piper_status() -> dict[str, Any]:
        """Is the local Piper engine installed? Which voices are on disk? What
        is the last install/download job doing?"""
        return {"ok": True, **_piper.detect(), "job": dict(state.piper_job)}

    @app.post("/api/tts/piper/install")
    async def api_piper_install() -> dict[str, Any]:
        """Start a background pip install of the Piper runtime."""
        if state.piper_job["running"]:
            raise HTTPException(409, "a Piper job is already running")
        state.piper_job.update(
            running=True, kind="install", voice="", progress=0, started_at=time.time(),
            finished_at=None, returncode=None, log="installing the Piper runtime…",
        )
        asyncio.create_task(_run_piper_job(_piper.install_argv()), name="piper-install")
        return {"ok": True, "started": True, "kind": "install"}

    @app.post("/api/tts/piper/download")
    async def api_piper_download(body: PiperDownloadBody) -> dict[str, Any]:
        """Download one voice from the HuggingFace catalogue into voices/."""
        if not _piper.INSTALL_SCRIPT.is_file():
            raise HTTPException(400, "helper script missing: scripts/download_piper_voice.py")
        if state.piper_job["running"]:
            raise HTTPException(409, "a Piper job is already running")
        voice = (body.voice or "").strip()
        try:
            argv = _piper.download_argv(voice)
        except ValueError as e:
            raise HTTPException(400, str(e))
        state.piper_job.update(
            running=True, kind="download", voice=voice, progress=0, started_at=time.time(),
            finished_at=None, returncode=None, log=f"downloading {voice}…",
        )
        asyncio.create_task(_run_piper_job(argv), name="piper-download")
        return {"ok": True, "started": True, "kind": "download", "voice": voice}

    @app.get("/api/tts/piper/catalog")
    async def api_piper_catalog(refresh: bool = False) -> dict[str, Any]:
        """Every voice published in the rhasspy/piper-voices catalogue, so the
        Voice tab can browse/download without leaving the dashboard."""
        try:
            voices = await _piper.fetch_catalog(force=bool(refresh))
        except _piper.PiperCatalogError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "count": len(voices), "voices": voices}

    # ---------- Voice Lab — saved voices + voice cloning ----------
    from tts import voice_lab as _vlab

    def _voice_library_payload(profile: str) -> dict[str, Any]:
        from config import get_runtime
        secrets_obj = get_runtime().secrets
        ready = {
            "elevenlabs": bool((secrets_obj.elevenlabs_api_key or "").strip()),
            "fish": bool((secrets_obj.fish_api_key or "").strip()),
        }
        return {
            "profile": profile,
            "presets": [p.to_dict() for p in _vlab.load_library(profile)],
            "providers": [
                {**spec, "ready": bool(ready.get(spec["id"]))}
                for spec in _vlab.CLONE_PROVIDERS
            ],
        }

    def _conflict_response(e: "_vlab.VoiceConflictError") -> HTTPException:
        """409 + the diff, so the UI can ask before replacing saved voices."""
        return HTTPException(409, detail={
            "message": str(e),
            "conflicts": e.plan.get("conflicts", []),
            "plan": e.plan,
        })

    @app.get("/api/voices/library")
    def api_voices_library(profile: str = "") -> dict[str, Any]:
        """Saved voices + which cloning backends have a key.

        ``?profile=<name>`` peeks at ANOTHER profile's library (read-only) so the
        Voice Lab can offer to pull voices from it; the default is the active one.
        """
        cfg = load_profile()
        if not profile or profile == cfg.profile_name:
            return _voice_library_payload(cfg.profile_name)
        from config import list_profiles
        if profile not in list_profiles():
            raise HTTPException(404, f"no profile named {profile}")
        return _voice_library_payload(profile)

    @app.post("/api/voices/library")
    def api_voices_library_add(body: VoicePresetBody) -> dict[str, Any]:
        cfg = load_profile()
        try:
            preset = _vlab.add_preset(cfg.profile_name, body.model_dump())
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "preset": preset.to_dict(),
                **_voice_library_payload(cfg.profile_name)}

    @app.delete("/api/voices/library/{name}")
    def api_voices_library_delete(name: str) -> dict[str, Any]:
        cfg = load_profile()
        if not _vlab.remove_preset(cfg.profile_name, name):
            raise HTTPException(404, f"no saved voice named {name}")
        return {"ok": True, **_voice_library_payload(cfg.profile_name)}

    # ---------- Local voice pickers — one saved list per local engine ----------
    from tts import local_voices as _lvoices

    @app.get("/api/voices/local")
    def api_local_voices() -> dict[str, Any]:
        """Saved voices bucket, one list per local engine (Piper / Kokoro)."""
        cfg = load_profile()
        return {
            "ok": True,
            "supported": list(_lvoices.LOCAL_VOICE_PROVIDERS),
            **_lvoices.providers_payload(cfg.profile_name),
        }

    @app.post("/api/voices/local/{provider}")
    def api_local_voices_add(provider: str, body: LocalVoiceBody) -> dict[str, Any]:
        cfg = load_profile()
        try:
            voice = _lvoices.add_local(cfg.profile_name, provider, body.model_dump())
        except _lvoices.LocalVoiceError as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "voice": voice.to_dict(),
            **_lvoices.providers_payload(cfg.profile_name),
        }

    @app.delete("/api/voices/local/{provider}/{name}")
    def api_local_voices_delete(provider: str, name: str) -> dict[str, Any]:
        cfg = load_profile()
        try:
            removed = _lvoices.remove_local(cfg.profile_name, provider, name)
        except _lvoices.LocalVoiceError as e:
            raise HTTPException(400, str(e))
        if not removed:
            raise HTTPException(404, f"no saved {provider} voice named {name}")
        return {"ok": True, **_lvoices.providers_payload(cfg.profile_name)}

    @app.post("/api/voices/export")
    def api_voices_export(body: VoiceExportBody) -> dict[str, Any]:
        """Copy the active profile's saved voices (including the local
        Piper/Kokoro ones) into another profile's library."""
        from config import list_profiles
        cfg = load_profile()
        target = (body.target_profile or "").strip()
        if not target:
            raise HTTPException(400, "pick a target profile")
        if target == cfg.profile_name:
            raise HTTPException(400, "pick a different profile")
        if target not in list_profiles():
            raise HTTPException(404, f"no profile named {target}")
        plan = _vlab.plan_export(cfg.profile_name, target, body.names or None)
        try:
            copied = _vlab.export_presets(
                cfg.profile_name, target, body.names or None, overwrite=body.overwrite
            )
        except _vlab.VoiceConflictError as e:
            raise _conflict_response(e)
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "copied": copied,
            "target": target,
            "total": len(_vlab.load_library(target)),
            "added": len(plan["added"]),
            "replaced": len(plan["replaced"]),
            "unchanged": len(plan["unchanged"]),
        }

    @app.get("/api/voices/backup")
    def api_voices_backup(names: str = "") -> dict[str, Any]:
        """The active profile's saved voices as a portable, downloadable
        ``.json`` bundle (``?names=a,b`` limits it to those voices)."""
        cfg = load_profile()
        wanted = [n.strip() for n in (names or "").split(",") if n.strip()]
        try:
            bundle = _vlab.export_bundle(cfg.profile_name, wanted or None)
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        if not bundle["presets"]:
            raise HTTPException(400, "no saved voices to export in this profile")
        return bundle

    @app.post("/api/voices/import")
    def api_voices_import(body: VoiceImportBody) -> dict[str, Any]:
        """Merge a voice backup file into the active profile's library.

        Same-named voices are replaced (upsert), so restoring a backup onto a
        library that already has some of the voices is safe to repeat."""
        cfg = load_profile()
        try:
            payload = _vlab.parse_bundle_text(body.content)
            result = _vlab.import_bundle(
                cfg.profile_name, payload, overwrite=body.overwrite
            )
        except _vlab.VoiceConflictError as e:
            raise _conflict_response(e)
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, **result, **_voice_library_payload(cfg.profile_name)}

    @app.post("/api/voices/pull")
    def api_voices_pull(body: VoicePullBody) -> dict[str, Any]:
        """The inverse of /api/voices/export: bring saved voices FROM another
        profile's library into this one (upsert by name).

        This is how a character's voices — including local Piper/Kokoro ones —
        move back without re-creating them.
        """
        from config import list_profiles
        cfg = load_profile()
        source = (body.source_profile or "").strip()
        if not source:
            raise HTTPException(400, "pick a source profile")
        if source == cfg.profile_name:
            raise HTTPException(400, "pick a different profile")
        if source not in list_profiles():
            raise HTTPException(404, f"no profile named {source}")
        if not _vlab.load_library(source):
            raise HTTPException(400, f"“{source}” has no saved voices to pull")
        plan = _vlab.plan_export(source, cfg.profile_name, body.names or None)
        try:
            copied = _vlab.export_presets(
                source, cfg.profile_name, body.names or None, overwrite=body.overwrite
            )
        except _vlab.VoiceConflictError as e:
            raise _conflict_response(e)
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        if body.names and not copied:
            raise HTTPException(400, f"none of those voices are saved in “{source}”")
        return {
            "ok": True,
            "copied": copied,
            "source": source,
            "added": len(plan["added"]),
            "replaced": len(plan["replaced"]),
            "unchanged": len(plan["unchanged"]),
            "total": len(_vlab.load_library(cfg.profile_name)),
            **_voice_library_payload(cfg.profile_name),
        }

    def _validate_other_profile(name: str) -> str:
        """A profile that exists and is not the active one (400/404 otherwise)."""
        from config import list_profiles
        other = (name or "").strip()
        if not other:
            raise HTTPException(400, "pick another profile")
        if other == load_profile().profile_name:
            raise HTTPException(400, "pick a different profile")
        if other not in list_profiles():
            raise HTTPException(404, f"no profile named {other}")
        return other

    @app.get("/api/voices/mirror")
    def api_voices_mirror_plan(other: str = "") -> dict[str, Any]:
        """The two-way diff between the active profile's saved voices and
        another profile's — what a mirror would add on each side, and which
        shared names differ."""
        other = _validate_other_profile(other)
        cfg = load_profile()
        return {"ok": True, **_vlab.mirror_plan(cfg.profile_name, other)}

    @app.post("/api/voices/mirror")
    def api_voices_mirror(body: VoiceMirrorBody) -> dict[str, Any]:
        """Give both profiles the same saved voices (missing ones copied each
        way; differing shared names follow the chosen ``conflicts`` policy)."""
        other = _validate_other_profile(body.other_profile)
        cfg = load_profile()
        try:
            result = _vlab.mirror_libraries(
                cfg.profile_name, other, conflicts=body.conflicts
            )
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "other": other,
            **result,
            **_voice_library_payload(cfg.profile_name),
        }

    @app.get("/api/voices/partners")
    def api_voices_partners() -> dict[str, Any]:
        """Every profile against its voice partner, for the profile picker's
        out-of-sync badge.

        Read-only. A partner that was renamed or deleted since it was chosen is
        reported as ``stale`` rather than silently cleared, so a wrong badge
        never becomes a silently rewritten profile.
        """
        cfg = load_profile()
        names = list_profiles() or [cfg.profile_name]
        profiles: dict[str, Any] = {}
        for name in names:
            partner = (load_profile(name).voice_partner or "").strip()
            entry: dict[str, Any] = {
                "partner": partner,
                "stale": False,
                "out_of_sync": False,
                "to_other": 0,
                "to_this": 0,
                "conflicts": 0,
            }
            if partner:
                if partner == name or partner not in names:
                    entry["stale"] = True
                else:
                    entry.update(_vlab.sync_state(name, partner))
            profiles[name] = entry
        return {"ok": True, "active": cfg.profile_name, "profiles": profiles}

    @app.post("/api/voices/partner")
    def api_voices_set_partner(body: VoicePartnerBody) -> dict[str, Any]:
        """Set this profile's voice partner (empty clears it).

        Only records the choice — the libraries are still equalised by
        ``POST /api/voices/mirror``, so nothing is copied behind the user's back.
        """
        cfg = load_profile()
        partner = (body.partner or "").strip()
        if partner:
            if partner == cfg.profile_name:
                raise HTTPException(400, "a profile cannot partner itself")
            if partner not in (list_profiles() or []):
                raise HTTPException(404, f"no profile named {partner}")
        cfg.voice_partner = partner
        save_profile(cfg, cfg.profile_name)
        state: dict[str, Any] = {
            "partner": "",
            "stale": False,
            "out_of_sync": False,
            "to_other": 0,
            "to_this": 0,
            "conflicts": 0,
        }
        if partner:
            state.update(_vlab.sync_state(cfg.profile_name, partner))
        return {"ok": True, "active": cfg.profile_name, **state}

    @app.post("/api/voices/clone")
    async def api_voices_clone(body: VoiceCloneBody) -> dict[str, Any]:
        """Create a real voice at ElevenLabs / Fish Audio from reference audio.

        The dashboard posts base64 samples (it has no multipart parser installed),
        which are decoded here and uploaded as multipart to the provider."""
        import base64 as _b64

        from config import get_runtime
        cfg = load_profile()
        runtime = get_runtime()
        samples: list[tuple[str, bytes]] = []
        for i, s in enumerate(body.samples, 1):
            raw = (s.data or "").split(",", 1)[-1]      # tolerate a data: URL prefix
            try:
                blob = _b64.b64decode(raw, validate=False)
            except Exception:
                raise HTTPException(400, f"sample {s.filename or i} is not valid base64")
            samples.append((s.filename or f"sample{i}.wav", blob))
        try:
            voice_id = await _vlab.clone_voice(
                provider=body.provider,
                name=body.name,
                description=body.description,
                samples=samples,
                secrets=runtime.secrets,
            )
        except _vlab.VoiceLabError as e:
            raise HTTPException(400, str(e))
        preset = None
        if body.save_preset:
            try:
                preset = _vlab.add_preset(cfg.profile_name, {
                    "name": body.name,
                    "provider": body.provider,
                    "voice_id": voice_id,
                    "source": f"clone:{body.provider}",
                    "notes": body.description,
                })
            except _vlab.VoiceLabError as e:
                # The voice EXISTS at the provider — report it, don't fail the call.
                logger.warning(f"voice lab: clone succeeded but preset save failed: {e}")
        return {
            "ok": True,
            "provider": body.provider,
            "voice_id": voice_id,
            "preset": preset.to_dict() if preset else None,
            **_voice_library_payload(cfg.profile_name),
        }

    @app.post("/api/voices/record")
    async def api_voices_record(body: VoiceRecordBody) -> dict[str, Any]:
        """Record a reference sample from the default microphone.

        Returns a WAV the browser can play back and feed straight into a clone,
        so a voice can be built without leaving the dashboard."""
        import base64 as _b64

        seconds = max(1.0, min(30.0, float(body.seconds or 8.0)))
        try:
            pcm = await asyncio.to_thread(_record_mic, seconds)
        except Exception as e:  # noqa: BLE001 — soundcard/COM failures are expected
            raise HTTPException(400, f"recording failed: {str(e)[:200]}")
        if getattr(pcm, "size", 0) == 0:
            raise HTTPException(400, "nothing was recorded — check the default microphone")
        wav = _vlab.pcm16_wav_bytes(pcm, 16000)
        return {
            "ok": True,
            "sample_rate": 16000,
            "seconds": round(len(pcm) / 16000.0, 2),
            "wav_b64": _b64.b64encode(wav).decode("ascii"),
        }

    @app.post("/api/voices/preview")
    async def api_voices_preview(body: VoicePreviewBody) -> dict[str, Any]:
        """Synthesize one line with a chosen voice and return it as a playable WAV.

        Unlike /api/test/voice this deliberately does NOT play anything: the Voice
        Lab A/B calls it once per voice and puts the two clips side by side in the
        browser, so nothing is routed through the live output device and the user
        can replay both. Nothing is written to the profile either.
        """
        import base64 as _b64

        from config import Runtime, get_runtime
        from wallie import _effective_tts

        text = (body.text or "").strip()
        if not text:
            raise HTTPException(400, "empty text")
        if len(text) > 600:
            raise HTTPException(400, "text too long (max 600 chars)")

        runtime = get_runtime()
        requested = body.name.strip()
        updates: dict[str, Any] = {}
        if requested:
            preset = next(
                (p for p in _vlab.load_library(runtime.config.profile_name)
                 if p.name.lower() == requested.lower()),
                None,
            )
            if preset is None:
                raise HTTPException(404, f"no saved voice named {requested}")
            updates = _vlab.tts_updates(preset)
            source = f"saved voice · {preset.name}"
        elif body.provider or body.voice_id or body.tts:
            updates = _vlab.sanitize_tts(body.tts)
            if body.provider:
                updates["provider"] = body.provider
            if body.voice_id:
                updates["voice_id"] = body.voice_id
            source = "unsaved Voice page settings"
        else:
            source = "saved config"

        if updates:
            tts_cfg = runtime.config.tts.model_copy(update=updates)
            # Re-resolve through the live build's OWN resolution: a preset may
            # switch the provider TO openai_compatible, and only that path knows
            # how to find the block's endpoint and key.
            runtime = Runtime(
                config=runtime.config.model_copy(update={"tts": tts_cfg}),
                secrets=runtime.secrets,
            )
        try:
            tts_cfg, tts_secrets = _effective_tts(runtime)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e)[:220])

        res = await _synthesize_pcm(tts_cfg, tts_secrets, text)
        if res.error:
            raise HTTPException(400, res.error[:300])

        seconds = len(res.pcm) / float(res.sample_rate * res.channels * 2)
        wav = _vlab.pcm16_bytes_to_wav(res.pcm, res.sample_rate, res.channels)
        return {
            "ok": True,
            "source": source,
            "provider": tts_cfg.provider,
            "voice_id": tts_cfg.voice_id,
            "voice_ref": tts_cfg.provider_ref,
            "sample_rate": res.sample_rate,
            "channels": res.channels,
            "bytes": len(res.pcm),
            "seconds": round(seconds, 2),
            "decoded": res.decoded,
            "wav_b64": _b64.b64encode(wav).decode("ascii"),
        }

    @app.post("/api/vision/models")
    async def api_vision_models(body: VisionModelsBody) -> dict[str, Any]:
        """Query the configured vision endpoint's /models for available VL models,
        so the Vision page can offer a dropdown instead of a blind text input."""
        from wallie import fetch_vision_models
        from config import get_runtime
        try:
            runtime = get_runtime()
            return await fetch_vision_models(
                runtime.config, runtime.secrets, ref=body.provider_ref)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e)[:220])
        except Exception as e:
            logger.exception("dashboard: vision models fetch failed")
            raise HTTPException(status_code=500, detail=str(e)[:220])

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
        store = _speaker_store()
        # Case-insensitive resolution, same contract as the note route
        # (labels like "owner" hit the stored "Owner").
        resolved = store.find_name(name) or name
        if not store.remove(resolved):
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
        # Validate with _resolve_field (NOT just the static SECRET_FIELDS) so
        # dynamic provider-block keys (PROVIDER_<ID>_API_KEY) are accepted too —
        # they live in the same .env, but only exist as metadata at runtime.
        from secrets_store import _resolve_field, set_secret
        if _resolve_field(body.env) is None:
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

    def _save_providers(blocks: list[ProviderBlock]) -> tuple[list[dict[str, Any]], list[str]]:
        cfg = load_profile()
        # Cascade: any provider_ref pointing at a block this save removes gets
        # blanked (→ runtime first-block-of-category fallback) instead of
        # silently dangling. Covers BOTH delete paths: the trash-can DELETE
        # route and removing a row on the page + "save blocks".
        old_ids = {p.id for p in cfg.providers}
        new_ids = {b.id for b in blocks}
        cleared_refs: list[str] = []
        for removed_id in sorted(old_ids - new_ids):
            cleared_refs.extend(clear_provider_refs(cfg, removed_id))
        try:
            cfg = cfg.model_copy(update={"providers": blocks})
            save_profile(cfg, cfg.profile_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid provider block: {e}")
        _sync_provider_context()
        return [p.model_dump() for p in blocks], cleared_refs

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
        saved, cleared_refs = _save_providers(blocks)
        return {"ok": True, "providers": saved, "cleared_refs": cleared_refs}

    @app.delete("/api/providers/{provider_id}")
    def api_providers_delete(provider_id: str) -> dict[str, Any]:
        cfg = load_profile()
        blocks = [p for p in cfg.providers if p.id != provider_id]
        if len(blocks) == len(cfg.providers):
            raise HTTPException(status_code=404, detail=f"unknown provider: {provider_id}")
        # Blank any provider_ref that pointed at the deleted block so no
        # subsystem keeps silently falling back; report them to the UI.
        cleared_refs = clear_provider_refs(cfg, provider_id)
        if cleared_refs:
            save_profile(cfg, cfg.profile_name)
        _save_providers(blocks)
        # Clean the block's key out of .env so deleted blocks don't leave
        # orphan PROVIDER_*_API_KEY rows piling up on the API Keys page.
        # Direct unset_key + pop (bypasses set_secret's load_dotenv reload).
        from secrets_store import ENV_FILE, _harden_perms
        key_env = provider_key_env(provider_id)
        try:
            from dotenv import unset_key
            unset_key(str(ENV_FILE), key_env)
        except Exception:
            pass  # best-effort; the block itself is already gone
        os.environ.pop(key_env, None)
        try:
            _harden_perms(ENV_FILE)
        except Exception:
            pass
        return {"ok": True, "deleted": provider_id, "cleared_refs": cleared_refs}

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
        key_env = provider_key_env(provider_id)
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
    async def test_vision(body: Optional[TestVisionBody] = None) -> dict[str, Any]:
        """Capture + ask, using the SAME vision provider a live session uses:
        a resolved dedicated vision block when one exists, otherwise the main
        engine. Mirrors build_orchestrator's resolution and gate.

        Optional {provider, model, base_url} overrides build a THROWAWAY
        provider, so a candidate vision model can be tried out from the UI
        without saving it into the profile — the way to recover when a model
        gets retired (e.g. gemini-2.5-flash answering 404)."""
        from config import get_runtime
        from wallie import effective_compat
        from llm.factory import build_provider
        runtime = get_runtime()
        cfg = runtime.config
        ov = body or TestVisionBody()
        ov_model = (ov.model or "").strip()
        ov_url = (ov.base_url or "").strip()
        provider = (ov.provider or "").strip() or cfg.llm.provider
        dedicated = None
        if ov.has_override():
            # ---- override: probe a model/provider WITHOUT touching the profile ----
            if provider == "openai_compatible":
                try:
                    vis_cfg, vis_sec = effective_compat(
                        cfg, runtime.secrets, "vision", cfg.llm,
                        ref=(ov.provider_ref or cfg.llm.vision_provider_ref),
                        allow_default=False,
                    )
                except RuntimeError as e:  # e.g. remote block without an API key
                    raise HTTPException(400, str(e)[:300])
                updates: dict[str, Any] = {"vision_provider": "openai_compatible"}
                if ov_model:
                    updates["vision_model"] = ov_model
                if ov_url:
                    updates["vision_openai_compatible_base_url"] = ov_url
                vis_cfg = vis_cfg.model_copy(update=updates)
                if not (vis_cfg.vision_model or "").strip():
                    raise HTTPException(
                        400, "pick a model for the OpenAI-compatible vision test"
                    )
                try:
                    from wallie import _build_vision_llm
                    dedicated = _build_vision_llm(vis_cfg, vis_sec)
                except RuntimeError as e:
                    raise HTTPException(400, str(e)[:300])
            else:
                test_model = ov_model or cfg.llm.model
                if not test_model:
                    raise HTTPException(400, f"pick a model to test {provider}")
                # vision_capable is forced ON: the user explicitly named the
                # model, and the request itself is the test of whether it
                # really accepts images.
                fields: dict[str, Any] = {
                    "provider": provider, "model": test_model, "vision_capable": True,
                }
                if ov_url:
                    # Ollama carries its own base-URL field; every other
                    # provider here reaches its endpoint through the generic one.
                    fields["ollama_base_url" if provider == "ollama"
                           else "openai_compatible_base_url"] = ov_url
                try:
                    dedicated = build_provider(
                        cfg.llm.model_copy(update=fields), runtime.secrets
                    )
                except Exception as e:  # unknown provider / missing key / bad URL
                    raise HTTPException(400, f"could not build {provider}: {e}"[:300])
        elif cfg.llm.vision_provider == "openai_compatible":
            try:
                vis_cfg, vis_sec = effective_compat(
                    cfg, runtime.secrets, "vision", cfg.llm,
                    ref=cfg.llm.vision_provider_ref, allow_default=False)
            except RuntimeError as e:  # e.g. remote block without an API key
                raise HTTPException(400, str(e)[:300])
            try:
                from wallie import _build_vision_llm
                dedicated = _build_vision_llm(vis_cfg, vis_sec)
            except RuntimeError as e:
                raise HTTPException(400, str(e)[:300])
        if dedicated is None and not cfg.llm.vision_capable:
            raise HTTPException(
                400,
                "No vision model available: Engine.vision_capable is OFF and no "
                "dedicated vision provider is configured (create a VISION block on "
                "the API Keys page or toggle vision_capable for the engine).",
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
        llm = dedicated if dedicated is not None else build_provider(cfg.llm, runtime.secrets)
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
            # 400, not 500: a retired model / rejected key is a configuration
            # problem the user fixes by picking another model in the test strip.
            raise HTTPException(400, f"LLM error: {_scrub_error(str(e), 400)}")
        finally:
            await llm.aclose()
        text = "".join(tokens).strip()
        return {
            "ok": True,
            "frame_size": [frame.width, frame.height],
            "frame_bytes": len(frame.jpeg),
            "model": getattr(llm, "model", "") or cfg.llm.model,
            "provider": getattr(llm, "name", "") or cfg.llm.provider,
            "dedicated": dedicated is not None,
            "override": ov.has_override(),
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
        """Synthesize a sample line with the CONFIGURED TTS and play it.

        Uses the same block/legacy resolution as the real build (via
        wallie._effective_tts), and reports what happened: bytes received,
        whether the payload needed ffmpeg decoding, and the real reason on
        failure (empty synthesis, non-PCM without ffmpeg, bad config…).
        """
        from wallie import _effective_tts
        from config import get_runtime

        text = (body.text or "").strip()
        if not text:
            return JSONResponse(status_code=400, content={"detail": "empty text"})
        if len(text) > 600:
            return JSONResponse(status_code=400, content={"detail": "text too long (max 600 chars)"})

        cfg = load_profile()
        runtime = get_runtime()
        try:
            tts_cfg, tts_secrets = _effective_tts(runtime)
        except (ValueError, RuntimeError) as e:
            return JSONResponse(status_code=400, content={"detail": str(e)[:220]})

        def _fail(msg: str) -> JSONResponse:
            return JSONResponse(status_code=400, content={"detail": msg})

        orch = state.orchestrator
        live = bool(orch and orch.status().get("running"))

        res = await _synthesize_pcm(tts_cfg, tts_secrets, text)
        if res.error:
            return _fail(res.error)
        raw = res.pcm

        # ---- playback ----
        if live:
            player = orch._player  # noqa: SLF001
            await player.write(raw)
        else:
            from audio import AudioPlayer
            player = AudioPlayer(sample_rate=res.sample_rate, channels=res.channels,
                                 device=(cfg.tts.output_device or None))
            player.start()
            try:
                await player.write(raw)
                await asyncio.sleep(0.3)
                await player.wait_drained()
            finally:
                player.close()

        routed = "live-player" if live else "preview-player"
        dur = len(raw) / (res.sample_rate * res.channels * 2)
        note = f"{len(raw)} bytes · {dur:.1f}s"
        if res.decoded:
            note += " · decoded via ffmpeg (gateway ignored pcm)"
        return {"ok": True, "routed": routed, "decoded": res.decoded,
                "bytes": len(raw), "duration_sec": round(dur, 2), "note": note}

    @app.post("/api/test/hearing")
    async def test_hearing(body: TestHearingBody) -> dict[str, Any]:
        """Record from an audio input and transcribe it with the CONFIGURED STT.

        The mirror of /api/test/voice for the ear. Two sources:
        - ``system`` — the WASAPI loopback Wallie actually listens through live;
        - ``mic`` — the default microphone, for a true round-trip test (loopback
          only hears what PLAYS on the machine, not your voice).
        The engine resolution (provider block → legacy fields) is identical to
        the build's. Returns transcript, level and duration — or the real
        reason on failure (device missing, silence, STT error…).
        """
        import numpy as np
        from config import get_runtime
        from wallie import _effective_stt

        try:
            from hearing.capture import SystemAudioCapture
        except Exception as e:
            return JSONResponse(status_code=400, content={"detail":
                f"capture unavailable: {e}"})

        source = (body.source or "system").strip().lower()
        if source not in ("system", "mic"):
            return JSONResponse(status_code=400, content={"detail":
                f"unknown source {source!r} — use 'system' or 'mic'"})
        seconds = min(15.0, max(2.0, float(body.seconds or 5.0)))
        cfg = load_profile()
        runtime = get_runtime()
        try:
            hearing_cfg, stt_secrets = _effective_stt(runtime)
        except (ValueError, RuntimeError) as e:
            return JSONResponse(status_code=400, content={"detail": str(e)[:220]})

        def _fail(msg: str) -> JSONResponse:
            return JSONResponse(status_code=400, content={"detail": msg})

        # ---- record (same capture stack as HearingLoop) ----
        try:
            if source == "mic":
                audio = await asyncio.to_thread(_record_mic, seconds)
            else:
                # Same loopback device the live loop listens through
                # (cfg.hearing.loopback_device; "" = system default speaker).
                # body.device lets the UI test an UNSAVED pick first.
                cap = SystemAudioCapture(
                    samplerate=16000,
                    device=((body.device or getattr(cfg.hearing, "loopback_device", "")) or ""),
                )
                cap.open()
                try:
                    # Let the ring buffer fill before grabbing — same rule the
                    # live loop uses before its first read.
                    await asyncio.sleep(seconds)
                    audio = cap.latest(seconds)
                finally:
                    cap.close()
        except Exception as e:
            return _fail(f"could not open audio input ({source}): {e}")

        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        if rms < 1e-4:
            return _fail(
                "recorded silence — check that the right device is capturing "
                "and that it is not muted"
            )

        # ---- transcribe with the CONFIGURED engine ----
        def _transcribe() -> str:
            if getattr(hearing_cfg, "engine", "") != "openai_compatible":
                from hearing.hearing_loop import HearingLoop
                loop = HearingLoop(hearing_cfg, __import__("asyncio").Queue(maxsize=2))
                loop._model = loop._load_model()
                return loop._transcribe(audio).strip()
            key = (stt_secrets.openai_compatible_stt_api_key or "").strip()
            # Same rule as the build: only a REMOTE endpoint demands a key —
            # localhost gateways run without one. _compat_api_key encodes it.
            from wallie import _compat_api_key
            key = _compat_api_key(
                key, getattr(hearing_cfg, "openai_compatible_base_url", ""),
                "STT provider block")
            import wave as _wave
            import io as _io
            from hearing.openai_stt import RemoteWhisperModel
            engine = RemoteWhisperModel(
                api_key=key,
                base_url=hearing_cfg.openai_compatible_base_url,
                model=getattr(hearing_cfg, "openai_compatible_model", "whisper-1"),
                language=getattr(hearing_cfg, "language", ""),
                prompt=getattr(hearing_cfg, "openai_compatible_prompt", ""),
                timeout=getattr(hearing_cfg, "openai_compatible_timeout", 20.0),
            )
            # The engine accepts numpy frames; the transcription call is
            # blocking (httpx sync) so it runs on this worker thread.
            try:
                segs, _info = engine.transcribe(audio)
            finally:
                engine.close()  # per-request engine — release its connection pool
            return " ".join(s.text.strip() for s in segs).strip()

        try:
            text = await asyncio.get_event_loop().run_in_executor(None, _transcribe)
        except Exception as e:
            return _fail(f"transcription failed: {str(e)[:300]}")

        from hearing.hearing_loop import _is_hallucination
        note = f"{len(text)} chars · {seconds:.0f}s ({source})"
        return {"ok": True, "text": text, "seconds": seconds, "source": source,
                "rms": round(rms, 4), "hallucination": bool(_is_hallucination(text)),
                "note": note}

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
        app.mount("/static", _NoCacheStatic(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        # Same reasoning as _NoCacheStatic: a stale index.html would keep
        # pointing the browser at an old app.js.
        return FileResponse(str(STATIC_DIR / "index.html"), headers={"Cache-Control": "no-cache"})

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

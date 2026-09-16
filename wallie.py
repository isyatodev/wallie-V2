"""Entrypoint — builds all subsystems from config and starts the pipeline."""
from __future__ import annotations

import asyncio
import re as _re
import sys
from typing import Optional

from loguru import logger

from audio import AudioPlayer
from chat import ChatManager
from config import Runtime, get_runtime
from core import Orchestrator, Persona, MemoryStore
from config import PROFILES_DIR
from donations import DonationDedupe, DonationEvent
from donations.queue import DonationQueue
from llm import build_provider
from tts import build_tts


_PROVIDER_SLUG_RE = _re.compile(r"[^a-z0-9_]+")


def _provider_slug(pid: str) -> str:
    """Env-safe slug for a provider id (matches secrets_store normalization)."""
    s = _PROVIDER_SLUG_RE.sub("_", (pid or "").lower()).strip("_")
    return s[:40] or "provider"


_CATEGORY_KEY_FIELD = {
    "llm": "openai_compatible_api_key",
    "vision": "openai_compatible_vision_api_key",
    "tts": "openai_compatible_tts_api_key",
    "stt": "openai_compatible_stt_api_key",
    "memory": "openai_compatible_memory_api_key",
    "thoughts": "openai_compatible_thoughts_api_key",
}


def _resolve_provider(cfg, secrets, category: str, ref: str = "", allow_default: bool = True) -> dict:
    """Resolve the endpoint settings for a subsystem (module-level, testable).

    Order: block named by `ref` → first block of the category → the block
    named "default" → the legacy per-subsystem fields. Returns a dict with
    base_url / model / api_key / from.
    """
    import os as _os
    blocks = [p.model_dump() for p in (getattr(cfg, "providers", None) or [])]
    ref = (ref or "").strip()
    block = None
    if ref:
        for b in blocks:
            if b.get("id") == ref:
                block = b
                break
    if block is None:
        for b in blocks:
            if b.get("category") == category:
                block = b
                break
    if block is None and allow_default:
        for b in blocks:
            if b.get("id") == "default":
                block = b
                break
    if block is not None:
        pid = block.get("id", "")
        key_env = "PROVIDER_" + _provider_slug(pid).upper() + "_API_KEY"
        return {
            "base_url": (block.get("base_url") or "").strip(),
            "model": (block.get("model") or "").strip(),
            "api_key": _os.getenv(key_env, ""),
            "from": f"provider:{pid}",
        }
    # Legacy fallback — the per-subsystem fields.
    cat = {
        "llm": cfg.llm, "vision": cfg.llm, "tts": cfg.tts, "stt": cfg.hearing,
        "memory": cfg.memory, "thoughts": cfg.random_thoughts,
    }.get(category)
    return {
        "base_url": getattr(cat, "openai_compatible_base_url", "") if cat else "",
        "model": getattr(cat, "openai_compatible_model", "") if cat else "",
        "api_key": getattr(secrets, _CATEGORY_KEY_FIELD.get(category, ""), ""),
        "from": "legacy",
    }


def _compat_api_key(api_key: str, base_url: str, purpose: str) -> str:
    """Local OpenAI-compatible servers (Ollama, LM Studio, kokoro-fastapi…)
    don't need a key; remote ones do. Send a placeholder for localhost so the
    provider constructor doesn't reject an empty credential."""
    key = (api_key or "").strip()
    if key:
        return key
    host = (base_url or "").split("://", 1)[-1].split("/", 1)[0].lower()
    if host.startswith(("localhost", "127.0.0.1", "[::1]", "0.0.0.0", "::1")):
        return "local-no-key"
    raise RuntimeError(
        f"{purpose}: remote endpoints need an API key "
        "(API Keys → the matching OpenAI-Compatible entry)"
    )


def _build_vision_llm(cfg, secrets):
    """Build the DEDICATED vision provider when llm.vision_provider is
    'openai_compatible'; None means "reuse the main LLM" (default behavior)."""
    if cfg.llm.vision_provider != "openai_compatible":
        return None
    from llm.openai_compat import OpenAICompatProvider
    base_url = (cfg.llm.vision_openai_compatible_base_url or "").strip()
    if not base_url:
        # Fall back to the generic LLM endpoint if vision's own is unset —
        # handy when one gateway serves both text and vision models.
        base_url = (cfg.llm.openai_compatible_base_url or "").strip()
    if not base_url:
        raise RuntimeError(
            "llm.vision_provider=openai_compatible: set vision_openai_compatible_base_url "
            '(or openai_compatible_base_url) — e.g. "https://dashscope.example/v1", version path included'
        )
    model = (cfg.llm.vision_model or "").strip()
    if not model:
        # No silent fallback to the brain model: a text-only model silently
        # receiving screenshots is exactly what this block exists to avoid.
        raise RuntimeError(
            "llm.vision_provider=openai_compatible: set llm.vision_model "
            "(e.g. qwen-vl-max) — it cannot be inferred from the brain model"
        )
    return OpenAICompatProvider(
        name="openai_compatible_vision",
        model=model,
        api_key=secrets.openai_compatible_vision_api_key,
        base_url=base_url,
        supports_vision=True,
        timeout=cfg.llm.vision_openai_compatible_timeout,
    )


def _configure_logger(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)


# ---------------------------------------------------------------------------
# Pre-start checks (dashboard Start button)
#
# `preflight()` mirrors, statically, what `build_orchestrator()` will do: same
# provider resolution (block → category → default → legacy fields), same
# localhost-doesn't-need-a-key rule, same required fields. It performs NO
# network calls so the dashboard gets an instant checklist instead of a
# 500 with a traceback after the user clicks Start.
# ---------------------------------------------------------------------------


def _pf_key_ok(api_key: str, base_url: str) -> bool:
    """Same rule as `_compat_api_key`: a key is optional only for localhost."""
    if (api_key or "").strip():
        return True
    host = (base_url or "").split("://", 1)[-1].split("/", 1)[0].lower()
    return host.startswith(("localhost", "127.0.0.1", "[::1]", "0.0.0.0", "::1"))


def _pf_compat(cfg, secrets, category: str, ref: str,
               legacy_base: str, legacy_model: str, legacy_key: str) -> tuple[str, str, str]:
    """Effective (base_url, model, api_key) for an openai_compatible subsystem,
    exactly as `_effective_compat` + `_compat_api_key` resolve them in the build."""
    prov = _resolve_provider(cfg, secrets, category, ref=ref)
    if prov["from"] == "legacy":
        return (legacy_base or "").strip(), (legacy_model or "").strip(), legacy_key
    base = (prov["base_url"] or "").strip() or (legacy_base or "").strip()
    model = (prov["model"] or "").strip() or (legacy_model or "").strip()
    return base, model, prov["api_key"]


def _pf_dep_missing(pkg: str) -> bool:
    import importlib.util
    import sys
    if pkg in sys.modules:
        return False  # already importable (covers test doubles too)
    try:
        return importlib.util.find_spec(pkg) is None
    except (ImportError, ValueError):
        return True


def preflight(runtime: Optional[Runtime] = None) -> list[dict]:
    """Static pre-start checklist for the dashboard.

    Returns a list of issues; empty list = everything will build. Each issue
    is {"level": "error"|"warn", "section": str, "message": str} where
    "error" means start() WILL fail and "warn" means a feature will run
    degraded or silently disabled.
    """
    runtime = runtime or get_runtime()
    cfg = runtime.config
    sec = runtime.secrets
    issues: list[dict] = []

    def add(level: str, section: str, message: str) -> None:
        issues.append({"level": level, "section": section, "message": message})

    # ---- Engine (LLM) — the build raises straight through, so all errors ----
    llm = cfg.llm
    if llm.provider == "openai_compatible":
        base, _model, key = _pf_compat(
            cfg, sec, "llm", llm.provider_ref,
            llm.openai_compatible_base_url, llm.model,
            sec.openai_compatible_api_key,
        )
        if not base:
            add("error", "Engine (LLM)",
                "provider is 'openai_compatible' but no endpoint is set — "
                "create a block in API Keys (category LLM) or fill the base URL field")
        elif not _pf_key_ok(key, base):
            add("error", "Engine (LLM)",
                f"endpoint {base} is remote and has no API key (API Keys → the LLM block)")
        if _pf_dep_missing("openai"):
            add("error", "Engine (LLM)", "package 'openai' is not installed (pip install openai)")
    elif llm.provider in ("openai", "groq", "openrouter", "anthropic", "gemini"):
        key_field = {
            "openai": "openai_api_key", "groq": "groq_api_key",
            "openrouter": "openrouter_api_key", "anthropic": "anthropic_api_key",
            "gemini": "gemini_api_key",
        }[llm.provider]
        if not (getattr(sec, key_field) or "").strip():
            add("error", "Engine (LLM)",
                f"provider '{llm.provider}' has no API key ({key_field.upper()} in .env)")

    # ---- Voice (TTS) — build raises, so errors ----
    tts = cfg.tts
    if tts.provider == "piper":
        import pathlib
        if not (tts.piper_model_path or "").strip():
            add("error", "Voice (TTS)",
                "piper has no model — download a .onnx voice and set the path, "
                "or switch the TTS provider (Voice section)")
        elif not pathlib.Path(tts.piper_model_path).is_file():
            add("error", "Voice (TTS)", f"piper model file not found: {tts.piper_model_path}")
    elif tts.provider == "fish" and not (sec.fish_api_key or "").strip():
        add("error", "Voice (TTS)", "provider 'fish' has no API key (FISH_API_KEY)")
    elif tts.provider == "elevenlabs" and not (sec.elevenlabs_api_key or "").strip():
        add("error", "Voice (TTS)", "provider 'elevenlabs' has no API key (ELEVENLABS_API_KEY)")
    elif tts.provider == "openai_compatible":
        base, model, key = _pf_compat(
            cfg, sec, "tts", tts.provider_ref,
            tts.openai_compatible_base_url, tts.openai_compatible_model,
            sec.openai_compatible_tts_api_key,
        )
        if not base:
            add("error", "Voice (TTS)",
                "provider is 'openai_compatible' but no endpoint is set (API Keys → TTS block)")
        elif not _pf_key_ok(key, base):
            add("error", "Voice (TTS)", f"endpoint {base} is remote and has no API key")
        if base and not model:
            add("warn", "Voice (TTS)", "no TTS model set — the endpoint's default will be used")

    # ---- Hearing (STT) — remote STT is lenient in the build, so warns there ----
    hearing = cfg.hearing
    if hearing.enabled:
        if hearing.engine == "openai_compatible":
            base, _m, key = _pf_compat(
                cfg, sec, "stt", hearing.provider_ref,
                hearing.openai_compatible_base_url, hearing.openai_compatible_model,
                sec.openai_compatible_stt_api_key,
            )
            if not base:
                add("error", "Hearing (STT)",
                    "engine is 'openai_compatible' but no endpoint is set (API Keys → STT block)")
            elif not _pf_key_ok(key, base):
                add("warn", "Hearing (STT)",
                    f"remote STT endpoint {base} has no API key — transcriptions will fail with 401")
        elif _pf_dep_missing("faster_whisper"):
            add("warn", "Hearing (STT)",
                "local engine selected but faster-whisper is not installed (pip install faster-whisper)")
        if _pf_dep_missing("soundcard"):
            add("warn", "Hearing (STT)", "package 'soundcard' not installed — audio capture will fail")

    # ---- Vision — build falls back / disables, so warns (except bad config) ----
    if cfg.vision.enabled:
        if not llm.vision_capable:
            add("warn", "Vision", "enabled but the LLM is not marked vision-capable — vision will be disabled")
        if llm.vision_provider == "openai_compatible":
            base = (llm.vision_openai_compatible_base_url or "").strip() or (llm.openai_compatible_base_url or "").strip()
            model = (llm.vision_model or "").strip()
            if not base:
                add("error", "Vision", "dedicated vision provider has no endpoint (Vision section)")
            if not model:
                add("error", "Vision", "dedicated vision provider has no model — set the vision model, "
                    "it cannot be inferred from the brain model")
            elif not _pf_key_ok(sec.openai_compatible_vision_api_key, base):
                add("warn", "Vision", "dedicated vision endpoint has no API key — will fall back to the main LLM")

    # ---- Memory — build disables the whole feature on failure ----
    mem = getattr(cfg, "memory", None)
    if mem is not None and mem.enabled:
        if mem.extractor == "openai_compatible":
            prov = _resolve_provider(cfg, sec, "memory", ref=mem.provider_ref)
            base = (prov["base_url"] or "").strip() if prov["from"] != "legacy" else ""
            base = base or (mem.openai_compatible_base_url or "").strip()
            if not base:
                add("error", "Memory", "extractor is 'openai_compatible' but no endpoint is set "
                    "(API Keys → memory block, or Memory section) — capture will be disabled")
        elif _pf_dep_missing("openai"):
            add("warn", "Memory", "package 'openai' not installed — extraction will fail")

    # ---- Thoughts — build disables on failure ----
    th = getattr(cfg, "random_thoughts", None)
    if th is not None and th.enabled and th.generator == "openai_compatible":
        prov = _resolve_provider(cfg, sec, "thoughts", ref=th.provider_ref)
        base = (prov["base_url"] or "").strip() if prov["from"] != "legacy" else ""
        base = base or (th.openai_compatible_base_url or "").strip()
        if not base:
            add("error", "Thoughts", "generator is 'openai_compatible' but no endpoint is set "
                "(API Keys → thoughts block, or Memory → Thought generator)")

    return issues


def build_orchestrator(runtime: Optional[Runtime] = None) -> Orchestrator:
    runtime = runtime or get_runtime()
    cfg = runtime.config
    import os
    _configure_logger(os.getenv("LOG_LEVEL", "INFO"))

    persona = Persona.from_config(cfg.persona)

    def _effective_compat(category: str, sub_cfg, *, ref: str = "", allow_default: bool = True):
        """(cfg_copy, secrets_copy) with a provider block's endpoint/model/key
        applied when the subsystem runs on 'openai_compatible'. Legacy fields
        remain untouched when no block matches."""
        prov = _resolve_provider(cfg, runtime.secrets, category, ref=ref, allow_default=allow_default)
        if prov["from"] == "legacy" or not prov["base_url"]:
            return sub_cfg, runtime.secrets
        updates: dict = {"openai_compatible_base_url": prov["base_url"]}
        if prov["model"]:
            updates["openai_compatible_model"] = prov["model"]
            # The block's model is the directed model for dedicated blocks.
            if category != "llm":
                updates["model"] = prov["model"]
        sub_cfg = sub_cfg.model_copy(update=updates)
        key_field = {
            "llm": "openai_compatible_api_key",
            "vision": "openai_compatible_vision_api_key",
            "tts": "openai_compatible_tts_api_key",
            "stt": "openai_compatible_stt_api_key",
            "memory": "openai_compatible_memory_api_key",
            "thoughts": "openai_compatible_thoughts_api_key",
        }[category]
        secrets_eff = runtime.secrets.model_copy(update={key_field: prov["api_key"]})
        return sub_cfg, secrets_eff

    llm_cfg_eff, llm_secrets_eff = (
        _effective_compat("llm", cfg.llm, ref=cfg.llm.provider_ref)
        if cfg.llm.provider == "openai_compatible" else (cfg.llm, runtime.secrets)
    )
    llm = build_provider(llm_cfg_eff, llm_secrets_eff)
    tts_cfg_eff, tts_secrets_eff = (
        _effective_compat("tts", cfg.tts, ref=cfg.tts.provider_ref)
        if cfg.tts.provider == "openai_compatible" else (cfg.tts, runtime.secrets)
    )
    tts = build_tts(tts_cfg_eff, tts_secrets_eff)
    vision_llm = None
    if cfg.llm.vision_provider == "openai_compatible":
        try:
            # A dedicated vision block only applies when one actually exists
            # (no "default" hijack — vision already falls back to the main LLM).
            vis_cfg_eff, vis_secrets_eff = _effective_compat(
                "vision", cfg.llm, ref=cfg.llm.vision_provider_ref, allow_default=False
            )
            vision_llm = _build_vision_llm(vis_cfg_eff, vis_secrets_eff)
            logger.info(f"vision: dedicated model {vision_llm.model} (OpenAI-compatible)")
        except Exception as e:
            logger.error(f"vision: dedicated provider failed, falling back to main LLM: {e}")
    player = AudioPlayer(sample_rate=tts.sample_rate, channels=tts.channels,
                         device=(cfg.tts.output_device or None))

    chat_manager: Optional[ChatManager] = None
    
    if cfg.chat.youtube_enabled or cfg.chat.twitch_enabled or cfg.chat.kick_enabled:
        chat_manager = ChatManager(cfg.chat, runtime.secrets)

    donations_cfg = getattr(cfg, "donations", None)
    donation_queue: Optional[DonationQueue] = None
    streamlabs_monitor = None
    if donations_cfg is not None and (
        donations_cfg.livepix_enabled or donations_cfg.streamlabs_enabled
    ):
        # The donation bridge needs the chat queue. Chat is usually enabled
        # alongside donations (Twitch) — if it isn't, spin up the manager anyway
        # so donations have their queue (no chat monitors get started).
        if chat_manager is None:
            chat_manager = ChatManager(cfg.chat, runtime.secrets)
            logger.info("donations: chat manager instantiated to host the donation queue")
        donation_queue = DonationQueue(chat_manager.queue)

    vision_queue = None
    vision_loop = None
    if cfg.vision.enabled:
        if not cfg.llm.vision_capable:
            logger.warning("vision enabled but llm.vision_capable is False; disabling vision")
        else:
            try:
                from vision import VisionEvent, VisionLoop
            except ModuleNotFoundError as e:
                logger.error(
                    f"vision enabled but a dep is missing: {e}. "
                    "Install: pip install mss pillow imagehash"
                )
            else:
                vision_queue = asyncio.Queue(maxsize=4)
                vision_loop = VisionLoop(cfg.vision, vision_queue)

    # ----- dynamic provider resolution (API Keys page) -------------------
    # Subsystem configs may reference a named provider block (provider="my-block").
    # Resolution order for the endpoint settings:
    #   1. the named block (cfg.providers), if it exists;
    #   2. the first block with matching category;
    #   3. the block named "default" (category-agnostic);
    #   4. the legacy per-subsystem fields (previous behavior).
    def _resolve_provider_inner(category: str, ref: str = "", allow_default: bool = True) -> dict:
        return _resolve_provider(cfg, runtime.secrets, category, ref=ref, allow_default=allow_default)

    hearing_queue = None
    hearing_loop = None
    _speaker_id = None
    _speaker_store = None
    _enrollment_buffer = None
    if cfg.hearing.enabled:
        try:
            from hearing import HearingEvent, HearingLoop
        except ModuleNotFoundError as e:
            logger.error(
                f"hearing enabled but a dep is missing: {e}. "
                "Install: pip install soundcard faster-whisper"
            )
        else:
            hearing_queue = asyncio.Queue(maxsize=8)
            # Mute hearing for the capture window PLUS a tail margin: the ear grabs
            # the last `window_sec` of audio, so to be sure none of it contains Wallie's
            # own voice we also cover the playback that's still draining after the write.
            self_mute_window = cfg.hearing.window_sec + 2.5
            # Voice-print speaker ID (owner vs others). Enrollment buffer is created
            # whenever the feature is on so the dashboard can start/finish enrollment
            # live; the identifier itself only scores when there are prints enrolled.
            _sid_cfg = getattr(cfg.hearing, "speaker_id", None)
            _speaker_id = None
            _enrollment_buffer = None
            if _sid_cfg is not None and _sid_cfg.enabled:
                from hearing.speaker_id import EnrollmentBuffer, SpeakerIdentifier, SpeakerPrintStore
                _speaker_store = SpeakerPrintStore(PROFILES_DIR / f"{cfg.profile_name or 'default'}.speakers.json")
                _speaker_id = SpeakerIdentifier(
                    _speaker_store,
                    threshold=_sid_cfg.threshold,
                    unknown_threshold=_sid_cfg.unknown_threshold,
                    collect_other_voices=_sid_cfg.collect_other_voices,
                )
                _enrollment_buffer = EnrollmentBuffer()
                logger.info(
                    f"hearing: speaker ID on ({len(_speaker_store.names())} enrolled, "
                    f"threshold {_sid_cfg.threshold})"
                )
            hearing_loop = HearingLoop(
                cfg.hearing, hearing_queue,
                is_self_speaking=lambda: player.speaking_recently(self_mute_window),
                speaker_identifier=_speaker_id,
                enrollment_buffer=_enrollment_buffer,
            )
            if getattr(cfg.hearing, "engine", "") == "openai_compatible":
                prov = _resolve_provider_inner("stt", ref=getattr(cfg.hearing, "provider_ref", ""))
                try:
                    hearing_loop._stt_api_key = _compat_api_key(
                        prov["api_key"], prov["base_url"], "STT endpoint"
                    )
                except RuntimeError:
                    # Don't kill the whole build over a missing key — the
                    # transcription request will just 401 and be logged.
                    hearing_loop._stt_api_key = ""
                    logger.warning("hearing: remote STT endpoint has no API key set")
                hearing_loop._stt_base_url = prov["base_url"]
                logger.info(f"hearing: remote STT via OpenAI-compatible transcription API ({prov['from']})")
            logger.info("hearing: enabled (system-audio loopback + STT, self-muted while speaking)")

    avatar = None
    if cfg.avatar.enabled:
        try:
            from avatar import VTubeStudioAvatar
            avatar = VTubeStudioAvatar(cfg.avatar)
            asyncio.create_task(avatar.connect(), name="vts-avatar")
            logger.info(f"avatar: VTube Studio enabled ({cfg.avatar.vts_host}:{cfg.avatar.vts_port})")
        except Exception as e:
            logger.error(f"avatar: failed to start VTS client: {e}")

    profile_name = cfg.profile_name or "default"
    memory_path = PROFILES_DIR / f"{profile_name}.memory.json"
    memory_store = MemoryStore(memory_path)

    # ----- durable fact memory (short/long tier) + extractor + thought scheduler -----
    ltm = None
    memory_capture = None
    memory_consolidator = None
    thought_scheduler = None
    memory_cfg = getattr(cfg, "memory", None)
    thoughts_cfg = getattr(cfg, "random_thoughts", None)
    if memory_cfg is not None and memory_cfg.enabled:
        try:
            from core.long_term_memory import LongTermMemory
            from core.memory_capture import (
                MemoryCapture,
                MemoryConsolidator,
                ThoughtScheduler,
            )

            ltm = LongTermMemory(PROFILES_DIR / f"{profile_name}.longterm.json")
            ltm.load()

            extractor = None
            if memory_cfg.extractor == "openai_compatible":
                prov = _resolve_provider_inner("memory", ref=memory_cfg.provider_ref)
                base_url = (prov["base_url"] or "").strip()
                if not base_url:
                    raise RuntimeError(
                        "memory.extractor=openai_compatible: set the endpoint in "
                        "API Keys → provider block (or Memory → base URL), "
                        "e.g. http://localhost:11434/v1"
                    )
                from llm.openai_compat import OpenAICompatProvider
                # A named provider block's model wins; else the Memory field; else a safe default.
                mem_model = (
                    prov["model"] if prov["from"] != "legacy" and prov["model"]
                    else (memory_cfg.model or "").strip() or prov["model"] or "gpt-4o-mini"
                )
                extractor = OpenAICompatProvider(
                    name="openai_compatible_memory",
                    model=mem_model,
                    api_key=_compat_api_key(prov["api_key"], base_url, "memory extractor"),
                    base_url=base_url,
                    supports_vision=False,
                    timeout=memory_cfg.timeout,
                )
                logger.info(f"memory: extraction via OpenAI-compatible block ({mem_model}, {prov['from']})")
            elif memory_cfg.extractor == "main":
                extractor = llm  # reuse the brain LLM
                logger.info("memory: extraction via the main engine LLM")

            memory_capture = MemoryCapture(memory_cfg, ltm, extractor)
            memory_consolidator = MemoryConsolidator(memory_cfg, ltm, extractor)
            logger.info(
                f"memory: durable fact store loaded "
                f"({ltm.stats()['long_term']} long / {ltm.stats()['short_term']} short)"
            )
        except Exception as e:
            logger.error(f"memory: failed to initialize ({e}); feature disabled")
            ltm = None
            memory_capture = None
            memory_consolidator = None

    if thoughts_cfg is not None and thoughts_cfg.enabled:
        try:
            from core.memory_capture import ThoughtScheduler

            generator = None
            if thoughts_cfg.style in ("context", "random", "mix") and thoughts_cfg.generator != "off":
                if thoughts_cfg.generator == "openai_compatible":
                    from llm.openai_compat import OpenAICompatProvider
                    t_prov = _resolve_provider_inner("thoughts", ref=thoughts_cfg.provider_ref)
                    t_base = (t_prov["base_url"] or "").strip()
                    if not t_base:
                        raise RuntimeError(
                            "random_thoughts.generator=openai_compatible: "
                            "set the endpoint in API Keys → provider block (or Memory → Thought generator)"
                        )
                    th_model = (
                        t_prov["model"] if t_prov["from"] != "legacy" and t_prov["model"]
                        else (thoughts_cfg.model or "").strip() or t_prov["model"] or "gpt-4o-mini"
                    )
                    generator = OpenAICompatProvider(
                        name="openai_compatible_thoughts",
                        model=th_model,
                        api_key=_compat_api_key(t_prov["api_key"], t_base, "thought generator"),
                        base_url=t_base,
                        supports_vision=False,
                        timeout=thoughts_cfg.timeout,
                    )
                elif thoughts_cfg.generator == "main":
                    generator = llm
            thought_scheduler = ThoughtScheduler(
                thoughts_cfg, generator,
                ltm=(ltm if (ltm is not None and memory_cfg is not None and memory_cfg.enabled) else None),
            )
            thought_scheduler.reset()
            logger.info(
                f"thoughts: spontaneous thoughts on ({thoughts_cfg.schedule} schedule, "
                f"style={thoughts_cfg.style})"
            )
        except Exception as e:
            logger.error(f"thoughts: failed to initialize ({e}); feature disabled")
            thought_scheduler = None

    # Caption bridge — serves the browser-source overlay at /captions.
    caption_hub = None
    if getattr(cfg, "captions", None) is not None and cfg.captions.enabled:
        try:
            from captions import CaptionHub
            caption_hub = CaptionHub()
            logger.info(
                f"captions: overlay enabled at {cfg.captions.path} "
                "(open it as an OBS browser source)"
            )
        except Exception as e:
            logger.error(f"captions: failed to start bridge: {e}")

    orch = Orchestrator(
        runtime=runtime,
        persona=persona,
        llm=llm,
        tts=tts,
        player=player,
        chat_manager=chat_manager,
        vision_loop=vision_loop,
        vision_queue=vision_queue,
        avatar=avatar,
        memory_store=memory_store,
        hearing_loop=hearing_loop,
        hearing_queue=hearing_queue,
    )
    orch._vision_llm = vision_llm
    orch._captions = caption_hub
    orch._ltm = ltm
    orch._memory_capture = memory_capture
    orch._memory_consolidator = memory_consolidator
    orch._thoughts = thought_scheduler
    # Voice-print speaker ID (owner vs others in voice chat).
    orch._speaker_id = _speaker_id
    orch._speaker_store = _speaker_store
    orch._enrollment_buffer = _enrollment_buffer
    _setup_donations(runtime, chat_manager, orch)
    return orch


def _setup_donations(
    runtime: Runtime,
    chat_manager: Optional[ChatManager],
    orch: Orchestrator,
) -> None:
    """Create LivePix/Streamlabs integrations and bridge them into the SAME
    queue the orchestrator already drains. Never touches the AI pipeline."""
    cfg = runtime.config
    dcfg = getattr(cfg, "donations", None)
    if dcfg is None:
        return
    if not (dcfg.livepix_enabled or dcfg.streamlabs_enabled):
        return
    if chat_manager is None:
        logger.warning("donations: no chat queue available; donations disabled")
        return

    from donations.queue import DonationQueue

    bridge = DonationQueue(
        chat_manager.queue,
        reply_probability=getattr(dcfg, "donation_reply_probability", 1.0),
        cooldown_sec=getattr(dcfg, "donation_cooldown_sec", 0.0),
    )
    dedupe = DonationDedupe()

    async def _on_donations(events: list) -> None:
        await bridge.enqueue(events)

    if dcfg.streamlabs_enabled:
        try:
            from donations.streamlabs import StreamlabsMonitor
            monitor = StreamlabsMonitor(
                cfg=dcfg,
                on_donations=_on_donations,
                access_token=runtime.secrets.streamlabs_access_token,
                socket_token=runtime.secrets.streamlabs_socket_token,
                dedupe=dedupe,
            )
            orch._streamlabs_monitor = monitor
            logger.info("donations: Streamlabs monitor created (starts with the orchestrator)")
        except Exception as e:
            logger.error(f"donations: Streamlabs setup failed: {e}")

    if dcfg.livepix_enabled:
        # LivePix is a webhook: it needs the HTTP app. When running with the
        # dashboard, attach_donations_to_app() mounts the webhook on that app.
        logger.info(
            f"donations: LivePix webhook will be served at {dcfg.livepix_webhook_path} "
            "(mounted on the dashboard app)"
        )


async def run_cli() -> None:
    orch = build_orchestrator()
    await orch.start()
    try:
        # Block forever; the orchestrator's main task handles everything.
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await orch.stop()


async def run_with_dashboard() -> None:
    from dashboard.server import serve

    await serve(None)


def attach_donations_to_app(app, orch: Orchestrator) -> None:
    """Mount the LivePix webhook on a FastAPI app for a running orchestrator.
    Called by the dashboard after it builds its app (idempotent per path)."""
    from config import Secrets
    from donations.livepix import LivePixClient, mount_livepix_webhook

    cfg = orch._runtime.config
    dcfg = getattr(cfg, "donations", None)
    if dcfg is None or not dcfg.livepix_enabled:
        return
    # Guard: restarting the orchestrator must not stack duplicate routes.
    try:
        if getattr(app.state, "_livepix_mounted_path", None) == dcfg.livepix_webhook_path:
            return
    except Exception:
        pass
    client_id = orch._runtime.secrets.livepix_client_id
    client_secret = orch._runtime.secrets.livepix_client_secret
    enrich_client: Optional[LivePixClient] = None
    if dcfg.livepix_enrich and client_id and client_secret:
        enrich_client = LivePixClient(client_id=client_id, client_secret=client_secret)
    elif dcfg.livepix_enrich:
        logger.warning(
            "[LivePix] enrichment enabled but LIVEPIX_CLIENT_ID/SECRET missing — "
            "donations will be queued without amount/message details"
        )
    mount_livepix_webhook(
        app,
        on_donation=_queue_single_donation(orch),
        webhook_path=dcfg.livepix_webhook_path,
        expected_user_id=orch._runtime.secrets.livepix_user_id,
        verify_user_id=dcfg.livepix_verify_user_id,
        enrich_client=enrich_client,
    )
    try:
        app.state._livepix_mounted_path = dcfg.livepix_webhook_path
    except Exception:
        pass
    logger.info(f"[LivePix] webhook mounted at {dcfg.livepix_webhook_path}")


def _queue_single_donation(orch: Orchestrator):
    """Async callable that validates idempotency and queues one DonationEvent
    into the orchestrator's existing chat-queue machinery."""
    from donations.queue import DonationQueue

    async def _handle(event: DonationEvent) -> None:
        chat_manager = orch._chat
        if chat_manager is None:
            logger.warning("[LivePix] donation received but no queue is available; dropping")
            return
        if not orch._donation_dedupe.first_time(event.event_id):
            logger.info("[LivePix] duplicate donation suppressed (already processed)")
            return
        bridge = DonationQueue(chat_manager.queue)
        await bridge.enqueue([event])
        logger.info("[LivePix] Donation queued")

    return _handle


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="wallie", description="AI streamer runtime")
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the web dashboard instead of the headless loop",
    )
    args = ap.parse_args()

    try:
        if args.dashboard:
            asyncio.run(run_with_dashboard())
        else:
            asyncio.run(run_cli())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

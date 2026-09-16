"""Entrypoint — builds all subsystems from config and starts the pipeline."""
from __future__ import annotations

import asyncio
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


def build_orchestrator(runtime: Optional[Runtime] = None) -> Orchestrator:
    runtime = runtime or get_runtime()
    cfg = runtime.config
    import os
    _configure_logger(os.getenv("LOG_LEVEL", "INFO"))

    persona = Persona.from_config(cfg.persona)
    llm = build_provider(cfg.llm, runtime.secrets)
    tts = build_tts(cfg.tts, runtime.secrets)
    vision_llm = None
    if cfg.llm.vision_provider == "openai_compatible":
        try:
            vision_llm = _build_vision_llm(cfg, runtime.secrets)
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

    hearing_queue = None
    hearing_loop = None
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
            hearing_loop = HearingLoop(
                cfg.hearing, hearing_queue,
                is_self_speaking=lambda: player.speaking_recently(self_mute_window),
            )
            if getattr(cfg.hearing, "engine", "") == "openai_compatible":
                hearing_loop._stt_api_key = runtime.secrets.openai_compatible_stt_api_key
                hearing_loop._stt_base_url = cfg.hearing.openai_compatible_base_url
                logger.info("hearing: remote STT via OpenAI-compatible transcription API")
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
                base_url = (memory_cfg.openai_compatible_base_url or "").strip()
                if not base_url:
                    raise RuntimeError(
                        "memory.extractor=openai_compatible: set the base URL in "
                        "Memory → Extraction model (e.g. http://localhost:11434/v1)"
                    )
                from llm.openai_compat import OpenAICompatProvider
                extractor = OpenAICompatProvider(
                    name="openai_compatible_memory",
                    model=memory_cfg.model or "gpt-4o-mini",
                    api_key=_compat_api_key(
                        runtime.secrets.openai_compatible_memory_api_key,
                        base_url, "memory extractor",
                    ),
                    base_url=base_url,
                    supports_vision=False,
                    timeout=memory_cfg.timeout,
                )
                logger.info(f"memory: extraction via OpenAI-compatible block ({memory_cfg.model})")
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
                    t_base = (thoughts_cfg.openai_compatible_base_url or "").strip()
                    if not t_base:
                        raise RuntimeError(
                            "random_thoughts.generator=openai_compatible: "
                            "set the base URL in Memory → Thought generator"
                        )
                    generator = OpenAICompatProvider(
                        name="openai_compatible_thoughts",
                        model=thoughts_cfg.model or "gpt-4o-mini",
                        api_key=_compat_api_key(
                            runtime.secrets.openai_compatible_thoughts_api_key,
                            t_base, "thought generator",
                        ),
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

"""Periodic hearing loop — captures system audio windows, transcribes speech,
measures loudness, and emits HearingEvents.

Mirrors vision.vision_loop: a background task that turns a raw stream (audio)
into discrete, meaningful events the orchestrator can fuse with vision. Silence
is skipped so Wallie only "hears" when there's something to hear.
"""
from __future__ import annotations

import asyncio
import glob
import os
import site
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger

from .capture import SystemAudioCapture

# Phrases Whisper notoriously hallucinates over silence / noise / a TV in the
# background. When the WHOLE transcript is just one of these, it's almost certainly
# not real speech aimed at Wallie — drop it so messy rooms don't trigger replies.
_HALLUCINATIONS = frozenset({
    "thank you", "thanks", "thank you very much", "thank you so much",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "please subscribe", "subscribe to my channel", "like and subscribe",
    "see you next time", "i'll see you next time", "see you in the next video",
    "bye bye", "you", "okay", "so", "the end",
})


def _is_hallucination(text: str) -> bool:
    t = text.strip().lower().strip(" .!?,-…")
    return t in _HALLUCINATIONS


def _register_cuda_dll_dirs() -> None:
    """Put the pip-installed NVIDIA cuBLAS/cuDNN DLLs on Windows' DLL search path so
    CTranslate2 (faster-whisper) can load them. The wheels drop their DLLs under
    site-packages/nvidia/*/bin, which isn't searched by default — without this the
    GPU path dies at first inference with 'cublas64_12.dll not found'."""
    if not hasattr(os, "add_dll_directory"):
        return
    roots = list(site.getsitepackages())
    sp = getattr(site, "getusersitepackages", lambda: None)()
    if sp:
        roots.append(sp)
    for root in roots:
        for bindir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bindir)
            except OSError:
                pass


def _cudnn_present() -> bool:
    """True only if the cuDNN DLLs are actually installed. CTranslate2's CUDA path
    needs them; attempting CUDA without cuDNN can HANG model load, so we gate on this."""
    roots = list(site.getsitepackages())
    sp = getattr(site, "getusersitepackages", lambda: None)()
    if sp:
        roots.append(sp)
    for root in roots:
        if glob.glob(os.path.join(root, "nvidia", "cudnn", "bin", "cudnn*.dll")):
            return True
    return False


@dataclass
class HearingEvent:
    transcript: str            # what was said (may be "" for non-speech sound)
    loudness: float            # RMS 0..1 of the window
    has_speech: bool
    sound_type: str = "speech"  # speech | music | sound | quiet
    descriptor: str = ""        # for music/sound: e.g. "upbeat, energetic, bright"
    # Numeric musical mood (0 when not music) — lets the MoodEngine FEEL the music,
    # not just read a word in the prompt. valence -1..1, the rest 0..1.
    is_music: bool = False
    music_valence: float = 0.0
    music_arousal: float = 0.0
    music_energy: float = 0.0
    captured_at: float = field(default_factory=time.time)


class HearingLoop:
    def __init__(self, cfg, out_queue: "asyncio.Queue[HearingEvent]",
                 is_self_speaking: Optional[Callable[[], bool]] = None,
                 is_self_echo: Optional[Callable[[str], bool]] = None) -> None:
        self._cfg = cfg
        self._queue = out_queue
        self._capture = SystemAudioCapture(samplerate=16000)
        self._model = None
        # Remote STT (OpenAI-compatible) credentials, injected by wallie.build_orchestrator.
        self._stt_api_key: str = ""
        self._stt_base_url: str = ""
        self._task: Optional[asyncio.Task] = None
        # Returns True if Wallie is currently speaking — those windows are skipped
        # so Wallie never transcribes/reacts to its own TTS (timing-based guard).
        self._is_self_speaking = is_self_speaking
        # Returns True if a transcript matches something Wallie recently SAID — the
        # definitive self-echo guard, immune to capture lag (content, not timing).
        self._is_self_echo = is_self_echo

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="hearing-loop")
            logger.info(
                "hearing: loop started (window={}s, model={}, silence<{})".format(
                    self._cfg.window_sec, self._cfg.model_size, self._cfg.silence_threshold
                )
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._capture.close()

    # --- internal ---
    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            self._model = await loop.run_in_executor(None, self._load_model)
            self._capture.open()
            if getattr(self._cfg, "vad_segmentation", False):
                await self._run_vad(loop)
                return
            window = self._cfg.window_sec
            silence = self._cfg.silence_threshold
            from .audio_analysis import analyze_music, analyze_window
            # Let the ring buffer fill once before the first read.
            await asyncio.sleep(window)
            while True:
                # The capture thread drains the device non-stop, so we just grab the
                # most recent window — always current, never a stale/overflowed chunk.
                audio = self._capture.latest(window)
                if self._is_self_speaking is not None and self._is_self_speaking():
                    await asyncio.sleep(window)
                    continue  # Wallie was speaking — don't hear ourselves
                rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
                if rms < silence:
                    await asyncio.sleep(window)
                    continue  # nothing worth hearing
                text = await loop.run_in_executor(None, self._transcribe, audio)
                # Guard against Whisper hallucinating a stray word over music/noise.
                has_speech = bool(text) and len(text.split()) >= 2
                # Definitive self-echo guard: if what we "heard" matches something
                # Wallie just said, it's our own voice bleeding through the loopback —
                # drop it. Content-based, so it works even if capture lags seconds behind.
                if has_speech and self._is_self_echo is not None and self._is_self_echo(text):
                    logger.debug(f"hearing: muted self-echo — {text[:60]!r}")
                    await asyncio.sleep(window)
                    continue
                # Analyze the MUSICAL character over a longer slice (more stable tempo/key)
                # than the STT window, pulled straight from the rolling buffer.
                music_audio = self._capture.latest(min(window * 1.8, 9.0))
                sound_type, descriptor = analyze_window(
                    music_audio, 16000, has_speech=has_speech, silence=silence
                )
                if sound_type == "quiet":
                    await asyncio.sleep(window)
                    continue
                # Pull the numeric musical mood for anything musical — a song with
                # lyrics (→ speech) OR an instrumental (→ music). This is what lets the
                # MoodEngine FEEL the track, not just read a word in the prompt.
                feats = None
                if sound_type in ("speech", "music"):
                    feats = analyze_music(music_audio, 16000)
                vibe = feats.descriptor if (sound_type == "speech" and feats) else ""
                is_music_ev = feats is not None and (sound_type == "music" or bool(vibe))
                ev = HearingEvent(
                    transcript=text if has_speech else "",
                    loudness=rms, has_speech=has_speech,
                    sound_type=sound_type,
                    descriptor=(vibe if sound_type == "speech" else descriptor),
                    is_music=is_music_ev,
                    music_valence=(feats.valence if feats else 0.0),
                    music_arousal=(feats.arousal if feats else 0.0),
                    music_energy=(feats.energy if feats else 0.0),
                )
                self._enqueue(ev)
                if sound_type == "speech":
                    tag = f" ♪({vibe})" if vibe else ""
                    logger.info(f"hear> [{rms:.2f}]{tag} {text[:90]}")
                elif sound_type == "music":
                    logger.info(f"hear> [{rms:.2f}] ♪ music: {descriptor}")
                else:
                    logger.info(f"hear> [{rms:.2f}] (sound: {descriptor})")
                await asyncio.sleep(window)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"hearing: loop crashed: {e}")

    async def _run_vad(self, loop) -> None:
        """Utterance-based hearing for live two-way conversation.

        Polls voice activity frequently; when the speaker pauses (or hits the cap),
        transcribes the WHOLE utterance at once and emits it. Result: replies fire
        right after the person stops talking, and full sentences aren't truncated.
        Speech-only (skips the music-character analysis the windowed path does).
        """
        silence = self._cfg.silence_threshold
        poll = max(0.1, self._cfg.poll_interval_sec)
        end_sil = max(0.2, self._cfg.end_silence_sec)
        max_utt = max(2.0, self._cfg.max_utterance_sec)
        probe = max(0.2, poll * 1.5)  # recent slice used for the voice/silence check
        await asyncio.sleep(0.5)

        speech_active = False
        speech_start = 0.0
        last_voice = 0.0
        while True:
            now = time.time()
            # While Wallie is talking, don't capture our own voice as an utterance.
            if self._is_self_speaking is not None and self._is_self_speaking():
                speech_active = False
                await asyncio.sleep(poll)
                continue

            recent = self._capture.latest(probe)
            rms = float(np.sqrt(np.mean(recent ** 2))) if recent.size else 0.0
            voiced = rms >= silence
            if voiced:
                if not speech_active:
                    speech_active = True
                    speech_start = now
                last_voice = now

            should_flush = False
            if speech_active:
                if now - speech_start >= max_utt:
                    should_flush = True                      # safety cap on long talkers
                elif not voiced and now - last_voice >= end_sil:
                    should_flush = True                      # natural end-of-utterance pause

            if not should_flush:
                await asyncio.sleep(poll)
                continue

            # Grab from a touch before speech onset through to now (the trailing pause
            # is harmless — Whisper's VAD trims it), capped to the ring buffer length.
            grab = min(now - speech_start + 0.5, max_utt + 0.5, 9.5)
            audio = self._capture.latest(grab)
            speech_active = False

            text = await loop.run_in_executor(None, self._transcribe, audio)
            has_speech = bool(text) and len(text.split()) >= 2
            if not has_speech:
                await asyncio.sleep(poll)
                continue
            if self._is_self_echo is not None and self._is_self_echo(text):
                logger.debug(f"hearing: muted self-echo — {text[:60]!r}")
                await asyncio.sleep(poll)
                continue

            self._enqueue(HearingEvent(
                transcript=text, loudness=rms, has_speech=True,
                sound_type="speech", descriptor="",
            ))
            logger.info(f"hear> [{rms:.2f}] {text[:90]}")
            await asyncio.sleep(poll)

    def _load_model(self):
        """Load the STT engine — remote (OpenAI-compatible transcription API) when
        ``hearing.engine == "openai_compatible"``, else local faster-whisper with the
        best available device.

        The local GPU path is *probed* with a tiny inference: CTranslate2 builds the
        model handle lazily, so a missing cublas/cudnn DLL only blows up on the first
        real transcribe. We trigger that here and fall back to CPU instead of crashing
        the loop mid-session."""
        if getattr(self._cfg, "engine", "") == "openai_compatible":
            from .openai_stt import RemoteWhisperModel
            return RemoteWhisperModel(
                api_key=self._stt_api_key,
                base_url=self._stt_base_url,
                model=getattr(self._cfg, "openai_compatible_model", "whisper-1"),
                language=getattr(self._cfg, "language", ""),
                prompt=getattr(self._cfg, "openai_compatible_prompt", ""),
                timeout=getattr(self._cfg, "openai_compatible_timeout", 20.0),
            )

        from faster_whisper import WhisperModel
        size = self._cfg.model_size
        _register_cuda_dll_dirs()

        # Only attempt CUDA when the GPU AND cuDNN are both genuinely available —
        # a half-installed CUDA stack can HANG model load and silently kill hearing.
        use_cuda = False
        if _cudnn_present():
            try:
                import ctranslate2
                use_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                use_cuda = False

        if use_cuda:
            try:
                m = WhisperModel(size, device="cuda", compute_type="float16")
                list(m.transcribe(np.zeros(16000, dtype="float32"))[0])  # force CUDA init
                logger.info(f"hearing: Whisper '{size}' on CUDA (float16)")
                return m
            except Exception as e:
                logger.warning(f"hearing: CUDA load failed ({str(e)[:80]}) — using CPU")

        m = WhisperModel(size, device="cpu", compute_type="int8")
        logger.info(f"hearing: Whisper '{size}' on CPU (int8)")
        return m

    def _transcribe(self, audio: "np.ndarray") -> str:
        lang = self._cfg.language or None
        # Sung vocals sit low under the instrumental — lift the window toward full
        # scale so Whisper gets a strong signal instead of a faint mumble.
        if audio.size:
            peak = float(np.abs(audio).max())
            if 0.0 < peak < 0.9:
                audio = (audio * (0.9 / peak)).astype("float32")
        if getattr(self._cfg, "low_latency", False):
            # Live conversation: greedy decode is much faster than beam=5/best_of=5,
            # and on clear spoken voice the accuracy cost is negligible.
            segs, _info = self._model.transcribe(
                audio, language=lang,
                beam_size=1, temperature=0.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, threshold=0.4),
            )
        else:
            segs, _info = self._model.transcribe(
                audio, language=lang,
                beam_size=5, best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6],   # fall back to softer decoding if stuck
                # Each window is independent — don't carry prior text, which makes Whisper
                # loop/hallucinate lyrics over music. Big accuracy win for songs.
                condition_on_previous_text=False,
                # Drop hallucinated garbage that music/noise provokes.
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, threshold=0.4),
            )
        return " ".join(s.text.strip() for s in segs).strip()

    def _enqueue(self, ev: HearingEvent) -> None:
        try:
            self._queue.put_nowait(ev)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(ev)
            except asyncio.QueueFull:
                pass

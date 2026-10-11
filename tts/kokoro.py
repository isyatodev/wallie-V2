"""Kokoro TTS — local, high-quality neural TTS (free, CPU or GPU).

A modern open-weight voice that sounds far better than Piper while still running
locally with no API key. Output is 24 kHz mono PCM16, streamed per segment.

Speed notes (why this file looks the way it does):
  * The pipeline (model weights, ~300 MB) is cached per ``(lang_code, device)``
    at module level. Building a fresh ``KPipeline`` used to re-load the model
    for every preview / A-B take / session, which is the single biggest cost.
  * A voice downloaded into ``voices/kokoro/`` is loaded from disk and passed
    to the pipeline directly, so it never re-downloads and works offline.
  * Synthesis runs on a worker thread and each chunk is handed to the caller as
    soon as it is produced, instead of computing the whole segment first —
    that is the time-to-first-audio win the streamer actually feels.
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import AsyncIterator

from .base import TTSError, TTSProvider

_SAMPLE_RATE = 24000  # Kokoro generates at 24 kHz

# Loading the model costs seconds and a few hundred MB. A live session, the
# Voice page's ▶ Play preview and each A/B take all build a KokoroTTS, so the
# pipeline is cached per (lang_code, device) and reused instead of rebuilt.
_PIPELINES: dict[tuple[str, str], object] = {}
_PIPELINE_LOCK = threading.Lock()

# One worker thread per in-flight synthesis; chunks queue up between it and the
# async consumer. Small bound so a slow consumer cannot balloon memory.
_QUEUE_MAX = 64


def _local_voice_path(voice: str) -> str:
    """Path to a downloaded voice in ``voices/kokoro/``, or "" when not on disk."""
    try:
        from .kokoro_setup import is_valid_voice_name, voice_path
    except Exception:  # pragma: no cover - defensive (setup module is optional here)
        return ""
    name = (voice or "").strip()
    if not is_valid_voice_name(name):
        return ""
    path = voice_path(name)
    return str(path) if path.is_file() else ""


def _get_pipeline(lang_code: str, device: str = ""):
    """Cached ``KPipeline`` for a language (and optional torch device)."""
    key = (lang_code or "a", device or "")
    pipe = _PIPELINES.get(key)
    if pipe is not None:
        return pipe
    with _PIPELINE_LOCK:
        pipe = _PIPELINES.get(key)
        if pipe is not None:
            return pipe
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise TTSError(
                "kokoro: not installed. Install with: pip install kokoro soundfile"
            ) from e
        kwargs: dict = {"lang_code": key[0]}
        if key[1]:
            kwargs["device"] = key[1]
        try:
            pipe = KPipeline(**kwargs)
        except Exception as e:
            raise TTSError(
                f"kokoro: failed to init pipeline (lang_code={key[0]!r}): {e}"
            ) from e
        _PIPELINES[key] = pipe
        return pipe


def clear_pipeline_cache() -> None:
    """Drop cached pipelines (tests, or after a device/language change)."""
    with _PIPELINE_LOCK:
        _PIPELINES.clear()


class KokoroTTS(TTSProvider):
    name = "kokoro"

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
        device: str = "",
    ) -> None:
        self._pipeline = _get_pipeline(lang_code, device)
        self._voice = voice or "af_heart"
        self._speed = float(speed) if speed else 1.0
        self._voice_path = _local_voice_path(self._voice)
        self.sample_rate = _SAMPLE_RATE
        self.channels = 1

    def _voice_arg(self):
        """Voice passed to the pipeline: a downloaded file when we have one
        (offline, no re-download), else the voice id (kokoro fetches/caches it)."""
        path = self._voice_path
        if path and os.path.isfile(path):
            try:
                import torch  # type: ignore

                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                # Torch missing/old, or a non-tensor payload: kokoro can also
                # resolve a plain path string.
                return path
        return self._voice

    def _iter_pcm(self, text: str):
        """Yield PCM16 chunks for ``text`` from the kokoro generator."""
        import numpy as np

        for result in self._pipeline(text, voice=self._voice_arg(), speed=self._speed):
            # KPipeline yields (graphemes, phonemes, audio); audio is a float
            # tensor/array in [-1, 1]. Support tuple or object forms.
            audio = (
                result[2]
                if isinstance(result, (tuple, list))
                else getattr(result, "audio", None)
            )
            if audio is None:
                continue
            arr = np.asarray(audio, dtype=np.float32)
            if arr.size == 0:
                continue
            pcm = np.clip(arr, -1.0, 1.0)
            yield (pcm * 32767.0).astype(np.int16).tobytes()

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        loop = asyncio.get_event_loop()
        chunks: "queue.Queue[object]" = queue.Queue(maxsize=_QUEUE_MAX)
        sentinel = object()

        def _worker() -> None:
            try:
                for chunk in self._iter_pcm(text):
                    chunks.put(chunk)
            except Exception as e:  # noqa: BLE001 - forwarded to the consumer
                chunks.put(e)
            finally:
                chunks.put(sentinel)

        threading.Thread(target=_worker, name="kokoro-synth", daemon=True).start()
        while True:
            item = await loop.run_in_executor(None, chunks.get)
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise TTSError(f"kokoro: synthesis failed: {item}") from item
            if item:
                yield item  # type: ignore[misc]

    async def aclose(self) -> None:
        # The pipeline itself stays in the module cache: another provider
        # instance (or the next segment) reuses it instead of re-loading 300 MB.
        self._pipeline = None  # type: ignore

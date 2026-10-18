"""Piper TTS — local CPU-based neural TTS adapter."""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, AsyncIterator

from .base import TTSError, TTSProvider


def _params(fn: Any) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):   # builtins / C callables
        return set()


class PiperTTS(TTSProvider):
    name = "piper"

    def __init__(
        self,
        *,
        model_path: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        if not model_path:
            raise TTSError("piper: model_path is required (download a .onnx voice file)")
        path = Path(model_path)
        if not path.is_file():
            raise TTSError(f"piper: model file not found: {path}")

        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise TTSError(
                "piper: 'piper-tts' is not installed. Install with: pip install piper-tts"
            ) from e

        try:
            self._voice = PiperVoice.load(str(path))
        except Exception as e:
            raise TTSError(f"piper: failed to load model {path.name}: {e}") from e

        # Sample rate from voice config, default if missing.
        sr = getattr(getattr(self._voice, "config", None), "sample_rate", None)
        self.sample_rate = int(sr) if sr else 22050
        self.channels = 1
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w = noise_w

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._synthesize_blocking, text)
        for c in chunks:
            if c:
                yield c

    def _synthesize_blocking(self, text: str) -> list[bytes]:
        out: list[bytes] = []
        try:
            for chunk in self._start_synthesis(text):
                buf = self._pcm(chunk)
                if buf:
                    out.append(buf)
            return out
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"piper: synthesis failed: {e}") from e

    # -- piper-tts API dispatch --------------------------------------------
    # piper-tts 1.3+ (piper1-gpl) moved tuning into a SynthesisConfig and
    # dropped `synthesize_stream_raw`; 1.2.x still has the kwargs/stream API.
    # Inspect the installed build instead of calling both blind: the old
    # TypeError-then-retry chain hid the real failure and then blamed the
    # missing legacy method — "'PiperVoice' object has no attribute
    # 'synthesize_stream_raw'" on every synthesis with piper-tts >= 1.3.
    def _start_synthesis(self, text: str) -> Any:
        synthesize = getattr(self._voice, "synthesize", None)
        if synthesize is not None and "syn_config" in _params(synthesize):
            return synthesize(text, self._synthesis_config())
        legacy = getattr(self._voice, "synthesize_stream_raw", None)
        if legacy is not None:
            return legacy(text, **self._knobs_for(legacy))
        if synthesize is None:
            raise TTSError("piper: this piper-tts build exposes no synthesize()")
        # Unknown layout: pass only the tuning kwargs it accepts, loudly.
        return synthesize(text, **self._knobs_for(synthesize))

    def _knobs_for(self, fn: Any) -> dict[str, Any]:
        """The tuning knobs `fn` actually accepts — 1.8 renamed `noise_w`
        to `noise_w_scale`, so filter by real parameter names."""
        params = _params(fn)
        knobs: dict[str, Any] = {}
        if "length_scale" in params:
            knobs["length_scale"] = self._length_scale
        if "noise_scale" in params:
            knobs["noise_scale"] = self._noise_scale
        if "noise_w_scale" in params:
            knobs["noise_w_scale"] = self._noise_w
        elif "noise_w" in params:
            knobs["noise_w"] = self._noise_w
        return knobs

    def _synthesis_config(self) -> Any:
        try:
            from piper.config import SynthesisConfig
        except ImportError as e:
            raise TTSError(
                "piper: this piper-tts wants a syn_config but piper.config "
                "could not be imported"
            ) from e
        return SynthesisConfig(**self._knobs_for(SynthesisConfig))

    @staticmethod
    def _pcm(chunk: Any) -> bytes:
        """Little-endian int16 mono PCM from one chunk, whatever shape the
        installed piper-tts hands back (raw bytes, AudioChunk, float array)."""
        if isinstance(chunk, (bytes, bytearray)):
            return bytes(chunk)
        buf = getattr(chunk, "audio_int16_bytes", None)
        if callable(buf):
            buf = buf()
        if isinstance(buf, (bytes, bytearray)) and buf:
            return bytes(buf)
        arr = getattr(chunk, "audio_float_array", None)
        if arr is None:
            return b""
        import numpy as np
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return b""
        return (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    async def aclose(self) -> None:
        self._voice = None  # type: ignore

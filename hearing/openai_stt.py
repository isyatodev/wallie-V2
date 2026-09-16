"""Remote STT engine — OpenAI-compatible /audio/transcriptions backend.

A drop-in replacement for the local faster-whisper model used by HearingLoop.
Accepts a numpy float32 PCM frame (16 kHz mono, exactly what the capture
already produces), encodes it to WAV in-memory, and POSTs it to any endpoint
implementing the OpenAI transcription API (OpenAI, Groq, local
faster-whisper servers, speaches, …).

The returned object mirrors the tiny slice of the Whisper API HearingLoop
relies on (``transcribe(audio) -> (segments, info)`` with ``seg.text``), so the
loop needs zero special-casing: swap the engine, keep every hallucination
guard, VAD path and self-echo filter.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger


@dataclass
class _Segment:
    text: str


class RemoteWhisperModel:
    """Mimics faster_whisper.WhisperModel over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "whisper-1",
        language: str = "",
        prompt: str = "",
        timeout: float = 20.0,
        samplerate: int = 16000,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "openai_compatible (stt): missing API key — set OPENAI_COMPATIBLE_STT_API_KEY"
            )
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError(
                'openai_compatible (stt): set hearing.openai_compatible_base_url '
                '(e.g. "https://api.groq.com/openai/v1") — it must include the version path'
            )
        self._model = (model or "whisper-1").strip()
        self._language = (language or "").strip()
        self._prompt = (prompt or "").strip()
        self._samplerate = samplerate
        self._base_url = base
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(max(timeout, 20.0), connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        logger.info(f"hearing: remote STT endpoint {base} (model={self._model})")

    # HearingLoop calls this from a worker thread via run_in_executor — httpx's
    # sync API is used intentionally here (same as faster-whisper's blocking call).
    def transcribe(self, audio: "np.ndarray", **_kwargs: Any) -> tuple[list[_Segment], Any]:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._samplerate)
            wf.writeframes(pcm16.tobytes())
        wav_bytes = buf.getvalue()

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: dict[str, str] = {"model": self._model, "response_format": "json"}
        if self._language:
            data["language"] = self._language
        if self._prompt:
            data["prompt"] = self._prompt

        resp = self._client.post(f"{self._base_url}/audio/transcriptions",
                                 files=files, data=data)
        if resp.status_code >= 400:
            raise RuntimeError(f"openai_compatible (stt) {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        text = (payload.get("text") or "").strip()
        # A remote engine returns one blob; expose it as a single segment so
        # HearingLoop's `for s in segs` join keeps working unchanged.
        return ([_Segment(text)] if text else []), {"language": payload.get("language", "")}

    # Parity with faster-whisper so any optional feature flags stay harmless.
    @property
    def supports_word_timestamps(self) -> bool:
        return False


class LocalWhisperModel:
    """Thin adapter for the existing local engine (kept for engine='' path)."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def transcribe(self, audio: "np.ndarray", **kwargs: Any) -> tuple[list[Any], Any]:
        return self._model.transcribe(audio, **kwargs)

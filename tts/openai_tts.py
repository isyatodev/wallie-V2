"""OpenAI-compatible TTS adapter — POST {base_url}/audio/speech, PCM16 output.

Works against OpenAI itself, self-hosted gateways (Kokoro-FastAPI,
openedai-speech, speaches,…) and aggregators that implement the same
speech API. The voice, model, speed and sample rate are all user-configured;
the API key lives in Secrets (.env), never in the profile.

Like every provider here it returns raw PCM16 mono chunks — the orchestrator
already guards against non-PCM payloads, so a misconfigured endpoint that
answers MP3 is detected and aborted instead of playing static.
"""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from .base import TTSError, TTSProvider
from .decode import decode_to_pcm16, find_ffmpeg, sniff_compressed


class OpenAICompatibleTTS(TTSProvider):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        voice: str = "alloy",
        speed: float = 1.0,
        sample_rate: int = 24000,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise TTSError(
                "openai_compatible (tts): missing API key — set OPENAI_COMPATIBLE_TTS_API_KEY"
            )
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise TTSError(
                'openai_compatible (tts): set tts.openai_compatible_base_url in the profile '
                '(e.g. "https://api.openai.com/v1" or your own gateway) — it must include '
                "the version path"
            )
        if not (model or "").strip():
            raise TTSError("openai_compatible (tts): tts.openai_compatible_model is required")
        self.sample_rate = sample_rate
        self.channels = 1
        self._base_url = base
        self._model = model.strip()
        self._voice = voice or "alloy"
        self._speed = max(0.25, min(4.0, float(speed or 1.0)))
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(max(timeout, 30.0), connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        payload = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            # Ask for raw PCM; endpoints that ignore this (some free gateways
            # always answer MP3) are detected below and decoded via ffmpeg.
            "response_format": "pcm",
            "speed": self._speed,
        }
        try:
            # Single pass over the stream: the first chunk decides between the
            # raw-PCM streaming path and the buffer+decode fallback (httpx
            # forbids restarting aiter_bytes, so no second pass is possible).
            async with self._client.stream(
                "POST", f"{self._base_url}/audio/speech", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise TTSError(f"openai_compatible (tts) {resp.status_code}: {body[:200]!r}")
                first_seen = False
                compressed = False
                parts: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    if not first_seen:
                        first_seen = True
                        kind = sniff_compressed(chunk[:12])
                        if kind is None:
                            pass  # genuine raw PCM — stream through, chunk by chunk
                        else:
                            # Gateway ignored response_format=pcm: buffer the
                            # whole payload and decode. Without ffmpeg this
                            # fails LOUD and actionable below — never static.
                            compressed = True
                    if compressed:
                        parts.append(chunk)
                    else:
                        yield chunk
                if compressed:
                    blob = b"".join(parts)
                    kind = sniff_compressed(blob[:12]) or "audio"
                    ffmpeg = find_ffmpeg()
                    if not ffmpeg:
                        raise TTSError(
                            f"openai_compatible (tts): endpoint returned {kind} instead "
                            "of the requested raw PCM. Install ffmpeg (e.g. `winget "
                            "install ffmpeg`, or `pip install imageio-ffmpeg`), or set "
                            "WALLIE_FFMPEG to the binary path, to enable automatic "
                            "decoding."
                        )
                    try:
                        pcm = await decode_to_pcm16(blob, ffmpeg, self.sample_rate)
                    except RuntimeError as e:
                        raise TTSError(f"openai_compatible (tts): {e}") from e
                    if pcm:
                        yield pcm
        except httpx.HTTPError as e:
            raise TTSError(f"openai_compatible (tts) network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

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
            # Ask for raw PCM; the endpoint's default PCM rate is applied server-side.
            # Wallie re-chunks whatever arrives — sample-rate mismatches surface as
            # pitch-shifted audio, so the profile rate should match the gateway's.
            "response_format": "pcm",
            "speed": self._speed,
        }
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/audio/speech", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise TTSError(f"openai_compatible (tts) {resp.status_code}: {body[:200]!r}")
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            raise TTSError(f"openai_compatible (tts) network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

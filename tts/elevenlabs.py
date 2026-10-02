"""ElevenLabs TTS adapter (streaming PCM)."""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from .base import TTSError, TTSProvider

_ALLOWED_SR = {16000, 22050, 24000, 44100}


def _pcm_format(sr: int) -> str:
    if sr not in _ALLOWED_SR:
        raise TTSError(f"elevenlabs: unsupported sample_rate {sr}; use one of {sorted(_ALLOWED_SR)}")
    return f"pcm_{sr}"


class ElevenLabsTTS(TTSProvider):
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        sample_rate: int = 24000,
        model_id: str = "eleven_turbo_v2_5",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        optimize_streaming_latency: int = 3,
    ) -> None:
        if not api_key:
            raise TTSError("elevenlabs: missing ELEVENLABS_API_KEY")
        if not voice_id:
            raise TTSError("elevenlabs: missing voice_id")
        self.name = "elevenlabs"
        self.sample_rate = sample_rate
        self.channels = 1
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._optimize_latency = max(0, min(4, int(optimize_streaming_latency)))
        self._voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True,
        }
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
            f"?output_format={_pcm_format(self.sample_rate)}"
            f"&optimize_streaming_latency={self._optimize_latency}"
        )
        body = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": self._voice_settings,
        }
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        try:
            async with self._client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    raise TTSError(f"elevenlabs {resp.status_code}: {err[:200]!r}")
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            raise TTSError(f"elevenlabs network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

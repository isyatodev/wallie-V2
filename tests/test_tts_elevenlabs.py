"""ElevenLabs TTS adapter — optimize_streaming_latency knob (0-4).

0 = fastest first byte, 4 = best quality, 3 was the historical hardcoded
default. The level must reach the request URL, be clamped to the valid
range, and flow through the factory from TTSConfig.
"""
from __future__ import annotations

import pytest


class _FakeResp:
    status_code = 200

    async def aiter_bytes(self):
        yield b"\x00\x01"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    def stream(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return _FakeResp()


def _adapter(**kw):
    from tts.elevenlabs import ElevenLabsTTS

    tts = ElevenLabsTTS(api_key="k-test", voice_id="v-test", **kw)
    fake = _FakeClient()
    tts._client = fake
    return tts, fake


@pytest.mark.asyncio
async def test_default_latency_level_is_three():
    """Backwards-compatible default: the old hardcoded 3."""
    tts, fake = _adapter()
    chunks = [c async for c in tts.synthesize("oi")]
    assert chunks == [b"\x00\x01"]
    url = fake.calls[0][1]
    assert "optimize_streaming_latency=3" in url
    assert "output_format=pcm_24000" in url


@pytest.mark.asyncio
async def test_latency_level_from_config_reaches_url():
    tts, fake = _adapter(optimize_streaming_latency=0)
    _ = [c async for c in tts.synthesize("oi")]
    assert "optimize_streaming_latency=0" in fake.calls[0][1]


@pytest.mark.parametrize("raw,clamped", [(9, 4), (-2, 0), ("2", 2)])
def test_latency_level_clamped_to_0_4(raw, clamped):
    tts, _ = _adapter(optimize_streaming_latency=raw)
    assert tts._optimize_latency == clamped


def test_factory_passes_latency_level_from_config():
    from config import Secrets, TTSConfig
    from tts.factory import build_tts

    cfg = TTSConfig(provider="elevenlabs", voice_id="v-test",
                    el_optimize_streaming_latency=1)
    tts = build_tts(cfg, Secrets(elevenlabs_api_key="k-test"))
    assert tts._optimize_latency == 1

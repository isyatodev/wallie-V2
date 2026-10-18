"""PiperTTS adapter — dispatch across piper-tts API generations.

piper-tts 1.3+ (piper1-gpl) replaced ``synthesize(kwarg...)`` /
``synthesize_stream_raw`` with ``synthesize(text, syn_config)``. The adapter
used to try the kwarg call, swallow its TypeError, then blame the now-missing
``synthesize_stream_raw`` — every synthesis on piper-tts 1.8 failed with
``'PiperVoice' object has no attribute 'synthesize_stream_raw'``.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from tts.base import TTSError
from tts.piper import PiperTTS

ONNX = Path(__file__).resolve().parent.parent / "voices" / "pt_BR-faber-medium.onnx"


def _find(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_PIPER = _find("piper")
HAS_PIPER_CONFIG = HAS_PIPER and _find("piper.config")


def _tts(voice, *, length_scale=1.0, noise_scale=0.667, noise_w=0.8) -> PiperTTS:
    """Build a PiperTTS around a fake voice, skipping __init__ (no onnx load)."""
    tts = PiperTTS.__new__(PiperTTS)
    tts._voice = voice
    tts.sample_rate = 22050
    tts.channels = 1
    tts._length_scale = length_scale
    tts._noise_scale = noise_scale
    tts._noise_w = noise_w
    return tts


def _collect(tts: PiperTTS, text: str = "oi") -> list[bytes]:
    async def go():
        return [c async for c in tts.synthesize(text)]

    return asyncio.run(go())


def _chunk(pcm: bytes):
    return type("AudioChunkLike", (), {"audio_int16_bytes": pcm})()


class NewApiVoice:
    """piper-tts 1.8 shape: synthesize(text, syn_config) -> AudioChunk.

    Deliberately has no ``synthesize_stream_raw``, exactly like 1.8.0 —
    this is the shape that crashed before the fix.
    """

    def __init__(self):
        self.seen = None

    def synthesize(self, text, syn_config=None, include_alignments=False):
        self.seen = (text, syn_config)
        yield _chunk(b"\x01\x02\x03\x04")
        yield _chunk(b"\x05\x06\x07\x08")


class LegacyVoice:
    """piper-tts 1.2 shape: kwargs in, raw byte stream out."""

    def __init__(self):
        self.kw = None

    def synthesize_stream_raw(self, text, length_scale=1.0, noise_scale=0.667, noise_w=0.8):
        self.kw = dict(length_scale=length_scale, noise_scale=noise_scale, noise_w=noise_w)
        yield b"\x10\x20"


def test_the_1_8_shape_has_no_legacy_stream_method():
    # The regression guard: dispatch must NOT need synthesize_stream_raw.
    assert not hasattr(NewApiVoice(), "synthesize_stream_raw")


@pytest.mark.skipif(not HAS_PIPER_CONFIG, reason="piper-tts with piper.config not installed")
def test_new_api_gets_the_tuning_knobs_in_a_synthesis_config():
    voice = NewApiVoice()
    out = _collect(_tts(voice, length_scale=1.1, noise_scale=0.55, noise_w=0.9))
    assert out == [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"]

    from piper.config import SynthesisConfig

    text, cfg = voice.seen
    assert text == "oi"
    assert isinstance(cfg, SynthesisConfig)
    assert cfg.length_scale == 1.1
    assert cfg.noise_scale == 0.55
    assert cfg.noise_w_scale == 0.9   # 1.8 renamed noise_w -> noise_w_scale


@pytest.mark.skipif(not HAS_PIPER_CONFIG, reason="piper-tts with piper.config not installed")
def test_float_array_chunks_become_int16_pcm():
    np = pytest.importorskip("numpy")

    class FloatVoice:
        def synthesize(self, text, syn_config=None, include_alignments=False):
            yield type(
                "C", (), {"audio_float_array": np.array([-1.0, 0.0, 1.0], dtype=np.float32)}
            )()

    out = _collect(_tts(FloatVoice()))
    assert len(out) == 1
    pcm = np.frombuffer(out[0], dtype="<i2")
    assert pcm.tolist() == [-32767, 0, 32767]


def test_legacy_stream_api_still_gets_the_knobs():
    voice = LegacyVoice()
    out = _collect(_tts(voice, length_scale=1.4, noise_scale=0.7, noise_w=0.9))
    assert out == [b"\x10\x20"]
    assert voice.kw == {"length_scale": 1.4, "noise_scale": 0.7, "noise_w": 0.9}


@pytest.mark.skipif(not HAS_PIPER_CONFIG, reason="piper-tts with piper.config not installed")
def test_inference_errors_surface_instead_of_a_masked_fallback():
    class Boom:
        def synthesize(self, text, syn_config=None, include_alignments=False):
            raise RuntimeError("onnx exploded")
            yield  # pragma: no cover

    with pytest.raises(TTSError) as exc:
        _collect(_tts(Boom()))
    # The real cause, not a phantom complaint about synthesize_stream_raw.
    assert str(exc.value) == "piper: synthesis failed: onnx exploded"


@pytest.mark.skipif(not HAS_PIPER, reason="piper-tts not installed")
@pytest.mark.skipif(not ONNX.is_file(), reason="no piper voice downloaded in this checkout")
def test_real_model_synthesizes_pcm():
    tts = PiperTTS(model_path=str(ONNX))
    chunks = _collect(tts, "olá, tudo bem?")
    assert chunks and all(isinstance(c, bytes) for c in chunks)
    assert sum(len(c) for c in chunks) > 4000   # well under one second of audio
    assert tts.sample_rate == 22050
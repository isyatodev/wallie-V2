"""Pitch shift for the local engines — semitones, duration preserved.

A pitch control has to do two things at once: move the fundamental and leave the
line exactly as long. Resampling alone fails the second half (the voice speeds
up), so the shifter time-stretches first. These tests check the contract with
synthetic tones, then confirm both local providers actually route their PCM
through it.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from tts.pitch import MAX_SEMITONES, clamp_semitones, semitones_to_ratio, shift_pcm16

SR = 24000


def _tone(freq: float, seconds: float = 0.6, sr: int = SR) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * math.pi * freq * t) * 32767).astype("<i2").tobytes()


def _peak_hz(pcm: bytes, sr: int = SR) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    freqs = np.fft.rfftfreq(samples.size, 1 / sr)
    return float(freqs[int(np.argmax(spectrum))])


def _seconds(pcm: bytes, sr: int = SR) -> float:
    return len(pcm) / 2 / sr


# ---------------------------------------------------------------------------
# The shifter itself
# ---------------------------------------------------------------------------

def test_zero_is_a_true_noop():
    pcm = _tone(220.0)
    assert shift_pcm16(pcm, 0) is pcm
    assert shift_pcm16(pcm, 1e-6) is pcm


@pytest.mark.parametrize("semitones", [-12.0, -7.0, -3.5, 3.5, 7.0, 12.0])
def test_pitch_moves_by_the_requested_ratio_and_keeps_duration(semitones):
    pcm = _tone(220.0)
    out = shift_pcm16(pcm, semitones)
    expected = 220.0 * semitones_to_ratio(semitones)
    assert _peak_hz(out) == pytest.approx(expected, rel=0.02)
    assert len(out) == len(pcm)
    assert _seconds(out) == pytest.approx(_seconds(pcm), abs=1 / SR)


def test_an_octave_down_does_not_alias():
    # 4 kHz is below the 12 kHz Nyquist at 24 kHz; after halving the spectrum
    # it must still sit at 2 kHz, not fold back to some other frequency.
    pcm = _tone(4000.0, seconds=0.4)
    out = shift_pcm16(pcm, -12)
    assert _peak_hz(out) == pytest.approx(2000.0, rel=0.02)


def test_output_is_int16_and_never_clips_outside_the_range():
    samples = np.frombuffer(shift_pcm16(_tone(180.0), 7), dtype="<i2")
    assert samples.dtype == np.dtype("<i2")
    assert samples.size > 0
    assert int(np.abs(samples).max()) <= 32767


def test_the_shift_is_clamped_to_one_octave():
    assert clamp_semitones(48) == MAX_SEMITONES
    assert clamp_semitones(-99) == -MAX_SEMITONES
    assert clamp_semitones(float("nan")) == 0.0
    assert clamp_semitones("nope") == 0.0
    assert shift_pcm16(_tone(220.0), 48) == shift_pcm16(_tone(220.0), MAX_SEMITONES)


def test_a_chunk_too_short_to_carry_pitch_is_left_alone():
    tiny = _tone(220.0, seconds=0.02)
    assert shift_pcm16(tiny, 7) == tiny
    assert shift_pcm16(b"", 7) == b""


def test_odd_length_pcm_is_trimmed_not_rejected():
    out = shift_pcm16(_tone(220.0) + b"\x00", 4)
    assert len(out) % 2 == 0
    assert _peak_hz(out) == pytest.approx(220.0 * semitones_to_ratio(4), rel=0.02)


# ---------------------------------------------------------------------------
# Piper — the shift happens after the voice hands back PCM
# ---------------------------------------------------------------------------

def _piper_with_pitch(semitones: float):
    from tts.piper import PiperTTS

    pcm = _tone(220.0)

    class Voice:
        def synthesize(self, text, syn_config=None, include_alignments=False):
            yield type("Chunk", (), {"audio_int16_bytes": pcm})()

    tts = PiperTTS.__new__(PiperTTS)
    tts._voice = Voice()
    tts.sample_rate = SR
    tts.channels = 1
    tts._length_scale = 1.0
    tts._noise_scale = 0.667
    tts._noise_w = 0.8
    tts._pitch = clamp_semitones(semitones)
    return tts


def _collect(tts) -> list[bytes]:
    async def go():
        return [chunk async for chunk in tts.synthesize("oi")]

    return asyncio.run(go())


def test_piper_applies_the_pitch_to_the_synthesized_pcm():
    out = b"".join(_collect(_piper_with_pitch(7)))
    assert _peak_hz(out) == pytest.approx(220.0 * semitones_to_ratio(7), rel=0.02)


def test_piper_without_pitch_passes_the_pcm_through_untouched():
    out = _collect(_piper_with_pitch(0))
    assert out == [_tone(220.0)]


# ---------------------------------------------------------------------------
# Kokoro — one shift per pipeline segment, still 24 kHz PCM16
# ---------------------------------------------------------------------------

def _kokoro_with_pitch(semitones: float):
    from tts.kokoro import KokoroTTS

    audio = np.frombuffer(_tone(220.0), dtype="<i2").astype(np.float32) / 32767.0

    def pipeline(text, voice=None, speed=1.0):
        yield ("oi", "oi", audio)

    tts = KokoroTTS.__new__(KokoroTTS)
    tts._pipeline = pipeline
    tts._voice = "af_heart"
    tts._speed = 1.0
    tts._pitch = clamp_semitones(semitones)
    tts._voice_path = ""
    tts.sample_rate = SR
    tts.channels = 1
    return tts


def test_kokoro_applies_the_pitch_to_each_segment():
    out = b"".join(_collect(_kokoro_with_pitch(-5)))
    assert _peak_hz(out) == pytest.approx(220.0 * semitones_to_ratio(-5), rel=0.02)
    assert _seconds(out) == pytest.approx(_seconds(_tone(220.0)), abs=1 / SR)


def test_kokoro_without_pitch_still_streams_plain_pcm16():
    out = b"".join(_collect(_kokoro_with_pitch(0)))
    assert len(out) == len(_tone(220.0))


# ---------------------------------------------------------------------------
# Voice page — presets are shortcuts that write the same slider value
# ---------------------------------------------------------------------------

def test_voice_page_offers_pitch_presets_for_both_local_engines():
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "dashboard" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    app = (static / "app.js").read_text(encoding="utf-8")

    assert "pitchPresets: PITCH_PRESETS" in app
    assert "{ label: \"Natural 0\", value: 0 }" in app
    for field in ("piper_pitch", "kokoro_pitch"):
        assert f'@click="cfg.tts.{field} = p.value"' in html, field
        assert f':class="{{on: cfg.tts.{field} === p.value}}"' in html, field
    assert html.count('x-for="p in pitchPresets"') == 2

    # Both engines can hear the unsaved settings before saving them.
    assert html.count('@click="previewPitch()"') == 2
    assert "pitchPreview.url" in html
    assert "\n    async previewPitch()" in app
    assert '"/api/voices/preview"' in app

    # A/B: the slider value against the untouched voice, back to back. The
    # second take must pin pitch to 0 rather than reuse the slider.
    assert html.count('@click="previewPitchAb()"') == 2
    assert 'x-ref="pitchAudioB"' in html
    assert "pitchPreview.abUrl" in html
    assert "\n    async previewPitchAb()" in app
    assert "{ [field]: 0 }" in app
    assert 'this.pitchField()' in app

"""Pitch shifting for local TTS engines — change the voice's pitch without
changing how long it takes to say the line.

Neither Piper nor Kokoro exposes a pitch control, so the effect is applied to
the PCM they produce. Naively resampling raises the pitch *and* speeds the line
up; this does the two-step the resampling trick needs:

1. **time-stretch** the chunk by ``r`` with WSOLA (waveform-similarity
   overlap-add) — same pitch, longer by ``r``;
2. **resample** the stretched audio by ``1/r`` with an FFT (band-limited, so
   pitching *down* does not alias) — back to the original duration, pitch × r.

WSOLA is used instead of a phase vocoder because it keeps speech transients
(plosives, consonants) crisp and is cheap enough to run on a live chunk.
Everything is numpy-only; ``scipy``/``librosa`` are not required.

Processing is per chunk on purpose: Piper emits one chunk per sentence and
Kokoro one per pipeline segment, so a chunk is a natural place to reset the
overlap-add state and the boundaries already sit on silence.
"""
from __future__ import annotations

import numpy as np

# Beyond an octave the formants move so far the result stops sounding like the
# same speaker; clamping here also keeps the UI and a hand-edited profile from
# asking for something the algorithm cannot do well.
MAX_SEMITONES = 12.0

_FRAME = 1024          # ~43 ms at 24 kHz
_OVERLAP = _FRAME // 4  # 75% overlap: smooth, and cheap to search
_TOLERANCE = _FRAME // 4
_SEARCH_STEP = 4       # 0.17 ms at 24 kHz — finer than any pitch period
_FADE = 48             # ~2 ms, kills the click at a chunk boundary


def semitones_to_ratio(semitones: float) -> float:
    """Equal-tempered pitch ratio for a shift in semitones (12 = one octave)."""
    return float(2.0 ** (float(semitones) / 12.0))


def clamp_semitones(semitones: float) -> float:
    try:
        value = float(semitones)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return max(-MAX_SEMITONES, min(MAX_SEMITONES, value))


def shift_pcm16(pcm: bytes, semitones: float, sample_rate: int = 24000) -> bytes:
    """Pitch-shift little-endian mono PCM16 by ``semitones``, keeping duration.

    ``0`` (and anything below a hundredth of a semitone) is a true no-op and
    returns the very same bytes, so an untouched profile pays nothing.
    """
    shift = clamp_semitones(semitones)
    if not pcm or abs(shift) < 0.01:
        return pcm
    if len(pcm) < 2:
        return pcm
    if len(pcm) % 2:
        pcm = pcm[:-1]

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    # A chunk shorter than two frames carries no usable pitch; leave it alone
    # rather than smearing it into the overlap-add window.
    if samples.size < 2 * _FRAME:
        return pcm

    ratio = semitones_to_ratio(shift)
    stretched = _wsola(samples, ratio)
    shifted = _resample_fft(stretched, 1.0 / ratio)
    shifted = _fit_length(shifted, samples.size)
    _fade_edges(shifted, _FADE)

    out = np.rint(np.clip(shifted, -1.0, 1.0) * 32767.0)
    return out.astype("<i2").tobytes()


def _wsola(x: np.ndarray, alpha: float) -> np.ndarray:
    """Time-stretch ``x`` by ``alpha`` (alpha > 1 = longer) keeping its pitch.

    Standard WSOLA: analysis hop is ``hs / alpha`` while the synthesis hop stays
    ``hs``; before copying each frame, search a small window around the nominal
    analysis position for the waveform that best continues the previous frame
    (normalized cross-correlation). The result is overlap-added with a Hann
    window and normalised by the window sum, so the amplitude is flat.
    """
    n = x.size
    hs = _FRAME - _OVERLAP
    ha = hs / alpha
    window = np.hanning(_FRAME).astype(np.float32)

    # Zero padding lets the last analysis frames run past the end instead of
    # truncating the tail of the stretched output.
    padded = np.concatenate([x.astype(np.float32, copy=False), np.zeros(_FRAME, np.float32)])
    out_len = int(round(n * alpha)) + _FRAME
    out = np.zeros(out_len, np.float32)
    norm = np.zeros(out_len, np.float32)

    prev_q = 0
    k = 0
    while True:
        nominal = int(round(k * ha))
        if nominal >= n:
            break
        q = 0
        if k:
            target_start = prev_q + hs
            target_len = min(_FRAME - hs, padded.size - target_start)
            if target_len <= 0:
                break
            lo = max(0, nominal - _TOLERANCE)
            hi = min(padded.size - _FRAME, nominal + _TOLERANCE)
            q = _best_offset(padded, target_start, target_len, lo, hi, nominal)
        start = k * hs
        out[start:start + _FRAME] += window * padded[q:q + _FRAME]
        norm[start:start + _FRAME] += window
        prev_q = q
        k += 1

    useful = int(round(n * alpha))
    out = out[:useful]
    norm = norm[:useful]
    norm[norm < 1e-6] = 1.0
    return out / norm


def _best_offset(
    padded: np.ndarray,
    target_start: int,
    target_len: int,
    lo: int,
    hi: int,
    fallback: int,
) -> int:
    """Start index of the frame that best continues the previous one."""
    if hi <= lo:
        return max(0, min(fallback, padded.size - _FRAME))

    target = padded[target_start:target_start + target_len]
    target = target - target.mean()
    norm_target = float(np.linalg.norm(target))
    if norm_target < 1e-6:
        return max(lo, min(fallback, hi))

    starts = np.arange(lo, hi + 1, _SEARCH_STEP)
    windows = np.lib.stride_tricks.sliding_window_view(padded, target_len)[starts]
    centered = windows - windows.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1)
    denom[denom < 1e-6] = 1e-6
    scores = (centered @ target) / denom
    return int(starts[int(np.argmax(scores))])


def _resample_fft(x: np.ndarray, factor: float) -> np.ndarray:
    """Band-limited resampling by ``factor`` (output length = len(x) × factor).

    Doing this in the frequency domain is what keeps a downward shift from
    aliasing: decimation truncates the spectrum instead of folding it back.
    """
    n = x.size
    m = max(1, int(round(n * factor)))
    if m == n:
        return x.astype(np.float32, copy=True)
    spectrum = np.fft.rfft(x)
    if m < n:
        spectrum = spectrum[: m // 2 + 1]
    elif m > n:
        spectrum = np.concatenate([
            spectrum,
            np.zeros(m // 2 + 1 - spectrum.size, dtype=spectrum.dtype),
        ])
    return np.fft.irfft(spectrum, m).astype(np.float32) * (m / n)


def _fit_length(x: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad to exactly ``length`` samples (rounding differences)."""
    if x.size == length:
        return x
    if x.size > length:
        return x[:length]
    return np.concatenate([x, np.zeros(length - x.size, np.float32)])


def _fade_edges(x: np.ndarray, fade: int) -> None:
    if x.size < 2 * fade or fade <= 0:
        return
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]

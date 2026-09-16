"""Voice-print speaker identification — who is talking, entirely on device.

Design (pure numpy, no heavy deps — runs alongside the existing STT loop):

- ``embed`` turns 16 kHz mono float32 PCM into a compact voice-print:
  log-mel spectrogram → DCT-II → MFCC mean+std, mean-variance normalized.
  It's a classical baseline, not a neural x-vector — but it is free, instant,
  private, and clearly separates the owner's mic/voice channel from other
  people in a voice chat in practice.
- ``SpeakerPrintStore`` persists per-speaker enrollment prints plus a small
  rolling list of recent embeddings per speaker (helps a lot in noise).
- ``SpeakerIdentifier`` scores each utterance against enrolled speakers:
  cosine similarity per stored print, take the median (robust to one bad
  print). Above ``threshold`` → that speaker; below ``unknown_threshold``
  for everyone → "unknown" (a voice we simply don't know).

Prints live in ``profiles/<name>.speakers.json`` next to the memory store —
raw audio never leaves the machine; only the derived (non-invertible) prints
are saved. While ``collect_other_voices`` is on, utterances from unknown
speakers are ALSO kept as short WAV blobs (on device) so the owner can enroll
new people by listening to them later from the Voice page.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_N_MFCC = 20
_N_MELS = 40
_FMIN, _FMAX = 80.0, 7600.0
_FFT = 512
_HOP = 160          # 10 ms
_MIN_EMBED_SEC = 0.35   # shorter audio produces garbage prints — skip scoring
_MIN_SPEECH_RMS = 0.004  # below this the clip is basically noise/silence
_PRINTS_PER_SPEAKER = 6   # rolling embeddings kept per enrolled speaker
_MAX_CLIPS = 12           # collected unknown-voice clips kept on disk
_CLIP_SEC = 8.0           # cap stored clip length

# ---------------------------------------------------------------------------
# DSP helpers (cached so the DCT matrix / mel filterbank are built once)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_pts = np.linspace(_hz_to_mel(_FMIN), _hz_to_mel(min(_FMAX, sr / 2 - 1)), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if hi <= lo:
            continue
        fb[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / max(1, mid - lo)
        fb[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / max(1, hi - mid)
    return fb


def _dct_matrix(n: int, m: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(m)[None, :]
    basis = np.cos(np.pi * (2 * k + 1) * i / (2 * n))
    basis *= np.sqrt(2.0 / n)
    basis[0, :] *= np.sqrt(0.5)
    return basis


class _Dsp:
    """Lazily-built shared DSP tables (identical for every 16 kHz caller)."""

    def __init__(self) -> None:
        self.fb = _mel_filterbank(SAMPLE_RATE, _FFT, _N_MELS)
        self.dct = _dct_matrix(_N_MELS, _N_MFCC)
        self.window = np.hanning(_FFT)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """float32 PCM (16 kHz mono) → normalized MFCC mean+std vector."""
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size < SAMPLE_RATE // 4:  # < 0.25 s: useless
            return np.array([])
        peak = float(np.max(np.abs(audio)))
        if peak <= 0:
            return np.array([])
        audio = audio / peak

        n_frames = 1 + (audio.size - _FFT) // _HOP
        if n_frames < 4:
            return np.array([])
        idx = np.arange(_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
        frames = audio[idx] * self.window[None, :]
        spec = np.abs(np.fft.rfft(frames, n=_FFT, axis=1)) ** 2
        mel = spec @ self.fb.T + 1e-10
        log_mel = np.log(mel)
        mfcc = log_mel @ self.dct  # (frames, n_mfcc)
        mfcc = mfcc[:, 1:]           # drop c0 (pure loudness) → timbre only

        # Frame-level VAD: keep the loudest 60% of frames (cheap energy gate)
        # so trailing silence doesn't wash out the print.
        energy = np.mean(mfcc, axis=1)
        if energy.size >= 5:
            keep = energy >= np.percentile(energy, 40)
            if keep.any():
                mfcc = mfcc[keep]

        feats = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0) + 1e-6])
        feats = (feats - feats.mean()) / (feats.std() + 1e-6)
        return feats.astype(np.float32)


_DSP: Optional[_Dsp] = None


def _dsp() -> _Dsp:
    global _DSP
    if _DSP is None:
        _DSP = _Dsp()
    return _DSP


def embed(audio: np.ndarray) -> np.ndarray:
    """Public helper — voice-print for a PCM clip (empty array if too short)."""
    try:
        rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2))) if np.asarray(audio).size else 0.0
        if rms < _MIN_SPEECH_RMS:
            return np.array([])
        return _dsp().embed(audio)
    except Exception as e:  # never let audio quirks kill the hearing loop
        logger.debug(f"speaker_id: embed failed ({e})")
        return np.array([])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Persistent store
# ---------------------------------------------------------------------------

class SpeakerPrintStore:
    """Enrolled voice prints + collected clips, persisted as JSON on device."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.speakers: dict[str, dict[str, Any]] = {}
        self.clips: list[dict[str, Any]] = []
        self.load()

    # -- persistence ---------------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning(f"speaker_id: could not read {self.path.name} ({e}); starting empty")
            return
        self.speakers = data.get("speakers", {})
        self.clips = data.get("clips", [])
        for sp in self.speakers.values():
            sp.setdefault("prints", [])
            sp.setdefault("label_count", 0)
            sp.setdefault("note", "")   # per-voice instruction for the prompt

    def save(self) -> None:
        try:
            self.path.parent.mkdir(exist_ok=True)
            self.path.write_text(
                json.dumps({"speakers": self.speakers, "clips": self.clips}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"speaker_id: could not save prints ({e})")

    # -- enrollment ----------------------------------------------------------
    def enroll(self, name: str, print_vec: np.ndarray, *, replace: bool = False) -> None:
        """Add a voice print for `name` (creates the speaker on first use)."""
        v = [float(x) for x in print_vec]
        sp = self.speakers.setdefault(
            name, {"prints": [], "label_count": 0, "created_at": time.time(), "note": ""}
        )
        sp.setdefault("note", "")
        if replace:
            sp["prints"] = []
        sp["prints"].append(v)
        sp["prints"] = sp["prints"][-_PRINTS_PER_SPEAKER:]
        self.save()

    # -- per-voice prompt instruction ---------------------------------------
    def set_note(self, name: str, note: str) -> bool:
        """Set the instruction injected into the prompt when this voice talks
        (e.g. "this is my mom — treat her warmly"). Empty string clears it."""
        if name not in self.speakers:
            return False
        self.speakers[name]["note"] = _clip_note(note)
        self.save()
        return True

    def get_note(self, name: str) -> str:
        """The stored instruction for this voice ('' when unset/unknown)."""
        sp = self.speakers.get(name)
        return (sp.get("note", "") if sp else "") or ""

    def find_name(self, label: str) -> str:
        """Case-insensitive resolved name for a voice-print label."""
        if label in self.speakers:
            return label
        low = label.strip().lower()
        for name in self.speakers:
            if name.lower() == low:
                return name
        return ""

    def remove(self, name: str) -> bool:
        if name in self.speakers:
            del self.speakers[name]
            self.save()
            return True
        return False

    def rename(self, old: str, new: str) -> bool:
        if old not in self.speakers or not new or new in self.speakers:
            return False
        self.speakers[new] = self.speakers.pop(old)
        self.save()
        return True

    def names(self) -> list[str]:
        return sorted(self.speakers)

    # -- scoring -------------------------------------------------------------
    def score(self, print_vec: np.ndarray) -> dict[str, float]:
        """Best median cosine similarity per speaker for this print."""
        out: dict[str, float] = {}
        for name, sp in self.speakers.items():
            sims = [cosine(print_vec, np.asarray(p, dtype=np.float32)) for p in sp.get("prints", [])]
            if sims:
                out[name] = float(np.median(sims))
        return out

    # -- collected clips (unknown voices kept for later enrollment) -----------
    def add_clip(self, audio: np.ndarray, print_vec: np.ndarray, suggestion: str) -> str:
        """Store a short WAV of an unknown voice; returns the clip id."""
        audio = np.asarray(audio, dtype=np.float32).ravel()
        audio = audio[: int(_CLIP_SEC * SAMPLE_RATE)]
        buf = io_bytes_wav(audio)
        clip = {
            "id": f"clip-{int(time.time() * 1000)}",
            "created_at": time.time(),
            "suggestion": suggestion,
            "print": [float(x) for x in print_vec],
            "wav_b64": base64.b64encode(buf).decode("ascii"),
        }
        self.clips.append(clip)
        self.clips = self.clips[-_MAX_CLIPS:]
        self.save()
        return clip["id"]

    def drop_clip(self, clip_id: str) -> bool:
        before = len(self.clips)
        self.clips = [c for c in self.clips if c.get("id") != clip_id]
        if len(self.clips) != before:
            self.save()
            return True
        return False

    def enroll_clip(self, clip_id: str, name: str) -> bool:
        """Promote a collected clip into an enrolled voice print."""
        for c in self.clips:
            if c.get("id") == clip_id:
                v = np.asarray(c.get("print", []), dtype=np.float32)
                if v.size:
                    self.enroll(name, v)
                    self.drop_clip(clip_id)
                    return True
        return False


_MAX_NOTE_CHARS = 240


def _clip_note(note: str) -> str:
    """Trim + sanitize a per-voice instruction (prompt-injection safety cap)."""
    return " ".join((note or "").split())[:_MAX_NOTE_CHARS]


def io_bytes_wav(audio: np.ndarray) -> bytes:
    """float32 PCM → 16-bit WAV bytes (for the collected-clip blobs)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Identifier — the object the hearing loop talks to
# ---------------------------------------------------------------------------

class SpeakerIdentifier:
    """Scores utterances against enrolled speakers, with a cooldown so one
    utterance can't spam clips, and honest "unknown" labeling."""

    def __init__(self, store: SpeakerPrintStore, *, threshold: float = 0.68,
                 unknown_threshold: float = 0.45,
                 collect_other_voices: bool = False) -> None:
        self.store = store
        self.threshold = float(threshold)
        self.unknown_threshold = float(unknown_threshold)
        self.collect_other_voices = bool(collect_other_voices)
        self._last_clip_ts = 0.0
        self._unknown_run = 0  # consecutive unknown utterances → better suggestion names

    def identify(self, audio: np.ndarray) -> tuple[str, float]:
        """→ (label, best_similarity). label ∈ speaker name | 'unknown' | ''."""
        v = embed(audio)
        if not v.size:
            return "", -1.0
        scores = self.store.score(v)
        if not scores:
            return "unknown", -1.0
        best_name = max(scores, key=scores.get)          # type: ignore[arg-type]
        best = scores[best_name]
        if best >= self.threshold:
            return best_name, best
        if best < self.unknown_threshold:
            return "unknown", best
        # In between: not the owner, but not confidently a stranger either.
        return "other", best

    def handle_utterance(self, audio: np.ndarray, *, collect: bool = True) -> tuple[str, float]:
        """Full pipeline entry: label + maybe keep an unknown-voice clip.

        Returns (label, similarity). Collection only happens for confident
        unknowns ('unknown'), never for 'other' — those could be the owner on
        a bad day and we must not enroll them accidentally.
        """
        label, sim = self.identify(audio)
        now = time.time()
        if (
            label == "unknown"
            and collect
            and self.collect_other_voices
            and now - self._last_clip_ts >= 45.0  # cooldown: at most ~1 clip/min
        ):
            self._unknown_run += 1
            suggestion = f"unknown voice #{self._unknown_run}"
            try:
                self.store.add_clip(audio, embed(audio), suggestion)
                self._last_clip_ts = now
                logger.info(f"speaker_id: kept clip of unknown voice ({suggestion})")
            except Exception as e:
                logger.debug(f"speaker_id: clip capture failed ({e})")
        return label, sim


class EnrollmentBuffer:
    """Collects consecutive utterances for enrolling a speaker from the live loop.

    The Voice page says "start enrollment", the owner talks normally for a few
    utterances, and each one is embedded here; finishing averages them into a
    single robust print stored under the chosen name.
    """

    def __init__(self, max_clips: int = 8, max_age_sec: float = 300.0) -> None:
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=max_clips)
        self._max_age = max_age_sec

    def add(self, audio: np.ndarray) -> bool:
        v = embed(audio)
        if v.size:
            self._buf.append((time.time(), v))
            return True
        return False

    @property
    def pending(self) -> int:
        self._expire()
        return len(self._buf)

    def _expire(self) -> None:
        cutoff = time.time() - self._max_age
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def finish(self, store: SpeakerPrintStore, name: str, *, replace: bool = False) -> bool:
        """Average the buffered prints into one and enroll under `name`."""
        self._expire()
        if not self._buf or not name.strip():
            return False
        vecs = np.stack([v for _, v in self._buf])
        mean = vecs.mean(axis=0)
        mean = (mean - mean.mean()) / (mean.std() + 1e-6)
        store.enroll(name.strip(), mean.astype(np.float32), replace=replace)
        self._buf.clear()
        logger.info(f"speaker_id: enrolled voice print for {name!r}")
        return True

    def cancel(self) -> None:
        self._buf.clear()

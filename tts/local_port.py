"""Local port — turn a saved cloud voice into a local dataset with no provider.

``tts.voice_port`` asks the provider to read a corpus: it needs an API key and
bills every clip. This module is the offline sibling. It takes the audio the
user already has — a downloaded sample set, takes saved from the stream, a
folder of clips — and builds the *same* package ``voice_port`` produces, so the
existing Train panel and the attach step finish the job unchanged.

The difference that matters is what is never touched:

* no provider, no key, no network call — the routes that call this never read
  ``Secrets``;
* decoding is local (stdlib ``wave`` for WAV, an ffmpeg subprocess for anything
  compressed);
* transcription is local faster-whisper — already a dependency of ``hearing`` —
  loaded with ``local_files_only=True``, so a missing model is reported instead
  of being downloaded mid-run;
* trimming, rate conversion and the package layout come from ``voice_port``, so
  a local package and a captured package are indistinguishable downstream.

The voice identity still comes from the saved ElevenLabs/Fish preset: the
result is registered with its name and provider, and the package manifest
records ``origin_mode: local-audio`` instead of pretending the provider spoke.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .decode import find_ffmpeg
from . import voice_port

# Where imported clips are kept, under the engines' own voices tree:
# voices/local-src/<voice>/ — the audio survives a re-package, so a second run
# (other engine, more clips, fixed transcripts) never needs the files again.
LOCAL_DIR_NAME = "local-src"

# What a user's folder realistically holds. ffmpeg decodes all of it locally;
# WAV works even without ffmpeg.
AUDIO_EXTS: tuple[str, ...] = (
    ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm", ".aac", ".mp4",
)
TRANSCRIPT_EXTS: tuple[str, ...] = (".txt", ".csv")

MAX_FILES = 60
MAX_FILE_BYTES = 40 * 1024 * 1024
# A single take longer than this is almost certainly a recording, not a sample;
# Piper wants short clips. It is cut (and reported), never silently kept whole.
MAX_CLIP_SECONDS = 60.0

# ``upload | sidecar | whisper`` — how each clip got its transcript.
TRANSCRIPT_SOURCES = ("you typed", "the file's sidecar", "local Whisper")


class LocalPortError(RuntimeError):
    """A failure worth showing to the user verbatim."""


# ---------------------------------------------------------------------------
# Where the clips live
# ---------------------------------------------------------------------------

def local_dir(name: str) -> Path:
    """Folder holding the audio already imported for a voice."""
    from . import piper_setup
    return piper_setup.voices_dir() / LOCAL_DIR_NAME / voice_port.slug(name or "voice")


def audio_name(filename: str) -> str:
    """A safe file name for an imported clip (no folders, no traversal)."""
    base = Path(str(filename or "clip").replace("\\", "/")).name
    stem = voice_port.slug(Path(base).stem or "clip")
    suffix = Path(base).suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise LocalPortError(
            f"{base} is not audio — use one of {', '.join(AUDIO_EXTS)}"
        )
    return f"{stem}{suffix}"


def _wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate() or 1
            return round(w.getnframes() / float(rate), 2)
    except Exception:          # noqa: BLE001 - a non-WAV clip just reports no duration
        return None


def scan(name: str) -> dict[str, Any]:
    """Clips already imported for this voice, reused by the next run."""
    directory = local_dir(name)
    files: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if path.suffix.lower() not in AUDIO_EXTS or not path.is_file():
                continue
            files.append({
                "filename": path.name,
                "bytes": path.stat().st_size,
                "seconds": _wav_seconds(path),
            })
    seconds = round(sum(f["seconds"] or 0.0 for f in files), 2)
    return {"dir": str(directory), "clips": len(files), "seconds": seconds, "files": files}


# ---------------------------------------------------------------------------
# Decoding — every byte stays on this machine
# ---------------------------------------------------------------------------

def _is_wav(blob: bytes) -> bool:
    return len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"


def _pcm16_from_wav(blob: bytes) -> tuple[bytes, int]:
    """Mono PCM16 + the file's own rate, from a WAV container (stdlib only)."""
    with wave.open(io.BytesIO(blob), "rb") as w:
        channels, width, rate, frames = (
            w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        )
        raw = w.readframes(frames)
    if width == 1:                       # unsigned 8-bit
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) << 8
    elif width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif width == 3:                     # packed 24-bit
        usable = (len(raw) // 3) * 3
        triples = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        arr = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        arr = np.where(arr >= (1 << 23), arr - (1 << 24), arr) >> 8
    elif width == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.int64) >> 16
    else:
        raise LocalPortError(f"unsupported WAV sample width ({width * 8}-bit)")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    pcm = np.clip(np.round(arr), -32768, 32767).astype("<i2").tobytes()
    return pcm, int(rate or 0)


def _decode_with_ffmpeg(blob: bytes, filename: str, rate: int) -> bytes:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise LocalPortError(
            f"{filename} is compressed and ffmpeg was not found — install ffmpeg, "
            "set WALLIE_FFMPEG, or convert the clip to WAV first"
        )
    cmd = [
        ffmpeg, "-v", "error", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=blob, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        detail = lines[-1][:180] if lines else f"exit={proc.returncode}"
        raise LocalPortError(f"ffmpeg could not decode {filename}: {detail}")
    out = proc.stdout
    return out[:-1] if len(out) % 2 else out


def decode_audio(blob: bytes, filename: str) -> bytes:
    """Any supported container → mono PCM16 at the port's capture rate, trimmed.

    Returns empty bytes for a clip that is all silence — the caller reports it
    and moves on instead of building a dataset with a blank row.
    """
    if not blob:
        raise LocalPortError(f"{filename} is empty")
    if _is_wav(blob):
        pcm, rate = _pcm16_from_wav(blob)
        pcm = voice_port.resample_pcm16(pcm, rate, voice_port.CAPTURE_SR)
    else:
        pcm = _decode_with_ffmpeg(blob, filename, voice_port.CAPTURE_SR)
    pcm = voice_port.trim_silence(pcm, voice_port.CAPTURE_SR)
    limit = int(MAX_CLIP_SECONDS * voice_port.CAPTURE_SR) * 2
    if len(pcm) > limit:
        pcm = pcm[:limit]
    return pcm


# ---------------------------------------------------------------------------
# Transcripts — typed, sidecar, or local Whisper
# ---------------------------------------------------------------------------

def _default_whisper_size() -> str:
    from config import HearingConfig
    return (HearingConfig().model_size or "small").strip()


def _whisper_cache_dir() -> Path:
    override = os.environ.get("WALLIE_WHISPER_DIR", "").strip()
    if override:
        return Path(override)
    root = os.environ.get("HF_HOME", "").strip()
    base = Path(root) if root else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def whisper_status(model_size: str = "") -> dict[str, Any]:
    """Whether local transcription can run *right now*, without a download."""
    size = (model_size or _default_whisper_size()).strip()
    try:
        import faster_whisper  # noqa: F401
    except Exception:          # noqa: BLE001 - reported, never raised from a panel
        return {
            "available": False, "model": size, "cached": False, "engine": "faster-whisper",
            "reason": "faster-whisper is not installed — pip install faster-whisper, "
                      "or type the transcripts",
        }
    cache = _whisper_cache_dir()
    cached = cache.is_dir() and any(cache.glob(f"models--*--faster-whisper-{size}"))
    return {
        "available": True,
        "model": size,
        "cached": cached,
        "engine": "faster-whisper",
        "reason": "" if cached else (
            f"the Whisper '{size}' model is not cached — play one Hearing test once "
            "(it downloads then), or type/paste the transcripts"
        ),
    }


_MODELS: dict[tuple[str, str], Any] = {}


def _model(size: str) -> Any:
    key = (size, str(_whisper_cache_dir()))
    if key in _MODELS:
        return _MODELS[key]
    status = whisper_status(size)
    if not status["available"]:
        raise LocalPortError(status["reason"])
    if not status["cached"]:
        raise LocalPortError(status["reason"])
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(
            size, device="cpu", compute_type="int8", local_files_only=True,
        )
    except Exception as e:     # noqa: BLE001 - the message is the user's next step
        raise LocalPortError(
            f"could not load the local Whisper '{size}' model: {str(e)[:180]}"
        ) from e
    _MODELS[key] = model
    return model


def transcribe_pcm(pcm: bytes, lang: str = "pt", model_size: str = "") -> str:
    """One clip → text, entirely on this machine."""
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if not audio.size:
        return ""
    peak = float(np.abs(audio).max())
    if 0.0 < peak < 0.9:                  # a quiet clip transcribes worse
        audio = (audio * (0.9 / peak)).astype(np.float32)
    model = _model((model_size or _default_whisper_size()).strip())
    segments, _info = model.transcribe(
        audio,
        language=(lang or None),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join((seg.text or "").strip() for seg in segments).strip()


def _read_transcripts(files: Sequence[tuple[str, bytes]]) -> tuple[dict[str, str], list[str]]:
    """Parse ``.txt`` / ``.csv`` uploaded (or saved) next to the clips.

    Two accepted shapes: ``name|text`` (Piper's own ``metadata.csv``) and one
    plain line per clip, in order. A header row and blank lines are ignored.
    """
    by_stem: dict[str, str] = {}
    ordered: list[str] = []
    for filename, blob in files:
        suffix = Path(filename).suffix.lower()
        if suffix not in TRANSCRIPT_EXTS:
            continue
        text = blob.decode("utf-8", "replace")
        for raw in text.splitlines():
            line = raw.strip().lstrip("\ufeff")
            if not line:
                continue
            if "|" in line:
                stem, _, body = line.partition("|")
            elif suffix == ".csv" and "," in line:
                stem, _, body = line.partition(",")
            else:
                ordered.append(line)
                continue
            stem_key = voice_port.slug(Path(stem.strip()).stem or "").lower()
            if stem_key in ("file", "filename", "audio", "wav", "id"):
                continue              # header row
            if stem.strip() and body.strip():
                by_stem[stem_key] = body.strip()
    return by_stem, [line for line in ordered if line.strip()]


# ---------------------------------------------------------------------------
# Preparing the package
# ---------------------------------------------------------------------------

def _collect_files(
    name: str, files: Sequence[tuple[str, bytes]], use_saved: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, bytes]]]:
    """Uploads + saved clips, one entry each, plus the transcript files.

    Uploads are written into the voice's folder first, so they are reused by the
    next run — and a clip with the same name is refreshed, not duplicated.
    """
    directory = local_dir(name)
    # Upload order is the order the typed transcript lines refer to — sorting by
    # name here would silently pair the wrong text with the wrong clip.
    uploaded: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    transcripts: list[tuple[str, bytes]] = []

    for filename, blob in files:
        base = Path(str(filename or "").replace("\\", "/")).name
        if not base:
            continue
        if Path(base).suffix.lower() in TRANSCRIPT_EXTS:
            transcripts.append((base, blob))
            continue
        if Path(base).suffix.lower() not in AUDIO_EXTS:
            raise LocalPortError(
                f"{base} is not audio — use one of {', '.join(AUDIO_EXTS)} "
                "(and .txt/.csv for the transcripts)"
            )
        if len(blob) > MAX_FILE_BYTES:
            raise LocalPortError(
                f"{base} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB — "
                "split it into shorter clips"
            )
        target = directory / audio_name(base)
        directory.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        if target.name.lower() in by_name:
            by_name[target.name.lower()]["path"] = target      # refreshed take
            continue
        entry = {"filename": target.name, "path": target, "uploaded": True}
        uploaded.append(entry)
        by_name[target.name.lower()] = entry

    saved: list[dict[str, Any]] = []
    if use_saved and directory.is_dir():
        for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() in TRANSCRIPT_EXTS:
                transcripts.append((path.name, path.read_bytes()))
                continue
            if path.suffix.lower() not in AUDIO_EXTS or path.name.lower() in by_name:
                continue
            saved.append({"filename": path.name, "path": path, "uploaded": False})

    ordered = uploaded + saved
    if len(ordered) > MAX_FILES:
        raise LocalPortError(
            f"{len(ordered)} clips is more than this handles ({MAX_FILES}) — "
            "import the best ones first"
        )
    return ordered, transcripts


def prepare(
    *,
    name: str,
    provider: str = "",
    voice_id: str = "",
    target: str = "piper",
    lang: str = "pt",
    files: Sequence[tuple[str, bytes]] = (),
    texts: Sequence[str] = (),
    transcribe: bool = True,
    use_saved: bool = True,
    model_size: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Import local audio into an engine package — no provider is contacted.

    The package is byte-identical in layout to the one a capture produces, which
    is what lets the dashboard's Train panel (and the attach step) work
    unchanged. Transcripts come from, in order: what the user typed, a sidecar
    ``.txt``/``.csv``, or local Whisper. A clip whose text cannot be established
    is refused for Piper rather than trained on a guessed line.
    """
    name = (name or "").strip()
    if not name:
        raise LocalPortError("pick the saved voice these clips belong to")
    spec = voice_port.target_spec(target)
    corpus = voice_port.lang_spec(lang)
    entries, transcript_files = _collect_files(name, files, use_saved)
    if not entries:
        raise LocalPortError(
            "no audio yet — attach the clips you already have (WAV/MP3/…); "
            "this path never calls ElevenLabs or Fish"
        )

    by_stem, ordered_lines = _read_transcripts(transcript_files)
    typed = [str(t or "").strip() for t in texts]

    transcriber = whisper_status(model_size) if spec["id"] == "piper" else {
        "available": False, "cached": False, "model": "", "engine": "", "reason": "",
    }
    used_whisper = 0
    takes: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    warnings: list[str] = []
    missing: list[str] = []

    for i, entry in enumerate(entries):
        filename = entry["filename"]
        pcm = decode_audio(entry["path"].read_bytes(), filename)
        if not pcm:
            skipped.append({"filename": filename, "reason": "silence — nothing to learn"})
            continue
        seconds = voice_port.pcm_seconds(pcm, voice_port.CAPTURE_SR)
        if seconds >= MAX_CLIP_SECONDS:
            warnings.append(f"{filename} was cut at {int(MAX_CLIP_SECONDS)}s")
        text = typed[i] if i < len(typed) and typed[i] else ""
        source = "you typed" if text else ""
        if not text:
            text = by_stem.get(voice_port.slug(Path(filename).stem).lower(), "")
            source = "the file's sidecar" if text else ""
        if not text and len(ordered_lines) == len(entries):
            text = ordered_lines[i].strip()
            source = "the file's sidecar" if text else ""
        if not text and transcribe and spec["id"] == "piper":
            if not transcriber["available"] or not transcriber["cached"]:
                missing.append(filename)
                continue
            text = transcribe_pcm(pcm, corpus["id"], model_size)
            if text:
                used_whisper += 1
                source = "local Whisper"
        if spec["id"] == "piper" and not text:
            missing.append(filename)
            continue
        takes.append({
            "index": len(takes) + 1,
            "text": text,
            "pcm": pcm,
            "seconds": seconds,
            "transcript": source or "not needed",
        })

    if missing:
        reason = transcriber.get("reason") or (
            "type one line per clip, or attach a .txt with one line per clip"
        )
        raise LocalPortError(
            f"{missing[0]} has no transcript and local transcription cannot run: {reason}"
        )
    if len(takes) < voice_port.MIN_CLIPS:
        raise LocalPortError(
            f"{len(takes)} usable clip(s) — a Piper dataset needs at least "
            f"{voice_port.MIN_CLIPS}; add more audio or fix the silent ones"
        )
    total = round(sum(float(t["seconds"]) for t in takes), 2)
    if spec["id"] == "piper" and total < voice_port.MIN_TRAIN_SECONDS:
        # The same verdict as the training preflight and the capture port: an
        # 8-second dataset is not a thin voice, it is pure noise. Refusing here
        # also keeps the Train panel from offering a run that cannot work.
        raise LocalPortError(
            f"only {len(takes)} clip(s) / {total}s of speech — a Piper voice needs "
            f"at least {int(voice_port.MIN_TRAIN_SECONDS)}s (10+ minutes recommended). "
            "Training on this would export noise, not a voice, so the package was "
            "not built. Your clips were kept — add more and build it again."
        )
    if spec["id"] == "piper" and total < voice_port.RECOMMENDED_TRAIN_SECONDS:
        warnings.append(
            f"only {total}s of speech — understandable but rough; "
            f"{int(voice_port.RECOMMENDED_TRAIN_SECONDS)}+ seconds (ideally 10–30 "
            "minutes) sounds much better"
        )

    origin = (
        f"Imported from {len(takes)} local clip(s) ({total}s of speech) for a "
        f"{corpus['label']} voice. No provider was contacted to build this package."
    )
    source = voice_port.PortSource(
        name=name,
        provider=(provider or "").strip().lower(),
        voice_id=(voice_id or "").strip(),
        notes="built from audio already on this machine",
        source=f"local:{(provider or 'audio').strip().lower()}",
    )
    report = voice_port.build_package(
        target=spec["id"],
        name=name,
        takes=takes,
        source=source,
        lang=corpus["id"],
        now=now,
        origin=origin,
    )

    # The manifest is the record of *how* the audio arrived; the package itself
    # stays identical so training/attach need no special case.
    manifest_path = Path(report["package"]) / voice_port.MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["origin_mode"] = "local-audio"
        manifest["network"] = False
        manifest["transcriber"] = (
            f"faster-whisper {transcriber['model']}" if used_whisper else ""
        )
        manifest["clips_meta"] = [
            {
                "index": t["index"], "text": t["text"], "seconds": t["seconds"],
                "transcript": t["transcript"],
            }
            for t in takes
        ]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (OSError, ValueError) as e:
        raise LocalPortError(f"could not finish the package manifest: {e}") from e

    report.update({
        "origin": "local-audio",
        "network": False,
        "keys_used": False,
        "name": name,
        "provider": source.provider,
        "saved_dir": str(local_dir(name)),
        "saved_clips": sum(1 for e in entries if not e.get("uploaded")),
        "skipped": skipped,
        "transcribed": used_whisper,
        "transcriber": f"faster-whisper {transcriber['model']}" if used_whisper else "",
        "warnings": warnings,
        "audition": {
            "url": "data:audio/wav;base64," + _b64(takes[0]["pcm"]),
            "text": takes[0]["text"],
            "seconds": takes[0]["seconds"],
        },
    })
    return report


def _b64(blob: bytes) -> str:
    import base64
    return base64.b64encode(blob).decode("ascii")


# ---------------------------------------------------------------------------
# Panel data for the dashboard
# ---------------------------------------------------------------------------

def panel_payload(name: str = "", model_size: str = "") -> dict[str, Any]:
    """What the local block needs: formats, limits, saved clips, transcriber."""
    return {
        "audio_exts": list(AUDIO_EXTS),
        "transcript_exts": list(TRANSCRIPT_EXTS),
        "max_files": MAX_FILES,
        "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
        "max_clip_seconds": MAX_CLIP_SECONDS,
        "min_clips": voice_port.MIN_CLIPS,
        "local_dir": str(local_dir(name)) if name else str(
            local_dir("voice").parent
        ),
        "saved": scan(name) if name else {"dir": "", "clips": 0, "seconds": 0, "files": []},
        "transcriber": whisper_status(model_size),
        "network": False,
    }

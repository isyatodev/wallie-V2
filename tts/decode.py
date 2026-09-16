"""ffmpeg decode bridge for OpenAI-compatible TTS endpoints that ignore
``response_format: "pcm"``.

Some free gateways always answer MP3/OGG (or wrap PCM in a WAV container)
regardless of the requested format. Playing those bytes as PCM = static;
aborting the sentence = silence. This module sits between: it detects a
non-PCM payload on the wire and decodes it to raw PCM16 mono with ffmpeg
(discovery order: WALLIE_FFMPEG env override -> PATH -> imageio-ffmpeg wheel).

ffmpeg is never a hard dependency: when it's absent the TTS pipeline keeps
its exact current behavior — compressed data is refused with an actionable
error instead of playing static.
"""
from __future__ import annotations

import asyncio
import os
import shutil

__all__ = ["sniff_compressed", "find_ffmpeg", "decode_to_pcm16"]


# ---------------------------------------------------------------------------
# Sniffing
# ---------------------------------------------------------------------------

def sniff_compressed(head: bytes) -> str | None:
    """Return a format label when `head` clearly isn't raw PCM16.

    Same thresholds as the orchestrator's ``_looks_non_pcm`` (which decides
    whether to abort a sentence): no false positives on real PCM — raw PCM
    bytes can be anything, so only unambiguous signatures count.
    """
    if not head:
        return None
    h = head[:12]
    if h[:4] == b"RIFF":          # WAV container
        return "wav"
    if h[:4] == b"OggS":          # OGG container
        return "ogg"
    if h[:3] == b"ID3":           # MP3 with tag
        return "mp3"
    if len(h) >= 2 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0 and (h[1] & 0x06) != 0:
        # MP3 sync word with valid layer bits (layer bits == 0 are reserved,
        # so 0xFF 0x0E is not a frame) — mirrors the orchestrator's heuristic.
        return "mp3"
    if h[:1] in (b"{", b"<"):     # JSON error body / XML — decode will fail
        return "json-or-xml"
    return None


# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str | None:
    """Locate an ffmpeg executable, or None when none is available."""
    override = os.environ.get("WALLIE_FFMPEG", "").strip()
    if override:
        return override
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg  # optional wheel that bundles a static binary

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

async def decode_to_pcm16(data: bytes, ffmpeg: str, sample_rate: int) -> bytes:
    """Decode arbitrary audio bytes (MP3/OGG/WAV/…) to raw PCM16 mono at
    `sample_rate` via an ffmpeg subprocess on stdin/stdout pipes."""
    cmd = [
        ffmpeg,
        "-v", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await proc.communicate(data)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0 or not out:
        lines = (err or b"").decode("utf-8", "replace").strip().splitlines()
        detail = lines[-1][:160] if lines else f"exit={proc.returncode}"
        raise RuntimeError(f"ffmpeg decode failed: {detail}")
    if len(out) % 2:  # a truncated final sample would crackle
        out = out[:-1]
    return out

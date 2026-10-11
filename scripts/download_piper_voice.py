"""Download a Piper voice model from the official HuggingFace repository.

Usage:
    python scripts/download_piper_voice.py en_US-amy-medium
    python scripts/download_piper_voice.py de_DE-thorsten-medium
    python scripts/download_piper_voice.py en_US-libritts_r-medium  --dest voices

Voice naming convention: ``<lang_REGION>-<voice>-<quality>``
  * lang_REGION : "en_US", "de_DE", "es_ES", ...
  * voice       : speaker / dataset name (e.g. "amy", "ryan", "dfki")
  * quality     : "low" (16 kHz), "medium" (22.05 kHz), "high" (22.05 kHz best)

Browse the catalogue at:
    https://huggingface.co/rhasspy/piper-voices
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# The progress glyphs below (↓ ✓ ·) crash a cp1252 Windows console, and the
# dashboard runs this script with stdout as a PIPE (also cp1252 by default) —
# which made the download die with UnicodeEncodeError before saving anything
# (the Voice tab looked like it downloaded, but no .onnx ever landed).
# Force UTF-8 on stdout/stderr where the interpreter allows it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _voice_urls(name: str) -> tuple[str, str]:
    """Return (onnx_url, json_url) for a voice name like 'en_US-amy-medium'."""
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"voice name must be lang_REGION-voice-quality (got {name!r})")
    lang_region = parts[0]
    voice = parts[1]
    quality = parts[-1]
    lang = lang_region.split("_")[0]
    base = f"{_BASE}/{lang}/{lang_region}/{voice}/{quality}/{name}"
    return f"{base}.onnx", f"{base}.onnx.json"


def _download(url: str, dest: Path) -> None:
    print(f"  ↓ {url}")
    with urllib.request.urlopen(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        total = int(resp.headers.get("Content-Length") or 0)
        with dest.open("wb") as f:
            read = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = read / total * 100
                    print(f"\r    {read/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)", end="")
            print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="download_piper_voice")
    ap.add_argument("voice", help="e.g. en_US-amy-medium  /  de_DE-thorsten-medium")
    ap.add_argument("--dest", default="voices", help="destination directory (default: voices/)")
    args = ap.parse_args(argv)

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    onnx = dest_dir / f"{args.voice}.onnx"
    cfg = dest_dir / f"{args.voice}.onnx.json"

    if onnx.exists() and cfg.exists():
        print(f"Voice already present: {onnx}")
        return 0

    try:
        onnx_url, json_url = _voice_urls(args.voice)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        if not onnx.exists():
            _download(onnx_url, onnx)
        if not cfg.exists():
            _download(json_url, cfg)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1

    print()
    print(f"✓ saved {onnx}")
    print(f"  set tts.provider = 'piper' and tts.piper_model_path = '{onnx}' in your profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

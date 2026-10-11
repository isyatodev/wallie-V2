"""Download one ready-made Kokoro voice pack (a single ``.pt`` file).

Usage:
    python scripts/download_kokoro_voice.py af_heart
    python scripts/download_kokoro_voice.py pf_dora --dest voices/kokoro

Voices come from the ``hexgrad/Kokoro-82M`` model repo. Nothing is bundled with
this project: you download just the voices you want, they land in
``voices/kokoro/`` by default, and the local file is used offline afterwards.

Voice id convention: ``<lang><gender>_<name>``
  * lang   : a=US English, b=UK English, e=Spanish, f=French, h=Hindi,
             i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin
  * gender : f=female, m=male

Browse the catalogue at:
    https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

# The progress glyphs below (↓ ✓ ·) crash a cp1252 Windows console, and the
# dashboard runs this script with stdout as a PIPE (also cp1252 by default) —
# which made the download die with UnicodeEncodeError before saving anything.
# Force UTF-8 on stdout/stderr where the interpreter allows it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_REPO = "hexgrad/Kokoro-82M"
_BASE = f"https://huggingface.co/{_REPO}/resolve/main/voices"
_NAME_RE = re.compile(r"^[a-z]{1,2}_[a-z0-9_]+$")


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
    ap = argparse.ArgumentParser(prog="download_kokoro_voice")
    ap.add_argument("voice", help="e.g. af_heart / pf_dora / zf_xiaobei")
    ap.add_argument("--dest", default="voices/kokoro",
                    help="destination directory (default: voices/kokoro)")
    args = ap.parse_args(argv)

    name = (args.voice or "").strip()
    if not _NAME_RE.match(name):
        print("error: voice name must look like <lang><gender>_<name>, e.g. af_heart",
              file=sys.stderr)
        return 2

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{name}.pt"
    if out.exists() and out.stat().st_size > 0:
        print(f"Voice already present: {out}")
        return 0

    try:
        _download(f"{_BASE}/{name}.pt", out)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1

    print()
    print(f"✓ saved {out}")
    print(f"  select voice '{name}' on the Voice page (provider: kokoro)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

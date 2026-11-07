"""KokoClone worker — one long-lived process that owns the clone pipeline.

KokoClone (`https://github.com/Ashish-Patnaik/kokoclone`) is a two-stage
pipeline: Kokoro-ONNX synthesises the line, then the Kanade voice-conversion
model reshapes it into the reference speaker. Loading those models costs
minutes the first time and several hundred MB, so they must be loaded ONCE
and reused — spawning a fresh process per sentence would reload gigabytes
every time.

This script is that process. It speaks JSON lines on stdout (one object in,
one object out), and every print from KokoClone itself is sent to stderr:
the library is chatty, and its "Synthesizing text…" lines would otherwise
corrupt the protocol.

Protocol (one job at a time, sequential on purpose — the models are not
thread-safe and there is one GPU):

    {"event": "starting"}
    {"event": "ready", "sample_rate": 24000}       # models loaded
    {"event": "error", "error": "…"}               # boot failed
    {"id": 1, "text": "…", "lang": "pt", "ref": "…", "out": "…"}
    {"ok": true,  "id": 1, "path": "…", "sample_rate": 24000}
    {"ok": false, "id": 1, "error": "…"}

Exit by closing stdin. Run with the interpreter that has kokoclone's
requirements: ``python tts/kokoclone_worker.py --root <checkout>``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from typing import Any


def _protocol_stream():
    """Duplicate stdout for the protocol, then point fd 1 at stderr.

    Everything KokoClone prints — and anything a dependency prints — lands on
    stderr, which the parent drains into a log tail. Without the dup, one
    library print between two JSON objects would make the parent's ``json.loads``
    fail and look exactly like a crashed worker.
    """
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    # fd 1 is a pipe now, so Python would block-buffer it and the parent's log
    # tail would stay empty until the buffer filled. Line buffering makes the
    # library's progress lines show up in the dashboard as they happen.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KokoClone worker (JSON lines)")
    parser.add_argument("--root", required=True, help="path to the kokoclone checkout")
    args = parser.parse_args(argv)

    proto = _protocol_stream()

    def emit(payload: dict[str, Any]) -> None:
        try:
            proto.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proto.flush()
        except (BrokenPipeError, OSError):
            os._exit(0)          # the parent went away — nothing left to serve

    root = os.path.abspath(args.root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # KokoClone downloads model/ and voice/ relative to the working directory.
    os.chdir(root)
    emit({"event": "starting"})

    state: dict[str, Any] = {"cloner": None, "error": "", "sample_rate": 24000}
    ready = threading.Event()

    def boot() -> None:
        try:
            from core.cloner import KokoClone  # type: ignore

            cloner = KokoClone()
            state["cloner"] = cloner
            state["sample_rate"] = int(getattr(cloner, "sample_rate", 24000) or 24000)
            emit({"event": "ready", "sample_rate": state["sample_rate"]})
        except Exception as e:  # noqa: BLE001 — reported, never a silent hang
            state["error"] = f"{type(e).__name__}: {e}"
            emit({"event": "error", "error": state["error"]})
        finally:
            ready.set()

    threading.Thread(target=boot, name="kokoclone-boot", daemon=True).start()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except ValueError:
            emit({"ok": False, "error": "worker received a non-JSON line"})
            continue

        job_id = job.get("id")
        ready.wait()
        if state["error"] or state["cloner"] is None:
            emit({"ok": False, "id": job_id,
                  "error": state["error"] or "KokoClone did not initialise"})
            continue

        text = str(job.get("text") or "").strip()
        if not text:
            emit({"ok": False, "id": job_id, "error": "empty text"})
            continue
        try:
            state["cloner"].generate(
                text=text,
                lang=str(job.get("lang") or "en"),
                reference_audio=str(job.get("ref") or ""),
                output_path=str(job.get("out") or "output.wav"),
            )
            emit({"ok": True, "id": job_id, "path": str(job.get("out") or ""),
                  "sample_rate": state["sample_rate"]})
        except Exception as e:  # noqa: BLE001 — the parent shows the reason
            detail = " ".join(traceback.format_exception_only(type(e), e)).strip()
            emit({"ok": False, "id": job_id, "error": detail[:400]})
    return 0


if __name__ == "__main__":
    sys.exit(main())

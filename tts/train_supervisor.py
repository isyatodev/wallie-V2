"""Detached Piper training supervisor — ``python -m tts.train_supervisor <run>``.

The dashboard starts this module in its own session and immediately forgets
about it. Everything the two need to share lives in the run folder:

* ``train-config.json`` — what to train (written by the dashboard);
* ``job.json``         — live status/progress/log tail (written here, read by
  every dashboard GET, so reopening the dashboard reconnects to the run);
* ``cancel``           — written by the dashboard to ask for a stop;
* ``train.log``        — the trainer's own output, appended live.

Usage is internal: only the dashboard's launch path calls it.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import piper_train


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m tts.train_supervisor <run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    if not run_dir.is_dir():
        print(f"run folder not found: {run_dir}", file=sys.stderr)
        return 2
    piper_train.supervise(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

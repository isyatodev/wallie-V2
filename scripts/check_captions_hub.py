#!/usr/bin/env python
"""Manual check for the CaptionHub (no network): produce, subscribe, clear."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captions import CaptionHub


async def main() -> None:
    hub = CaptionHub()

    # 1) Mid-speech: a client must see hello + the current line.
    hub.note_sentence("hello there, this is a caption")
    sub = hub.subscribe()
    frames = [await sub.__anext__() for _ in range(2)]   # hello + replayed update
    await sub.aclose()
    assert any("hello there" in f for f in frames), frames
    print("mid-speech replay OK:", len(frames), "frames")

    # 2) End-of-speech: a LATE client gets hello + clear — never the stale line.
    hub.speech_ended()
    sub = hub.subscribe()
    frames = [await sub.__anext__() for _ in range(2)]   # hello + clear
    await sub.aclose()
    assert any('"clear"' in f for f in frames), frames
    assert not any("hello there" in f for f in frames), frames
    print("post-speech clear OK:", frames)

    print("captions hub self-test PASSED")


if __name__ == "__main__":
    asyncio.run(main())

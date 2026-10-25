"""Smoke: the exact Build & Train path that died in the first training_step."""
import asyncio
import sys
from pathlib import Path

from tts import piper_train as pt

PKG = str(Path("voices/ported/luna-piper").resolve())


async def main() -> None:
    job = pt.new_job()
    pt.begin(job, kind="train", package=PKG, epochs=1, python=sys.executable)
    await pt.run_train(job, package=PKG, epochs=1, batch_size=8,
                       accelerator="cpu", python="")
    print("STATUS:", job["status"])
    if job.get("error"):
        print("ERROR:", job["error"])
    print("FLAVOR:", job.get("flavor"), "EPOCH:", job.get("epoch"))
    print("EXPORTED:", job.get("exported"))
    print("---- log tail ----")
    print("\n".join(job.get("log", "").splitlines()[-40:]))
    sys.stdout.flush()


asyncio.run(main())

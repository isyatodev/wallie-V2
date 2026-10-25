"""Smoke: the dashboard's own setup job (install train deps button)."""
import asyncio
import sys

from tts import piper_train as pt


async def main() -> None:
    job = pt.new_job()
    pt.begin(job, kind="setup", python=pt.interpreter(""), epochs=0)
    await pt.run_setup(job, "")
    print("STATUS:", job["status"])
    if job.get("error"):
        print("ERROR:", job["error"])
    print("---- log ----")
    print(job.get("log", ""))
    sys.stdout.flush()


asyncio.run(main())

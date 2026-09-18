"""Regenerate broken venv launcher executables (.exe shims on Windows).

If the project folder is renamed or moved, every .exe in .venv/Scripts keeps
the OLD absolute path to python.exe baked in and dies with
"Fatal error in launcher: Unable to create process". `python -m venv
--upgrade` rewrites pyvenv.cfg but NOT these launchers — the only supported
fix is reinstalling each owner package (pip regenerates its console scripts
on install).

This script maps each .exe in the venv's Scripts dir to the installed
distribution that declares it as a console script, then reinstalls that
distribution with --no-deps --force-reinstall at the SAME version, so no
package actually changes — only the launchers get rewritten with the
current path.

Usage: .venv/Scripts/python.exe scripts/repair_venv.py
Exit codes: 0 = launchers healthy (nothing to do or repaired), 1 = failure.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def _exe_targets(venv_scripts: pathlib.Path) -> set[str]:
    """Stems of .exe launchers present in the venv Scripts dir."""
    return {p.stem.lower() for p in venv_scripts.glob("*.exe")}


def _owner_packages(scripts_dir: pathlib.Path, exes: set[str]) -> dict[str, str]:
    """Map dist name -> version for dists owning one of the broken launchers."""
    from importlib.metadata import entry_points

    owners: dict[str, str] = {}
    for ep in entry_points(group="console_scripts"):
        dist = ep.dist
        if ep.name.lower() in exes and dist is not None:
            name = dist.metadata["Name"]
            if name:  # skip dists with no normalized name
                owners[name] = dist.version
    return owners


def _probe(venv_scripts: pathlib.Path) -> bool:
    """True if the venv's pip launcher can spawn a process (i.e. paths are sane)."""
    pip_exe = venv_scripts / "pip.exe"
    if not pip_exe.exists():
        return True  # nothing to repair; caller handles missing pip separately
    try:
        subprocess.run(
            [str(pip_exe), "--version"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return True
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    if sys.prefix == sys.base_prefix:
        print("[ERROR] repair_venv.py must run inside the project's venv "
              "(e.g. .venv\\Scripts\\python.exe scripts\\repair_venv.py)")
        return 1

    scripts_dir = pathlib.Path(sys.prefix) / (
        "Scripts" if os.name == "nt" else "bin"
    )
    if _probe(scripts_dir):
        print("[OK] venv launchers are healthy — no repair needed.")
        return 0

    print("[*] Broken venv launchers detected (stale absolute paths?). Repairing...")
    subprocess.run(
        [os.path.join(sys.base_prefix, "python.exe"), "-m", "venv",
         "--upgrade", sys.prefix],
        check=False,
    )

    exes = _exe_targets(scripts_dir)
    owners = _owner_packages(scripts_dir, exes)
    if not owners:
        print("[ERROR] Could not map any launcher to an owner package; "
              "recreate the venv:  rmdir /s /q .venv && install.bat")
        return 1

    print(f"[*] Regenerating {len(owners)} launcher(s) by reinstalling their owners...")
    failed: list[str] = []
    for name, ver in sorted(owners.items()):
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps",
               "--force-reinstall", "-q", f"{name}=={ver}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] {name}=={ver}")
        else:
            failed.append(name)
            print(f"  [!] {name}: {result.stderr.strip()[-200:]}")

    if failed:
        print(f"[ERROR] Could not repair launchers for: {', '.join(failed)}")
        return 1
    if not _probe(scripts_dir):
        print("[ERROR] pip.exe still failing after repair — recreate the venv.")
        return 1
    print("[OK] All venv launchers regenerated. (Packages untouched, same versions.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

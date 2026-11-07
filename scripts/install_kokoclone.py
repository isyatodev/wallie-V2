"""Clone and install KokoClone, the optional zero-shot cloner behind Kokoro.

KokoClone (https://github.com/Ashish-Patnaik/kokoclone) is a separate project:
Kokoro speaks the line and Kanade reshapes it into a short reference clip. It is
not vendored here and nothing is imported from it at import time, so it has to
be cloned and its requirements installed into the interpreter that runs Wallie.
This script is that install, in one command — and the dashboard's
**Voice Lab → KokoClone → install** button runs exactly this file.

Usage:
    python scripts/install_kokoclone.py --check        # is it ready? no changes
    python scripts/install_kokoclone.py --install      # clone + pip install
    python scripts/install_kokoclone.py --install --root D:\\tools\\kokoclone

Exit codes (so a wrapper can tell the cases apart):
    0  ready          2  packages missing      3  interpreter too old
    4  git not found  5  clone failed          6  no requirements.txt
    7  pip failed     8  the target is not an empty folder

What "ready" means: a checkout whose ``core/cloner.py`` exists, importable
``torch`` / ``torchaudio`` / ``kokoro_onnx`` / ``kanade_tokenizer`` /
``soundfile`` / ``misaki``, and Python 3.10-3.12 (KokoClone's own wheel range).
The ~1 GB of model weights are NOT downloaded here: KokoClone fetches them on
the first synthesis, which is why the Voice Lab says the first line is slow.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The status glyphs below (→ ✓ ✗ ·) crash a cp1252 Windows console, so force
# UTF-8 on stdout/stderr where the interpreter allows it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

REPO_URL = "https://github.com/Ashish-Patnaik/kokoclone.git"
# What the checkout imports to run (``core/cloner.py``). Kept in sync with
# tts/kokoclone.py's REQUIRED_PACKAGES — these are the import names.
PACKAGES = ("torch", "torchaudio", "kokoro_onnx", "kanade_tokenizer", "soundfile", "misaki")
# The pip specs that satisfy them. Deliberately NOT ``-r requirements.txt``:
# upstream also asks for ``misaki[ja,zh]`` (which builds pyopenjtalk with
# CMake/MSVC — it fails outright on a machine without a C++ toolchain) and for
# ``gradio``, their own web UI. Wallie's worker imports neither, and the Voice
# Lab offers the espeak languages (pt/en/es/fr/it/hi). ``--upstream`` installs
# their file verbatim for anyone who wants the whole app.
REQUIREMENTS = (
    "torch",
    "torchaudio",
    "soundfile",
    "huggingface_hub",
    "kokoro-onnx",
    "misaki",
    "git+https://github.com/frothywater/kanade-tokenizer",
)
MIN_PY = (3, 10)
MAX_PY = (3, 13)              # exclusive, per KokoClone's own badge
PY_RANGE_LABEL = "3.10–3.12"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _python_ok() -> bool:
    vi = sys.version_info
    return MIN_PY <= (vi[0], vi[1]) < MAX_PY


def _default_root() -> Path:
    override = (os.environ.get("WALLIE_KOKOCLONE_DIR") or "").strip()
    return Path(override).expanduser() if override else REPO_ROOT / "kokoclone"


def _is_checkout(root: Path) -> bool:
    """The one file that proves the clone is the project we expect."""
    return (root / "core" / "cloner.py").is_file()


def _missing_packages() -> list[str]:
    missing: list[str] = []
    for package in PACKAGES:
        try:
            found = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            found = False
        if not found:
            missing.append(package)
    return missing


def _print_python_problem() -> None:
    vi = sys.version_info
    print()
    print(f"✗ KokoClone needs Python {PY_RANGE_LABEL}, but this interpreter is "
          f"{vi[0]}.{vi[1]}.{vi[2]}.")
    print("  Its wheels have no builds for this version, so pip cannot install them.")
    print("  Run this script with the .venv Python of the profile that will speak —")
    print("  or point WALLIE_KOKOCLONE_DIR at a checkout that runs on a supported one.")


def _report(root: Path) -> int:
    """Print the status and return the matching exit code (no changes)."""
    checkout = _is_checkout(root)
    missing = _missing_packages()
    print(f"checkout   {root} — {'✓ found' if checkout else '✗ missing'}")
    print(f"python     {'✓' if _python_ok() else '✗'} "
          f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]} "
          f"(needs {PY_RANGE_LABEL})")
    print(f"packages   {'✓ all present' if not missing else '✗ missing ' + ', '.join(missing)}")
    if not _python_ok():
        return 3
    if not checkout:
        return 4
    return 0 if not missing else 2


def _clone(root: Path) -> int:
    """git clone the checkout into ``root`` (a missing or empty folder)."""
    git = shutil.which("git")
    if not git:
        print("✗ git is not on PATH — install git, or clone the repository by hand:")
        print(f"    git clone {REPO_URL} \"{root}\"")
        return 4
    if root.exists() and any(root.iterdir()):
        print(f"✗ {root} already exists and is not a KokoClone checkout.")
        print("  Move it away (or set WALLIE_KOKOCLONE_DIR to another folder) and retry.")
        return 8
    print(f"→ git clone --depth 1 {REPO_URL} \"{root}\"")
    try:
        rc = subprocess.call([git, "clone", "--depth", "1", REPO_URL, str(root)])
    except OSError as e:
        print(f"✗ could not run git: {e}")
        return 4
    if rc != 0:
        print(f"✗ git clone failed (exit {rc}) — check the connection and retry.")
        return 5
    if not _is_checkout(root):
        print(f"✗ the clone at {root} has no core/cloner.py — is that the right repository?")
        return 5
    print("✓ checkout ready")
    return 0


def _pip_install(root: Path, *, upstream: bool = False) -> int:
    requirements = root / "requirements.txt"
    if upstream:
        if not requirements.is_file():
            print(f"✗ {requirements} is missing — cannot install its pinned packages.")
            return 6
        print(f"→ pip install -r \"{requirements}\"")
        specs = ["-r", str(requirements)]
    else:
        print(f"→ pip install {' '.join(REQUIREMENTS)}")
        print("  (not requirements.txt: it also asks for misaki[ja,zh], which builds")
        print("   pyopenjtalk with CMake, and for gradio, their web UI — this worker needs")
        print("   neither. Use --upstream for their file verbatim.)")
        specs = list(REQUIREMENTS)
    print("  (the first install pulls PyTorch — a few GB; go make a coffee)")
    cmd = [sys.executable, "-m", "pip", "install", *specs]
    try:
        rc = subprocess.call(cmd, cwd=str(root))
    except OSError as e:
        print(f"✗ could not run pip: {e}")
        return 7
    if rc != 0:
        print(f"✗ pip install failed (exit {rc}).")
        print("  The log above has pip's own reason; a proxy or a missing wheel is the usual one.")
        return 7
    return 0


def install(root: Path, *, upstream: bool = False) -> int:
    """Clone (when needed) and install (when needed). Returns an exit code."""
    print("KokoClone — optional zero-shot cloning on top of Kokoro")
    print()
    if not _python_ok():
        _print_python_problem()
        return 3

    missing = _missing_packages()
    if not _is_checkout(root):
        rc = _clone(root)
        if rc != 0:
            return rc
        missing = _missing_packages()      # a clone does not install anything

    if missing:
        rc = _pip_install(root, upstream=upstream)
        if rc != 0:
            return rc
        missing = _missing_packages()

    print()
    if missing:
        print(f"✗ still missing after the install: {', '.join(missing)}")
        print("  Some of these may need a system package (espeak-ng, libsndfile) — see the")
        print("  KokoClone README, then run --check again.")
        return 2
    print("✓ KokoClone is installed.")
    print("  Next: refresh the dashboard's Voice Lab — the backend probes the packages on")
    print("  every request — then add a reference clip. The first synthesis downloads")
    print("  ~1 GB of models before it speaks.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="install_kokoclone",
        description="Clone and install KokoClone (optional zero-shot cloner behind Kokoro).",
    )
    ap.add_argument("--install", action="store_true",
                    help="clone the repository and pip-install its requirements")
    ap.add_argument("--check", action="store_true",
                    help="only report whether KokoClone can run here (no changes)")
    ap.add_argument("--root", default="",
                    help="checkout folder (default: WALLIE_KOKOCLONE_DIR, else ./kokoclone)")
    ap.add_argument("--upstream", action="store_true",
                    help="install the checkout's requirements.txt verbatim instead of the "
                         "subset Wallie's worker imports (needs a C++ toolchain for "
                         "pyopenjtalk, and pulls gradio)")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser() if (args.root or "").strip() else _default_root()
    if args.install:
        return install(root, upstream=args.upstream)
    return _report(root)


if __name__ == "__main__":
    sys.exit(main())

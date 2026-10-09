"""Detect the optional Kokoro local-TTS stack (and where its installer lives).

The dashboard shows a live install badge in the Voice tab, so this module answers
"is Kokoro usable right now?" cheaply — versions from package metadata, no heavy
imports (a bare ``import torch`` alone costs seconds) — and points at the
project's installer script for the one-click button.
"""
from __future__ import annotations

import platform
import sys
from importlib import metadata, util
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_kokoro.py"
MIN_KOKORO = "0.9.4"

# kokoro's multilingual releases (>= 0.9.4, needed for pt-BR) declare
# Requires-Python >=3.10,<3.13. On 3.13+ pip has no wheel to install, so the
# badge must say WHY the install cannot work instead of offering a button that
# would fail.
MIN_PY = (3, 10)
MAX_PY = (3, 13)          # exclusive
PY_RANGE_LABEL = "3.10–3.12"


def python_ok(version_info: tuple[int, ...] | None = None) -> bool:
    vi = version_info or sys.version_info
    return MIN_PY <= (vi[0], vi[1]) < MAX_PY


def python_version_str() -> str:
    return platform.python_version()

# lang_code -> extra pip requirement that language's front needs. Kept in sync
# with scripts/install_kokoro.py and dashboard/static/app.js KOKORO_LANGS.
LANG_EXTRAS: dict[str, str] = {"j": "misaki[ja]", "z": "misaki[zh]"}

# Languages the installer accepts — the endpoint validates against this.
LANG_CODES: tuple[str, ...] = ("a", "b", "p", "e", "f", "h", "i", "j", "z")


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive (broken dist-info)
        return None


def _importable(module: str) -> bool:
    try:
        return util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def detect() -> dict[str, Any]:
    """Cheap status snapshot: metadata only, never imports the heavy packages."""
    have_kokoro = _importable("kokoro")
    have_soundfile = _importable("soundfile")
    missing = [
        name
        for name, ok in (("kokoro", have_kokoro), ("soundfile", have_soundfile))
        if not ok
    ]
    py_ok = python_ok()
    return {
        "installed": have_kokoro,
        "kokoro_version": _version("kokoro"),
        "soundfile_version": _version("soundfile"),
        "torch_version": _version("torch"),
        "missing": missing,
        "installer_present": INSTALL_SCRIPT.is_file(),
        "min_version": MIN_KOKORO,
        "python_version": python_version_str(),
        "python_ok": py_ok,
        "python_range": PY_RANGE_LABEL,
        # A one-click install can only work on a supported interpreter with the
        # installer script present.
        "can_install": py_ok and INSTALL_SCRIPT.is_file(),
    }


def install_argv(lang: str = "a", voice: str = "") -> list[str]:
    """Command that installs the project-compatible Kokoro packages and
    pre-downloads the model + the voice the profile is configured to use."""
    argv = [sys.executable, str(INSTALL_SCRIPT), "--install", "--lang", lang]
    voice = (voice or "").strip()
    # Never let a crafted voice string masquerade as an installer flag.
    if voice and not voice.startswith("-"):
        argv += ["--voice", voice]
    return argv

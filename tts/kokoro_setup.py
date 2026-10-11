"""Detect the optional Kokoro local-TTS stack (and where its installer lives).

The dashboard shows a live install badge in the Voice tab, so this module answers
"is Kokoro usable right now?" cheaply — versions from package metadata, no heavy
imports (a bare ``import torch`` alone costs seconds) — and points at the
project's installer script for the one-click button.
"""
from __future__ import annotations

import platform
import re
import sys
import time
from importlib import metadata, util
from pathlib import Path
from typing import Any

import httpx

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


# ---------------------------------------------------------------------------
# Ready-made voices — list the catalogue, download only what the user picks
# ---------------------------------------------------------------------------
# Kokoro's voice packs live as individual ``<voice>.pt`` files in the model repo.
# Nothing is bundled with this project: the Voice tab lists the catalogue and
# the user downloads the ones they want into ``voices/kokoro/`` (the same
# ``voices/`` tree the Piper downloads use), so a downloaded voice works
# offline and never re-downloads.
HF_REPO = "hexgrad/Kokoro-82M"
HF_TREE_URL = f"https://huggingface.co/api/models/{HF_REPO}/tree/main/voices"
HF_VOICE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/voices/{{name}}.pt"
# Where downloaded ``<voice>.pt`` files live by default (module-level so tests
# can repoint it, like piper_setup.DEFAULT_VOICES_DIR).
DEFAULT_VOICES_DIR = REPO_ROOT / "voices" / "kokoro"
# The project's voice-download helper (the CLI alternative to the button).
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_kokoro_voice.py"

# A Kokoro voice id is ``<lang><gender>_<name>`` (af_heart, pf_dora, zf_xiaobei).
# Validated strictly so a crafted id can never be read as a CLI flag.
VOICE_NAME_RE = re.compile(r"^[a-z]{1,2}_[a-z0-9_]+$")

CATALOG_TTL_SEC = 3600.0
# name/size list is tiny; cached in-process like the Piper catalogue.
_catalog_cache: dict[str, Any] = {"at": 0.0, "data": None}


class KokoroCatalogError(RuntimeError):
    """The online voice list could not be loaded (network / bad payload)."""


def voices_dir() -> Path:
    """Directory scanned for downloaded ``*.pt`` voice files."""
    return DEFAULT_VOICES_DIR


def voice_path(name: str) -> Path:
    """On-disk path for one voice id (whether or not it exists yet)."""
    return voices_dir() / f"{name}.pt"


def is_valid_voice_name(name: str) -> bool:
    name = (name or "").strip()
    return bool(VOICE_NAME_RE.match(name))


def list_local_voices(directory: Path | None = None) -> list[dict[str, Any]]:
    """Voices already downloaded, sorted by name."""
    d = Path(directory) if directory is not None else voices_dir()
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(d.glob("*.pt")):
        try:
            size = p.stat().st_size
        except OSError:  # pragma: no cover - defensive
            size = 0
        out.append({"name": p.stem, "path": str(p), "size_bytes": size})
    return out


def download_argv(voice: str, dest: str = "") -> list[str]:
    """Command that downloads one voice pack. ``voice`` is validated first; a
    name that fails the check raises ValueError rather than reaching the shell."""
    name = (voice or "").strip()
    if not is_valid_voice_name(name):
        raise ValueError(
            "voice name must look like <lang><gender>_<name>, e.g. af_heart"
        )
    argv = [sys.executable, str(DOWNLOAD_SCRIPT), name]
    dest = (dest or "").strip()
    if dest:
        argv += ["--dest", dest]
    return argv


def catalog_voices(raw: Any) -> list[dict[str, Any]]:
    """Normalize the HuggingFace tree payload into a flat, sorted list."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path.endswith(".pt") or "/" not in path:
            continue
        name = path.rsplit("/", 1)[-1][:-3]
        if not is_valid_voice_name(name):
            continue
        out.append({"name": name, "size_bytes": int(item.get("size") or 0)})
    out.sort(key=lambda v: v["name"])
    return out


async def fetch_catalog(
    *,
    force: bool = False,
    ttl: float = CATALOG_TTL_SEC,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Every Kokoro voice pack the project can download, cached for ``ttl``.

    Raises :class:`KokoroCatalogError` with an actionable message on a network
    or payload failure — the UI shows it verbatim.
    """
    now = time.time()
    cached = _catalog_cache.get("data")
    if not force and cached is not None and (now - float(_catalog_cache.get("at") or 0)) < ttl:
        return cached

    own = client is None
    if own:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True
        )
    try:
        resp = await client.get(HF_TREE_URL)
    except httpx.HTTPError as e:
        raise KokoroCatalogError(f"could not reach the Kokoro voice list: {e}") from e
    finally:
        if own:
            await client.aclose()

    if resp.status_code >= 400:
        raise KokoroCatalogError(
            f"the voice list returned HTTP {resp.status_code} — try again shortly"
        )
    try:
        raw = resp.json()
    except ValueError as e:
        raise KokoroCatalogError("the voice list returned invalid JSON") from e

    voices = catalog_voices(raw)
    if not voices:
        raise KokoroCatalogError("the voice list was empty — try again shortly")
    _catalog_cache.update(data=voices, at=now)
    return voices

"""Detect the optional local Piper TTS stack and manage its voice files.

Mirrors :mod:`tts.kokoro_setup`: the dashboard shows a live badge in the Voice
tab (installed? which voice files are on disk?) and offers a one-click install
plus one-click voice downloads. Detection is cheap — package metadata and a
directory listing, never ``import piper`` (loading the native runtime is slow
and can fail on a broken install).
"""
from __future__ import annotations

import json
import re
import sys
import time
from importlib import metadata, util
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
# The project's voice-download helper (same script the docs tell users to run).
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "download_piper_voice.py"
# Where downloaded ``*.onnx`` (+ ``*.onnx.json``) voices live by default.
DEFAULT_VOICES_DIR = REPO_ROOT / "voices"

# A Piper voice name is ``<lang_REGION>-<voice>-<quality>`` (e.g.
# ``en_US-amy-medium``). Validated strictly so a crafted name can never be read
# as a command-line flag by the downloader.
VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9]+[A-Za-z0-9_]*-[A-Za-z0-9_]+-[A-Za-z0-9_]+$")

# The official catalogue of every published voice (name → files/metadata).
CATALOG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
CATALOG_TTL_SEC = 3600.0
# name -> {at, data}; the JSON is a few hundred KB, so it is cached in-process.
_catalog_cache: dict[str, Any] = {"at": 0.0, "data": None}


class PiperCatalogError(RuntimeError):
    """The online voice catalogue could not be loaded (network / bad payload)."""


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


def voices_dir() -> Path:
    """Directory scanned for ``*.onnx`` voice files (module-level so tests can
    repoint it)."""
    return DEFAULT_VOICES_DIR


def _voice_meta(onnx: Path) -> dict[str, Any]:
    """Cheap metadata for one voice: config presence + sample rate/language
    pulled from ``<name>.onnx.json`` when it parses."""
    cfg = Path(str(onnx) + ".json")
    meta: dict[str, Any] = {
        "name": onnx.stem,
        "path": str(onnx),
        "has_config": cfg.is_file(),
        "size_bytes": onnx.stat().st_size,
        "sample_rate": None,
        "language": "",
        "quality": onnx.stem.rsplit("-", 1)[-1] if "-" in onnx.stem else "",
    }
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            audio = data.get("audio") or {}
            meta["sample_rate"] = int(audio.get("sample_rate") or 0) or None
            meta["language"] = str(
                (data.get("language") or {}).get("code")
                or data.get("espeak", {}).get("voice")
                or ""
            )
        except (OSError, ValueError, TypeError):
            pass
    return meta


def list_voices(directory: Path | None = None) -> list[dict[str, Any]]:
    """Voices already on disk, sorted by name. A voice needs BOTH the ``.onnx``
    weights and the ``.onnx.json`` config to be usable; missing config is
    reported but the entry is still listed so the user can see the damage."""
    d = Path(directory) if directory is not None else voices_dir()
    if not d.is_dir():
        return []
    return [_voice_meta(p) for p in sorted(d.glob("*.onnx"))]


def is_valid_voice_name(name: str) -> bool:
    name = (name or "").strip()
    return bool(VOICE_NAME_RE.match(name))


def detect() -> dict[str, Any]:
    """Status snapshot for the Voice-tab badge: is ``piper`` importable, which
    voices exist, and can we install/download from here."""
    installed = _importable("piper")
    voices = list_voices()
    return {
        "installed": installed,
        "piper_tts_version": _version("piper-tts"),
        "piper_version": _version("piper"),
        "voices": voices,
        "voice_count": len(voices),
        "voices_dir": str(voices_dir()),
        "installer_present": INSTALL_SCRIPT.is_file(),
        "can_install": INSTALL_SCRIPT.is_file(),
    }


def install_argv() -> list[str]:
    """Install the Piper runtime into the running interpreter."""
    return [sys.executable, "-m", "pip", "install", "piper-tts"]


def download_argv(voice: str, dest: str = "") -> list[str]:
    """Command that downloads one voice from the HuggingFace catalogue.

    ``voice`` is validated first; a name that fails the check raises
    :class:`ValueError` rather than reaching the subprocess.
    """
    name = (voice or "").strip()
    if not is_valid_voice_name(name):
        raise ValueError(
            "voice name must look like <lang_REGION>-<voice>-<quality>, "
            "e.g. en_US-amy-medium"
        )
    argv = [sys.executable, str(INSTALL_SCRIPT), name]
    dest = (dest or "").strip()
    if dest:
        argv += ["--dest", dest]
    return argv


# ---------------------------------------------------------------------------
# Online catalogue (HuggingFace rhasspy/piper-voices)
# ---------------------------------------------------------------------------

def _normalize_catalog(raw: Any) -> list[dict[str, Any]]:
    """Turn the catalogue's ``name -> entry`` mapping into a flat, sorted list
    the UI can search. ``size_bytes`` covers just the model files (the .onnx
    weights plus its .onnx.json config), not the optional MODEL_CARD."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return out
    for key, item in raw.items():
        if not isinstance(item, dict):
            continue
        lang = item.get("language") or {}
        files = item.get("files") or {}
        size = 0
        for path, meta in (files.items() if isinstance(files, dict) else []):
            if not isinstance(meta, dict):
                continue
            if str(path).endswith((".onnx", ".onnx.json")):
                size += int(meta.get("size_bytes") or 0)
        name = str(item.get("key") or key)
        out.append({
            "name": name,
            "voice": str(item.get("name") or ""),
            "language": str(lang.get("code") or ""),
            "language_name": str(lang.get("name_english") or lang.get("name_native") or ""),
            "quality": str(item.get("quality") or ""),
            "speakers": int(item.get("num_speakers") or 1),
            "size_bytes": size,
        })
    out.sort(key=lambda v: (v["language"], v["name"]))
    return out


def normalize_catalog(raw: Any) -> list[dict[str, Any]]:
    """Public alias (tests and callers that already hold the JSON)."""
    return _normalize_catalog(raw)


async def fetch_catalog(
    *,
    force: bool = False,
    ttl: float = CATALOG_TTL_SEC,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Every voice the project can download, cached for ``ttl`` seconds.

    Raises :class:`PiperCatalogError` with an actionable message on a network or
    payload failure — the UI shows it verbatim instead of a traceback.
    """
    now = time.time()
    cached = _catalog_cache.get("data")
    if not force and cached is not None and (now - float(_catalog_cache.get("at") or 0)) < ttl:
        return cached

    own = client is None
    if own:
        # follow_redirects: the catalogue URL 302-redirects to the resolve-cache
        # host, so without this httpx would hand us the (non-JSON) redirect body.
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True
        )
    try:
        resp = await client.get(CATALOG_URL)
    except httpx.HTTPError as e:
        raise PiperCatalogError(f"could not reach the voice catalogue: {e}") from e
    finally:
        if own:
            await client.aclose()

    if resp.status_code >= 400:
        raise PiperCatalogError(
            f"the voice catalogue returned HTTP {resp.status_code} — try again shortly"
        )
    try:
        raw = resp.json()
    except ValueError as e:
        raise PiperCatalogError("the voice catalogue returned invalid JSON") from e

    voices = _normalize_catalog(raw)
    if not voices:
        raise PiperCatalogError("the voice catalogue was empty — try again shortly")
    _catalog_cache.update(data=voices, at=now)
    return voices

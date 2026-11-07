"""Piper training — turn a ported package into a local voice, from the dashboard.

A port package (see :mod:`tts.voice_port`) is a Piper dataset: 22.05 kHz WAVs
plus ``metadata.csv``. This module is the missing last mile — it runs the
trainer as a child process, streams its output into the Voice Lab, and cancels
it cleanly, so the user never has to leave a terminal open for hours.

Two trainers exist in the wild and both are supported:

* **piper1-gpl** (maintained, ``pip install piper-tts``) —
  ``python -m piper.train fit`` then ``python -m piper.train.export_onnx``.
  Its CSV is ``file.wav|text`` with the audio in ``--data.audio_dir``, so the
  package's ``wav/0001.wav|text`` rows are normalised into the run directory
  (the package itself is never rewritten).
* **rhasspy/piper** (archived, ``piper_train``) —
  ``piper_train.preprocess`` → ``piper_train`` → ``piper_train.export_onnx``.

Everything here is opt-in and heavy: training is CPU/GPU work measured in
minutes, torch is a multi-hundred-megabyte install, and a run is only tracked
while the dashboard process lives. The run directory therefore carries a
``run.json`` + ``train.log``, so a finished voice can still be attached after a
restart, and a training run is never a black box.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import piper_setup, voice_port

# One log tail is shipped on every poll; the full log stays in train.log.
MAX_LOG_CHARS = 12000

# The supervisor's file protocol — the dashboard and the detached trainer talk
# through these, never through a pipe that closes when the dashboard exits.
RUN_CONFIG_FILE = "train-config.json"
JOB_FILE = "job.json"
CANCEL_FILE = "cancel"
SUPERVISOR_LOG = "supervisor.log"
SUPERVISOR_MODULE = "tts.train_supervisor"
# Windows blocks an atomic replace while a reader has the file open; a few
# short retries ride over the dashboard's poll instead of losing the update.
WRITE_ATTEMPTS = 6
DEFAULT_EPOCHS = 200
MIN_EPOCHS = 1
MAX_EPOCHS = 5000
DEFAULT_BATCH = 8
MIN_BATCH = 1
MAX_BATCH = 64
ACCELERATORS = ["auto", "cpu", "gpu"]

# How long a child may take. Training is open-ended (hours on CPU); setup and
# export are bounded so a hung pip/ONNX step cannot wedge the panel forever.
SETUP_TIMEOUT = 3600.0
EXPORT_TIMEOUT = 1800.0

# The training extra pulls torch + lightning, which the inference runtime does
# not need — that is why the panel says so before starting the install.
TRAIN_EXTRA = "piper-tts[train]"

# An inference .onnx cannot be resumed like a checkpoint (no optimizer, no
# discriminator), so it is converted into a warm-start checkpoint inside the run
# and the trainer copies the weights whose name and shape still match.
REFERENCE_FILE = "reference.ckpt"


class TrainError(RuntimeError):
    """A failure worth showing verbatim (bad package, no trainer, …)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Environment probe — is there a trainer behind this interpreter?
# ---------------------------------------------------------------------------

# Runs in a CHILD interpreter: probing must never import torch here (that costs
# seconds in the dashboard process, and the user may point at another venv).
_PROBE_CODE = (
    "import json,platform,importlib.util as u\n"
    "def h(m):\n"
    "    try:\n"
    "        return u.find_spec(m) is not None\n"
    "    except Exception:\n"
    "        return False\n"
    "def ver(m):\n"
    "    try:\n"
    "        from importlib.metadata import version\n"
    "        return version(m)\n"
    "    except Exception:\n"
    "        return ''\n"
    "def piper_root():\n"
    "    try:\n"
    "        s = u.find_spec('piper')\n"
    "        locs = list(s.submodule_search_locations or []) if s else []\n"
    "        return str(locs[0]) if locs else ''\n"
    "    except Exception:\n"
    "        return ''\n"
    "def warmstart():\n"
    "    try:\n"
    "        s = u.find_spec('piper.train.vits.lightning')\n"
    "        with open(s.origin, encoding='utf-8') as f:\n"
    "            return 'warmstart_ckpt' in f.read()\n"
    "    except Exception:\n"
    "        return False\n"
    "print(json.dumps({\n"
    "    'version': platform.python_version(),\n"
    "    'gpl': h('piper.train'),\n"
    "    'legacy': h('piper_train'),\n"
    "    'torch': h('torch'),\n"
    "    'torch_v': ver('torch'),\n"
    "    'onnxscript': h('onnxscript'),\n"
    "    'lightning': h('lightning'),\n"
    "    'piper_root': piper_root(),\n"
    "    'warmstart': warmstart(),\n"
    "}))\n"
)

_PROBE_TTL = 20.0
_probe_cache: dict[str, dict[str, Any]] = {}


def _probe_sync(python: str) -> dict[str, Any]:
    import subprocess

    try:
        proc = subprocess.run(
            [python, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=str(piper_setup.REPO_ROOT),
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"version": "", "gpl": False, "legacy": False, "torch": False,
                "lightning": False, "error": str(e)[:200]}
    if proc.returncode != 0:
        return {"version": "", "gpl": False, "legacy": False, "torch": False,
                "lightning": False,
                "error": (proc.stderr or proc.stdout or "").strip()[:200]
                or f"exit {proc.returncode}"}
    try:
        raw = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"version": "", "gpl": False, "legacy": False, "torch": False,
                "lightning": False, "error": "the probe returned no JSON"}
    return raw if isinstance(raw, dict) else {}


def _torch_needs_onnxscript(torch_version: str) -> bool:
    """torch >= 2.9 exports through dynamo, which imports ``onnxscript``.

    piper's ``[train]`` extra only says ``torch>=2,<3``, so a fresh install
    resolves to a torch whose default export path needs a package the extra
    never declares. No torch at all (``""``) also counts: pip will pull the
    latest, which is past 2.9.
    """
    try:
        major, minor = (int(x) for x in re.split(r"[.+]", torch_version)[:2])
    except (TypeError, ValueError):
        return True
    return (major, minor) >= (2, 9)


def _env_payload(python: str, raw: dict[str, Any]) -> dict[str, Any]:
    flavor = "piper1-gpl" if raw.get("gpl") else ("legacy" if raw.get("legacy") else "")
    root = str(raw.get("piper_root") or "")
    # The legacy trainer has no Cython core to be missing; piper1-gpl's wheel
    # ships without it (see ensure_monotonic_align), so check the slot itself.
    monotonic = flavor != "piper1-gpl" or _core_installed(root)
    # The ONNX export runs through torch.onnx.export; with torch >= 2.9 that
    # imports onnxscript, which piper's [train] extra does not declare.
    export_missing = (not raw.get("onnxscript")) and _torch_needs_onnxscript(
        str(raw.get("torch_v") or "")
    )
    missing: list[str] = []
    if not flavor:
        missing.append(TRAIN_EXTRA)
    if not raw.get("torch"):
        missing.append("torch")
    if not raw.get("lightning"):
        missing.append("lightning")
    if export_missing:
        missing.append("onnxscript")
    ok = not missing
    if ok:
        hint = (
            "trainer ready — "
            + ("piper1-gpl (python -m piper.train)" if flavor == "piper1-gpl"
               else "legacy piper_train (rhasspy/piper)")
        )
        if flavor == "piper1-gpl" and not monotonic:
            hint += (
                " — note: this wheel has no monotonic_align core (an upstream "
                "packaging bug); training installs a pure-Python fallback"
            )
    elif not flavor:
        hint = (
            f"no Piper trainer behind this interpreter — install {TRAIN_EXTRA} "
            "or point at a venv that has it"
        )
    else:
        hint = (
            f"{flavor} is importable but {', '.join(m for m in missing if m != TRAIN_EXTRA)} "
            f"is missing — install {TRAIN_EXTRA} in that interpreter"
        )
    if export_missing:
        hint += (
            " — also onnxscript: torch 2.9+ exports the .onnx through it and "
            "piper's train extra never installs it"
        )
    return {
        "python": python,
        "version": str(raw.get("version") or ""),
        "flavor": flavor,
        "flavor_label": {
            "piper1-gpl": "piper1-gpl",
            "legacy": "legacy piper_train",
            "": "none",
        }[flavor],
        "torch": bool(raw.get("torch")),
        "lightning": bool(raw.get("lightning")),
        "monotonic": monotonic,
        "piper_root": root,
        "warmstart": bool(raw.get("warmstart")),
        "export_missing": export_missing,
        "missing": missing,
        "can_train": ok,
        "error": str(raw.get("error") or "")[:200],
        "hint": hint,
        "setup_command": f'"{python}" -m pip install --upgrade "{TRAIN_EXTRA}"'
        + (" onnxscript" if export_missing else ""),
    }


def interpreter(python: str = "") -> str:
    """The interpreter a run will use — the dashboard's own by default."""
    return (python or "").strip() or sys.executable


async def probe(python: str = "", *, force: bool = False) -> dict[str, Any]:
    """Which trainer/torch/lightning that interpreter has (cached, cheap)."""
    exe = interpreter(python)
    now = time.time()
    if not force:
        hit = _probe_cache.get(exe)
        if hit and (now - float(hit["at"])) < _PROBE_TTL:
            return dict(hit["data"])
    raw = await asyncio.to_thread(_probe_sync, exe)
    data = _env_payload(exe, raw)
    _probe_cache[exe] = {"at": now, "data": data}
    return dict(data)


def clear_probe_cache() -> None:
    _probe_cache.clear()


# ---------------------------------------------------------------------------
# monotonic_align — the alignment core piper's wheel never ships
# ---------------------------------------------------------------------------
#
# piper-tts (<=1.8.0, every platform) installs train/vits/monotonic_align
# with only __init__.py + setup.py: the compiled Cython core is absent from
# both the wheels and the sdist, so the first training step dies with
# ModuleNotFoundError and no pip reinstall fixes it. tts.monotonic_core is a
# numpy implementation of the same alignment search; when the slot is empty
# it is copied in as core.py, and a locally compiled core extension would
# still win, because extensions load before source files.

def _core_installed(root: str) -> bool:
    """Whether the trainer's alignment core slot has something importable."""
    if not root:
        return False
    slot = Path(root) / "train" / "vits" / "monotonic_align" / "monotonic_align"
    if not slot.is_dir():
        return False
    # core.py, core.pyd, core.cp312-win_amd64.pyd, core.cpython-312-…so —
    # but never the unbuilt leftovers (core.c, core.pyx).
    return any(p.suffix in {".py", ".pyd", ".so"} for p in slot.glob("core*"))


def ensure_monotonic_align(python: str, root: str = "") -> dict[str, Any]:
    """Install the pure-Python alignment core if the wheel left it out.

    Idempotent, and it reports instead of raising: the caller decides
    whether an empty core is fatal (it is, right before training). ``state``
    is ``present`` (nothing to do), ``installed`` (wrote core.py), ``absent``
    (no piper1-gpl install to heal) or ``error`` — with ``detail`` saying
    why, e.g. a read-only site-packages.
    """
    root = (root or "").strip() or _piper_root(python)
    if not root:
        return {"state": "absent"}
    base = Path(root) / "train" / "vits" / "monotonic_align"
    if not base.is_dir():
        return {"state": "absent"}
    if _core_installed(root):
        return {"state": "present"}
    from . import monotonic_core  # numpy — kept out of the dashboard's import

    slot = base / "monotonic_align"
    try:
        slot.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(monotonic_core.__file__, slot / "core.py")
    except OSError as e:
        return {"state": "error", "detail": str(e)[:200],
                "target": str(slot / "core.py")}
    return {"state": "installed", "target": str(slot / "core.py")}


def _piper_root(python: str) -> str:
    """The piper package root behind ``python`` (``""`` when not installed)."""
    return str(_probe_sync(interpreter(python)).get("piper_root") or "")


def ensure_export_compat(python: str, root: str = "") -> dict[str, Any]:
    """Make piper's exporter work on torch >= 2.9 — its own call does not.

    piper's ``export_onnx`` was written for the TorchScript exporter (it
    passes ``dynamic_axes`` at opset 15) but omits ``dynamo``, so on torch
    2.9+, where ``dynamo=True`` is the default, it runs the new exporter —
    which refuses this model (a data-dependent ``assert`` in the spline
    flow). Stock piper never mentions ``dynamo``, so the check is a plain
    marker scan and the patch is idempotent; a piper that someday passes the
    flag itself is left alone. Same report contract as
    :func:`ensure_monotonic_align`; callers pass ``root`` from the probe (an
    empty root means no known install, so nothing is touched).
    """
    root = (root or "").strip() or _piper_root(python)
    if not root:
        return {"state": "absent"}
    src = Path(root) / "train" / "export_onnx.py"
    if not src.is_file():
        return {"state": "absent"}
    try:
        with open(src, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
        if "dynamo=" in text:
            return {"state": "present"}
        anchor = text.find("torch.onnx.export(")
        if anchor < 0:
            return {
                "state": "error",
                "detail": "export_onnx.py no longer calls torch.onnx.export( — "
                          "this piper build needs dynamo=False and the file "
                          "changed shape",
                "target": str(src),
            }
        # Insert the flag as the call's first argument, in the file's own
        # indentation and line endings, so everything else stays byte-identical.
        nl = text.find("\n", anchor)
        eol = "\r\n" if text[nl - 1:nl] == "\r" else "\n"
        pos = nl + 1
        indent = re.match(r"[ \t]*", text[pos:]).group(0)
        line = (
            f"{indent}dynamo=False,  # torch >= 2.9 defaults to dynamo, which "
            f"cannot export this model (this call targets the legacy exporter){eol}"
        )
        with open(src, "w", encoding="utf-8", newline="") as fh:
            fh.write(text[:pos] + line + text[pos:])
    except OSError as e:
        return {"state": "error", "detail": str(e)[:200], "target": str(src)}
    return {"state": "installed", "target": str(src)}


def ensure_checkpoint_compat(python: str, root: str = "") -> dict[str, Any]:
    """Drop piper's optional ``val_mos`` ModelCheckpoint callback.

    piper's CLI always adds a second checkpoint callback monitoring
    ``val_mos``. That metric is only logged when the UTMOS predictor scores a
    test split, and this trainer deliberately sets
    ``num_test_examples=0`` — so as soon as validation runs, Lightning raises
    ``ModelCheckpoint(monitor='val_mos') could not find the monitored key`` at
    the end of the first epoch. A run without a validation split (the old
    three-clip case) never reaches the hook, which is why this stayed hidden.

    The ``val_mel`` callback is the one used for resume and export, so the
    optional callback is removed. Idempotent: a piper whose CLI no longer has
    it is left alone. Same report contract as
    :func:`ensure_export_compat`.
    """
    root = (root or "").strip() or _piper_root(python)
    if not root:
        return {"state": "absent"}
    src = Path(root) / "train" / "__main__.py"
    if not src.is_file():
        return {"state": "absent"}
    try:
        text = src.read_text(encoding="utf-8")
        if 'monitor="val_mos"' not in text:
            return {"state": "present"}
        patched, count = re.subn(
            r'\n    ModelCheckpoint\(\n        monitor="val_mos",.*?\n    \),?\n',
            "\n",
            text,
            count=1,
            flags=re.S,
        )
        if not count:
            return {
                "state": "error",
                "detail": "train/__main__.py has a val_mos callback this patch "
                          "cannot locate — this piper build changed shape",
                "target": str(src),
            }
        src.write_text(patched, encoding="utf-8")
    except OSError as e:
        return {"state": "error", "detail": str(e)[:200], "target": str(src)}
    return {"state": "installed", "target": str(src)}


def _heal_trainer(job: dict[str, Any], env: dict[str, Any]) -> None:
    """Heal the trainer's known gaps during setup, leaving a trace."""
    root = str(env.get("piper_root") or "")
    if env.get("flavor") == "piper1-gpl" and not env.get("monotonic", True):
        healed = ensure_monotonic_align(env["python"], root)
        if healed["state"] == "installed":
            job["log"] += (
                "• installed the pure-Python monotonic_align fallback — piper's "
                "wheel ships without the compiled core\n"
            )
        elif healed["state"] == "error":
            job["log"] += (
                f"⚠ could not write the alignment fallback: {healed['detail']}\n"
            )
    if root:
        compat = ensure_export_compat("", root)
        if compat["state"] == "installed":
            job["log"] += (
                "• patched piper's export with dynamo=False — torch 2.9+ "
                "defaults to the exporter this model cannot pass\n"
            )
        elif compat["state"] == "error":
            job["log"] += (
                f"⚠ could not patch piper's export_onnx.py: {compat['detail']}\n"
            )
        ckpt = ensure_checkpoint_compat("", root)
        if ckpt["state"] == "installed":
            job["log"] += (
                "• removed piper's optional val_mos checkpoint — this trainer "
                "has no test split to score it\n"
            )
        elif ckpt["state"] == "error":
            job["log"] += (
                f"⚠ could not remove piper's val_mos checkpoint: {ckpt['detail']}\n"
            )


# ---------------------------------------------------------------------------
# Packages — only Piper ports have a trainable dataset
# ---------------------------------------------------------------------------

def default_voice_name(manifest: dict[str, Any]) -> str:
    """The name Piper's loader expects: ``<lang_REGION>-<voice>-<quality>``."""
    lang_id = str((manifest.get("lang") or {}).get("id") or "pt").lower()
    region = {"pt": "pt_BR", "en": "en_US"}.get(lang_id, lang_id)
    # Piper's loader allows letters/digits/underscore inside each segment, so the
    # slug's hyphens would make the name invalid — turn them into underscores.
    stem = re.sub(
        r"[^A-Za-z0-9]+", "_", str(manifest.get("name") or "voice")
    ).strip("_").lower()[:40] or "voice"
    name = f"{region}-{stem}-medium"
    if not piper_setup.is_valid_voice_name(name):
        name = f"{region}-voice-medium"
    return name


def resolve_package(package: str) -> Path:
    """A package path, validated to live under this profile's ported folder."""
    wanted = (package or "").strip()
    if not wanted:
        raise TrainError("pick a ported package first — convert a voice to Piper")
    try:
        manifest = voice_port.read_manifest(wanted)
    except voice_port.VoicePortError as e:
        raise TrainError(str(e))
    target = (manifest.get("target") or {}).get("id")
    if target != "piper":
        raise TrainError(
            f"this package was made for {target or 'another engine'} — only a Piper "
            "package carries a trainable dataset"
        )
    path = Path(wanted).resolve()
    dataset = path / "dataset"
    if not (dataset / "metadata.csv").is_file():
        raise TrainError("this package has no dataset/metadata.csv — convert it again")
    if not any((dataset / "wav").glob("*.wav")):
        raise TrainError("this package has no clips in dataset/wav — convert it again")
    return path


def package_spec(package: str) -> dict[str, Any]:
    """Everything the panel needs about one selected package."""
    path = resolve_package(package)
    manifest = voice_port.read_manifest(path)
    lang = manifest.get("lang") or {}
    return {
        "package": str(path),
        "name": str(manifest.get("name") or path.name),
        "clips": int(manifest.get("clips") or 0),
        "seconds": float(manifest.get("seconds") or 0),
        "created_at": str(manifest.get("created_at") or ""),
        "lang": str(lang.get("label") or lang.get("id") or ""),
        "espeak": str(lang.get("espeak") or "en-us"),
        "sample_rate": int(
            (manifest.get("target") or {}).get("sample_rate") or 22050
        ),
        "source_provider": str((manifest.get("source") or {}).get("provider") or ""),
        "voice_name": default_voice_name(manifest),
        "last_run": read_run(path),
        "resume": latest_resume(path),
    }


def list_packages() -> list[dict[str, Any]]:
    """Every Piper port on disk, newest first (so a reload still trains)."""
    root = voice_port.ported_dir()
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for manifest_path in root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (manifest.get("target") or {}).get("id") != "piper":
            continue
        pkg = manifest_path.parent
        if not (pkg / "dataset" / "metadata.csv").is_file():
            continue
        lang = manifest.get("lang") or {}
        out.append({
            "package": str(pkg),
            "name": str(manifest.get("name") or pkg.name),
            "clips": int(manifest.get("clips") or 0),
            "seconds": float(manifest.get("seconds") or 0),
            "created_at": str(manifest.get("created_at") or ""),
            "lang": str(lang.get("label") or lang.get("id") or ""),
            "espeak": str(lang.get("espeak") or "en-us"),
            "sample_rate": int((manifest.get("target") or {}).get("sample_rate") or 22050),
            "source_provider": str((manifest.get("source") or {}).get("provider") or ""),
            "voice_name": default_voice_name(manifest),
            "last_run": read_run(pkg),
            "resume": latest_resume(pkg),
        })
    out.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# Recipe — the exact argv for the trainer that interpreter has
# ---------------------------------------------------------------------------

def normalize_csv(csv_path: Path, dest: Path) -> int:
    """Rewrite ``wav/0001.wav|text`` as ``0001.wav|text`` (piper1-gpl shape).

    piper1-gpl joins the CSV's first column onto ``--data.audio_dir``; keeping
    ``wav/`` would look for ``wav/wav/0001.wav``. The package's own CSV is left
    untouched (the legacy preprocessor expects that layout), so the normalised
    copy lives in the run directory.
    """
    rows: list[str] = []
    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "|" not in line:
            continue
        name, text = line.split("|", 1)
        name = Path(name.strip()).name
        if not name or not text.strip():
            continue
        rows.append(f"{name}|{text.strip()}")
    if not rows:
        raise TrainError(
            "dataset/metadata.csv has no usable rows — expected file|text per line"
        )
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def new_run_dir(package: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = package / "train"
    base.mkdir(parents=True, exist_ok=True)
    run = base / f"run-{stamp}"
    suffix = 1
    while run.exists():
        run = base / f"run-{stamp}-{suffix}"
        suffix += 1
    run.mkdir(parents=True)
    return run


def prepare_run(package: Path, run_dir: Path, flavor: str, voice_name: str) -> dict[str, Any]:
    """Write the run's own CSV/config paths; return what the argv needs."""
    manifest = voice_port.read_manifest(package)
    dataset = package / "dataset"
    lang = manifest.get("lang") or {}
    rate = int((manifest.get("target") or {}).get("sample_rate") or 22050)
    prepared: dict[str, Any] = {
        "dataset_dir": str(dataset),
        "audio_dir": str(dataset / "wav"),
        "csv_path": "",
        "config_path": str(run_dir / f"{voice_name}.json"),
        "espeak": str(lang.get("espeak") or "en-us"),
        "sample_rate": rate,
        # Measured from the clips the trainer will actually read, not trusted
        # from the manifest: a missing/renamed WAV must not look like speech.
        "seconds": dataset_seconds(dataset / "metadata.csv", dataset / "wav"),
    }
    if flavor == "piper1-gpl":
        csv_path = run_dir / "train.csv"
        prepared["clips"] = normalize_csv(dataset / "metadata.csv", csv_path)
        prepared["csv_path"] = str(csv_path)
    else:
        prepared["clips"] = count_csv_rows(dataset / "metadata.csv")
    return prepared


def count_csv_rows(csv_path: Path) -> int:
    """Usable ``file|text`` rows (what the split is computed from)."""
    count = 0
    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        if "|" in raw and raw.split("|", 1)[1].strip():
            count += 1
    return count


def _wav_seconds(path: Path) -> float:
    """Duration of one dataset clip (0.0 when it cannot be read as WAV)."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / float(rate) if rate else 0.0
    except (OSError, EOFError, wave.Error):
        return 0.0


def dataset_seconds(csv_path: Path, audio_dir: Path) -> float:
    """Total speech in a Piper dataset, from the CSV rows the trainer reads.

    The package manifest records what was captured, but training reads
    ``metadata.csv`` + the WAVs, so this measures those. A row whose file is
    missing (or is not a WAV) contributes nothing — that is exactly the case a
    duration gate has to catch, and the trainer will skip it anyway.
    """
    total = 0.0
    try:
        lines = csv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0
    for raw in lines:
        line = raw.strip()
        if "|" not in line:
            continue
        name, _, text = line.partition("|")
        if not text.strip():
            continue
        name = Path(name.strip()).name
        if name:
            total += _wav_seconds(audio_dir / name)
    return round(total, 2)


def preprocess_argv(env: dict[str, Any], prepared: dict[str, Any], run_dir: Path) -> list[str]:
    """Legacy piper_train.preprocess — cache the dataset for the trainer."""
    return [
        env["python"], "-m", "piper_train.preprocess",
        "--language", prepared["espeak"],
        "--input-dir", prepared["dataset_dir"],
        "--output-dir", str(run_dir / "dataset"),
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", str(prepared["sample_rate"]),
        "--cache-dir", str(run_dir / "cache"),
    ]


def reference_argv(python: str, onnx: str, out: Path) -> list[str]:
    """Convert an inference .onnx into a warm-start checkpoint.

    Runs with the TRAINING interpreter: piper-tts[train] is what installs torch
    and onnx there, and the dashboard's own venv is kept free of torch. The
    script is called by path so the training venv does not need this package on
    its import path.
    """
    script = Path(__file__).with_name("piper_reference.py")
    return [python, str(script), "--onnx", onnx, "--out", str(out)]


def train_argv(
    env: dict[str, Any],
    prepared: dict[str, Any],
    run_dir: Path,
    *,
    voice_name: str,
    epochs: int,
    batch_size: int,
    accelerator: str,
    checkpoint: str = "",
    warmstart: bool = False,
) -> list[str]:
    """The training command for the detected trainer.

    ``checkpoint`` is either a Lightning .ckpt (strict resume: optimizer and
    epoch come along) or a checkpoint converted from an inference .onnx
    (``warmstart``: piper copies the weights by name and shape and starts a
    fresh optimizer, which is all a .onnx can support).
    """
    if env["flavor"] == "piper1-gpl":
        argv = [
            env["python"], "-m", "piper.train", "fit",
            "--data.voice_name", voice_name,
            "--data.csv_path", prepared["csv_path"],
            "--data.audio_dir", prepared["audio_dir"],
            "--model.sample_rate", str(prepared["sample_rate"]),
            "--data.espeak_voice", prepared["espeak"],
            "--data.cache_dir", str(run_dir / "cache"),
            "--data.config_path", prepared["config_path"],
            "--data.batch_size", str(batch_size),
            # One held-out clip is enough to make "last.ckpt" meaningful; the
            # test split is dead weight on a 1–3 minute dataset, so drop it.
            "--data.validation_split", f"{_validation_split(prepared):.4f}",
            "--data.num_test_examples", "0",
            # Workers would be extra processes to orphan on Windows cancel.
            "--data.num_workers", "0",
            "--trainer.accelerator", accelerator,
            "--trainer.devices", "1",
            "--trainer.max_epochs", str(epochs),
            "--trainer.default_root_dir", str(run_dir),
        ]
        if checkpoint:
            argv += (["--model.warmstart_ckpt", checkpoint] if warmstart
                     else ["--ckpt_path", checkpoint])
        return argv
    if warmstart:
        raise TrainError(
            "the legacy piper_train has no warm start — use the piper1-gpl "
            "trainer (piper-tts[train]) or a .ckpt checkpoint"
        )
    argv = [
        env["python"], "-m", "piper_train",
        "--dataset-dir", str(run_dir / "dataset"),
        "--accelerator", accelerator,
        "--devices", "1",
        "--batch-size", str(batch_size),
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(epochs),
        "--checkpoint-epochs", "1",
        "--precision", "32",
        "--default_root_dir", str(run_dir),
    ]
    if checkpoint:
        argv += ["--resume_from_checkpoint", checkpoint]
    return argv


def _validation_split(prepared: dict[str, Any]) -> float:
    """Hold out at least one clip, even on a 3-clip package.

    piper1-gpl computes ``valid_set_size = int(n * validation_split)`` — it
    floors. Asking for exactly ``1 / n`` is 0.3333… for three clips, which
    floors to **zero**: the val_dataloader ends up empty, Lightning never runs
    validation, ``val_mel``/``val_mos`` are never logged, ModelCheckpoint has
    nothing to monitor and the run silently exports its last (worst) epoch.
    Asking for 1.5 clips leaves the floor at one despite float error, while
    staying within the 0.1–0.5 range piper expects.
    """
    clips = max(1, int(prepared.get("clips") or 0))
    return min(0.5, max(0.1, 1.5 / clips))


def export_argv(env: dict[str, Any], checkpoint: Path, onnx: Path) -> list[str]:
    if env["flavor"] == "piper1-gpl":
        return [
            env["python"], "-m", "piper.train.export_onnx",
            "--checkpoint", str(checkpoint),
            "--output-file", str(onnx),
        ]
    return [
        env["python"], "-m", "piper_train.export_onnx",
        str(checkpoint), str(onnx),
    ]


def find_checkpoint(run_dir: Path) -> Path | None:
    """Newest checkpoint under the run — ``last.ckpt`` wins when it exists.

    ``reference.ckpt`` is the warm-start copy of an inference .onnx, not a
    training checkpoint: exporting or resuming it would ship the reference
    instead of the run.
    """
    ckpts = [p for p in run_dir.rglob("*.ckpt")
             if p.is_file() and p.name != REFERENCE_FILE]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    for path in reversed(ckpts):
        if "last" in path.name.lower():
            return path
    return ckpts[-1]


def sidecar_config(env: dict[str, Any], prepared: dict[str, Any], run_dir: Path) -> Path:
    """The config JSON the exporter does not write (it is produced by training)."""
    candidates = [Path(prepared["config_path"])]
    if env["flavor"] == "legacy":
        candidates.append(run_dir / "dataset" / "config.json")
    candidates += sorted(run_dir.glob("*.json"))
    for path in candidates:
        if path.is_file() and path.name != "run.json":
            return path
    raise TrainError(
        "training finished but wrote no config.json — the .onnx needs it as "
        ".onnx.json; check the log for the trainer's own error"
    )


# ---------------------------------------------------------------------------
# Progress — Lightning writes "Epoch 3: 100%|…| 8/8 [00:02<…]"
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"\bEpoch\s+(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def parse_epoch(line: str, epochs: int) -> dict[str, Any] | None:
    """Epoch number + fraction done, from one log line (None if unrelated)."""
    m = _EPOCH_RE.search(line or "")
    if not m:
        return None
    epoch = int(m.group(1))
    frac = 0.0
    step = _STEP_RE.search(line, m.end())
    if step:
        done, total = int(step.group(1)), int(step.group(2))
        if total > 0:
            frac = min(1.0, done / total)
    total_epochs = max(1, int(epochs))
    pct = min(100.0, ((epoch + frac) / total_epochs) * 100.0)
    return {"epoch": min(epoch, total_epochs), "percent": round(pct, 1)}


# ---------------------------------------------------------------------------
# Job state — one run at a time, like the Piper/Kokoro download jobs
# ---------------------------------------------------------------------------

def new_job() -> dict[str, Any]:
    return {
        "running": False,
        "status": "idle",          # idle | running | done | error | cancelled
        "kind": "",                # train | setup
        "phase": "",               # preprocess | training | export
        "phase_label": "",
        "progress": 0,
        "epoch": 0,
        "epochs": 0,
        "voice_name": "",
        "package": "",
        "run_dir": "",
        "flavor": "",
        "python": "",
        "clips": 0,
        "seconds": 0.0,
        "batch_size": DEFAULT_BATCH,
        "started_at": None,
        "finished_at": None,
        "error": "",
        "cancelled": False,
        "resumed_from": {},
        "warmstart": False,
        "reference": {},
        "log": "",
        "steps": [],
        "exported": {},
        # Auto-port: the supervisor installs the .onnx itself when the run was
        # started with an attach spec (server-side, no dashboard needed).
        "attached": {},
        "attach_error": "",
    }


def snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Serialisable view (the live process handle stays inside the job)."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def begin(job: dict[str, Any], *, kind: str, package: str = "", voice_name: str = "",
          epochs: int = DEFAULT_EPOCHS, python: str = "", flavor: str = "",
          resumed_from: dict[str, Any] | None = None) -> None:
    """Mark the job started before the task is spawned (no double-start race)."""
    job.update(
        running=True, status="running", kind=kind, phase="", phase_label="starting…",
        progress=0, epoch=int((resumed_from or {}).get("epoch") or 0), epochs=epochs,
        voice_name=voice_name, package=package, run_dir="", flavor=flavor,
        python=python, clips=0, seconds=0.0, batch_size=DEFAULT_BATCH,
        started_at=time.time(), finished_at=None, error="",
        cancelled=False, resumed_from=dict(resumed_from or {}), log="",
        warmstart=False, reference={},
        steps=[], exported={}, notified={}, attached={}, attach_error="",
    )
    job["_saved_epoch"] = -1          # per-run cache for the epoch checkpoint write


async def cancel(job: dict[str, Any]) -> bool:
    """Ask the run to stop; returns False when there was nothing to stop."""
    if not job.get("running"):
        return False
    job["cancelled"] = True
    event = job.get("_cancel")
    if event is not None:
        event.set()
    proc = job.get("_proc")
    if proc is not None:
        await _terminate(proc)
    return True


async def _terminate(proc: Any) -> None:
    """Stop the child, then its process group where the platform has one."""
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
    except (ProcessLookupError, OSError):
        pass


async def _spawn(argv: list[str], *, cwd: Path, env: dict[str, str]) -> Any:
    """Child process factory (monkeypatched by tests)."""
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        import subprocess
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **kwargs,
    )


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


async def _stream(
    job: dict[str, Any],
    argv: list[str],
    *,
    cwd: Path,
    log_file: Any | None,
    on_line: Callable[[str], None],
    timeout: float | None = None,
    on_spawn: Callable[[], None] | None = None,
) -> int:
    """Run one step, feeding each line to the job; -1 when it was cancelled."""
    try:
        proc = await _spawn(argv, cwd=cwd, env=_child_env())
    except Exception as e:  # noqa: BLE001 — a bad interpreter path is user input
        raise TrainError(f"could not start the trainer: {str(e)[:220]}")
    job["_proc"] = proc
    if on_spawn is not None:
        # The child pid must reach job.json before the first log line: a cancel
        # from a restarted dashboard has nothing else to signal.
        try:
            on_spawn()
        except Exception:  # noqa: BLE001
            pass
    pending = ""
    deadline = time.time() + timeout if timeout else None
    try:
        while True:
            if job["cancelled"]:
                await _terminate(proc)
                return -1
            if deadline is not None and time.time() > deadline:
                await _terminate(proc)
                raise TrainError(
                    f"the step took longer than {int(timeout or 0)}s and was stopped"
                )
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=0.4)
            except asyncio.TimeoutError:
                continue
            except Exception as e:  # noqa: BLE001 — the pipe can close on a crash
                raise TrainError(f"could not read the trainer output: {str(e)[:200]}")
            if not chunk:
                break
            pending += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\r\n]", pending)
            pending = parts.pop()
            for line in parts:
                if not line.strip():
                    continue
                on_line(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()
        if pending.strip():
            on_line(pending)
            if log_file is not None:
                log_file.write(pending + "\n")
                log_file.flush()
        await proc.wait()
        if job["cancelled"]:
            return -1
        return int(proc.returncode or 0)
    finally:
        job["_proc"] = None


def _stopped_by_user(job: dict[str, Any], run_dir: Path) -> bool:
    """Whether a failed step is really the user's stop.

    ``cancel`` writes the marker and terminates the child directly, so the
    process can exit before the supervisor's 0.5 s watcher has copied the flag
    onto the job — without reading the marker here, the stop would surface as
    "training exited with an error".
    """
    if job.get("cancelled"):
        return True
    try:
        return (Path(run_dir) / CANCEL_FILE).is_file()
    except OSError:
        return False


def _write_run(run_dir: Path, record: dict[str, Any]) -> None:
    """Persist the run so a restart can still show/attach its model."""
    target = run_dir / "run.json"
    tmp = run_dir / "run.json.tmp"
    try:
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        pass


def read_run(package: Path | str) -> dict[str, Any] | None:
    """The newest finished/attempted run of a package, or None."""
    try:
        base = Path(package) / "train"
    except TypeError:
        return None
    if not base.is_dir():
        return None
    runs = sorted(
        (p for p in base.glob("run-*") if (p / "run.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return None

    def _load(path: Path) -> dict[str, Any] | None:
        try:
            record = json.loads((path / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        record["run_dir"] = str(path)
        exported = record.get("exported") or {}
        record["attachable"] = bool(
            record.get("status") == "done"
            and exported.get("onnx")
            and Path(exported["onnx"]).is_file()
            and exported.get("config")
            and Path(exported["config"]).is_file()
        )
        return record

    attempts = [r for r in (_load(path) for path in runs) if r is not None]
    if not attempts:
        return None
    # A failed newest attempt must not hide an older voice that did finish:
    # attach is what the panel offers, so the newest attachable run wins.
    for record in attempts:
        if record["attachable"]:
            return record
    return attempts[0]


def _epoch_from_checkpoint(path: Path) -> int:
    """The epoch a Lightning checkpoint filename carries (``epoch=137-…``)."""
    match = re.search(r"epoch=(\d+)", path.name)
    return int(match.group(1)) if match else 0


def latest_resume(package: Path | str) -> dict[str, Any] | None:
    """Offer to continue the package's **last** run when it stopped short.

    A cancelled run, a failed one and a run whose dashboard died all share this
    shape: not ``done``, a ``run.json`` and at least one ``.ckpt`` — exactly
    what makes a resume button honest instead of retraining from zero.

    Only the newest run counts. Once a later run finishes, the model that run
    exported supersedes the older checkpoint, and resuming the old branch would
    silently throw away the newer one.
    """
    try:
        base = Path(package) / "train"
    except TypeError:
        return None
    if not base.is_dir():
        return None
    runs = sorted(
        (p for p in base.glob("run-*") if (p / "run.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return None
    run = runs[0]
    try:
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("status") == "done":
        return None
    checkpoint = find_checkpoint(run)
    if checkpoint is None:
        return None
    return {
        "run_dir": str(run),
        "checkpoint": str(checkpoint),
        "epoch": int(record.get("epoch") or 0) or _epoch_from_checkpoint(checkpoint),
        "epochs": int(record.get("epochs") or DEFAULT_EPOCHS),
        "voice_name": str(record.get("voice_name") or ""),
        "status": str(record.get("status") or ""),
    }


def resolve_resume(package: Path | str, run_dir: str) -> dict[str, Any]:
    """Validate a run the user wants to continue and find its checkpoint."""
    pkg = resolve_package(str(package))
    train_root = (pkg / "train").resolve()
    try:
        run = Path(run_dir or "").resolve()
    except OSError as e:
        raise TrainError(f"bad run folder: {e}")
    if train_root not in run.parents:
        raise TrainError("that run folder is not one of this package's training runs")
    record: dict[str, Any] = {}
    meta = run / "run.json"
    if meta.is_file():
        try:
            record = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
    if record.get("status") == "done":
        raise TrainError("that run already finished — attach its model or start a new run")
    checkpoint = find_checkpoint(run)
    if checkpoint is None:
        raise TrainError("that run has no .ckpt to resume from")
    return {
        "run_dir": str(run),
        "checkpoint": str(checkpoint),
        "epoch": int(record.get("epoch") or 0) or _epoch_from_checkpoint(checkpoint),
        "epochs": int(record.get("epochs") or DEFAULT_EPOCHS),
        "voice_name": str(record.get("voice_name") or ""),
    }


def validate_start(
    *,
    package: str,
    voice_name: str,
    epochs: int,
    batch_size: int,
    accelerator: str,
    python: str,
    checkpoint: str,
    resume_from: str = "",
) -> dict[str, Any]:
    """Normalise/validate the request; raises :class:`TrainError` with a fix."""
    spec = package_spec(package)
    resume = resolve_resume(spec["package"], resume_from) if (resume_from or "").strip() else {}
    name = ((voice_name or "").strip()
            or (resume.get("voice_name") if resume else "")
            or spec["voice_name"])
    if not piper_setup.is_valid_voice_name(name):
        raise TrainError(
            f"“{name}” is not a Piper voice name — use <lang_REGION>-<voice>-<quality>, "
            f"e.g. {spec['voice_name']}"
        )
    try:
        epochs = int(epochs)
    except (TypeError, ValueError):
        epochs = DEFAULT_EPOCHS
    epochs = max(MIN_EPOCHS, min(MAX_EPOCHS, epochs))
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH
    batch_size = max(MIN_BATCH, min(MAX_BATCH, batch_size))
    accel = (accelerator or "auto").strip().lower()
    if accel not in ACCELERATORS:
        raise TrainError(f"unknown accelerator {accelerator!r} — use {', '.join(ACCELERATORS)}")
    exe = interpreter(python)
    if python.strip():
        path = Path(python.strip())
        if not path.is_file():
            raise TrainError(f"training interpreter not found: {python.strip()}")
        exe = str(path.resolve())
    warmstart = False
    if resume:
        # A resumed run continues the checkpoint it stopped at; a separate
        # checkpoint field would be a second, contradictory starting point.
        ckpt = resume["checkpoint"]
    else:
        ckpt = (checkpoint or "").strip()
        if ckpt:
            path = Path(ckpt)
            if not path.is_file():
                raise TrainError(f"reference not found: {ckpt}")
            suffix = path.suffix.lower()
            if suffix == ".onnx":
                # An inference model: warm start from its weights.
                warmstart = True
            elif suffix != ".ckpt":
                raise TrainError(
                    "the reference must be a .ckpt training checkpoint or an "
                    ".onnx voice"
                )
            # The child runs with cwd=run_dir, so a relative path would not resolve.
            ckpt = str(path.resolve())
    return {
        "package": spec["package"],
        "voice_name": name,
        "epochs": epochs,
        "batch_size": batch_size,
        "accelerator": accel,
        "python": exe,
        "checkpoint": ckpt,
        "warmstart": warmstart,
        "resume": resume,
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------

async def run_train(
    job: dict[str, Any],
    *,
    package: str,
    voice_name: str = "",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    accelerator: str = "auto",
    python: str = "",
    checkpoint: str = "",
    warmstart: bool = False,
    resume_from: str = "",
    run_dir: Path | None = None,
    on_change: Callable[[], None] | None = None,
) -> None:
    """Train + export, streaming progress and the log onto ``job``.

    ``run_dir`` lets the detached supervisor own the folder it already created;
    ``on_change`` is called whenever the job snapshot changes, so the supervisor
    can mirror it to ``job.json`` for a dashboard that may not be running.
    """
    job["_cancel"] = asyncio.Event()
    log_file = None
    if job.get("cancelled"):
        job.update(status="cancelled", running=False, finished_at=time.time(),
                   phase="cancelled", phase_label="cancelled")
        return
    record: dict[str, Any] = {
        "format": "wallie.train",
        "version": 1,
        "created_at": _now_iso(),
        "status": "running",
        "package": package,
        "voice_name": voice_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "accelerator": accelerator,
        "python": interpreter(python),
        "checkpoint": checkpoint,
        "warmstart": bool(warmstart),
        "reference": {},
        "resumed_from": {},
        "epoch": 0,
        "error": "",
        "exported": {},
        "steps": [],
    }

    def log(line: str) -> None:
        job["log"] = (job["log"] + line + "\n")[-MAX_LOG_CHARS:]

    def log_failure(message: str) -> None:
        """Put a failure where the user will look: the job and train.log.

        The panel reads ``job["error"]``, but a restarted dashboard only reads
        train.log for its log tail — an error that exists only in memory is
        gone exactly when the user goes looking for it.
        """
        banner = "=" * 72
        log(f"✗ TRAINING FAILED — {message}")
        if log_file is not None:
            try:
                log_file.write(f"\n{banner}\nTRAINING FAILED\n{message}\n{banner}\n")
                log_file.flush()
            except OSError:
                pass

    def notify() -> None:
        if on_change is None:
            return
        try:
            on_change()
        except Exception:  # noqa: BLE001 — persistence must never break training
            pass

    try:
        options = validate_start(
            package=package,
            voice_name=voice_name,
            epochs=job.get("epochs") or epochs,
            batch_size=batch_size,
            accelerator=accelerator,
            python=python,
            checkpoint=checkpoint,
            resume_from=resume_from,
        )
        job["voice_name"] = options["voice_name"]
        job["epochs"] = options["epochs"]
        job["python"] = options["python"]
        warmstart = bool(options.get("warmstart"))
        job["warmstart"] = warmstart
        record["warmstart"] = warmstart
        resume = options.get("resume") or {}
        job["resumed_from"] = dict(resume)

        env = await probe(options["python"], force=True)
        job["flavor"] = env["flavor"]
        if not env["can_train"]:
            raise TrainError(env["hint"] + " — press “install train deps” or set another interpreter")
        if warmstart and env["flavor"] != "piper1-gpl":
            raise TrainError(
                "an .onnx reference needs the piper1-gpl trainer (python -m "
                "piper.train), but this interpreter has the legacy piper_train — "
                "install piper-tts[train] in it, or start from a .ckpt"
            )
        if warmstart and not env.get("warmstart"):
            raise TrainError(
                "this piper-tts build has no warm-start option "
                "(--model.warmstart_ckpt) — update it with "
                'pip install -U "piper-tts[train]" or start from a .ckpt'
            )
        if env["flavor"] == "piper1-gpl" and not env.get("monotonic", True):
            # The wheel ships without the alignment core, so fit would die in
            # its first training_step. Heal it before any prep, and refuse to
            # start at all when the fallback cannot be written either.
            healed = ensure_monotonic_align(
                options["python"], str(env.get("piper_root") or "")
            )
            if healed["state"] == "installed":
                log(
                    "• piper's wheel ships without its monotonic_align core — "
                    "installed the pure-Python alignment fallback (same MAS)"
                )
            elif healed["state"] == "error":
                raise TrainError(
                    "the trainer has no monotonic_align core and the fallback "
                    f"could not be written ({healed['detail']}) — make the venv "
                    "writable or build the core with MSVC, then start again"
                )
        if env["flavor"] == "piper1-gpl" and env.get("piper_root"):
            # Export only runs an hour in; patch piper's call now, so a torch
            # whose default exporter cannot pass this model fails the run
            # before any training happens instead of after it.
            compat = ensure_export_compat("", str(env["piper_root"]))
            if compat["state"] == "installed":
                log(
                    "• patched piper's export with dynamo=False — torch 2.9+ "
                    "defaults to the exporter this model cannot pass"
                )
            elif compat["state"] == "error":
                raise TrainError(
                    "piper's export must pass dynamo=False on torch 2.9+ and "
                    f"export_onnx.py could not be patched ({compat['detail']})"
                )
            ckpt = ensure_checkpoint_compat("", str(env["piper_root"]))
            if ckpt["state"] == "installed":
                log(
                    "• removed piper's optional val_mos checkpoint — this "
                    "trainer has no test split to score it"
                )
            elif ckpt["state"] == "error":
                raise TrainError(
                    "piper's val_mos checkpoint would abort the first validation "
                    f"epoch and train/__main__.py could not be patched "
                    f"({ckpt['detail']})"
                )

        pkg = Path(options["package"])
        run_dir = Path(run_dir) if run_dir is not None else new_run_dir(pkg)
        run_dir.mkdir(parents=True, exist_ok=True)
        job["run_dir"] = str(run_dir)
        job["package"] = str(pkg)
        notify()
        log_file = open(run_dir / "train.log", "a", encoding="utf-8")  # noqa: SIM115
        record.update(
            run_dir=str(run_dir),
            voice_name=options["voice_name"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            accelerator=options["accelerator"],
            flavor=env["flavor"],
            python=env["python"],
            resumed_from=dict(resume),
            log_path=str(run_dir / "train.log"),
        )

        prepared = prepare_run(pkg, run_dir, env["flavor"], options["voice_name"])
        clips = int(prepared.get("clips") or 0)
        seconds = float(prepared.get("seconds") or 0.0)
        record.update(clips=clips, seconds=seconds)
        job["clips"], job["seconds"] = clips, seconds
        log(f"• dataset: {clips} clip(s) · {seconds:.1f}s of speech")
        if clips < 3:
            raise TrainError(
                "a Piper dataset needs at least 3 clips — re-run the conversion "
                "with more minutes"
            )
        if seconds < voice_port.MIN_TRAIN_SECONDS:
            raise TrainError(
                f"the dataset is too small to train a voice: {clips} clip(s) / "
                f"{seconds:.1f}s of speech. Piper needs at least "
                f"{int(voice_port.MIN_TRAIN_SECONDS)}s "
                "(10+ minutes recommended); below that it only learns noise — that is "
                "the robotic audio a run like this exports. Add more audio and build "
                "the package again."
            )
        if seconds < voice_port.RECOMMENDED_TRAIN_SECONDS:
            log(
                f"⚠ {seconds:.1f}s is below the recommended "
                f"{int(voice_port.RECOMMENDED_TRAIN_SECONDS)}s — the voice may still "
                "sound rough when the run finishes"
            )
        # Mirror piper's own floor (valid_set_size = int(n * split)) so the batch
        # can never swallow the whole train split. With 3 clips and batch_size 8
        # the trainer logged "Train split (3) is not larger than batch_size (8)"
        # and trained one ragged batch per epoch.
        split = _validation_split(prepared) if env["flavor"] == "piper1-gpl" else 0.0
        held_out = int(clips * split)
        train_clips = max(1, clips - held_out)
        requested_batch = int(options["batch_size"])
        effective_batch = max(1, min(requested_batch, train_clips))
        if effective_batch != requested_batch:
            log(
                f"• batch size {requested_batch} is larger than the "
                f"{train_clips}-clip train split — using {effective_batch}"
            )
        options["batch_size"] = effective_batch
        job["batch_size"] = effective_batch
        record["batch_size"] = effective_batch
        log(
            f"• split: {train_clips} train / {held_out} validation clip(s) · "
            f"batch {effective_batch}"
        )
        job["package"] = str(pkg)
        steps: list[dict[str, str]] = []
        if env["flavor"] == "legacy":
            steps.append({"id": "preprocess", "label": "prepare the dataset", "status": "pending"})
        steps.append({"id": "train", "label": "training", "status": "pending"})
        steps.append({"id": "export", "label": "export the .onnx", "status": "pending"})
        job["steps"] = steps
        record["steps"] = steps
        _write_run(run_dir, record)
        notify()

        def mark(step_id: str, status: str) -> None:
            for step in job["steps"]:
                if step["id"] == step_id:
                    step["status"] = status

        if env["flavor"] == "legacy":
            mark("preprocess", "running")
            job.update(phase="preprocess", phase_label="preparing the dataset…", progress=1.0,
                       epoch=0)
            _write_run(run_dir, {**record, "steps": job["steps"], "status": "running"})
            notify()
            rc = await _stream(
                job, preprocess_argv(env, prepared, run_dir),
                cwd=run_dir, log_file=log_file, on_line=log, on_spawn=notify,
            )
            if rc != 0:
                if job["cancelled"]:
                    raise _Cancelled()
                raise TrainError(
                    "the dataset preprocess failed — the log has the trainer's reason"
                )
            mark("preprocess", "done")

        # An inference .onnx has no optimizer/discriminator state, so it cannot
        # be resumed — it is converted inside the run and handed to piper's
        # non-strict warm start, which copies whatever matches the new model.
        reference_ckpt = ""
        if warmstart:
            reference_ckpt = str(run_dir / REFERENCE_FILE)
            source = Path(options["checkpoint"]).name
            job.update(phase="reference", phase_label=f"reading {source}…", progress=6.0)
            _write_run(run_dir, {**record, "steps": job["steps"], "status": "running"})
            notify()
            rc = await _stream(
                job, reference_argv(env["python"], options["checkpoint"], Path(reference_ckpt)),
                cwd=run_dir, log_file=log_file, on_line=log, on_spawn=notify,
            )
            if rc != 0:
                if _stopped_by_user(job, run_dir):
                    raise _Cancelled()
                raise TrainError(
                    "the .onnx reference could not be read — the log above has the "
                    "converter's reason (a corrupted export or a model that is not "
                    "a Piper voice)"
                )
            if not Path(reference_ckpt).is_file():
                raise TrainError(
                    "the .onnx reference produced no checkpoint — the log above has "
                    "the converter's output"
                )
            job["reference"] = {"onnx": options["checkpoint"], "checkpoint": reference_ckpt}
            record["reference"] = job["reference"]
            log(
                "• the trainer now copies the reference's matching weights "
                "(non-strict warm start) and trains a fresh optimizer"
            )

        mark("train", "running")
        start_epoch = int(resume.get("epoch") or 0) if resume else 0
        job.update(
            phase="training",
            phase_label=(f"resuming from epoch {start_epoch}…" if resume else "training…"),
            progress=round(8.0 + min(1.0, start_epoch / max(1, options["epochs"])) * 85.0, 1),
            epoch=start_epoch,
        )
        if resume:
            log(
                f"↻ resuming from epoch {start_epoch} — "
                f"{Path(resume['checkpoint']).name} (run {Path(resume['run_dir']).name})"
            )
        _write_run(run_dir, {**record, "steps": job["steps"], "status": "running",
                             "epoch": job.get("epoch", 0)})
        notify()

        def on_train_line(line: str) -> None:
            log(line)
            parsed = parse_epoch(line, job["epochs"])
            if parsed is not None:
                job["epoch"] = parsed["epoch"]
                job["progress"] = round(8.0 + parsed["percent"] * 0.85, 1)
                # Persist the epoch as it advances: a cancelled run must be
                # resumable at the right place even after a restart.
                if parsed["epoch"] != job.get("_saved_epoch"):
                    job["_saved_epoch"] = parsed["epoch"]
                    _write_run(run_dir, {**record, "steps": job["steps"],
                                         "status": "running",
                                         "epoch": parsed["epoch"]})
                    notify()

        rc = await _stream(
            job,
            train_argv(
                env, prepared, run_dir,
                voice_name=options["voice_name"],
                epochs=options["epochs"],
                batch_size=effective_batch,
                accelerator=options["accelerator"],
                checkpoint=reference_ckpt or options["checkpoint"],
                warmstart=warmstart,
            ),
            cwd=run_dir,
            log_file=log_file,
            on_line=on_train_line,
            on_spawn=notify,
        )
        if rc != 0:
            if _stopped_by_user(job, run_dir):
                raise _Cancelled()
            raise TrainError(
                "training exited with an error — the log above has the trainer's reason"
            )
        mark("train", "done")

        ckpt = find_checkpoint(run_dir)
        if ckpt is None:
            raise TrainError(
                "training finished but wrote no .ckpt — there is nothing to export"
            )
        onnx = run_dir / f"{options['voice_name']}.onnx"
        config = sidecar_config(env, prepared, run_dir)

        mark("export", "running")
        job.update(phase="export", phase_label="exporting the .onnx…", progress=93.0)
        _write_run(run_dir, {**record, "steps": job["steps"], "status": "running"})
        notify()
        rc = await _stream(
            job, export_argv(env, ckpt, onnx),
            cwd=run_dir, log_file=log_file, on_line=log, timeout=EXPORT_TIMEOUT,
            on_spawn=notify,
        )
        if rc != 0:
            if _stopped_by_user(job, run_dir):
                raise _Cancelled()
            raise TrainError("the ONNX export failed — the log has the exporter's reason")
        if not onnx.is_file():
            raise TrainError("the exporter exited cleanly but wrote no .onnx")
        config_out = Path(str(onnx) + ".json")
        try:
            config_out.write_bytes(config.read_bytes())
        except OSError as e:
            raise TrainError(f"could not write the .onnx.json config: {e}")
        mark("export", "done")

        job.update(status="done", progress=100.0, phase="done", phase_label="ready",
                   exported={"onnx": str(onnx), "config": str(config_out)},
                   finished_at=time.time(), running=False)
        record.update(status="done", exported=job["exported"], steps=job["steps"],
                      epoch=job.get("epoch", 0), finished_at=time.time())
        _write_run(run_dir, record)
        log(f"✓ ready — {onnx.name} (+ .json). Attach it below to use the voice.")
        notify()
    except _Cancelled:
        job.update(status="cancelled", running=False, finished_at=time.time(),
                   phase="cancelled", phase_label="cancelled")
        log("■ cancelled — the process was stopped")
        if run_dir is not None:
            _write_run(run_dir, {**record, "steps": job.get("steps") or [],
                                 "status": "cancelled", "epoch": job.get("epoch", 0),
                                 "finished_at": time.time()})
    except TrainError as e:
        job.update(status="error", running=False, finished_at=time.time(),
                   phase="error", phase_label="failed", error=str(e))
        log_failure(str(e))
        if run_dir is not None:
            _write_run(run_dir, {**record, "steps": job.get("steps") or [],
                                 "status": "error", "error": str(e),
                                 "epoch": job.get("epoch", 0),
                                 "finished_at": time.time()})
    except Exception as e:  # noqa: BLE001 — never leave a job "running" forever
        msg = f"unexpected training failure: {str(e)[:220]}"
        job.update(status="error", running=False, finished_at=time.time(),
                   phase="error", phase_label="failed", error=msg)
        log_failure(msg)
        if run_dir is not None:
            _write_run(run_dir, {**record, "steps": job.get("steps") or [],
                                 "status": "error", "error": msg,
                                 "epoch": job.get("epoch", 0),
                                 "finished_at": time.time()})
    finally:
        job.pop("_cancel", None)
        job.pop("_proc", None)
        if log_file is not None:
            log_file.close()
        notify()


class _Cancelled(Exception):
    """Internal signal: the user pressed stop, not a trainer failure."""


async def run_setup(job: dict[str, Any], python: str = "") -> None:
    """Install the training extra into the chosen interpreter (with cancel)."""
    exe = interpreter(python)
    job["_cancel"] = asyncio.Event()
    job.update(python=exe, phase="setup", phase_label="installing the trainer…",
               progress=1.0, log="")
    log_file = None
    try:
        env = await probe(exe, force=True)
        if env["can_train"]:
            job.update(status="done", running=False, finished_at=time.time(),
                       progress=100.0, phase="done", phase_label="ready", exported={})
            job["log"] = "✓ the trainer is already installed — nothing to do\n"
            # "Installed" is exactly where the wheel-without-core state hides:
            # finish the job, but fill the slot if it is still empty.
            _heal_trainer(job, env)
            return
        # The command is exactly the one shown in the panel (nothing hidden);
        # only the tail of its output is kept, on the job itself.
        argv = [exe, "-m", "pip", "install", "--upgrade", TRAIN_EXTRA]
        if env.get("export_missing"):
            argv.append("onnxscript")
        job["log"] = "$ " + " ".join(argv) + "\n"
        rc = await _stream(
            job,
            argv,
            cwd=piper_setup.REPO_ROOT,
            log_file=None,
            on_line=lambda line: job.__setitem__(
                "log", (job["log"] + line + "\n")[-MAX_LOG_CHARS:]
            ),
            timeout=SETUP_TIMEOUT,
        )
        if rc != 0:
            if job["cancelled"]:
                raise _Cancelled()
            raise TrainError("pip could not install the trainer — see the log above")
        env = await probe(exe, force=True)
        job.update(status="done", running=False, finished_at=time.time(),
                   progress=100.0, phase="done", phase_label="ready",
                   flavor=env["flavor"], exported={})
        if not env["can_train"]:
            job.update(status="error", error=env["hint"], phase="error",
                       phase_label="failed")
            job["log"] += f"\n✗ {env['hint']}\n"
        else:
            _heal_trainer(job, env)
    except _Cancelled:
        job.update(status="cancelled", running=False, finished_at=time.time(),
                   phase="cancelled", phase_label="cancelled")
        job["log"] += "\n■ cancelled — pip was stopped\n"
    except TrainError as e:
        job.update(status="error", running=False, finished_at=time.time(),
                   phase="error", phase_label="failed", error=str(e))
        job["log"] += f"\n✗ {e}\n"
    except Exception as e:  # noqa: BLE001
        msg = f"unexpected setup failure: {str(e)[:220]}"
        job.update(status="error", running=False, finished_at=time.time(),
                   phase="error", phase_label="failed", error=msg)
        job["log"] += f"\n✗ {msg}\n"
    finally:
        job.pop("_cancel", None)
        job.pop("_proc", None)
        if log_file is not None:
            log_file.close()


# ---------------------------------------------------------------------------
# Detached supervisor — the run outlives the dashboard
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace ``path``, retrying a transient Windows lock.

    Windows refuses the replace while another process has the file open (the
    dashboard reads ``job.json`` every poll), so a single attempt can lose an
    update — and for the supervisor that would also mean a crashed run.
    """
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    for attempt in range(WRITE_ATTEMPTS):
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
            return
        except OSError:
            if attempt + 1 < WRITE_ATTEMPTS:
                time.sleep(0.05 * (attempt + 1))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _pid_alive(pid: int) -> bool:
    """Is that process still there? (No psutil: the dep list stays as-is.)"""
    if not pid or int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> bool:
    """Stop one process — its group on POSIX — without a shell or taskkill."""
    if not _pid_alive(pid):
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_TERMINATE = 0x0001
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
        if not handle:
            return False
        try:
            return bool(kernel32.TerminateProcess(handle, 1))
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(int(pid), signal.SIGTERM)
            return True
        except OSError:
            return False


def write_job_snapshot(run_dir: Path, job: dict[str, Any]) -> None:
    """Mirror the in-memory job to ``job.json`` (the dashboard's only source)."""
    record = snapshot(job)
    record.setdefault("pid", os.getpid())
    proc = job.get("_proc")
    child = int(getattr(proc, "pid", 0) or 0) if proc is not None else 0
    record["child_pid"] = child or int(record.get("child_pid") or 0) or None
    record["updated_at"] = time.time()
    _write_json(Path(run_dir) / JOB_FILE, record)


def _announce(run_dir: Path, job: dict[str, Any], prefs: dict[str, Any] | None) -> None:
    """Tell the user how the run ended (Windows toast / Discord), best effort.

    Called by the supervisor with no dashboard in sight, so every failure is
    absorbed: the run's outcome must never depend on a notification.
    """
    if not prefs:
        return
    try:
        from . import train_notify

        results = train_notify.notify_run(run_dir, job, prefs)
    except Exception:  # noqa: BLE001 — notification is a courtesy, not a step
        return
    if results:
        job["notified"] = results


def supervise(run_dir: Path | str) -> None:
    """Run one training config to the end, with no dashboard attached.

    This is the entry point of ``python -m tts.train_supervisor``. It owns the
    run folder it is given, keeps ``job.json`` current, honors the ``cancel``
    marker, and always leaves a terminal status behind.
    """
    run_dir = Path(run_dir)
    config = _read_json(run_dir / RUN_CONFIG_FILE)
    if not config:
        print(f"missing {RUN_CONFIG_FILE} in {run_dir}", file=sys.stderr)
        return
    job = new_job()
    begin(
        job,
        kind="train",
        package=str(config.get("package") or ""),
        voice_name=str(config.get("voice_name") or ""),
        epochs=int(config.get("epochs") or DEFAULT_EPOCHS),
        python=str(config.get("python") or ""),
        flavor=str(config.get("flavor") or ""),
        resumed_from=config.get("resumed_from") or {},
    )
    job.update(run_dir=str(run_dir), pid=os.getpid(), status="running",
               phase_label="starting the trainer…", started_at=time.time())
    cancel_path = run_dir / CANCEL_FILE
    try:
        cancel_path.unlink()
    except OSError:
        pass

    def persist() -> None:
        write_job_snapshot(run_dir, job)

    async def watch() -> None:
        while job.get("running"):
            if cancel_path.is_file():
                job["cancelled"] = True
                proc = job.get("_proc")
                if proc is not None:
                    await _terminate(proc)
                return
            await asyncio.sleep(0.5)

    async def main() -> None:
        watcher = asyncio.create_task(watch())
        try:
            await run_train(
                job,
                package=str(config["package"]),
                voice_name=str(config.get("voice_name") or ""),
                epochs=int(config.get("epochs") or DEFAULT_EPOCHS),
                batch_size=int(config.get("batch_size") or DEFAULT_BATCH),
                accelerator=str(config.get("accelerator") or "auto"),
                python=str(config.get("python") or ""),
                checkpoint=str(config.get("checkpoint") or ""),
                warmstart=bool(config.get("warmstart")),
                resume_from=str(config.get("resume_from") or ""),
                run_dir=run_dir,
                on_change=persist,
            )
        finally:
            watcher.cancel()
            persist()
        if job.get("running"):
            job.update(status="error", running=False, phase="error", phase_label="interrupted",
                       error="the trainer stopped without a result")
        persist()

    persist()
    asyncio.run(main())
    # The dashboard may not be running: the run finishes its own last mile and
    # says how it ended by itself.
    auto_attach(run_dir, job, config.get("attach") or {})
    _announce(run_dir, job, config.get("notify") or {})
    persist()


def _popen_supervisor(argv: list[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    """Start the supervisor in its own session, output on supervisor.log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    kwargs: dict[str, Any] = dict(
        cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        close_fds=True, env=_child_env(),
    )
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(argv, **kwargs)
    finally:
        log.close()


def start_run(job: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Validate-free launch of the detached supervisor (options pre-checked).

    The dashboard writes the config and the first ``job.json``, then returns.
    From here on the run is its own process: closing the dashboard (or losing
    it) no longer interrupts training, and a new dashboard can reconnect just by
    reading the folder.
    """
    pkg = Path(options["package"])
    run_dir = new_run_dir(pkg)
    config = {
        "format": "wallie.train.config",
        "version": 1,
        "created_at": _now_iso(),
        "package": str(pkg),
        "run_dir": str(run_dir),
        "voice_name": options["voice_name"],
        "epochs": options["epochs"],
        "batch_size": options["batch_size"],
        "accelerator": options["accelerator"],
        "python": options["python"],
        "checkpoint": options.get("checkpoint", ""),
        "warmstart": bool(options.get("warmstart")),
        "resume_from": options.get("resume_from", ""),
        "flavor": options.get("flavor", ""),
        "resumed_from": options.get("resume") or {},
        # Carried with the run: the supervisor may finish long after the
        # dashboard that started it is gone.
        "notify": dict(options.get("notify") or {}),
        # Auto-port: where the finished .onnx should be installed. Empty = the
        # user attaches by hand, exactly as before.
        "attach": dict(options.get("attach") or {}),
    }
    _write_json(run_dir / RUN_CONFIG_FILE, config)
    try:
        (run_dir / CANCEL_FILE).unlink()
    except OSError:
        pass
    try:
        proc = _popen_supervisor(
            [sys.executable, "-m", SUPERVISOR_MODULE, str(run_dir)],
            cwd=piper_setup.REPO_ROOT,
            log_path=run_dir / SUPERVISOR_LOG,
        )
    except (OSError, subprocess.SubprocessError) as e:
        begin(job, kind="train", package=str(pkg), voice_name=options["voice_name"],
              epochs=options["epochs"], python=options["python"])
        job.update(status="error", running=False, phase="error", phase_label="failed",
                   error=f"could not start the training process: {str(e)[:200]}")
        return snapshot(job)

    begin(
        job,
        kind="train",
        package=str(pkg),
        voice_name=options["voice_name"],
        epochs=options["epochs"],
        python=options["python"],
        flavor=options.get("flavor", ""),
        resumed_from=options.get("resume") or {},
    )
    job.update(status="running", run_dir=str(run_dir), pid=proc.pid,
               phase_label="starting the trainer…", started_at=time.time(),
               warmstart=bool(options.get("warmstart")))
    write_job_snapshot(run_dir, job)
    return snapshot(job)


def _log_tail(run_dir: Path) -> str:
    path = Path(run_dir) / "train.log"
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            handle.seek(max(0, size - MAX_LOG_CHARS))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def active_job() -> dict[str, Any] | None:
    """The live run, found on disk — what a restarted dashboard reconnects to."""
    root = voice_port.ported_dir()
    if not root.is_dir():
        return None
    best: dict[str, Any] | None = None
    for path in root.glob(f"*/train/run-*/{JOB_FILE}"):
        data = _read_json(path)
        if not data or data.get("status") != "running":
            continue
        if not _pid_alive(int(data.get("pid") or 0)):
            # The supervisor vanished (machine reboot, kill -9): record it once
            # instead of pretending a run is still alive forever.
            data.update(status="error", running=False, phase="error",
                        phase_label="interrupted",
                        error="the trainer process is gone — resume it from its checkpoint",
                        updated_at=time.time())
            _write_json(path, data)
            continue
        run_dir = path.parent
        data["run_dir"] = str(run_dir)
        data["package"] = str(run_dir.parents[1])
        data["running"] = True
        data["log"] = _log_tail(run_dir)
        if best is None or float(data.get("updated_at") or 0) > float(best.get("updated_at") or 0):
            best = data
    return best


def sync_job(job: dict[str, Any]) -> dict[str, Any]:
    """Adopt the detached run's disk state into the dashboard's job dict."""
    data = active_job()
    run_dir = Path(str(job.get("run_dir") or ""))
    # No live run: refresh from this job's own folder once it is finished, so the
    # panel shows the supervisor's last word (final error, and whether it managed
    # to announce the result) instead of freezing on the last poll before it ended.
    if data is None and run_dir.name:
        disk = _read_json(run_dir / JOB_FILE)
        if disk:
            run_dir = Path(str(disk.get("run_dir") or run_dir))
            disk["run_dir"] = str(run_dir)
            disk["running"] = disk.get("status") == "running"
            disk["log"] = _log_tail(run_dir)
            data = disk
    if data:
        for key, value in data.items():
            job[key] = value
        job["running"] = bool(data.get("running"))
        if not job.get("log"):
            job["log"] = str(data.get("log") or "")
    return snapshot(job)


def cancel_job(job: dict[str, Any]) -> bool:
    """Stop a detached run; works even after the dashboard restarted.

    The marker is what the supervisor polls, and terminating the trainer process
    is what makes the stop immediate — the supervisor then finalizes the status.
    """
    run_dir = Path(str(job.get("run_dir") or ""))
    data = _read_json(run_dir / JOB_FILE) if run_dir.name else None
    if data is None:
        active = active_job() or {}
        if active.get("run_dir"):
            run_dir = Path(str(active["run_dir"]))
            data = active
    if not run_dir.name or not (run_dir / JOB_FILE).is_file():
        return False
    if (data or {}).get("status") != "running":
        return False
    try:
        (run_dir / CANCEL_FILE).write_text("1", encoding="utf-8")
    except OSError:
        pass
    child = int((data or {}).get("child_pid") or 0)
    if child:
        _terminate_pid(child)
    job["cancelled"] = True
    job["running"] = True          # until the supervisor reports the final state
    return True


# ---------------------------------------------------------------------------
# Attach — the exported model goes where Piper looks for it
# ---------------------------------------------------------------------------

def attach_run(
    *,
    profile: str,
    package: str,
    run_dir: str,
    name: str = "",
    display_name: str = "",
) -> dict[str, Any]:
    """Register a finished run's .onnx/.onnx.json as a saved Piper voice."""
    pkg = resolve_package(package)
    train_root = (pkg / "train").resolve()
    try:
        run = Path(run_dir).resolve()
    except OSError as e:
        raise TrainError(f"bad run folder: {e}")
    if train_root not in run.parents:
        raise TrainError("that run folder is not one of this package's training runs")
    manifest_path = run / "run.json"
    if not manifest_path.is_file():
        raise TrainError("that run has no run.json — start the training from here")
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TrainError(f"could not read that run: {e}")
    if record.get("status") != "done":
        raise TrainError("that run did not finish — there is no model to attach")
    exported = record.get("exported") or {}
    onnx = Path(str(exported.get("onnx") or ""))
    config = Path(str(exported.get("config") or ""))
    for path, what in ((onnx, ".onnx"), (config, ".onnx.json")):
        if not path.is_file():
            raise TrainError(f"the run's {what} file is gone — train it again")
        if run not in path.resolve().parents:
            raise TrainError("the run's exported files point outside its folder")
    if config.name != onnx.name + ".json":
        raise TrainError(
            "the .onnx and .onnx.json names must match (Piper loads them as a pair)"
        )
    manifest = voice_port.read_manifest(pkg)
    seconds = float(record.get("seconds") or manifest.get("seconds") or 0.0)
    if 0.0 < seconds < voice_port.MIN_TRAIN_SECONDS:
        # Installing this would put noise in the voice picker under a real
        # name, and the next test would sound broken instead of explaining why.
        raise TrainError(
            f"this run was trained on only {seconds:.1f}s of speech — the exported "
            "model is noise, not a voice. A Piper voice needs at least "
            f"{int(voice_port.MIN_TRAIN_SECONDS)}s (10+ minutes recommended). Add "
            "more audio, train again, then attach that run."
        )
    stem = (name or "").strip() or str(record.get("voice_name") or onnx.stem)
    return voice_port.attach(
        profile=profile,
        target="piper",
        name=stem,
        display_name=(display_name or "").strip() or str(manifest.get("name") or stem),
        source_provider=str((manifest.get("source") or {}).get("provider") or ""),
        files=[(onnx.name, onnx.read_bytes()), (config.name, config.read_bytes())],
    )


def auto_attach(
    run_dir: Path | str,
    job: dict[str, Any],
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Install a finished run's export as a saved voice — from the supervisor.

    The run may end hours after the dashboard that started it is gone, so the
    install spec travels in ``train-config.json`` instead of depending on a
    browser. Best effort by design: a failure is recorded (``attach_error``) and
    never turns a finished training run into a failed one — the .onnx stays
    attachable by hand in the Train panel.
    """
    spec = spec or {}
    job["attached"] = {}
    job["attach_error"] = ""
    if not spec:
        return {"attached": {}, "error": ""}
    run_dir = Path(run_dir)
    profile = str(spec.get("profile") or "").strip()
    if job.get("status") != "done":
        job["attach_error"] = (
            "the run did not finish — there is no model to install automatically"
        )
    elif not profile:
        job["attach_error"] = (
            "no profile in the run config — attach the .onnx from the Train panel"
        )
    else:
        try:
            result = attach_run(
                profile=profile,
                package=str(job.get("package") or ""),
                run_dir=str(run_dir),
                name=str(spec.get("name") or ""),
                display_name=str(spec.get("display_name") or ""),
            )
        except Exception as e:  # noqa: BLE001 — manual attach stays available
            job["attach_error"] = f"automatic install failed: {str(e)[:240]}"
        else:
            job["attached"] = {
                "name": str(result.get("name") or ""),
                "path": str(result.get("path") or ""),
                "profile": profile,
            }
    # run.json is what a restarted dashboard reads; keep the install state there
    # too, so the Train panel can say "already installed" instead of offering it.
    record = _read_json(run_dir / "run.json")
    if record is not None:
        record["attached"] = job["attached"]
        record["attach_error"] = job["attach_error"]
        _write_json(run_dir / "run.json", record)
    return {"attached": job["attached"], "error": job["attach_error"]}


# ---------------------------------------------------------------------------
# Panel payload
# ---------------------------------------------------------------------------

MAX_REFERENCES = 40


def list_references() -> list[dict[str, Any]]:
    """Every Piper ``.onnx`` on this machine that can seed a training run.

    Downloaded voices, attached models and models exported by earlier runs all
    live under the voices tree, so the panel can offer a picker instead of
    asking for a path. Newest first: the voice you trained last is usually the
    one worth fine-tuning from. The file is only a *reference* — training never
    modifies it.
    """
    root = Path(piper_setup.DEFAULT_VOICES_DIR)
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in root.rglob("*.onnx"):
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append({
            "path": str(path.resolve()),
            "name": path.name,
            "size": int(stat.st_size),
            "modified": float(stat.st_mtime),
            # The exported model and its config travel together; a lone .onnx
            # can still warm-start, this is only a hint for the panel.
            "config": path.with_name(path.name + ".json").is_file(),
            "trained": any(p.name.startswith("run-") for p in path.parents),
        })
    found.sort(key=lambda r: r["modified"], reverse=True)
    return found[:MAX_REFERENCES]


async def status_payload(python: str = "") -> dict[str, Any]:
    env = await probe(python)
    return {
        "env": env,
        "packages": list_packages(),
        "references": list_references(),
        "defaults": {
            "epochs": DEFAULT_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "max_epochs": MAX_EPOCHS,
            "batch_size": DEFAULT_BATCH,
            "min_batch": MIN_BATCH,
            "max_batch": MAX_BATCH,
            "accelerator": "auto",
            "accelerators": list(ACCELERATORS),
        },
        "train_extra": TRAIN_EXTRA,
    }

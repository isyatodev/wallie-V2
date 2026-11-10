"""KokoClone — zero-shot cloning on top of Kokoro, as an optional TTS backend.

Kokoro cannot learn a voice from an id; KokoClone gets around that the honest
way: **Kokoro synthesises the line, then a voice-conversion model (Kanade)
reshapes it to sound like a short reference clip**. Nothing is trained, the
reference never leaves this machine, and the result is a real Kokoro voice —
which is why it belongs in the Voice Lab next to the blend panel.

It is a separate project with its own (heavy) dependencies
(`torch`, `kokoro-onnx`, `kanade-tokenizer`), so it is never imported here.
The dashboard talks to :mod:`tts.kokoclone_worker` — one long-lived child
process that loads the models once and answers JSON lines — which gives three
things a direct import cannot: the models survive between segments, a model
download cannot freeze the event loop, and a crashed pipeline takes only the
worker down.

Reference clips live in ``voices/kokoclone/<slug>/ref.wav`` (24 kHz mono, the
rate Kanade works at), next to a small ``meta.json``. Saved voices point at
them, so a KokoClone voice is portable across profiles exactly like a Kokoro
one — and ``voices/`` is git-ignored, so a clip of someone's voice never ends
up in a commit.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, AsyncIterator

from .base import TTSError, TTSProvider
from .pitch import clamp_semitones, shift_pcm16

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = Path(__file__).resolve().parent / "kokoclone_worker.py"
# Where a checkout lives unless WALLIE_KOKOCLONE_DIR says otherwise.
DEFAULT_ROOT = REPO_ROOT / "kokoclone"
# The installer the panel's button runs (the same file a person runs by hand).
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_kokoclone.py"
# Reference clips + their metadata (module-level so tests can repoint it, like
# piper_setup.DEFAULT_VOICES_DIR).
DEFAULT_VOICES_DIR = REPO_ROOT / "voices" / "kokoclone"

SAMPLE_RATE = 24000           # Kanade's own rate; the wav header is authoritative
MIN_REF_SEC = 2.0
MAX_REF_SEC = 30.0
RECOMMENDED_REF = "3–10 s"

# KokoClone's languages → the Kokoro front end each one needs.
LANGUAGES: tuple[dict[str, str], ...] = (
    {"id": "pt", "label": "Portuguese (Brazil)", "kokoro": "p"},
    {"id": "en", "label": "English", "kokoro": "a"},
    {"id": "es", "label": "Spanish", "kokoro": "e"},
    {"id": "fr", "label": "French", "kokoro": "f"},
    {"id": "it", "label": "Italian", "kokoro": "i"},
    {"id": "hi", "label": "Hindi", "kokoro": "h"},
    {"id": "ja", "label": "Japanese", "kokoro": "j"},
    {"id": "zh", "label": "Mandarin", "kokoro": "z"},
)
LANG_IDS: tuple[str, ...] = tuple(lang["id"] for lang in LANGUAGES)
LANG_LABEL: dict[str, str] = {lang["id"]: lang["label"] for lang in LANGUAGES}

# What the checkout needs before a clone can run. Probed with find_spec (no
# imports: a bare `import torch` costs seconds and the dashboard must stay snappy).
REQUIRED_PACKAGES: tuple[str, ...] = (
    "torch", "torchaudio", "kokoro_onnx", "kanade_tokenizer", "soundfile", "misaki",
)
MIN_PY = (3, 10)
MAX_PY = (3, 13)              # exclusive, per KokoClone's own badge
PY_RANGE_LABEL = "3.10–3.12"

REF_NAME = "ref.wav"
META_NAME = "meta.json"
_META_MAX_BYTES = 64 * 1024      # a recipe of a dozen fields, not a document
SLUG_RE = re.compile(r"[^a-z0-9]+")

# One reference clip, however it arrives (an upload or a path on disk). The Voice
# port caps files at 40 MB and this is the same ceiling: 30 s of speech is under
# 2 MB, so anything bigger is either an uncompressed long take (the page asks for
# 3–10 s) or not a voice at all — and it never becomes a large allocation.
MAX_REF_BYTES = 40 * 1024 * 1024

# The worker runs third-party code (the checkout's ``core.cloner``), which also
# downloads models. It gets its own environment with credentials removed: those
# keys belong to Wallie's providers, not to a library installed from the internet.
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
_SECRET_ENV_NAMES = frozenset({"DASHBOARD_PIN"})

# How long to wait for the worker's first "ready" (a cold start may download
# the Kokoro + Kanade weights, which is minutes, not seconds).
READY_TIMEOUT = 900.0
JOB_TIMEOUT = 240.0
_CHUNK = 16 * 1024


class KokoCloneError(RuntimeError):
    """A failure worth showing the user verbatim (missing checkout, bad clip…)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any, default: float = 0.0) -> float:
    """A number read back from JSON that may have been hand-edited or corrupt."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _worker_env() -> dict[str, str]:
    """This process's environment without anything that authenticates it.

    Paths, the model caches, proxies, ``CUDA_VISIBLE_DEVICES`` and the rest of
    the functional settings are kept — the worker needs them to download and
    load its models. Keys, tokens, secrets, passwords and the dashboard PIN are
    dropped, so a checkout can never read what it was not given.
    """
    env = {
        key: value for key, value in os.environ.items()
        if key not in _SECRET_ENV_NAMES
        and not key.upper().endswith(_SECRET_ENV_SUFFIXES)
    }
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def root_dir() -> Path:
    override = (os.environ.get("WALLIE_KOKOCLONE_DIR") or "").strip()
    return Path(override).expanduser() if override else DEFAULT_ROOT


def install_argv(root: Path | None = None) -> list[str]:
    """Command that clones the checkout and pip-installs its requirements.

    The root travels as an argument (not only through the environment) so the
    button installs exactly where :func:`detect` will look next time.
    """
    target = root or root_dir()
    return [sys.executable, str(INSTALL_SCRIPT), "--install", "--root", str(target)]


def voices_dir() -> Path:
    return DEFAULT_VOICES_DIR


def slug(name: str) -> str:
    """A safe folder name for a voice ("Voz Quente!" -> "voz_quente")."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    out = SLUG_RE.sub("_", text).strip("_")[:40].strip("_")
    if not out:
        raise KokoCloneError("give the voice a name with at least one letter or number")
    return out


def voice_dir(name: str) -> Path:
    return voices_dir() / slug(name)


def ref_path(name: str) -> Path:
    return voice_dir(name) / REF_NAME


def resolve_ref(reference: str) -> Path:
    """A reference clip as an absolute path.

    Accepts three spellings, because three callers produce them: an absolute
    path (a hand-typed file), a repo-relative one (what a saved voice stores,
    so a checkout can move), and a bare voice name (what a person types).
    """
    raw = str(reference or "").strip().strip('"')
    if not raw:
        raise KokoCloneError(
            "pick a KokoClone voice first — it carries the reference clip"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        if "/" in raw or "\\" in raw:
            path = REPO_ROOT / raw
        else:
            path = ref_path(raw)
    return path.resolve()


def _relative(path: Path) -> str:
    """Store paths inside the project relative to it; anything else absolute."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Is it installed?
# ---------------------------------------------------------------------------

def _importable(module: str) -> bool:
    try:
        return importlib_util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def python_ok(version_info: tuple[int, ...] | None = None) -> bool:
    vi = version_info or sys.version_info
    return MIN_PY <= (vi[0], vi[1]) < MAX_PY


def detect() -> dict[str, Any]:
    """Cheap status snapshot for the Voice Lab badge (no heavy imports)."""
    root = root_dir()
    checkout = (root / "core" / "cloner.py").is_file()
    py_ok = python_ok()
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _importable(pkg)]
    worker_ok = WORKER_SCRIPT.is_file()
    installed = checkout and py_ok and not missing and worker_ok
    return {
        "installed": installed,
        "root": str(root),
        "checkout": checkout,
        # NOT "worker": that key is the running worker's status everywhere else
        # in the panel payload, and a bool there would be a nasty surprise.
        "worker_present": worker_ok,
        "missing": missing,
        "python_ok": py_ok,
        # The panel offers the one-click install only when it could work here:
        # a supported interpreter and the installer script present.
        "installer_present": INSTALL_SCRIPT.is_file(),
        "can_install": py_ok and INSTALL_SCRIPT.is_file(),
        "git": shutil.which("git") is not None,
        "python_version": ".".join(str(p) for p in sys.version_info[:3]),
        "python_range": PY_RANGE_LABEL,
        "languages": [dict(lang) for lang in LANGUAGES],
        "sample_rate": SAMPLE_RATE,
        "recommended_seconds": RECOMMENDED_REF,
        "min_seconds": MIN_REF_SEC,
        "max_seconds": MAX_REF_SEC,
        "hint": _install_hint(root, checkout, py_ok, missing),
    }


def _install_hint(root: Path, checkout: bool, py_ok: bool, missing: list[str]) -> str:
    if not py_ok:
        return (
            f"KokoClone needs Python {PY_RANGE_LABEL} — this dashboard runs "
            f"{'.'.join(str(p) for p in sys.version_info[:3])}. Point "
            "WALLIE_KOKOCLONE_DIR at a checkout that runs on a supported interpreter."
        )
    if not checkout:
        hint = (
            "KokoClone is a separate project, cloned next to this one: press "
            "**⬇ install KokoClone** to clone it and pip-install its requirements "
            "(it runs scripts/install_kokoclone.py). By hand: "
            f"git clone https://github.com/Ashish-Patnaik/kokoclone.git \"{root}\" "
            "then pip install -r requirements.txt."
        )
        if not shutil.which("git"):
            hint += (
                " git is not on PATH, so the button cannot clone — install git or "
                "copy a checkout in by hand."
            )
        return hint
    if missing:
        return (
            "the checkout is there but these packages are missing in this "
            f"interpreter: {', '.join(missing)} — press **⬇ install KokoClone** "
            "(or run python scripts/install_kokoclone.py --install); it installs the "
            "subset this backend imports, skipping the checkout's CMake-only "
            "Japanese/Chinese extras and its gradio UI"
        )
    return (
        "KokoClone is ready. The first synthesis loads (and on the very first "
        "run downloads, ~1 GB) the Kokoro + Kanade models, so it can take a few minutes."
    )


# ---------------------------------------------------------------------------
# Reference clips
# ---------------------------------------------------------------------------

def _meta(name: str) -> dict[str, Any]:
    path = voice_dir(name) / META_NAME
    if not path.is_file():
        return {}
    try:
        if path.stat().st_size > _META_MAX_BYTES:
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def list_voices() -> list[dict[str, Any]]:
    """Every reference clip on disk, newest first."""
    base = voices_dir()
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for folder in base.iterdir():
        if not folder.is_dir() or not (folder / REF_NAME).is_file():
            continue
        meta = _meta(folder.name)
        out.append({
            "name": str(meta.get("name") or folder.name),
            "slug": folder.name,
            "ref": _relative(folder / REF_NAME),
            "lang": str(meta.get("lang") or "en"),
            "seconds": _as_float(meta.get("seconds")),
            "created_at": str(meta.get("created_at") or ""),
            "notes": str(meta.get("notes") or ""),
        })
    out.sort(key=lambda v: v["created_at"], reverse=True)
    return out


def store_voice(
    name: str,
    blob: bytes,
    *,
    lang: str = "en",
    filename: str = "",
    notes: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Validate a reference clip and keep it as a 24 kHz mono WAV.

    Decoding, trimming and resampling happen here (the same local pipeline the
    Voice port uses), so KokoClone always gets the one format it wants and a
    compressed upload never depends on ``soundfile`` being importable at clone
    time. The clip is never uploaded anywhere.
    """
    label = str(name or "").strip()[:60]
    if not label:
        raise KokoCloneError("give the voice a name")
    lang = str(lang or "en").strip().lower()
    if lang not in LANG_IDS:
        raise KokoCloneError(
            f"unknown KokoClone language {lang!r} — use one of {', '.join(LANG_IDS)}"
        )
    if not blob:
        raise KokoCloneError("attach a reference clip (or record one)")
    if len(blob) > MAX_REF_BYTES:
        raise KokoCloneError(
            f"the reference clip is {len(blob) // (1024 * 1024)} MB — keep it "
            f"under {MAX_REF_BYTES // (1024 * 1024)} MB ({RECOMMENDED_REF} of "
            "clean speech is all KokoClone needs)"
        )

    from . import local_port

    try:
        pcm = local_port.decode_audio(blob, filename or "reference.wav")
    except local_port.LocalPortError as e:
        raise KokoCloneError(str(e)) from e
    seconds = len(pcm) / 2.0 / SAMPLE_RATE
    if seconds < MIN_REF_SEC:
        raise KokoCloneError(
            f"the reference clip is {seconds:.1f}s — KokoClone needs at least "
            f"{int(MIN_REF_SEC)}s, and {RECOMMENDED_REF} of clean speech works best"
        )
    if seconds > MAX_REF_SEC:
        raise KokoCloneError(
            f"the reference clip is {seconds:.1f}s — trim it to about "
            f"{RECOMMENDED_REF} (max {int(MAX_REF_SEC)}s)"
        )

    from .voice_lab import pcm16_bytes_to_wav

    folder = voice_dir(label)
    folder.mkdir(parents=True, exist_ok=True)
    wav = pcm16_bytes_to_wav(pcm, SAMPLE_RATE, 1)
    (folder / REF_NAME).write_bytes(wav)
    meta = {
        "name": label,
        "slug": folder.name,
        "lang": lang,
        "seconds": round(seconds, 2),
        "notes": str(notes or "")[:400],
        "sample_rate": SAMPLE_RATE,
        "created_at": now or _now_iso(),
    }
    (folder / META_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    return {**meta, "ref": _relative(folder / REF_NAME),
            "wav_b64_length": len(wav)}


def remove_voice(name: str) -> bool:
    """Delete a voice's clip folder. Only ever inside ``voices/kokoclone/``."""
    folder = voice_dir(name).resolve()
    base = voices_dir().resolve()
    if base not in folder.parents:
        raise KokoCloneError("refusing to delete a folder outside voices/kokoclone")
    if not folder.is_dir():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------

class _Worker:
    """One KokoClone child process, and the JSON-lines conversation with it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ready = threading.Event()
        self.dead = threading.Event()
        self.error = ""
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self.tail: deque[str] = deque(maxlen=120)
        self.started_at = time.time()
        # Boot progress, reported by the worker itself (imports and weight
        # loading are otherwise a minute of silence that reads as a hang).
        self.stage = ""
        self.stage_label = ""
        self.stage_since = time.time()
        self._stage_labels: dict[str, str] = {}
        self._pending: Any = None
        self._reply_event = threading.Event()
        self._reply: dict[str, Any] | None = None
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), "--root", str(root)],
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_worker_env(),
        )
        threading.Thread(target=self._pump_out, name="kokoclone-out", daemon=True).start()
        threading.Thread(target=self._pump_err, name="kokoclone-err", daemon=True).start()

    # -- plumbing ---------------------------------------------------------

    def _push(self, text: str) -> None:
        text = text.replace("\x00", "").strip()
        if text:
            self.tail.append(text[-300:])

    def _pump_err(self) -> None:
        """Drain stderr, splitting on ``\\r`` as well as ``\\n``.

        Progress bars (torch.hub, huggingface) redraw a single line with ``\\r``
        and never end it until 100 %, so line-based iteration would show
        NOTHING during a 360 MB download — the exact silence that looks like a
        frozen backend. Reading the raw pipe in chunks and splitting on both
        keeps the frames, throttled so the tail stays readable.
        """
        stream = self.proc.stderr
        if stream is None:
            return
        raw = getattr(stream, "buffer", None) or stream
        rest = ""
        last_partial = 0.0
        while True:
            try:
                chunk = raw.read1(8192) if hasattr(raw, "read1") else raw.read(8192)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk
            rest += text
            frames = re.split(r"[\r\n]+", rest)
            rest = frames.pop()
            for frame in frames:
                self._push(frame)
            # Keep the in-flight bar frame visible, at most twice a second.
            if rest.strip() and time.time() - last_partial > 0.5:
                self._push(rest)
                last_partial = time.time()
        self._push(rest)

    def _pump_out(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.tail.append(line[-300:])
                continue
            event = msg.get("event")
            if event == "stage":
                stage = str(msg.get("stage") or "")
                label = str(msg.get("label") or stage)
                if label:
                    self._stage_labels[stage] = label
                if stage != self.stage:
                    self.stage = stage
                    self.stage_label = label
                    self.stage_since = time.time()
                    if label:
                        self._push(f"→ {label}…")
            elif event == "heartbeat":
                # "still on this step, N seconds in" — keeps a silent import or
                # weight load from looking like a hang.
                if str(msg.get("stage") or "") == self.stage:
                    self.stage_since = time.time() - _as_float(msg.get("seconds"))
            elif event == "stage_done":
                stage = str(msg.get("stage") or "")
                label = self._stage_labels.get(stage, stage)
                took = _as_float(msg.get("seconds"))
                self._push(f"✓ {label}" + (f" ({took:.1f}s)" if took else ""))
            elif event == "ready":
                self.sample_rate = int(msg.get("sample_rate") or SAMPLE_RATE)
                self.ready.set()
            elif event == "error":
                self.error = str(msg.get("error") or "KokoClone failed to start")
                self.ready.set()
                self.dead.set()
            elif msg.get("id") is not None and msg.get("id") == self._pending:
                self._reply = msg
                self._reply_event.set()
        # stdout closed: the child is gone.
        self.dead.set()
        self.ready.set()
        self._reply_event.set()

    def alive(self) -> bool:
        return self.proc.poll() is None and not self.dead.is_set()

    def log_tail(self, lines: int = 20) -> str:
        return "\n".join(list(self.tail)[-lines:])

    # -- requests ---------------------------------------------------------

    def request(self, payload: dict[str, Any], timeout: float = JOB_TIMEOUT) -> dict[str, Any]:
        job_id = int(time.time() * 1000) % 2_000_000_000
        with self.lock:
            if not self.alive():
                raise KokoCloneError(
                    "the KokoClone worker stopped"
                    + (f" — {self.error}" if self.error else "")
                    + (f"\n{self.log_tail(8)}" if self.tail else "")
                )
            self._pending = job_id
            self._reply = None
            self._reply_event.clear()
            assert self.proc.stdin is not None
            try:
                self.proc.stdin.write(json.dumps({**payload, "id": job_id},
                                                 ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise KokoCloneError(f"the KokoClone worker closed its input: {e}") from e
            if not self._reply_event.wait(timeout):
                raise KokoCloneError(
                    f"KokoClone did not answer within {int(timeout)}s"
                    + (f"\n{self.log_tail(8)}" if self.tail else "")
                )
            reply = self._reply or {}
            self._pending = None
            if not reply:
                raise KokoCloneError(
                    "the KokoClone worker exited mid-request"
                    + (f"\n{self.log_tail(8)}" if self.tail else "")
                )
            return reply

    def stop(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
            except OSError:
                pass


_WORKERS: dict[str, _Worker] = {}
_WORKERS_LOCK = threading.Lock()


def _worker_for(root: Path) -> _Worker:
    key = str(root)
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is not None and worker.alive():
            return worker
        if worker is not None:
            worker.stop()
        if not (root / "core" / "cloner.py").is_file():
            raise KokoCloneError(
                "KokoClone is not set up — " + detect()["hint"]
            )
        worker = _Worker(root)
        _WORKERS[key] = worker
        return worker


def _shutdown_workers() -> None:
    with _WORKERS_LOCK:
        for worker in _WORKERS.values():
            worker.stop()
        _WORKERS.clear()


atexit.register(_shutdown_workers)


def worker_status() -> dict[str, Any]:
    """What the panel shows about the background process (never starts one)."""
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
    running = [w for w in workers if w.alive()]
    if not running:
        return {"running": False, "ready": False, "log": ""}
    worker = running[0]
    ready = worker.ready.is_set() and not worker.error
    return {
        "running": True,
        "ready": ready,
        "error": worker.error,
        # What it is doing right now (empty once the models are up).
        "stage": "" if ready else worker.stage,
        "stage_label": "" if ready else worker.stage_label,
        "stage_seconds": 0.0 if ready else round(time.time() - worker.stage_since, 1),
        "seconds": round(time.time() - worker.started_at, 1),
        "pid": worker.proc.pid,
        "log": worker.log_tail(),
    }


def warm() -> dict[str, Any]:
    """Start the worker and its model loading, without synthesizing anything.

    The panel's "load models" button: it lets a person watch the boot (and pay
    the first-run download) before pressing test, instead of discovering a
    two-minute wait inside a request that looks frozen.
    """
    if not detect()["installed"]:
        raise KokoCloneError(detect()["hint"])
    _worker_for(root_dir())
    return worker_status()


def stop_worker() -> bool:
    """Kill the background process (the panel's "stop" button)."""
    with _WORKERS_LOCK:
        had = bool(_WORKERS)
        for worker in _WORKERS.values():
            worker.stop()
        _WORKERS.clear()
    return had


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _read_wav_pcm(path: Path) -> tuple[bytes, int]:
    """Read a PCM16 WAV the worker wrote, as raw little-endian bytes."""
    import wave

    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except (OSError, EOFError, wave.Error) as e:
        raise KokoCloneError(f"KokoClone wrote no readable wav: {e}") from e
    if width != 2:
        raise KokoCloneError(f"KokoClone wrote {width * 8}-bit audio — expected 16-bit")
    if channels > 1:
        # Kanade is mono; a stereo writer would still mean two of the same take.
        import numpy as np

        arr = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
        frames = arr[:, 0].astype("<i2").tobytes()
    return frames, rate


def synthesize_pcm(
    text: str,
    *,
    reference: str,
    lang: str = "en",
    pitch: float = 0.0,
    timeout: float = JOB_TIMEOUT,
) -> bytes:
    """Render one line in the reference voice, as PCM16 mono at 24 kHz."""
    import tempfile

    ref = resolve_ref(reference)
    if not ref.is_file():
        raise KokoCloneError(
            f"the reference clip is missing: {ref} — re-create the KokoClone voice"
        )
    try:
        ref_bytes = ref.stat().st_size
    except OSError as e:
        raise KokoCloneError(f"could not read the reference clip {ref.name}: {e}") from e
    if ref_bytes > MAX_REF_BYTES:
        raise KokoCloneError(
            f"the reference clip {ref.name} is {ref_bytes // (1024 * 1024)} MB — "
            f"keep it under {MAX_REF_BYTES // (1024 * 1024)} MB"
        )
    lang = str(lang or "en").strip().lower()
    if lang not in LANG_IDS:
        raise KokoCloneError(
            f"unknown KokoClone language {lang!r} — use one of {', '.join(LANG_IDS)}"
        )
    leap = clamp_semitones(pitch)

    worker = _worker_for(root_dir())
    if not worker.ready.wait(READY_TIMEOUT):
        where = worker.stage_label or worker.stage
        raise KokoCloneError(
            "KokoClone is still loading its models"
            + (f" (step: {where})" if where else "")
            + f" — {int(time.time() - worker.started_at)}s so far. Watch it in the "
            "Voice Lab panel and press test again when it says ready."
        )
    if worker.error:
        raise KokoCloneError(worker.error + (f"\n{worker.log_tail(8)}" if worker.tail else ""))

    with tempfile.TemporaryDirectory(prefix="kokoclone-") as tmp:
        out = Path(tmp) / "out.wav"
        reply = worker.request(
            {"text": text, "lang": lang, "ref": str(ref), "out": str(out)},
            timeout=timeout,
        )
        if not reply.get("ok"):
            detail = str(reply.get("error") or "unknown error")
            raise KokoCloneError(f"KokoClone failed: {detail}")
        pcm, rate = _read_wav_pcm(out)

    if int(rate) != SAMPLE_RATE:
        from . import voice_port

        pcm = voice_port.resample_pcm16(pcm, int(rate), SAMPLE_RATE)
    if leap:
        pcm = shift_pcm16(pcm, leap, SAMPLE_RATE)
    return pcm


class KokoCloneTTS(TTSProvider):
    """TTS provider: Kokoro + Kanade voice conversion against a reference clip."""

    name = "kokoclone"

    def __init__(
        self,
        *,
        reference: str = "",
        lang: str = "en",
        pitch_semitones: float = 0.0,
    ) -> None:
        try:
            self._ref = str(resolve_ref(reference))
        except KokoCloneError as e:
            raise TTSError(f"kokoclone: {e}") from e
        self._lang = str(lang or "en").strip().lower()
        if self._lang not in LANG_IDS:
            raise TTSError(
                f"kokoclone: unknown language {self._lang!r} — "
                f"use one of {', '.join(LANG_IDS)}"
            )
        self._pitch = clamp_semitones(pitch_semitones)
        self.sample_rate = SAMPLE_RATE
        self.channels = 1

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        try:
            pcm = await asyncio.to_thread(
                synthesize_pcm,
                text,
                reference=self._ref,
                lang=self._lang,
                pitch=self._pitch,
            )
        except KokoCloneError as e:
            raise TTSError(f"kokoclone: {e}") from e
        if not pcm:
            raise TTSError("kokoclone: the worker produced no audio")
        for start in range(0, len(pcm), _CHUNK):
            yield pcm[start:start + _CHUNK]

    async def aclose(self) -> None:
        # The worker (and its loaded models) stays up for the next segment —
        # reloading them per sentence is exactly what this design avoids.
        self._ref = ""

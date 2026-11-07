"""Kokoro voice blending — in-between local voices built from the packs on disk.

A Kokoro voice is a *style* tensor: one 256-float row per text length, and the
pipeline picks the row that matches the token count. Averaging two packs
therefore averages the timbre and delivery the model learned, so a midpoint
between two published voices becomes a voice of its own — no training, no
cloud, no cloning. That is the whole trick behind the Voice Lab's blender.

A blend is written next to the downloaded packs as ``voices/kokoro/<id>.pt``,
with ``<id>`` = ``kb_<slug>``. That shape satisfies the same name rule as a
real Kokoro voice (``<letters>_<name>``), which matters: it is what makes
:class:`tts.kokoro.KokoroTTS` load the blend from disk without any special
case. A sidecar registry — ``voices/kokoro/blends.json`` — remembers the
recipe (which packs, which weights, which language), so the dashboard can show
it, and the phoneme language is never guessed from an id that no longer carries
one.

Both files live under ``voices/``, which is git-ignored: a blend is generated
on this machine and can be rebuilt from its recipe.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import kokoro_setup

# A blend id must look like a Kohoro voice id: ``<1-2 letters>_<name>``.
# ``kb_`` keeps blends recognisable in the pickers while the regex still
# matches (that is what ``kokoro._local_voice_path`` checks before loading).
BLEND_PREFIX = "kb_"
BLEND_ID_RE = re.compile(r"^kb_[a-z0-9_]+$")
MAX_SLUG = 40

MIN_SOURCES = 2
MAX_SOURCES = 4
MAX_BLENDS = 40

REGISTRY_FORMAT = "wallie.kokoro.blends"
REGISTRY_VERSION = 1
# 40 blends of recipes are a few KB; a bigger file is not one we wrote, and it
# is read on every blend route, so it is never loaded whole.
REGISTRY_MAX_BYTES = 2 * 1024 * 1024

# Loading a .pt is torch work — done through this indirection so tests can feed
# synthetic tensors without importing torch or downloading a single voice.
StyleLoader = Callable[[str], Any]


class KokoroBlendError(RuntimeError):
    """A failure worth showing the user verbatim (missing pack, bad weights…)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any, default: float = 0.0) -> float:
    """A recipe weight that may have been hand-edited or corrupted.

    The registry is a plain file on disk, so a bad number in it must not make
    every blend unreadable (or raise out of a GET route).
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def blend_id(name: str) -> str:
    """The on-disk id for a blend called ``name`` (``kb_<slug>``).

    A non-ASCII name is transliterated first, so "Voz Quente" and "voz-quente"
    land on the same id — and an empty result is refused instead of writing a
    file called ``kb_.pt``.
    """
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:MAX_SLUG].strip("_")
    if not slug:
        raise KokoroBlendError("give the blend a name with at least one letter or number")
    return f"{BLEND_PREFIX}{slug}"


def is_blend_id(voice: str) -> bool:
    return bool(BLEND_ID_RE.match(str(voice or "").strip()))


def voices_dir() -> Path:
    # Read the module attribute every time (not at import): tests repoint it,
    # exactly like piper_setup.DEFAULT_VOICES_DIR.
    return kokoro_setup.voices_dir()


def voice_path(voice: str) -> Path:
    return voices_dir() / f"{voice}.pt"


def _registry_path() -> Path:
    return voices_dir() / "blends.json"


# ---------------------------------------------------------------------------
# Registry — what went into each blend
# ---------------------------------------------------------------------------

def _read_registry() -> list[dict[str, Any]]:
    path = _registry_path()
    if not path.is_file():
        return []
    try:
        if path.stat().st_size > REGISTRY_MAX_BYTES:
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("blends") if isinstance(raw, dict) else None
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        blend = str(item.get("id") or "").strip()
        if not is_blend_id(blend):
            continue
        out.append({
            "id": blend,
            "name": str(item.get("name") or blend),
            "lang_code": str(item.get("lang_code") or "a"),
            "notes": str(item.get("notes") or ""),
            "created_at": str(item.get("created_at") or ""),
            "sources": [
                {"voice": str(s.get("voice") or ""), "weight": _as_float(s.get("weight"))}
                for s in (item.get("sources") or [])
                if isinstance(s, dict) and str(s.get("voice") or "").strip()
            ],
        })
    return out


def _write_registry(blends: Sequence[dict[str, Any]]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": REGISTRY_FORMAT,
        "version": REGISTRY_VERSION,
        "blends": list(blends)[:MAX_BLENDS],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)          # atomic — a crash never eats the recipe book


def list_blends() -> list[dict[str, Any]]:
    """Every registered blend whose pack is still on disk (newest first)."""
    return [b for b in _read_registry() if voice_path(b["id"]).is_file()]


def get_blend(voice: str) -> dict[str, Any] | None:
    target = str(voice or "").strip()
    for blend in _read_registry():
        if blend["id"] == target:
            return blend
    return None


def lang_code_for(voice: str) -> str:
    """The phoneme language a voice id belongs to.

    A plain Kokoro id carries it in the first letter (``pf_dora`` -> ``p``). A
    blend id does not, so its language comes from the registry — a blend of
    pt-BR voices is read with the pt-BR front end, which is the only one whose
    phonemes those style rows were trained for.
    """
    name = str(voice or "").strip()
    if is_blend_id(name):
        blend = get_blend(name)
        if blend is None:
            raise KokoroBlendError(
                f"{name} is not in the blend registry — rebuild the blend"
            )
        return blend["lang_code"] or "a"
    if not kokoro_setup.is_valid_voice_name(name):
        raise KokoroBlendError(
            f"{name!r} does not look like a Kokoro voice id (af_heart, pf_dora…)"
        )
    return name[0]


# ---------------------------------------------------------------------------
# Loading / mixing the style tensors
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise KokoroBlendError(
            "blending Kokoro voices needs PyTorch — it comes with the Kokoro "
            "install: python scripts/install_kokoro.py --install"
        ) from e
    return torch


def load_style_pack(voice: str, *, loader: StyleLoader | None = None) -> Any:
    """The style tensor of one voice pack, from disk (or a test loader)."""
    name = str(voice or "").strip()
    if not (kokoro_setup.is_valid_voice_name(name) or is_blend_id(name)):
        raise KokoroBlendError(
            f"{name!r} does not look like a Kokoro voice id (af_heart, pf_dora…)"
        )
    path = voice_path(name)
    if not path.is_file():
        raise KokoroBlendError(
            f"{name} is not downloaded yet — download the voice first "
            "(Download a Kokoro voice on the Voice page), then blend it"
        )
    if loader is not None:
        tensor = loader(str(path))
    else:
        torch = _torch()
        try:
            tensor = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001 — a corrupt pack is user-visible
            raise KokoroBlendError(f"could not read {path.name}: {e}") from e
    if not _is_tensor(tensor):
        raise KokoroBlendError(
            f"{path.name} does not hold a Kokoro style tensor — re-download the voice"
        )
    return tensor


def _is_tensor(value: Any) -> bool:
    """Duck-typed so the module never needs torch (or numpy) at import time."""
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _to_numpy(value: Any) -> Any:
    """A style pack as a float32 numpy array — works for torch and numpy."""
    import numpy as np

    source = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    return np.asarray(source, dtype=np.float32)


def _to_style_tensor(array: Any) -> Any:
    """Back to the form Kokoro's pipeline (and ``torch.save``) expects.

    The arithmetic itself stays in numpy: it keeps the mixing testable on a
    machine without PyTorch, and the only torch-specific step is this one.
    """
    import numpy as np

    return _torch().from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def blend_tensors(
    items: Iterable[tuple[str, float, Any]],
) -> Any:
    """Weighted average of style tensors — the actual blending.

    Weights are normalized to sum to 1, so ``[1, 1]`` and ``[0.5, 0.5]`` are
    the same blend and a slider's absolute scale never changes the result.

    Kokoro indexes the style by token count, so the packs must agree on how
    many rows they carry. Different releases have used slightly different
    lengths; repeating the final row (the "longest text" style) is the least
    destructive way to make them add up, and trimming keeps the shorter one's
    real rows instead of dropping the pair.
    """
    import numpy as np

    parts = [(name, float(weight), tensor) for name, weight, tensor in items]
    if len(parts) < MIN_SOURCES:
        raise KokoroBlendError(f"a blend needs at least {MIN_SOURCES} voices")
    weights = [w for _, w, _ in parts]
    if any(w <= 0 for w in weights):
        raise KokoroBlendError("every voice in the blend needs a weight above zero")
    total = sum(weights)
    if total <= 0:
        raise KokoroBlendError("the blend weights add up to zero")

    arrays = [(name, weight, _to_numpy(tensor)) for name, weight, tensor in parts]
    rows = max(int(arr.shape[0]) for _, _, arr in arrays)
    rank = arrays[0][2].ndim
    tail = tuple(int(d) for d in arrays[0][2].shape[1:])
    for name, _, arr in arrays:
        if arr.ndim != rank or tuple(int(d) for d in arr.shape[1:]) != tail:
            raise KokoroBlendError(
                f"{name} has a different style shape than the other voices — "
                "re-download both packs and try again"
            )

    acc = None
    for _, weight, arr in arrays:
        if arr.shape[0] < rows:
            pad = np.repeat(arr[-1:], rows - int(arr.shape[0]), axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > rows:
            arr = arr[:rows]
        piece = arr * (weight / total)
        acc = piece if acc is None else acc + piece
    return _to_style_tensor(acc)


def _normalize_sources(sources: Any) -> list[tuple[str, float]]:
    """Validate a UI payload into ``[(voice, weight), …]``."""
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise KokoroBlendError("pick the voices to blend")
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in sources:
        if not isinstance(raw, dict):
            raise KokoroBlendError("each blend source needs a voice and a weight")
        voice = str(raw.get("voice") or raw.get("name") or "").strip()
        if not voice:
            raise KokoroBlendError("each blend source needs a voice id")
        if not (kokoro_setup.is_valid_voice_name(voice) or is_blend_id(voice)):
            raise KokoroBlendError(
                f"{voice!r} does not look like a Kokoro voice id (af_heart, pf_dora…)"
            )
        if voice in seen:
            raise KokoroBlendError(f"{voice} is already in the blend")
        seen.add(voice)
        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            raise KokoroBlendError(f"the weight of {voice} is not a number") from None
        if weight <= 0:
            raise KokoroBlendError(f"the weight of {voice} must be above zero")
        out.append((voice, min(weight, 100.0)))
    if len(out) < MIN_SOURCES:
        raise KokoroBlendError(f"a blend needs at least {MIN_SOURCES} voices")
    if len(out) > MAX_SOURCES:
        raise KokoroBlendError(f"at most {MAX_SOURCES} voices per blend")
    return out


def blend_sources(
    sources: Any,
    *,
    loader: StyleLoader | None = None,
) -> dict[str, Any]:
    """Mix ``sources`` into a style tensor plus the recipe that produced it.

    Every source must be downloaded and belong to the same language: Kokoro's
    style rows carry no phonemes, so the language (and therefore the G2P front
    end) has to be one single choice for the whole blend.
    """
    pairs = _normalize_sources(sources)
    langs = {lang_code_for(voice) for voice, _ in pairs}
    if len(langs) > 1:
        labels = ", ".join(sorted(langs))
        raise KokoroBlendError(
            f"these voices speak different languages ({labels}) — a blend keeps "
            "one phoneme language, so pick voices from the same one"
        )
    tensors = [(voice, weight, load_style_pack(voice, loader=loader)) for voice, weight in pairs]
    tensor = blend_tensors(tensors)
    total = sum(w for _, w in pairs)
    return {
        "tensor": tensor,
        "lang_code": langs.pop(),
        "voices": [voice for voice, _ in pairs],
        "weights": [round(weight / total, 4) for _, weight in pairs],
    }


# ---------------------------------------------------------------------------
# Creating / deleting blends
# ---------------------------------------------------------------------------

def save_blend(
    name: str,
    sources: Any,
    *,
    loader: StyleLoader | None = None,
    notes: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Write the blend pack and register its recipe. Returns the registry entry."""
    label = str(name or "").strip()[:60]
    if not label:
        raise KokoroBlendError("give the blend a name")
    blend = blend_id(label)
    mixed = blend_sources(sources, loader=loader)

    existing = _read_registry()
    if len([b for b in existing if b["id"] != blend]) >= MAX_BLENDS:
        raise KokoroBlendError(f"too many blends ({MAX_BLENDS}) — delete one first")

    blend_tensor = mixed["tensor"]
    path = voice_path(blend)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        _torch().save(blend_tensor, str(tmp))
        tmp.replace(path)       # atomic — a half-written pack never loads
    except KokoroBlendError:
        raise
    except Exception as e:  # noqa: BLE001 — torch write failures are user-visible
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise KokoroBlendError(f"could not write the blend: {e}") from e

    entry = {
        "id": blend,
        "name": label,
        "lang_code": mixed["lang_code"],
        "notes": str(notes or "")[:400],
        "created_at": now or _now_iso(),
        "sources": [
            {"voice": voice, "weight": weight}
            for voice, weight in zip(mixed["voices"], mixed["weights"])
        ],
    }
    kept = [b for b in existing if b["id"] != blend]
    kept.insert(0, entry)
    _write_registry(kept)
    return entry


def remove_blend(voice: str) -> bool:
    """Delete a blend pack and its recipe. A plain Kokoro voice is left alone."""
    blend = str(voice or "").strip()
    if not is_blend_id(blend):
        raise KokoroBlendError(f"{blend!r} is not a blend id (kb_…)")
    removed = False
    path = voice_path(blend)
    if path.is_file():
        try:
            path.unlink()
            removed = True
        except OSError as e:
            raise KokoroBlendError(f"could not delete {path.name}: {e}") from e
    registry = _read_registry()
    kept = [b for b in registry if b["id"] != blend]
    if len(kept) != len(registry):
        _write_registry(kept)
        removed = True
    return removed


def available_voices() -> list[dict[str, Any]]:
    """The downloaded packs a blend can draw from (including other blends)."""
    voices = kokoro_setup.list_local_voices()
    by_id = {b["id"]: b for b in list_blends()}
    for voice in voices:
        blend = by_id.get(voice["name"])
        if blend is not None:
            voice["blend"] = True
            voice["lang_code"] = blend["lang_code"]
            voice["sources"] = blend["sources"]
        else:
            voice["blend"] = is_blend_id(voice["name"])
            voice["lang_code"] = voice["name"][:1] if voice["name"] else ""
    return voices


def describe(blend: dict[str, Any]) -> str:
    """``pf_dora 60% + pm_alex 40% (pt)`` — the recipe in one line."""
    bits = " + ".join(
        f"{s['voice']} {round(float(s['weight']) * 100)}%"
        for s in blend.get("sources") or []
    )
    lang = blend.get("lang_code") or "?"
    return f"{bits} ({lang})" if bits else f"({lang})"

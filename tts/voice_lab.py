"""Voice Lab — per-profile voice presets and provider voice cloning.

Two independent capabilities, both driven from the dashboard's Voice Lab page:

* **Presets** — a named ``(provider, voice_id, tuning)`` tuple kept per profile in
  ``profiles/<name>.voices.json``. Saving one snapshots the current voice; applying
  one writes the TTS fields back, so a voice can be recalled without re-typing ids.
* **Cloning** — create a REAL voice from reference audio on the providers that
  support it. The dashboard sends samples as base64 (no multipart parsing dep in
  the dashboard itself); the multipart upload is built here.

Nothing in this module talks to the profile or the orchestrator: the routes
compose these pieces, which keeps the provider HTTP shapes unit-testable.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

from config import Secrets, TTSConfig

_ELEVENLABS_ADD_URL = "https://api.elevenlabs.io/v1/voices/add"
_FISH_MODEL_URL = "https://api.fish.audio/model"

# Providers the dashboard can create a voice on. `needs_key` mirrors the .env
# field the account lives in (Secrets), so the UI can say what to paste.
CLONE_PROVIDERS: list[dict[str, Any]] = [
    {
        "id": "elevenlabs",
        "label": "ElevenLabs — Instant Voice Clone",
        "key_env": "ELEVENLABS_API_KEY",
        "help_url": "https://elevenlabs.io/app/settings/api-keys",
        "needs_key": True,
        "notes": "Instant cloning, works from ~1 minute of clean speech.",
    },
    {
        "id": "fish",
        "label": "Fish Audio — voice model",
        "key_env": "FISH_API_KEY",
        "help_url": "https://fish.audio/go-api/",
        "needs_key": True,
        "notes": "Creates a private model; the returned id goes in Voice ID.",
    },
]

MAX_SAMPLES = 5
MAX_SAMPLE_BYTES = 12 * 1024 * 1024      # providers cap around 10-11 MB per file
MAX_PRESETS = 60
ALLOWED_AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4")

# Presets may retune anything the TTS engine reads EXCEPT the output device —
# that routes audio and is not a property of a voice.
PRESET_TTS_FIELDS: frozenset[str] = frozenset(TTSConfig.model_fields) - {"output_device"}

# Portable backup file written by the Voice Lab (a .json the user downloads and
# uploads again). The format tag lets an import refuse a file it cannot read
# instead of guessing at its shape.
EXPORT_FORMAT = "wallie.voices"
EXPORT_VERSION = 1
MAX_IMPORT_CHARS = 2 * 1024 * 1024      # a backup is a few KB; this is a guard


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preset_from_raw(raw: dict[str, Any], *, now: str = "") -> VoicePreset:
    """Normalize one raw dict — from the UI, a library file or a backup file —
    into a VoicePreset, truncating the fields and keeping only real TTS knobs."""
    return VoicePreset(
        name=str(raw.get("name") or "").strip()[:60],
        provider=str(raw.get("provider") or ""),
        voice_id=str(raw.get("voice_id") or ""),
        notes=str(raw.get("notes") or "")[:400],
        source=str(raw.get("source") or "manual")[:40],
        created_at=str(raw.get("created_at") or now or _now_iso()),
        tts=sanitize_tts(raw.get("tts")),
    )


class VoiceLabError(RuntimeError):
    """A failure worth showing the user verbatim (missing key, bad samples…)."""


@dataclass
class VoicePreset:
    """A named voice + its tuning, scoped to one profile."""

    name: str
    provider: str = ""
    voice_id: str = ""
    notes: str = ""
    source: str = "manual"                              # manual | clone:elevenlabs | clone:fish
    created_at: str = ""
    tts: dict[str, Any] = field(default_factory=dict)   # extra TTSConfig overrides

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_tts(raw: Any) -> dict[str, Any]:
    """Keep only real TTSConfig fields, so a hand-edited library file (or a
    stale UI) can never inject arbitrary keys into the profile."""
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in PRESET_TTS_FIELDS}


def library_path(profile: str) -> Path:
    # Imported lazily (repo convention): tests repoint config.PROFILES_DIR at a
    # tmp dir, which a module-level `from config import PROFILES_DIR` would
    # freeze at import time.
    from config import PROFILES_DIR
    return PROFILES_DIR / f"{profile or 'default'}.voices.json"


def load_library(profile: str) -> list[VoicePreset]:
    path = library_path(profile)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("presets") if isinstance(raw, dict) else None
    out: list[VoicePreset] = []
    for item in items or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        out.append(VoicePreset(
            name=str(item["name"]).strip(),
            provider=str(item.get("provider") or ""),
            voice_id=str(item.get("voice_id") or ""),
            notes=str(item.get("notes") or ""),
            source=str(item.get("source") or "manual"),
            created_at=str(item.get("created_at") or ""),
            tts=sanitize_tts(item.get("tts")),
        ))
    return out


def save_library(profile: str, presets: Iterable[VoicePreset]) -> None:
    path = library_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile or "default",
        "presets": [p.to_dict() for p in presets][:MAX_PRESETS],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)          # atomic — a crash mid-write never eats the library


def add_preset(profile: str, raw: dict[str, Any], *, now: str = "") -> VoicePreset:
    """Upsert by name (case-insensitive) so re-saving a voice updates it."""
    if not str(raw.get("name") or "").strip():
        raise VoiceLabError("the saved voice needs a name")
    preset = _preset_from_raw(raw, now=now)
    presets = [p for p in load_library(profile) if p.name.lower() != preset.name.lower()]
    presets.insert(0, preset)
    if len(presets) > MAX_PRESETS:
        raise VoiceLabError(f"voice library is full ({MAX_PRESETS} saved voices) — delete one first")
    save_library(profile, presets)
    return preset


def remove_preset(profile: str, name: str) -> bool:
    presets = load_library(profile)
    kept = [p for p in presets if p.name.lower() != (name or "").strip().lower()]
    if len(kept) == len(presets):
        return False
    save_library(profile, kept)
    return True


def get_preset(profile: str, name: str) -> VoicePreset | None:
    target = (name or "").strip().lower()
    if not target:
        return None
    for p in load_library(profile):
        if p.name.lower() == target:
            return p
    return None


# ---------------------------------------------------------------------------
# Merging libraries — what a copy/pull/import would do to the saved voices
# ---------------------------------------------------------------------------

# Two presets are "the same voice" when these match. ``created_at`` is
# deliberately excluded — it is stamped on every save, so comparing it would
# flag every re-save as a change.
_COMPARE_FIELDS: tuple[str, ...] = ("provider", "voice_id", "notes", "tts")


def _same_settings(a: VoicePreset, b: VoicePreset) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in _COMPARE_FIELDS)


def _descriptor(preset: VoicePreset) -> str:
    """Short human description of a voice, for the replace warning."""
    label = f"{preset.provider or '?'}"
    if preset.voice_id:
        label += f" : {preset.voice_id}"
    if preset.tts:
        label += " (" + ", ".join(f"{k}={v}" for k, v in sorted(preset.tts.items())) + ")"
    return label


def _limited(
    presets: Iterable[VoicePreset],
    names: Iterable[str] | None,
) -> list[VoicePreset]:
    """``names`` selects presets (case-insensitive); ``None`` keeps them all."""
    listed = list(presets)
    if names is None:
        return listed
    wanted = {str(n).strip().lower() for n in names if str(n).strip()}
    return [p for p in listed if p.name.lower() in wanted]


def merge_plan(
    existing: Iterable[VoicePreset],
    incoming: Iterable[VoicePreset],
) -> dict[str, Any]:
    """What applying ``incoming`` over ``existing`` (upsert by name) would do.

    * ``added`` — new names that would be created.
    * ``replaced`` / ``conflicts`` — same name but DIFFERENT settings, i.e. the
      writes that would overwrite what the profile already has (the UI asks
      before allowing these).
    * ``unchanged`` — same name and same settings, a harmless re-save.
    """
    by_name = {p.name.lower(): p for p in existing}
    latest: dict[str, VoicePreset] = {}
    for preset in incoming:
        latest[preset.name.lower()] = preset      # the same voice twice: last wins
    plan: dict[str, Any] = {"added": [], "replaced": [], "unchanged": [], "conflicts": []}
    for preset in latest.values():
        current = by_name.get(preset.name.lower())
        if current is None:
            plan["added"].append(preset.name)
        elif _same_settings(current, preset):
            plan["unchanged"].append(preset.name)
        else:
            plan["replaced"].append(preset.name)
            plan["conflicts"].append({
                "name": preset.name,
                "current": _descriptor(current),
                "incoming": _descriptor(preset),
            })
    return plan


def _apply_merge(existing: Iterable[VoicePreset], incoming: Iterable[VoicePreset]) -> list[VoicePreset]:
    """Upsert ``incoming`` into ``existing`` by name; new voices are appended."""
    merged = list(existing)
    by_name = {p.name.lower(): i for i, p in enumerate(merged)}
    for preset in incoming:
        key = preset.name.lower()
        if key in by_name:
            merged[by_name[key]] = preset
        else:
            by_name[key] = len(merged)
            merged.append(preset)
    return merged


class VoiceConflictError(VoiceLabError):
    """Applying this merge would overwrite saved voices that differ.

    Carries :attr:`plan` (see :func:`merge_plan`) so the caller can show exactly
    which voices would change before asking the user to confirm.
    """

    def __init__(self, message: str, plan: dict[str, Any]) -> None:
        super().__init__(message)
        self.plan = plan


def conflict_message(target: str, plan: dict[str, Any]) -> str:
    """Plain-language summary of the voices an operation would overwrite."""
    conflicts = plan.get("conflicts") or []
    names = ", ".join(c["name"] for c in conflicts[:5])
    more = "" if len(conflicts) <= 5 else f" +{len(conflicts) - 5} more"
    return (
        f"{len(conflicts)} saved voice(s) in “{target or 'default'}” already use "
        f"that name with different settings: {names}{more}"
    )


def export_presets(
    src_profile: str,
    dst_profile: str,
    names: Iterable[str] | None = None,
    *,
    overwrite: bool = False,
) -> int:
    """Copy saved voices from one profile's library into another (upsert by
    name), returning how many were written. ``names`` limits the export;
    ``None`` copies the whole library.

    This is how a voice — including a local Piper/Kokoro voice saved on the
    Voice page — moves between characters without re-creating it.

    Overwriting a same-named voice whose settings DIFFER raises
    :class:`VoiceConflictError` unless ``overwrite`` is set, so a copy can never
    silently replace a tuned voice with a different one.
    """
    src = _limited(load_library(src_profile), names)
    if not src:
        return 0

    dst = load_library(dst_profile)
    plan = merge_plan(dst, src)
    if plan["conflicts"] and not overwrite:
        raise VoiceConflictError(conflict_message(dst_profile, plan), plan)
    if len(dst) + len(plan["added"]) > MAX_PRESETS:
        raise VoiceLabError(
            f"“{dst_profile}” already has {MAX_PRESETS} saved voices — delete one there first"
        )
    save_library(dst_profile, _apply_merge(dst, src))
    return len(src)


def plan_export(
    src_profile: str,
    dst_profile: str,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """What :func:`export_presets` would do, without writing anything — so the
    caller can report how many voices it added or replaced."""
    return merge_plan(
        load_library(dst_profile),
        _limited(load_library(src_profile), names),
    )


# ---------------------------------------------------------------------------
# Backup file — the whole library (or one voice) as a portable .json
# ---------------------------------------------------------------------------

def export_bundle(
    profile: str,
    names: Iterable[str] | None = None,
    *,
    now: str = "",
) -> dict[str, Any]:
    """A portable, profile-independent snapshot of saved voices.

    Unlike :func:`export_presets` — which copies into another profile's library —
    this returns a plain dict the dashboard downloads as a file, so a voice
    library can be backed up, versioned or handed to someone else. ``names``
    limits the bundle to those voices; ``None`` takes the whole library.
    """
    presets = _limited(load_library(profile), names)
    if names is not None and not presets:
        raise VoiceLabError("none of those voices are saved in this profile")
    return {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "exported_at": now or _now_iso(),
        "source_profile": profile or "default",
        "count": len(presets),
        "presets": [p.to_dict() for p in presets],
    }


def parse_bundle_text(text: str) -> Any:
    """Parse an uploaded ``.json`` backup, with a message worth showing the user."""
    if not (text or "").strip():
        raise VoiceLabError("that file is empty — pick an exported voice backup")
    if len(text) > MAX_IMPORT_CHARS:
        raise VoiceLabError(
            f"that file is too large ({len(text) // 1024} KB) — expected a small voice backup"
        )
    try:
        return json.loads(text)
    except ValueError as e:
        raise VoiceLabError(f"that file is not valid JSON ({str(e)[:100]})") from e


def _bundle_items(payload: Any) -> list[dict[str, Any]]:
    """The saved-voice dicts inside a backup payload: an exported bundle, a raw
    library file (same shape), a bare list of presets, or a single saved voice."""
    if isinstance(payload, dict):
        try:
            newer = int(payload.get("version")) > EXPORT_VERSION
        except (TypeError, ValueError):
            newer = False
        if payload.get("format") and newer:
            raise VoiceLabError(
                f"that backup was written by a newer Wallie (format v{payload.get('version')})"
            )
        items = payload.get("presets")
        if items is None and payload.get("name"):
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        items = None
    return [item for item in (items or []) if isinstance(item, dict)]


def import_bundle(
    profile: str,
    payload: Any,
    *,
    now: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge saved voices from a backup into this profile's library (upsert by
    name), returning ``{imported, skipped, replaced, unchanged, total}``.

    Malformed entries are skipped instead of failing the whole file, but a file
    with no usable voice — or one that would overflow the library — is refused
    with an actionable message. Nothing is written unless the whole file fits,
    so a refused import never leaves a half-merged library behind.

    Same-named voices with DIFFERENT settings raise :class:`VoiceConflictError`
    unless ``overwrite`` is set — restoring a backup never silently replaces a
    voice this profile has tuned differently.
    """
    items = _bundle_items(payload)
    if not items:
        raise VoiceLabError(
            "no saved voices found in that file — expected a Wallie voice backup "
            "(or a profiles/<name>.voices.json)"
        )
    now = now or _now_iso()
    incoming = [
        _preset_from_raw(raw, now=now)
        for raw in items
        if str(raw.get("name") or "").strip()
    ]
    skipped = len(items) - len(incoming)
    library = load_library(profile)
    plan = merge_plan(library, incoming)
    if plan["conflicts"] and not overwrite:
        raise VoiceConflictError(conflict_message(profile, plan), plan)
    if len(library) + len(plan["added"]) > MAX_PRESETS:
        raise VoiceLabError(
            f"“{profile or 'default'}” already has {MAX_PRESETS} saved voices — "
            "delete some before importing"
        )
    merged = _apply_merge(library, incoming)
    save_library(profile, merged)
    return {
        "imported": len(incoming),
        "added": len(plan["added"]),
        "skipped": skipped,
        "replaced": len(plan["replaced"]),
        "unchanged": len(plan["unchanged"]),
        "total": len(merged),
    }


# ---------------------------------------------------------------------------
# Mirroring — make two profiles hold the same saved voices
# ---------------------------------------------------------------------------

# What to do with a shared name whose settings DIFFER. "skip" is the default
# because it is the only policy that can never discard a setting.
MIRROR_POLICIES: tuple[str, ...] = ("skip", "this", "other")


def mirror_plan(this_profile: str, other_profile: str) -> dict[str, Any]:
    """The two-way diff between two libraries, computed without writing anything.

    * ``to_other`` — voices this profile has and the other one lacks.
    * ``to_this`` — the other way round.
    * ``identical`` — shared names with the same settings.
    * ``conflicts`` — shared names whose settings DIFFER; a mirror needs a policy
      (:func:`mirror_libraries`) before those can be equalised.
    """
    mine = load_library(this_profile)
    theirs = load_library(other_profile)
    their_by = {p.name.lower(): p for p in theirs}
    my_by = {p.name.lower(): p for p in mine}
    to_other: list[str] = []
    identical: list[str] = []
    conflicts: list[dict[str, str]] = []
    for key, preset in my_by.items():
        twin = their_by.get(key)
        if twin is None:
            to_other.append(preset.name)
        elif _same_settings(preset, twin):
            identical.append(preset.name)
        else:
            conflicts.append({
                "name": preset.name,
                "this": _descriptor(preset),
                "other": _descriptor(twin),
            })
    return {
        "this_profile": this_profile or "default",
        "other_profile": other_profile or "default",
        "to_other": to_other,
        "to_this": [p.name for key, p in their_by.items() if key not in my_by],
        "identical": identical,
        "conflicts": conflicts,
        "this_count": len(mine),
        "other_count": len(theirs),
    }


def mirror_libraries(
    this_profile: str,
    other_profile: str,
    *,
    conflicts: str = "skip",
) -> dict[str, Any]:
    """Copy each profile's missing voices into the other one, both ways.

    ``conflicts`` decides the shared names that differ: ``skip`` leaves them
    exactly as they are, ``this`` makes this profile's version win (overwriting
    theirs), ``other`` makes theirs win. Returns the counts plus the diff AFTER
    applying, so the caller can show the resulting state without a second call.

    Both libraries are checked for capacity first, so a mirror that cannot fit
    changes nothing at all.
    """
    policy = (conflicts or "skip").strip().lower()
    if policy not in MIRROR_POLICIES:
        raise VoiceLabError(
            f"unknown conflict policy {conflicts!r} — use one of {', '.join(MIRROR_POLICIES)}"
        )
    plan = mirror_plan(this_profile, other_profile)
    clash = {c["name"].lower() for c in plan["conflicts"]}
    mine = load_library(this_profile)
    theirs = load_library(other_profile)
    my_keys = {p.name.lower() for p in mine}
    their_keys = {p.name.lower() for p in theirs}
    # Only the voices that actually CHANGE something travel: the missing ones
    # each way, plus the differing ones when a policy picks a winner (an
    # identical voice is left alone instead of being re-saved).
    push = [p for p in mine if p.name.lower() not in their_keys]
    pull = [p for p in theirs if p.name.lower() not in my_keys]
    if policy == "this":
        push += [p for p in mine if p.name.lower() in clash]
    elif policy == "other":
        pull += [p for p in theirs if p.name.lower() in clash]

    def _fits(dst_profile: str, dst: Sequence[VoicePreset], incoming: Sequence[VoicePreset]) -> None:
        known = {p.name.lower() for p in dst}
        adds = sum(1 for p in incoming if p.name.lower() not in known)
        if len(dst) + adds > MAX_PRESETS:
            raise VoiceLabError(
                f"“{dst_profile}” would hold more than {MAX_PRESETS} saved voices — "
                "delete some there first"
            )

    _fits(other_profile, theirs, push)
    _fits(this_profile, mine, pull)
    merged_theirs = _apply_merge(theirs, push)
    merged_mine = _apply_merge(mine, pull)
    save_library(other_profile, merged_theirs)
    save_library(this_profile, merged_mine)
    return {
        "policy": policy,
        "to_other": len(push),
        "to_this": len(pull),
        "skipped": len(clash) if policy == "skip" else 0,
        "this_total": len(merged_mine),
        "other_total": len(merged_theirs),
        "plan": mirror_plan(this_profile, other_profile),
    }


def sync_state(this_profile: str, partner: str) -> dict[str, Any]:
    """One profile measured against its **voice partner** — the data behind the
    dashboard's out-of-sync badge.

    A partner is simply the profile this one is supposed to mirror (the
    ``voice_partner`` field on the profile). ``out_of_sync`` is True when a
    mirror under the default ``skip`` policy would still change something:
    either side holding a voice the other lacks, or a shared name whose
    settings differ. Two libraries that merely *disagree* about a shared voice
    count as out of sync too — that is usually what the badge is for, since
    ``skip`` alone would never resolve it. Read-only.
    """
    plan = mirror_plan(this_profile, partner)
    return {
        "partner": partner,
        "out_of_sync": bool(plan["to_other"] or plan["to_this"] or plan["conflicts"]),
        "to_other": len(plan["to_other"]),
        "to_this": len(plan["to_this"]),
        "conflicts": len(plan["conflicts"]),
        "identical": len(plan["identical"]),
        "this_count": plan["this_count"],
        "other_count": plan["other_count"],
    }


def tts_updates(preset: VoicePreset) -> dict[str, Any]:
    """TTSConfig fields a preset applies (provider + voice id + its tuning)."""
    updates = dict(preset.tts)
    if preset.provider:
        updates["provider"] = preset.provider
    if preset.voice_id:
        updates["voice_id"] = preset.voice_id
    return updates


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def _mime_for(filename: str) -> str:
    return {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
        ".mp4": "audio/mp4",
    }.get(Path(filename or "").suffix.lower(), "application/octet-stream")


def _safe_detail(resp: httpx.Response, limit: int = 300) -> str:
    """Provider error text, trimmed and stripped of anything key-shaped."""
    try:
        payload = resp.json()
        detail: Any = payload.get("detail") if isinstance(payload, dict) else payload
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("reason") or detail
        elif isinstance(detail, list) and detail:
            first = detail[0]
            detail = first.get("msg", first) if isinstance(first, dict) else first
        text = str(detail if detail not in (None, "") else payload)
    except ValueError:
        text = resp.text
    text = re.sub(r"\b(sk-|sk_|gsk_|xi-|xi_)[A-Za-z0-9_\-]{8,}", "[redacted]", text)
    return text[:limit]


def validate_samples(samples: Sequence[tuple[str, bytes]]) -> None:
    if not samples:
        raise VoiceLabError("attach at least one audio sample (or record one)")
    if len(samples) > MAX_SAMPLES:
        raise VoiceLabError(f"at most {MAX_SAMPLES} samples per voice")
    for filename, blob in samples:
        label = filename or "sample"
        if not blob:
            raise VoiceLabError(f"{label} is empty")
        if len(blob) > MAX_SAMPLE_BYTES:
            raise VoiceLabError(
                f"{label} is larger than {MAX_SAMPLE_BYTES // (1024 * 1024)} MB — "
                "trim it to under a minute of speech"
            )


async def clone_voice(
    *,
    provider: str,
    name: str,
    samples: Sequence[tuple[str, bytes]],
    secrets: Secrets,
    description: str = "",
) -> str:
    """Create a voice at the provider from reference audio. Returns its id.

    Only the two providers with a documented create-a-voice API are supported;
    local engines (Piper, Kokoro) cannot be cloned into, and saying so beats a
    silent no-op.
    """
    provider = (provider or "").strip().lower()
    name = (name or "").strip()
    if not name:
        raise VoiceLabError("give the voice a name")
    validate_samples(samples)
    if provider == "elevenlabs":
        return await _clone_elevenlabs(name, description, samples, secrets)
    if provider == "fish":
        return await _clone_fish(name, description, samples, secrets)
    known = ", ".join(p["id"] for p in CLONE_PROVIDERS)
    raise VoiceLabError(
        f"{provider or 'that provider'} can't create voices from the dashboard — "
        f"supported: {known}"
    )


async def _clone_elevenlabs(
    name: str,
    description: str,
    samples: Sequence[tuple[str, bytes]],
    secrets: Secrets,
) -> str:
    key = (secrets.elevenlabs_api_key or "").strip()
    if not key:
        raise VoiceLabError(
            "ElevenLabs cloning needs ELEVENLABS_API_KEY — add it on the API Keys page"
        )
    files = [
        ("files", (fn or f"sample{i}.wav", blob, _mime_for(fn)))
        for i, (fn, blob) in enumerate(samples, 1)
    ]
    data = {
        "name": name,
        "description": description or f"Cloned from Wallie ({len(samples)} sample(s))",
    }
    try:  # docs: https://elevenlabs.io/docs/api-reference/voices/ivc/create
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            resp = await client.post(
                _ELEVENLABS_ADD_URL, headers={"xi-api-key": key}, data=data, files=files
            )
    except httpx.HTTPError as e:
        raise VoiceLabError(f"elevenlabs network error: {e}") from e
    if resp.status_code >= 400:
        raise VoiceLabError(
            f"elevenlabs rejected the clone (HTTP {resp.status_code}): {_safe_detail(resp)}"
        )
    try:
        payload = resp.json()
    except ValueError as e:
        raise VoiceLabError("elevenlabs returned a non-JSON response") from e
    voice_id = str(payload.get("voice_id") or "").strip()
    if not voice_id:
        raise VoiceLabError("elevenlabs created no voice_id — check the sample quality")
    return voice_id


async def _clone_fish(
    name: str,
    description: str,
    samples: Sequence[tuple[str, bytes]],
    secrets: Secrets,
) -> str:
    key = (secrets.fish_api_key or "").strip()
    if not key:
        raise VoiceLabError(
            "Fish Audio cloning needs FISH_API_KEY — add it on the API Keys page"
        )
    # Docs (docs.fish.audio/api-reference/endpoint/model/create-model) list the
    # fields as JSON but note that uploads must be multipart/form-data, with the
    # reference clips under `voices`.
    files = [
        ("voices", (fn or f"sample{i}.wav", blob, _mime_for(fn)))
        for i, (fn, blob) in enumerate(samples, 1)
    ]
    data = {
        "type": "tts",
        "title": name,
        "train_mode": "fast",
        "visibility": "private",
        "description": description or f"Cloned from Wallie ({len(samples)} sample(s))",
        "enhance_audio_quality": "true",
        "generate_sample": "false",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            resp = await client.post(
                _FISH_MODEL_URL,
                headers={"Authorization": f"Bearer {key}"},
                data=data,
                files=files,
            )
    except httpx.HTTPError as e:
        raise VoiceLabError(f"fish network error: {e}") from e
    if resp.status_code >= 400:
        raise VoiceLabError(
            f"fish rejected the clone (HTTP {resp.status_code}): {_safe_detail(resp)}"
        )
    try:
        payload = resp.json()
    except ValueError as e:
        raise VoiceLabError("fish returned a non-JSON response") from e
    voice_id = str(payload.get("_id") or payload.get("id") or "").strip()
    if not voice_id:
        raise VoiceLabError("fish created no model id — check the sample quality")
    return voice_id


# ---------------------------------------------------------------------------
# Reference recording
# ---------------------------------------------------------------------------

def pcm16_bytes_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw little-endian PCM16 in a RIFF/WAVE container.

    Stdlib only: the cloning and A/B paths must not depend on ``soundfile`` (it
    ships with the optional Kokoro install, not with the base requirements).
    A trailing odd byte is dropped — WAV frames are 2 bytes per sample.
    """
    import io
    import wave

    channels = max(1, int(channels or 1))
    sample_rate = max(1, int(sample_rate or 24000))
    body = pcm if len(pcm) % 2 == 0 else pcm[:-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(body)
    return buf.getvalue()


def pcm16_wav_bytes(samples: Any, sample_rate: int = 16000) -> bytes:
    """Wrap mono float32 [-1, 1] samples in a RIFF/WAVE container."""
    import numpy as np

    arr = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return pcm16_bytes_to_wav((arr * 32767.0).astype("<i2").tobytes(), sample_rate)

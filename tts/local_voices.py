"""Per-model local voice lists — a *view* over the Voice Lab library.

Earlier this module kept its own ``profiles/<name>.localvoices.json`` file, one
bucket per local engine. That duplicated the Voice Lab's saved-voice library
and split a user's voices across two stores: a Piper voice saved on the Voice
page never showed up in the Voice Lab, the A/B comparison, or an export.

Now there is one store — the Voice Lab library (``profiles/<name>.voices.json``)
— and this module is just the per-engine projection of it:

* :func:`list_local` / :func:`providers_payload` return only the presets whose
  provider matches, with their tuning narrowed to that engine's knobs.
* :func:`add_local` / :func:`remove_local` write through to the shared library.

Because local voices ARE library presets, they automatically appear in the A/B
dropdowns and travel with ``voice_lab.export_presets`` when a voice library is
copied between profiles.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import voice_lab

# Local (no-API-key) engines the studio can keep per-model voice pickers for.
LOCAL_VOICE_PROVIDERS: tuple[str, ...] = ("piper", "kokoro")

# TTSConfig field that holds the voice/model identity for each local provider.
VOICE_ID_FIELD: dict[str, str] = {
    "piper": "piper_model_path",
    "kokoro": "kokoro_voice",
}

# Delivery knobs that belong to each local provider. A saved local voice may
# override only these (voice_id is stored separately), which is what keeps the
# per-model views clean even when the shared library holds other engines' presets.
PROVIDER_FIELDS: dict[str, frozenset[str]] = {
    "piper": frozenset({"piper_length_scale", "piper_noise_scale", "piper_noise_w"}),
    "kokoro": frozenset({"kokoro_lang_code", "kokoro_speed"}),
}

MAX_LOCAL_VOICES = 40
# Presets written by this module carry this source prefix, so the UI can tell a
# local voice from a hand-saved or cloned one.
SOURCE_PREFIX = "local:"


class LocalVoiceError(RuntimeError):
    """A failure worth showing to the user verbatim."""


@dataclass
class LocalVoice:
    """One saved voice for a single local engine (a projection of a preset)."""

    name: str
    voice_id: str = ""
    notes: str = ""
    created_at: str = ""
    tts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(provider: str) -> str:
    return (provider or "").strip().lower()


def is_supported(provider: str) -> bool:
    return _norm(provider) in LOCAL_VOICE_PROVIDERS


def _require_provider(provider: str) -> str:
    p = _norm(provider)
    if p not in PROVIDER_FIELDS:
        raise LocalVoiceError(f"{p or 'that engine'} has no local voice list")
    return p


def sanitize_tts(provider: str, raw: Any) -> dict[str, Any]:
    """Keep only the provider's own TTSConfig knobs from a raw mapping."""
    allowed = PROVIDER_FIELDS.get(_norm(provider), frozenset())
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in allowed}


def _to_local(provider: str, preset: voice_lab.VoicePreset) -> LocalVoice:
    return LocalVoice(
        name=preset.name,
        voice_id=preset.voice_id,
        notes=preset.notes,
        created_at=preset.created_at,
        tts=sanitize_tts(provider, preset.tts),
    )


def local_path(profile: str) -> Path:
    """The single store local voices now live in (the Voice Lab library)."""
    return voice_lab.library_path(profile)


def _legacy_path(profile: str) -> Path:
    from config import PROFILES_DIR
    return PROFILES_DIR / f"{profile or 'default'}.localvoices.json"


_migrated: set[str] = set()


def _migrate_legacy(profile: str) -> None:
    """Best-effort, one-shot import of a pre-unification ``.localvoices.json``.

    The split file only ever existed in a single development branch, but folding
    it in costs nothing and means nobody silently loses a saved voice. Runs at
    most once per profile per process; failures are ignored.
    """
    key = profile or "default"
    if key in _migrated:
        return
    _migrated.add(key)
    legacy = _legacy_path(profile)
    if not legacy.exists():
        return
    try:
        raw = json.loads(legacy.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    providers = raw.get("providers") if isinstance(raw, dict) else None
    if isinstance(providers, dict):
        for provider, items in providers.items():
            provider = _norm(provider)
            if provider not in PROVIDER_FIELDS or not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                    continue
                try:
                    voice_lab.add_preset(profile, {
                        "name": item["name"],
                        "provider": provider,
                        "voice_id": item.get("voice_id", ""),
                        "notes": item.get("notes", ""),
                        "source": f"{SOURCE_PREFIX}{provider}",
                        "tts": sanitize_tts(provider, item.get("tts")),
                    })
                except Exception:  # noqa: BLE001 - migration must never break a read
                    continue
    try:
        legacy.unlink()
    except OSError:
        pass


def list_local(profile: str, provider: str) -> list[LocalVoice]:
    provider = _require_provider(provider)
    _migrate_legacy(profile)
    return [
        _to_local(provider, p)
        for p in voice_lab.load_library(profile)
        if _norm(p.provider) == provider
    ]


def load_local(profile: str) -> dict[str, list[LocalVoice]]:
    """Every local bucket for the profile."""
    _migrate_legacy(profile)
    library = voice_lab.load_library(profile)
    return {
        provider: [
            _to_local(provider, p)
            for p in library
            if _norm(p.provider) == provider
        ]
        for provider in LOCAL_VOICE_PROVIDERS
    }


def add_local(profile: str, provider: str, raw: dict[str, Any], *, now: str = "") -> LocalVoice:
    """Upsert by name inside one provider's view — written to the shared
    library so the voice also shows up in the Voice Lab and A/B."""
    provider = _require_provider(provider)
    name = str(raw.get("name") or "").strip()
    if not name:
        raise LocalVoiceError("give the voice a name")

    existing = [
        p for p in voice_lab.load_library(profile)
        if _norm(p.provider) == provider and p.name.lower() != name.lower()
    ]
    if len(existing) >= MAX_LOCAL_VOICES:
        raise LocalVoiceError(
            f"the {provider} voice list is full ({MAX_LOCAL_VOICES} saved) — delete one first"
        )

    payload = {
        "name": name,
        "provider": provider,
        "voice_id": str(raw.get("voice_id") or "").strip(),
        "notes": raw.get("notes", ""),
        "source": str(raw.get("source") or f"{SOURCE_PREFIX}{provider}"),
        "tts": sanitize_tts(provider, raw.get("tts")),
    }
    if raw.get("created_at"):
        payload["created_at"] = raw["created_at"]
    try:
        preset = voice_lab.add_preset(profile, payload, now=now)
    except voice_lab.VoiceLabError as e:
        raise LocalVoiceError(str(e)) from e
    return _to_local(provider, preset)


def remove_local(profile: str, provider: str, name: str) -> bool:
    """Delete a saved voice, but only if it belongs to this provider — a
    same-named preset on another engine must not vanish from the Voice Lab."""
    provider = _require_provider(provider)
    target = (name or "").strip().lower()
    presets = voice_lab.load_library(profile)
    kept = [
        p for p in presets
        if not (p.name.lower() == target and _norm(p.provider) == provider)
    ]
    if len(kept) == len(presets):
        return False
    voice_lab.save_library(profile, kept)
    return True


def tts_updates(provider: str, voice: LocalVoice) -> dict[str, Any]:
    """TTSConfig fields a saved local voice applies: switch to the engine, set
    its voice identity, then layer the stored knobs on top."""
    provider = _require_provider(provider)
    updates: dict[str, Any] = dict(voice.tts)
    updates["provider"] = provider
    id_field = VOICE_ID_FIELD[provider]
    if voice.voice_id:
        updates[id_field] = voice.voice_id
    return updates


def providers_payload(profile: str) -> dict[str, Any]:
    """Everything the Voice tab needs for the per-model pickers in one call.

    ``in_library`` mirrors the whole Voice Lab list (all providers) so the UI can
    tell a local voice from the broader library without another request.
    """
    buckets = load_local(profile)
    library = voice_lab.load_library(profile)
    return {
        "profile": profile,
        "providers": {
            provider: [v.to_dict() for v in buckets.get(provider, [])]
            for provider in LOCAL_VOICE_PROVIDERS
        },
        "library_count": len(library),
    }

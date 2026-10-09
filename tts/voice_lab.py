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
    name = str(raw.get("name") or "").strip()
    if not name:
        raise VoiceLabError("the saved voice needs a name")
    if not now:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preset = VoicePreset(
        name=name[:60],
        provider=str(raw.get("provider") or ""),
        voice_id=str(raw.get("voice_id") or ""),
        notes=str(raw.get("notes") or "")[:400],
        source=str(raw.get("source") or "manual")[:40],
        created_at=str(raw.get("created_at") or now),
        tts=sanitize_tts(raw.get("tts")),
    )
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

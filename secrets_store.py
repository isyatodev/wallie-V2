"""Secrets management — UI-editable API keys, written to .env safely."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from config import BASE_DIR

ENV_FILE = BASE_DIR / ".env"


SECRET_FIELDS: dict[str, dict[str, str]] = {
    "OPENAI_API_KEY": {
        "label": "OpenAI", "kind": "llm",
        "url": "https://platform.openai.com/api-keys",
        "hint": "starts with sk-…",
    },
    "GROQ_API_KEY": {
        "label": "Groq", "kind": "llm",
        "url": "https://console.groq.com/keys",
        "hint": "starts with gsk_…",
    },
    "OPENROUTER_API_KEY": {
        "label": "OpenRouter", "kind": "llm",
        "url": "https://openrouter.ai/keys",
        "hint": "starts with sk-or-…",
    },
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic (Claude)", "kind": "llm",
        "url": "https://console.anthropic.com/settings/keys",
        "hint": "starts with sk-ant-…",
    },
    "GEMINI_API_KEY": {
        "label": "Google Gemini", "kind": "llm",
        "url": "https://aistudio.google.com/apikey",
        "hint": "free tier available",
    },
    "FISH_API_KEY": {
        "label": "Fish Audio", "kind": "tts",
        "url": "https://fish.audio/go-api/api-keys/",
        "hint": "from fish.audio dashboard",
    },
    "ELEVENLABS_API_KEY": {
        "label": "ElevenLabs", "kind": "tts",
        "url": "https://elevenlabs.io/app/settings/api-keys",
        "hint": "starts with sk_…",
    },
    "YOUTUBE_API_KEY": {
        "label": "YouTube API", "kind": "stream",
        "url": "https://console.cloud.google.com/apis/credentials",
        "hint": "Google Cloud Console",
    },
    "YOUTUBE_LIVE_CHAT_ID": {
        "label": "YouTube Live Chat ID", "kind": "stream",
        "url": "",
        "hint": "auto-detected when blank",
    },
    "TWITCH_OAUTH_TOKEN": {
        "label": "Twitch OAuth Token", "kind": "stream",
        "url": "https://twitchtokengenerator.com",
        "hint": "needs chat:read scope",
    },
    "TWITCH_CHANNEL": {
        "label": "Twitch Channel", "kind": "stream",
        "url": "",
        "hint": "the channel you stream on",
    },
    "TWITCH_NICK": {
        "label": "Twitch Nick", "kind": "stream",
        "url": "",
        "hint": "leave blank for anonymous",
    },
    "KICK_CHANNEL": {
        "label": "Kick Channel", "kind": "stream",
        "url": "",
        "hint": "channel slug",
    },
    "OPENAI_COMPATIBLE_API_KEY": {
        "label": "OpenAI-Compatible (generic)", "kind": "llm",
        "url": "",
        "hint": "any OpenAI-compatible endpoint; set base URL + model in Engine",
    },
    "OPENAI_COMPATIBLE_VISION_API_KEY": {
        "label": "OpenAI-Compatible (vision)", "kind": "llm",
        "url": "",
        "hint": "dedicated vision model (e.g. qwen-vl); base URL + model in Engine → Vision block",
    },
    "OPENAI_COMPATIBLE_TTS_API_KEY": {
        "label": "OpenAI-Compatible (TTS)", "kind": "tts",
        "url": "",
        "hint": "speech gateway ({base}/audio/speech); base URL + voice in Voice",
    },
    "OPENAI_COMPATIBLE_STT_API_KEY": {
        "label": "OpenAI-Compatible (STT)", "kind": "llm",
        "url": "",
        "hint": "transcription gateway ({base}/audio/transcriptions); configure in Hearing",
    },
    "OPENAI_COMPATIBLE_MEMORY_API_KEY": {
        "label": "OpenAI-Compatible (memory)", "kind": "llm",
        "url": "",
        "hint": "memory-extraction model; base URL + model in the Memory section",
    },
    "OPENAI_COMPATIBLE_THOUGHTS_API_KEY": {
        "label": "OpenAI-Compatible (thoughts)", "kind": "llm",
        "url": "",
        "hint": "spontaneous-thought generator; base URL + model in the Memory section",
    },
    "LIVEPIX_CLIENT_ID": {
        "label": "LivePix Client ID", "kind": "stream",
        "url": "https://docs.livepix.gg",
        "hint": "from your LivePix account settings → apps",
    },
    "LIVEPIX_CLIENT_SECRET": {
        "label": "LivePix Client Secret", "kind": "stream",
        "url": "https://docs.livepix.gg",
        "hint": "OAuth2 client_credentials",
    },
    "LIVEPIX_USER_ID": {
        "label": "LivePix User ID", "kind": "stream",
        "url": "https://docs.livepix.gg",
        "hint": "optional; rejects webhooks for other accounts",
    },
    "STREAMLABS_ACCESS_TOKEN": {
        "label": "Streamlabs Access Token", "kind": "stream",
        "url": "https://streamlabs.com/dashboard#/settings/api-settings",
        "hint": "OAuth token with donations.read + socket.token scopes",
    },
    "STREAMLABS_SOCKET_TOKEN": {
        "label": "Streamlabs Socket Token", "kind": "stream",
        "url": "https://streamlabs.com/dashboard#/settings/api-settings",
        "hint": "API Settings → API Tokens → Socket API Token (simplest)",
    },
}


def mask(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    if len(v) <= 8:
        return "•" * len(v)
    return f"{v[:3]}{'•' * 6}{v[-3:]}"


def list_secrets() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for env_name, meta in SECRET_FIELDS.items():
        raw = os.getenv(env_name, "") or ""
        out.append(
            {
                "env": env_name,
                "label": meta["label"],
                "kind": meta["kind"],
                "url": meta["url"],
                "hint": meta["hint"],
                "is_set": bool(raw.strip()),
                "masked": mask(raw),
            }
        )
    return out


def set_secret(env_name: str, value: str) -> None:
    if env_name not in SECRET_FIELDS:
        raise ValueError(f"refused write to unknown env name: {env_name!r}")

    value = (value or "").strip()
    _ensure_env_file()

    from dotenv import load_dotenv, set_key, unset_key

    if value:
        set_key(str(ENV_FILE), env_name, value, quote_mode="auto")
    else:
        try:
            unset_key(str(ENV_FILE), env_name)
        except Exception:
            # If the key wasn't there, that's fine.
            pass

    _harden_perms(ENV_FILE)
    load_dotenv(str(ENV_FILE), override=True)


def update_many(values: dict[str, str]) -> list[str]:
    accepted: list[str] = []
    for env, val in values.items():
        if env not in SECRET_FIELDS:
            continue
        set_secret(env, val)
        accepted.append(env)
    return accepted


def envs_for_kind(kind: str) -> Iterable[str]:
    return [env for env, meta in SECRET_FIELDS.items() if meta["kind"] == kind]


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _ensure_env_file() -> None:
    if not ENV_FILE.exists():
        ENV_FILE.write_text("", encoding="utf-8")
        _harden_perms(ENV_FILE)


def _harden_perms(path: Path) -> None:
    try:
        if os.name != "nt":
            os.chmod(path, 0o600)
    except Exception:
        pass

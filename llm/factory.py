"""Provider factory — maps LLMConfig + Secrets to a concrete LLMProvider."""
from __future__ import annotations

from config import LLMConfig, Secrets

from .base import LLMError, LLMProvider


def _missing_sdk(name: str, pkg: str) -> LLMError:
    return LLMError(
        f"{name} provider selected but '{pkg}' is not installed. "
        f"Install it with: pip install {pkg}"
    )


def build_provider(cfg: LLMConfig, secrets: Secrets) -> LLMProvider:
    p = cfg.provider

    if p in ("openai", "groq", "openrouter"):
        try:
            from .openai_compat import OpenAICompatProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk(p, "openai") from e

        if p == "openai":
            return OpenAICompatProvider(
                name="openai",
                model=cfg.model,
                api_key=secrets.openai_api_key,
                supports_vision=cfg.vision_capable,
            )
        if p == "groq":
            return OpenAICompatProvider(
                name="groq",
                model=cfg.model,
                api_key=secrets.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                supports_vision=cfg.vision_capable,
            )
        return OpenAICompatProvider(
            name="openrouter",
            model=cfg.model,
            api_key=secrets.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            supports_vision=cfg.vision_capable,
            extra_headers={
                "HTTP-Referer": "https://github.com/wallie-ai/wallie",
                "X-Title": "Wallie",
            },
        )

    if p == "openai_compatible":
        # GENERIC OpenAI-compatible endpoint: any provider implementing the
        # chat/completions API. Base URL (with version path), model and API key
        # are all user-configured — no provider is hardcoded here.
        try:
            from .openai_compat import OpenAICompatProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk(p, "openai") from e
        base_url = (cfg.openai_compatible_base_url or "").strip()
        if not base_url:
            raise LLMError(
                "openai_compatible: set llm.openai_compatible_base_url in the profile "
                '(e.g. "https://example.com/v1") — it must include the version path'
            )
        return OpenAICompatProvider(
            name="openai_compatible",
            model=cfg.model,
            api_key=secrets.openai_compatible_api_key,
            base_url=base_url,
            supports_vision=cfg.vision_capable,
            timeout=cfg.openai_compatible_timeout,
        )

    if p == "anthropic":
        try:
            from .anthropic import AnthropicProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk("anthropic", "anthropic") from e
        return AnthropicProvider(
            model=cfg.model,
            api_key=secrets.anthropic_api_key,
            supports_vision=cfg.vision_capable,
        )

    if p == "gemini":
        try:
            from .gemini import GeminiProvider
        except ModuleNotFoundError as e:
            raise _missing_sdk("gemini", "google-generativeai") from e
        return GeminiProvider(
            model=cfg.model,
            api_key=secrets.gemini_api_key,
            supports_vision=cfg.vision_capable,
        )

    if p == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(
            model=cfg.model,
            base_url=cfg.ollama_base_url,
            keep_alive=cfg.ollama_keep_alive,
            supports_vision=cfg.vision_capable,
        )

    raise LLMError(f"Unknown LLM provider: {p}")
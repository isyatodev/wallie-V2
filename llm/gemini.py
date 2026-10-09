"""Google Gemini provider via google-generativeai."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import google.generativeai as genai

from .base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

# Gemini 2.5+ models are "thinking" models: they spend hidden reasoning tokens
# that are billed against ``max_output_tokens``. The installed (deprecated)
# ``google-generativeai`` SDK exposes no ``thinking_config``/``thinking_budget``,
# so a small visible-output budget — the vision loop caps at ~100-200 tokens —
# gets eaten by reasoning and the reply is cut off mid-sentence. We add explicit
# headroom on top of the caller's visible budget so the answer always fits.
_THINKING_MODEL_PREFIXES = ("gemini-3", "gemini-2.5")
_THINKING_HEADROOM = 1024
_MIN_OUTPUT_TOKENS = 40  # Gemini needs a floor to avoid safety-filter triggers.
_FINISH_MAX_TOKENS = 2  # proto FinishReason.MAX_TOKENS


class GeminiProvider(LLMProvider):
    def __init__(self, *, model: str, api_key: str, supports_vision: bool = True) -> None:
        if not api_key:
            raise LLMError("gemini: missing API key")
        self.name = "gemini"
        self.model = model
        self.supports_vision = supports_vision
        genai.configure(api_key=api_key)
        self._model_name = model
        self._thinking_headroom = (
            _THINKING_HEADROOM
            if (model or "").lower().startswith(_THINKING_MODEL_PREFIXES)
            else 0
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 500,
        presence_penalty: float = 0.0,  # accepted for signature parity
        frequency_penalty: float = 0.0,
    ) -> AsyncIterator[str]:
        system_prompt, rest = self._split_system(messages)
        # Reasoning headroom + a floor: ``max_tokens`` is the VISIBLE budget the
        # caller asked for, not the total the model may spend.
        effective_tokens = max(_MIN_OUTPUT_TOKENS, max_tokens) + self._thinking_headroom
        model = genai.GenerativeModel(
            self._model_name,
            system_instruction=system_prompt or None,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": effective_tokens,
            },
        )
        contents = [self._encode_message(m) for m in rest]
        try:
            response = await model.generate_content_async(contents, stream=True)
        except Exception as e:
            raise LLMError(f"gemini request failed: {e}") from e

        last_finish: int | None = None
        try:
            async for chunk in response:
                text = self._chunk_text(chunk)
                if text:
                    yield text
                fr = self._finish_reason(chunk)
                if fr is not None:
                    last_finish = fr
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"gemini stream failed: {e}") from e

        if last_finish == _FINISH_MAX_TOKENS:
            logger.warning(
                "gemini: reply hit MAX_TOKENS (budget=%d, model=%s) — increase "
                "max_tokens or headroom",
                effective_tokens,
                self._model_name,
            )

    async def aclose(self) -> None:
        await asyncio.sleep(0)

    # ----- helpers -----
    @classmethod
    def _chunk_text(cls, chunk: Any) -> str:
        try:
            text = getattr(chunk, "text", None)
        except ValueError:
            text = None
        if text:
            return text
        # ``.text`` raises when a chunk carries no plain text part (e.g. a
        # thought-only or non-text chunk). Fall back to the text parts so one
        # odd chunk does not silently end the stream mid-reply.
        out: list[str] = []
        for cand in getattr(chunk, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "thought", False):
                    continue  # hidden reasoning must never be spoken
                part_text = getattr(part, "text", None)
                if part_text:
                    out.append(part_text)
        return "".join(out)

    @staticmethod
    def _finish_reason(chunk: Any) -> int | None:
        for cand in getattr(chunk, "candidates", None) or []:
            fr = getattr(cand, "finish_reason", None)
            if fr is None:
                continue
            try:
                return int(fr)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        rest: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if isinstance(m["content"], str) else ""
            else:
                rest.append(m)
        return system, rest

    def _encode_message(self, m: dict[str, Any]) -> dict[str, Any]:
        role = "user" if m["role"] == "user" else "model"
        content = m.get("content")
        parts: list[Any] = []
        if isinstance(content, list):
            for b in content:
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    parts.append(
                        {"mime_type": b.get("mime", "image/jpeg"), "data": b["data"]}
                    )
        else:
            parts.append(content or "")
        return {"role": role, "parts": parts}

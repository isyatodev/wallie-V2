"""GeminiProvider: thinking-model budget headroom + stream robustness.

Gemini 2.5/3.x are "thinking" models whose hidden reasoning tokens are billed
against ``max_output_tokens``. The vision loop asks for a small visible budget
(100-200 tokens), so without headroom the reasoning ate the whole allowance and
replies came back truncated mid-sentence. These tests pin the headroom, the
stream-survives-a-non-text-chunk behaviour, and error wrapping.
"""
from __future__ import annotations

import sys
import pathlib
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import llm.gemini as gemini_mod
from llm.base import LLMError
from llm.gemini import GeminiProvider


# ─────────────────────────────────────────────────────────────────────
# Fake google-generativeai surface
# ─────────────────────────────────────────────────────────────────────

class _Chunk:
    def __init__(self, text=None, *, text_raises=False, parts=None, finish_reason=None):
        self._text = text
        self._text_raises = text_raises
        self.candidates = [
            types.SimpleNamespace(
                content=types.SimpleNamespace(parts=parts or []),
                finish_reason=finish_reason,
            )
        ]

    @property
    def text(self):
        if self._text_raises:
            raise ValueError("no text part")
        return self._text


class _FakeModel:
    def __init__(self, chunks, capture, *, boom=None):
        self._chunks = chunks
        self._capture = capture
        self._boom = boom

    async def generate_content_async(self, contents, stream=False):
        if self._boom is not None:
            raise self._boom
        self._capture["contents"] = contents
        self._capture["stream"] = stream

        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


class _FakeGenai:
    def __init__(self, capture, *, chunks=(), model_boom=None, sdk_boom=None):
        self._capture = capture
        self._chunks = chunks
        self._model_boom = model_boom
        self._sdk_boom = sdk_boom

    def configure(self, api_key):
        self._capture["api_key"] = api_key

    def GenerativeModel(self, name, system_instruction=None, generation_config=None):
        self._capture["name"] = name
        self._capture["generation_config"] = generation_config
        return _FakeModel(self._chunks, self._capture, boom=self._model_boom)


def _install(monkeypatch, **kw) -> dict:
    capture: dict = {}
    monkeypatch.setattr(gemini_mod, "genai", _FakeGenai(capture, **kw))
    return capture


async def _drain(provider, *, max_tokens=500):
    out = []
    async for tok in provider.stream(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
    ):
        out.append(tok)
    return out


# ─────────────────────────────────────────────────────────────────────
# Budget headroom
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_thinking_model_gets_reasoning_headroom(monkeypatch):
    """A gemini-3 model budgets max_tokens + headroom, not just max_tokens."""
    cap = _install(monkeypatch, chunks=[_Chunk(text="hi")])
    p = GeminiProvider(model="gemini-3.8-flash", api_key="k")
    tokens = await _drain(p, max_tokens=100)  # vision-loop-scale budget
    assert tokens == ["hi"]
    cfg = cap["generation_config"]
    assert cfg["max_output_tokens"] == 100 + gemini_mod._THINKING_HEADROOM
    assert cfg["temperature"] is not None and cfg["top_p"] is not None


@pytest.mark.asyncio
async def test_legacy_25_model_also_gets_headroom(monkeypatch):
    cap = _install(monkeypatch, chunks=[_Chunk(text="ok")])
    p = GeminiProvider(model="gemini-2.5-flash", api_key="k")
    await _drain(p, max_tokens=80)
    assert cap["generation_config"]["max_output_tokens"] == 80 + gemini_mod._THINKING_HEADROOM


@pytest.mark.asyncio
async def test_non_thinking_model_has_no_headroom(monkeypatch):
    cap = _install(monkeypatch, chunks=[_Chunk(text="ok")])
    p = GeminiProvider(model="gemini-1.5-flash", api_key="k")
    await _drain(p, max_tokens=80)
    assert cap["generation_config"]["max_output_tokens"] == 80


@pytest.mark.asyncio
async def test_tiny_budget_respects_floor(monkeypatch):
    """A tiny max_tokens still gets the SDK-safety floor (no headroom here)."""
    cap = _install(monkeypatch, chunks=[_Chunk(text="ok")])
    p = GeminiProvider(model="gemini-1.5-flash", api_key="k")
    await _drain(p, max_tokens=4)
    assert cap["generation_config"]["max_output_tokens"] == gemini_mod._MIN_OUTPUT_TOKENS


# ─────────────────────────────────────────────────────────────────────
# Stream robustness
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_survives_a_non_text_chunk(monkeypatch):
    """A chunk whose .text raises must NOT end the stream (old bug: truncation)."""
    _install(monkeypatch, chunks=[_Chunk(text_raises=True), _Chunk(text="completed")])
    p = GeminiProvider(model="gemini-3.8-flash", api_key="k")
    tokens = await _drain(p)
    assert tokens == ["completed"]


@pytest.mark.asyncio
async def test_thought_parts_are_never_spoken(monkeypatch):
    """Hidden reasoning parts are skipped; only real text is yielded."""
    parts = [
        types.SimpleNamespace(text="let me reason...", thought=True),
        types.SimpleNamespace(text="the actual answer", thought=False),
    ]
    _install(monkeypatch, chunks=[_Chunk(text_raises=True, parts=parts)])
    p = GeminiProvider(model="gemini-3.8-flash", api_key="k")
    tokens = await _drain(p)
    assert tokens == ["the actual answer"]


@pytest.mark.asyncio
async def test_max_tokens_finish_reason_warns(monkeypatch, caplog):
    _install(monkeypatch, chunks=[_Chunk(text="cut", finish_reason=2)])
    p = GeminiProvider(model="gemini-3.8-flash", api_key="k")
    with caplog.at_level("WARNING", logger="llm.gemini"):
        await _drain(p)
    assert any("MAX_TOKENS" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_mid_stream_error_is_wrapped(monkeypatch):
    class _BoomModel(_FakeModel):
        async def generate_content_async(self, contents, stream=False):
            async def _gen():
                yield _Chunk(text="partial")
                raise RuntimeError("connection reset")

            return _gen()

    cap: dict = {}

    class _BoomGenai(_FakeGenai):
        def GenerativeModel(self, *a, **kw):
            return _BoomModel((), cap)

    monkeypatch.setattr(gemini_mod, "genai", _BoomGenai(cap))
    p = GeminiProvider(model="gemini-3.8-flash", api_key="k")
    with pytest.raises(LLMError) as ei:
        await _drain(p)
    assert "gemini stream failed" in str(ei.value)
    assert "connection reset" in str(ei.value)


@pytest.mark.asyncio
async def test_request_error_is_wrapped(monkeypatch):
    _install(monkeypatch, model_boom=RuntimeError("503 high demand"))
    p = GeminiProvider(model="gemini-3.8-flash", api_key="k")
    with pytest.raises(LLMError) as ei:
        await _drain(p)
    assert "gemini request failed" in str(ei.value)


def test_missing_api_key_raises():
    with pytest.raises(LLMError):
        GeminiProvider(model="gemini-3.8-flash", api_key="")

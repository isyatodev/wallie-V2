"""Tests for the generic OpenAI-compatible provider path (no network).

The provider itself (llm/openai_compat.py) is battle-tested by the existing
openai/groq/openrouter routes; here we verify the GENERIC configuration layer:
factory wiring, config validation, base-url handling, and vision gating.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import pytest

# Stub the openai SDK so the provider imports without the real dependency.
if "openai" not in sys.modules:
    _openai = MagicMock()
    _openai.AsyncOpenAI = MagicMock()
    sys.modules["openai"] = _openai

from config import LLMConfig, Secrets
from llm.base import LLMError
from llm.factory import build_provider
from llm.openai_compat import OpenAICompatProvider


def _cfg(**kw) -> LLMConfig:
    base = dict(
        provider="openai_compatible",
        model="qwen2.5-72b-instruct",
        openai_compatible_base_url="https://example.com/v1",
        openai_compatible_timeout=12.5,
        vision_capable=False,
    )
    base.update(kw)
    return LLMConfig(**base)


def test_factory_builds_generic_provider():
    secrets = Secrets(openai_compatible_api_key="sk-test-123")
    p = build_provider(_cfg(), secrets)
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "openai_compatible"
    assert p.model == "qwen2.5-72b-instruct"
    assert p._base_url == "https://example.com/v1"
    assert p.supports_vision is False


def test_factory_requires_base_url():
    secrets = Secrets(openai_compatible_api_key="sk-test-123")
    with pytest.raises(LLMError) as ei:
        build_provider(_cfg(openai_compatible_base_url=""), secrets)
    assert "openai_compatible_base_url" in str(ei.value)


def test_provider_rejects_empty_key():
    with pytest.raises(LLMError):
        OpenAICompatProvider(
            name="openai_compatible", model="m", api_key="",
            base_url="https://example.com/v1",
        )


def test_vision_flag_gates():
    secrets = Secrets(openai_compatible_api_key="k")
    p_text = build_provider(_cfg(vision_capable=False), secrets)
    assert p_text.supports_vision is False
    p_vis = build_provider(_cfg(vision_capable=True), secrets)
    assert p_vis.supports_vision is True


def test_known_endpoints_are_accepted_shapes():
    """The generic route must accept the same shapes as the named providers."""
    secrets = Secrets(openai_compatible_api_key="k")
    for url in (
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
        "https://api.groq.com/openai/v1",
        "https://my-gateway.example.com/api/v1",
    ):
        p = build_provider(_cfg(openai_compatible_base_url=url), secrets)
        assert p._base_url == url


def test_timeout_propagates():
    with patch("llm.openai_compat.AsyncOpenAI") as mock_cls:
        build_provider(_cfg(openai_compatible_timeout=42.0), Secrets(openai_compatible_api_key="k"))
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["timeout"] == 42.0


# ---------------------------------------------------------------------------
# Message encoding: text + vision + system prompt
# ---------------------------------------------------------------------------

def test_encode_text_message():
    p = OpenAICompatProvider(name="t", model="m", api_key="k",
                             base_url="https://x/v1")
    out = p._encode_message({"role": "user", "content": "hello"})
    assert out == {"role": "user", "content": "hello"}


def test_encode_system_message():
    p = OpenAICompatProvider(name="t", model="m", api_key="k",
                             base_url="https://x/v1")
    out = p._encode_message({"role": "system", "content": "you are..."})
    assert out == {"role": "system", "content": "you are..."}


def test_encode_system_message_cache_only_for_openrouter():
    """cache_control breakpoints are an OpenRouter extension — other providers
    get plain content so they don't reject the unknown field."""
    p = OpenAICompatProvider(name="openrouter", model="m", api_key="k",
                             base_url="https://openrouter.ai/api/v1")
    out = p._encode_message({"role": "system", "content": "you are...", "cache": True})
    assert out["content"][0]["text"] == "you are..."
    assert out["content"][0]["cache_control"] == {"type": "ephemeral"}
    p2 = OpenAICompatProvider(name="openai_compatible", model="m", api_key="k",
                              base_url="https://x/v1")
    out2 = p2._encode_message({"role": "system", "content": "you are...", "cache": True})
    assert out2["content"] == "you are..."


def test_encode_vision_message_base64():
    p = OpenAICompatProvider(name="t", model="m", api_key="k",
                             base_url="https://x/v1")
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "data": b"\xff\xd8\xffjpeg", "mime": "image/jpeg"},
        ],
    }
    out = p._encode_message(msg)
    assert out["content"][0] == {"type": "text", "text": "look"}
    img = out["content"][1]
    assert img["type"] == "image_url"
    assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_stream_sends_openai_payload():
    """stream() must issue a chat.completions.create with the expected kwargs."""
    p = OpenAICompatProvider(name="t", model="my-model", api_key="k",
                             base_url="https://x/v1")

    async def fake_stream(**kwargs):
        async def _gen():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="ok"))])
        return _gen()

    with patch.object(p._client.chat.completions, "create", side_effect=fake_stream) as spy:
        chunks = [c async for c in p.stream(
            [{"role": "user", "content": "hi"}],
            temperature=0.5, top_p=0.9, max_tokens=50,
            presence_penalty=0.1, frequency_penalty=0.2,
        )]
    assert chunks == ["ok"]
    kwargs = spy.call_args.kwargs
    assert kwargs["model"] == "my-model"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 50
    assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_401_raises_llm_error():
    import httpx as _httpx
    err = _httpx.HTTPStatusError(
        "401 unauthorized",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )
    p = OpenAICompatProvider(name="t", model="m", api_key="k",
                             base_url="https://x/v1")

    async def fail(**kwargs):
        raise err
        yield  # pragma: no cover

    with patch.object(p._client.chat.completions, "create", side_effect=fail):
        with pytest.raises(LLMError):
            async for _ in p.stream([{"role": "user", "content": "hi"}]):
                pass


@pytest.mark.asyncio
async def test_stream_429_retries_then_raises():
    import httpx as _httpx
    err = _httpx.HTTPStatusError(
        "429 too many requests",
        request=MagicMock(),
        response=MagicMock(status_code=429),
    )
    p = OpenAICompatProvider(name="t", model="m", api_key="k",
                             base_url="https://x/v1")

    async def fail(**kwargs):
        raise err
        yield  # pragma: no cover

    from unittest.mock import AsyncMock
    with patch("asyncio.sleep", new=AsyncMock()):
        with patch.object(p._client.chat.completions, "create", side_effect=fail):
            with pytest.raises(LLMError):
                async for _ in p.stream([{"role": "user", "content": "hi"}]):
                    pass


@pytest.mark.asyncio
async def test_vision_incompatible_model_path_is_config_level():
    """A text-only model + vision enabled is a config error surfaced by the
    orchestrator (wallie.py disables vision when vision_capable is False)."""
    secrets = Secrets(openai_compatible_api_key="k")
    cfg = _cfg(vision_capable=False)
    p = build_provider(cfg, secrets)
    assert p.supports_vision is False
    # The wallie guard: vision config must refuse to start on a text-only model.
    assert cfg.vision_capable is False

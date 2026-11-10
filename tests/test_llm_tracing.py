"""Tests for the optional Respan tracing hook (no network, no real key).

Tracing must stay invisible unless ``RESPAN_API_KEY`` is set: a checkout
without a Respan account keeps working, and a provider build never fails
because tracing could not start.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import llm.tracing as tracing
from config import LLMConfig, Secrets
from llm.factory import build_provider


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "_respan", None)
    monkeypatch.setattr(tracing, "_init_failed", False)


def test_disabled_without_key():
    assert tracing.tracing_enabled() is False
    # A missing package would raise on import; no key means we never get there.
    with patch.dict(sys.modules, {"respan": None}):
        assert tracing.init_tracing() is None
    assert tracing._respan is None


def test_initializes_once_with_the_app_name(monkeypatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    facade = MagicMock(name="Respan facade")
    ctor = MagicMock(return_value=facade)
    with patch.dict(sys.modules, {"respan": SimpleNamespace(Respan=ctor)}):
        assert tracing.init_tracing() is facade
        assert tracing.init_tracing() is facade
    assert ctor.call_count == 1
    assert ctor.call_args.kwargs == {"app_name": "wallie"}


def test_missing_package_leaves_the_app_running(monkeypatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    with patch.dict(sys.modules, {"respan": None}):
        assert tracing.init_tracing() is None
    assert tracing._init_failed is True


def test_failed_init_is_swallowed_and_not_retried(monkeypatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    ctor = MagicMock(side_effect=RuntimeError("boom"))
    with patch.dict(sys.modules, {"respan": SimpleNamespace(Respan=ctor)}):
        assert tracing.init_tracing() is None
        assert tracing.init_tracing() is None
    assert ctor.call_count == 1


def test_flush_is_a_no_op_when_tracing_is_off():
    tracing.flush_tracing()  # must not raise

    facade = MagicMock()
    with patch.object(tracing, "_respan", facade):
        tracing.flush_tracing()
    facade.flush.assert_called_once_with()


def test_build_provider_initializes_tracing(monkeypatch):
    """The factory is the single place a model client is born."""
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    ctor = MagicMock(return_value=MagicMock())
    cfg = LLMConfig(
        provider="openai_compatible",
        model="m",
        openai_compatible_base_url="https://example.com/v1",
        vision_capable=False,
    )
    with patch.dict(sys.modules, {"respan": SimpleNamespace(Respan=ctor)}):
        build_provider(cfg, Secrets(openai_compatible_api_key="dummy"))
    assert ctor.call_count == 1

def test_workflow_decorator_is_transparent_without_key():
    async def step():
        return "ok"

    assert tracing.workflow(name="wallie.reply")(step) is step


def test_workflow_decorator_wraps_with_respan_when_enabled(monkeypatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    wrapped = object()
    apply = MagicMock(return_value=wrapped)
    factory = MagicMock(return_value=apply)

    async def step():
        return "ok"

    with patch.dict(sys.modules, {"respan": SimpleNamespace(workflow=factory)}):
        out = tracing.workflow(name="wallie.reply")(step)
    assert out is wrapped
    factory.assert_called_once_with(name="wallie.reply")
    apply.assert_called_once_with(step)

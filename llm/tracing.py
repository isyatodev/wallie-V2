"""Respan tracing — one SDK init before any LLM client exists.

Respan auto-instruments the provider SDKs the app already imports (OpenAI,
Anthropic, ...), so the router in ``llm/factory.py`` keeps its provider
selection untouched: this is observability, not a gateway. Without
``RESPAN_API_KEY`` in the environment everything here is a no-op and the app
runs exactly as before.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from loguru import logger

# The initialized Respan facade, or None while tracing is off/unavailable.
_respan: Optional[Any] = None
# One failed attempt is enough: don't retry on every provider build.
_init_failed = False


def tracing_enabled() -> bool:
    """True when a Respan key is configured (the SDK reads it from the env)."""
    return bool(os.getenv("RESPAN_API_KEY", "").strip())


def init_tracing() -> Optional[Any]:
    """Initialize Respan once, before the first model client is created.

    Safe to call from every provider build: the SDK is initialized a single
    time and later calls are cheap. Returns the facade, or None when tracing is
    off or the optional package is unavailable.
    """
    global _respan, _init_failed
    if _respan is not None or _init_failed:
        return _respan
    if not tracing_enabled():
        return None
    try:
        from respan import Respan  # imported lazily — tracing is optional
    except ImportError:
        _init_failed = True
        logger.warning(
            "respan: RESPAN_API_KEY is set but 'respan-ai' is not installed — "
            "tracing off (add respan-ai==4.2.4 to the venv)"
        )
        return None
    try:
        # app_name groups these calls in the Respan platform; provider routing
        # stays exactly as configured in the profile.
        _respan = Respan(app_name="wallie")
    except Exception as e:  # noqa: BLE001 — tracing must never break the stream
        _init_failed = True
        logger.warning(f"respan: tracing init failed, continuing without traces: {e}")
        return None
    logger.info("respan: tracing enabled")
    return _respan


def workflow(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate one pipeline step so Respan records it as a workflow trace.

    The decorator is applied while the module that owns the step is imported,
    so a key present at startup is enough. Without tracing (no key, or the
    optional package absent) the decorated function is handed back untouched:
    the step keeps its exact behaviour and cost.
    """
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not tracing_enabled():
            return fn
        try:
            from respan import workflow as respan_workflow  # lazy: optional
        except ImportError:
            return fn
        return respan_workflow(name=name)(fn)

    return decorate


def flush_tracing() -> None:
    """Flush buffered spans now (the app calls this on shutdown)."""
    if _respan is None:
        return
    try:
        _respan.flush()
    except Exception as e:  # noqa: BLE001 — a lost span must not block shutdown
        logger.debug(f"respan: flush failed (non-fatal): {e}")

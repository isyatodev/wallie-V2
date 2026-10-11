"""Dashboard UI wiring for the API Keys page + per-feature provider blocks.

These are drift guards for markup that has no runtime to exercise in CI: a
renamed Alpine method or a dropped selector only fails in the browser, which
is exactly the kind of silent break these tests exist to catch. Every
assertion reads the shipped static assets directly.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "dashboard" / "static"


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _defines(js: str, name: str) -> bool:
    return bool(re.search(rf"\b{name}\s*\(", js))


# ---------------------------------------------------------------------------
# API Keys page: collapsible service categories
# ---------------------------------------------------------------------------

def test_provider_blocks_are_grouped_into_collapsible_categories():
    """Each service category (TTS/LLM/STT/…) collapses, so a long block list
    costs one line per category instead of a wall of rows."""
    html = _html()
    assert "x-for=\"cat in providerCategories\"" in html
    assert "groupOpen('provider', cat)" in html
    assert "toggleGroup('provider', cat)" in html
    js = _js()
    for name in ("groupOpen", "toggleGroup", "providerCatLabel", "providerCatHint", "providerBlocks"):
        assert _defines(js, name), name


def test_builtin_platform_keys_are_collapsible_and_exclude_block_keys():
    """The platform-key groups collapse too — and they must filter out dynamic
    provider-block keys, which are edited on their own block rows (duplicating
    them would show the same secret twice)."""
    html = _html()
    assert "groupOpen('secret', kind)" in html
    assert "builtinSecrets(kind)" in html
    assert "secretsByKind(kind)" not in html   # replaced by builtinSecrets
    js = _js()
    assert "s.kind === kind && !s.dynamic" in js


# ---------------------------------------------------------------------------
# Rename + delete + API key per block
# ---------------------------------------------------------------------------

def test_every_provider_block_row_can_be_renamed_deleted_and_keyed():
    html = _html()
    assert "providerRename(p)" in html
    assert "providerConfirmDelete(p)" in html
    # The key editor is inline on the row itself.
    assert "providerKeyEnv(p)" in html
    assert "startEditSecret(providerKeyEnv(p))" in html
    assert "saveSecret(providerKeyEnv(p))" in html
    js = _js()
    for name in ("providerRename", "providerKeySet", "providerKeyMasked", "secretRow"):
        assert _defines(js, name), name


def test_rename_keeps_the_block_id_so_refs_never_dangle():
    """Renaming a block only changes the display name: the id (and therefore
    PROVIDER_<ID>_API_KEY) stays, so no subsystem ref is orphaned."""
    js = _js()
    body = re.search(r"providerRename\(p\) \{(.*?)\n    \},", js, re.S)
    assert body, "providerRename not found in app.js"
    assert "p.name = name" in body.group(1)
    assert "p.id =" not in body.group(1)   # never regenerates the id


# ---------------------------------------------------------------------------
# Engine page: create a vision block + enter its key without leaving the page
# ---------------------------------------------------------------------------

def test_engine_page_can_create_a_vision_block_and_enter_its_key():
    html = _html()
    assert "providerAdd('vision')" in html
    assert "visionBlock()" in html
    assert "providerKeyEnv(visionBlock())" in html
    assert _defines(_js(), "visionBlock")


def test_vision_block_mirrors_the_runtime_fallback():
    """visionBlock() falls back to the first VISION block when no ref is set —
    the same order _resolve_provider uses, so the key shown is the key used."""
    js = _js()
    body = re.search(r"visionBlock\(\) \{(.*?)\n    \},", js, re.S)
    assert body, "visionBlock not found in app.js"
    assert "category === \"vision\"" in body.group(1)


# ---------------------------------------------------------------------------
# Every AI feature can select a created block
# ---------------------------------------------------------------------------

def test_memory_and_thoughts_offer_provider_block_selection():
    """The two subsystems that had no block picker now select one, like
    LLM/TTS/STT/Vision already did. Without it a memory/thoughts block was
    impossible to point a feature at."""
    html = _html()
    assert "cfg.memory.provider_ref" in html
    assert "providers.filter(x => x.category==='memory')" in html
    assert "cfg.random_thoughts.provider_ref" in html
    assert "providers.filter(x => x.category==='thoughts')" in html

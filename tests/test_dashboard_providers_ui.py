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


def _template(html: str, tpl_id: str) -> str:
    """Markup of one <provider-blocks> template.

    The templates nest real <template x-if/x-for> elements, so there is no
    reliable closing-tag match; instead take everything up to the next
    template id (the manager is the last one, followed by the component's
    script tag).
    """
    start = html.index(f'<template id="{tpl_id}">')
    rest = html[start + 1:]
    cuts = [i for i in (rest.find('<template id='),
                        rest.find('<script src="/static/provider-blocks.js">'))
            if i != -1]
    return rest[:min(cuts)]


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
    manager = _template(html, "tpl-provider-manager")
    assert "providerRename(p)" in manager
    assert "providerConfirmDelete(p)" in manager
    # Every row plugs the shared key editor in for its own block.
    assert '<provider-key-slot block="p"></provider-key-slot>' in manager
    key = _template(html, "tpl-provider-key")
    assert "providerKeyEnv(%%block%%)" in key
    assert "startEditSecret(providerKeyEnv(%%block%%))" in key
    assert "saveSecret(providerKeyEnv(%%block%%))" in key
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
    """The Engine shortcut adds a block, saves it and edits its key in place.

    That whole strip belongs to the picker component now: the Engine call site
    only asks for features="key new" for the vision category.
    """
    html = _html()
    usages = re.findall(r"<provider-blocks\b([^>]*)>", html)
    engine = [u for u in usages if 'category="vision"' in u and 'features="key new"' in u]
    assert len(engine) == 1, "the Engine vision picker lost its inline block/key strip"
    assert 'ref="cfg.llm.vision_provider_ref"' in engine[0]
    picker = _template(html, "tpl-provider-picker")
    assert "providerAdd(%%cat%%)" in picker
    assert 'data-feature="new"' in picker
    assert '<provider-key-slot data-feature="key"' in picker
    key = _template(html, "tpl-provider-key")
    assert "providerKeyEnv(%%block%%)" in key
    assert _defines(_js(), "visionBlock")


def test_vision_block_mirrors_the_runtime_fallback():
    """visionBlock() falls back to the first VISION block when no ref is set —
    the same order _resolve_provider uses, so the key shown is the key used."""
    js = _js()
    body = re.search(r"visionBlock\(\) \{(.*?)\n    \},", js, re.S)
    assert body, "visionBlock not found in app.js"
    assert "category === \"vision\"" in body.group(1)


# ---------------------------------------------------------------------------
# Every picker can probe the block it would use
# ---------------------------------------------------------------------------

def test_every_picker_probes_its_block_without_saving():
    """The test button in a picker reuses the existing provider probe.

    The target must be the block the runtime would really call (ref → first of
    the category → "default"), and the click must not save: one probe cannot
    silently commit edits from the other pages. The manager row keeps its
    save-first behaviour, since unsaved rows only exist there.
    """
    html = _html()
    js = _js()
    picker = _template(html, "tpl-provider-picker")
    assert "providerTestPick('%%ref%%', %%cat%%, %%ref%%)" in picker
    assert "providerTestTarget(%%cat%%, %%ref%%)" in picker
    # `undefined` is truthy for Alpine's boolean binding: without the !! the
    # button ships disabled and can never be clicked.
    assert ":disabled=\"!!pickerTestBusy['%%ref%%']\"" in picker
    assert "pickerTestMsg['%%ref%%']" in picker
    for name in ("providerProbe", "providerTestTarget", "providerTestPick"):
        assert _defines(js, name), name

    target = re.search(r"providerTestTarget\(category, ref\) \{(.*?)\n    \},", js, re.S)
    assert target, "providerTestTarget not found in app.js"
    body = target.group(1)
    assert "p.id === wanted" in body
    assert "p.category === category" in body
    assert 'p.id === "default"' in body

    pick = re.search(r"async providerTestPick\(refPath, category, ref\) \{(.*?)\n    \},", js, re.S)
    assert pick, "providerTestPick not found in app.js"
    assert "providerSaveAll" not in pick.group(1)
    assert "providerProbe(block)" in pick.group(1)

    manager = re.search(r"async providerTest\(p\) \{(.*?)\n    \},", js, re.S)
    assert manager, "providerTest not found in app.js"
    assert "providerSaveAll()" in manager.group(1)
    assert "providerProbe(p)" in manager.group(1)


def test_manager_can_probe_every_block_and_summarise_the_failures():
    """One click on API Keys saves, probes everything, and names what broke.

    The summary is the deliverable: a bare count leaves the user hunting the
    failing row. Saving is part of the contract because the server only probes
    blocks that exist in the profile — and a failed save must abort the sweep
    instead of reporting results for the previous config.
    """
    html = _html()
    js = _js()
    assert '@click="providerTestAll()"' in html
    assert "providerTestSummary" in html
    assert ":disabled=\"providerBusy || testingAll\"" in html

    save = re.search(r"async providerSaveAll\(\) \{(.*?)\n    \},", js, re.S)
    assert save, "providerSaveAll not found in app.js"
    assert "return true" in save.group(1)
    assert "return false" in save.group(1)

    sweep = re.search(r"async providerTestAll\(\) \{(.*?)\n    \},", js, re.S)
    assert sweep, "providerTestAll not found in app.js"
    body = sweep.group(1)
    assert "providerSaveAll()" in body
    assert "if (!saved)" in body
    # Parallel probes: a sequential sweep would cost the sum of the timeouts.
    assert "Promise.all" in body
    assert "providerProbe(p)" in body
    # Failures are reported by name, with the server's reason.
    assert "r.block.name || r.block.id" in body
    assert "r.text" in body
    assert "no blocks to test" in body
    assert "okCount" in body and "failed.length" in body


def test_topbar_badge_reports_the_last_test_all():
    """The badge must be a snapshot of the sweep, not a guess.

    A green pill with no timestamp is a promise the dashboard cannot keep, so
    the title carries the time; and the badge starts hidden, so nobody reads
    "all ok" before any block was ever probed. Clicking lands on the page that
    fixes a red one.
    """
    html = _html()
    js = _js()
    badge = re.search(r'<button[^>]*class="health-badge"[^>]*>', html, re.S)
    assert badge, "the provider health badge is gone from the top bar"
    tag = badge.group(0)
    assert 'x-show="providerHealth"' in tag
    assert ':class="providerHealthBadge().cls"' in tag
    assert ':title="providerHealthBadge().title"' in tag
    assert '@click="openProviderHealth()"' in tag

    sweep = re.search(r"async providerTestAll\(\) \{(.*?)\n    \},", js, re.S)
    assert sweep, "providerTestAll not found in app.js"
    body = sweep.group(1)
    assert 'state: "testing"' in body
    assert 'state: failed.length ? "failed" : "ok"' in body
    assert "providerHealth = null" in body          # no blocks -> hide the badge

    badge_fn = re.search(r"providerHealthBadge\(\) \{(.*?)\n    \},", js, re.S)
    assert badge_fn, "providerHealthBadge not found in app.js"
    fn = badge_fn.group(1)
    assert "toLocaleTimeString()" in fn
    assert "Last check" in fn
    assert "failing" in fn and "failed" in fn

    click = re.search(r"openProviderHealth\(\) \{(.*?)\n    \},", js, re.S)
    assert click, "openProviderHealth not found in app.js"
    target = re.search(r'this\.section = "([^"]+)"', click.group(1))
    assert target, "the badge no longer navigates anywhere"
    section_id = target.group(1)
    assert f"section==='{section_id}'" in html
    assert f'id: "{section_id}"' in js       # reachable from the nav too

    # A snapshot from another profile (or one naming deleted blocks) is a lie.
    switch = re.search(r"async switchProfile\(name\) \{(.*?)\n    \},", js, re.S)
    assert switch and "providerHealth = null" in switch.group(1)
    delete = re.search(r"async providerDelete\(id, localRefs\) \{(.*?)\n    \},", js, re.S)
    assert delete and "providerHealth = null" in delete.group(1)


# ---------------------------------------------------------------------------
# Every AI feature can select a created block
# ---------------------------------------------------------------------------

def test_memory_and_thoughts_offer_provider_block_selection():
    """The two subsystems that had no block picker now select one, like
    LLM/TTS/STT/Vision already did. Without it a memory/thoughts block was
    impossible to point a feature at."""
    html = _html()
    assert 'category="memory" ref="cfg.memory.provider_ref"' in html
    assert 'category="thoughts" ref="cfg.random_thoughts.provider_ref"' in html
    # The picker filters by the category it was given, so each site gets only
    # its own blocks.
    assert "providers.filter(x => x.category===%%cat%%)" in _template(html, "tpl-provider-picker")

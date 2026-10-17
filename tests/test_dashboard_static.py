"""Dashboard stylesheet ↔ markup guards, and the asset cache policy.

The dashboard has exactly one stylesheet: dashboard/static/style.css. These
tests keep it that way and keep it honest, because every failure mode here is
silent:

* a ``var(--token)`` the stylesheet never declares compiles to nothing — the
  divider it painted simply vanishes (``--border`` did exactly that);
* a class with no rule renders as the raw browser default — ``.mini-btn`` was a
  grey UA button in a black cockpit, and ``.modal-backdrop`` had no overlay at
  all, so the engagement-gate inspector opened in normal flow behind
  ``body{overflow:hidden}`` and could never be seen;
* styling that creeps back into inline attributes is styling the theme switch
  cannot reach (the setup wizard kept a whole second palette in index.html);
* assets served without ``Cache-Control`` get heuristic freshness from the
  browser, so an edited stylesheet keeps rendering from cache for minutes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "dashboard" / "static"
CSS = STATIC / "style.css"

# Alpine binds these to data values, not to classes; they only ever appear as
# modifier names next to a styled base class (e.g. .chip.on, .status-pill.on).
DYNAMIC_CLASSES = {"drawer-open"}


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _declared_tokens(css: str) -> set[str]:
    return {m[:-1].strip() for m in re.findall(r"--[A-Za-z0-9_-]+\s*:", css)}


def _referenced_tokens(*sources: str) -> set[str]:
    out: set[str] = set()
    for src in sources:
        out.update(re.findall(r"var\((--[A-Za-z0-9_-]+)", src))
    return out


def _classes_with_rules(css: str) -> set[str]:
    return set(re.findall(r"\.([A-Za-z][A-Za-z0-9_-]*)", css))


def _classes_in_markup(*sources: str) -> set[str]:
    """Literal class names in ``class="…"`` attributes of HTML and in the
    template strings app.js builds.

    Deliberately literal-only: ``:class="running ? 'on' : 'off'"`` and
    ``:class="{ 'drawer-open': drawerOpen }"`` carry expressions and data
    values, which are not declarations of styling intent.
    """
    out: set[str] = set()
    for src in sources:
        for attr in re.findall(r'(?<!:)class="([^"]*)"', src):
            if "{" in attr:
                continue
            out.update(t for t in attr.split() if re.fullmatch(r"[a-z][a-z0-9-]*", t))
    return out


# ---------------------------------------------------------------------------
# Drift: tokens
# ---------------------------------------------------------------------------

def test_stylesheet_declares_every_token_the_ui_references():
    """A renamed or mistyped token must fail here, not silently paint nothing."""
    css = _read("style.css")
    referenced = _referenced_tokens(_read("index.html"), _read("app.js"))
    assert referenced, "no var(--token) found — did the markup change shape?"
    assert referenced - _declared_tokens(css) == set()


def test_every_class_in_the_markup_has_a_rule():
    """No component may go back to relying on browser defaults."""
    css = _read("style.css")
    used = _classes_in_markup(_read("index.html"), _read("app.js"))
    assert len(used) > 150, f"only found {len(used)} classes — the scanner is broken"
    unstyled = sorted(used - _classes_with_rules(css) - DYNAMIC_CLASSES)
    assert unstyled == []


def test_visual_styling_stays_out_of_the_markup():
    """No inline palette/type in index.html, and no second <style> block: the
    theme switch and the token layer can only govern what they can see."""
    html = _read("index.html")
    blocks = re.findall(r"<style>(.*?)</style>", html, re.S)
    assert len(blocks) == 1
    # The one inline block exists so Alpine's cloak applies before it boots.
    assert "[x-cloak]" in blocks[0]
    assert len(re.findall(r"\{", blocks[0])) == 1

    visual = re.findall(r'style="[^"]*(?:color:\s*#|background:\s*#|font-size:\s*[\d.]+rem)', html)
    assert visual == []


# ---------------------------------------------------------------------------
# Click targets
# ---------------------------------------------------------------------------

def test_partner_badge_shortcut_points_at_a_real_handler_and_section():
    """The ⇄ badge opens the Voice Lab with the partner already selected.

    Both halves of that shortcut fail silently: delete/rename
    ``openPartnerMirror()`` and the click does nothing; rename the section id
    and it switches to a section that isn't there.
    """
    html = _read("index.html")
    app = _read("app.js")

    badge = re.search(r"<button[^>]*class=\"sync-badge\"[^>]*>", html, re.S)
    assert badge, "the ⇄ partner badge is no longer a clickable button"
    assert '@click="openPartnerMirror()"' in badge.group(0)

    handler = re.search(r"openPartnerMirror\(\)\s*\{", app)
    assert handler, "openPartnerMirror() is gone from app.js"
    end = app.find("\n    },", handler.end())      # the method's own closing brace
    body = app[handler.end():end if end != -1 else len(app)]
    target = re.search(r'this\.section = "([^"]+)"', body)
    assert target, "openPartnerMirror() no longer switches section"
    section_id = target.group(1)
    assert f"section==='{section_id}'" in html, f"no section with id '{section_id}'"
    # The nav must offer it too, or the shortcut lands on an unreachable page.
    assert f'id: "{section_id}"' in app, f"'{section_id}' is missing from the nav sections"
    # And it scrolls to the mirror block, which is why that id has to exist.
    assert 'this.scrollToMirror()' in body
    scroller = re.search(r"scrollToMirror\(\)\s*\{", app)
    assert scroller, "scrollToMirror() is gone from app.js"
    assert 'getElementById("voice-mirror")' in app[scroller.end():scroller.end() + 400]
    assert 'id="voice-mirror"' in html


# ---------------------------------------------------------------------------
# One provider-block component
# ---------------------------------------------------------------------------

def _provider_block_usages(html: str) -> list[dict[str, str]]:
    return [dict(re.findall(r'([a-z-]+)="([^"]*)"', raw))
            for raw in re.findall(r"<provider-blocks\b([^>]*)>", html)]


def test_provider_blocks_render_from_one_component():
    """Engine, Vision and API Keys must use the same <provider-blocks>.

    Eight hand-rolled copies of the selector is how one of them silently
    forgets to warn about a deleted block, or the manager's inline key editor
    drifts from the picker's. These assertions pin the single component: the
    markup exists once (templates in index.html), every call site names a real
    mode and ref, and the API Keys manager is not a second implementation.
    """
    html = _read("index.html")
    module = _read("provider-blocks.js")

    # Loaded before app.js, so the elements are upgraded before Alpine boots.
    assert '<script src="/static/provider-blocks.js"></script>' in html
    assert html.index("/static/provider-blocks.js") < html.index("/static/app.js")

    usages = _provider_block_usages(html)
    assert usages, "<provider-blocks> disappeared from the markup"

    pickers: dict[str, int] = {}
    managers = 0
    for attrs in usages:
        assert attrs.get("mode") in ("picker", "manager"), attrs
        if attrs["mode"] == "manager":
            managers += 1
            continue
        assert attrs.get("label"), attrs
        pickers[attrs.get("ref", "")] = pickers.get(attrs.get("ref", ""), 0) + 1
        features = set((attrs.get("features") or "").split())
        assert features <= {"key", "new"}, attrs
        # The module only knows how to resolve the vision block's key itself.
        if "key" in features:
            assert attrs.get("category") == "vision" or attrs.get("key-block"), attrs

    # One place per subsystem, and Vision is the one that exists twice
    # (the Engine shortcut and the Vision section's own controls).
    assert pickers == {
        "cfg.llm.provider_ref": 1,
        "cfg.llm.vision_provider_ref": 2,
        "cfg.tts.provider_ref": 1,
        "cfg.hearing.provider_ref": 1,
        "cfg.memory.provider_ref": 1,
        "cfg.random_thoughts.provider_ref": 1,
    }
    assert managers == 1, "the API Keys manager must be the same component"

    # The category decides which blocks the picker lists — a mismatch would
    # offer LLM blocks to the STT engine. stt is the one name that does not
    # match its config section (Hearing), which is exactly why it is pinned.
    expected_categories = {
        "cfg.llm.provider_ref": "llm",
        "cfg.llm.vision_provider_ref": "vision",
        "cfg.tts.provider_ref": "tts",
        "cfg.hearing.provider_ref": "stt",
        "cfg.memory.provider_ref": "memory",
        "cfg.random_thoughts.provider_ref": "thoughts",
    }
    for attrs in usages:
        if attrs["mode"] != "picker":
            continue
        assert attrs.get("category") == expected_categories[attrs["ref"]], attrs

    # No hand-rolled selector is left: every stale-ref warning in the markup
    # comes from the picker template, and the ref selects are parameterized.
    assert html.count("applySuggestedRef(") == 1
    assert html.count("staleRef(") == 2      # the x-if guard + its message
    assert not re.search(r'x-model="cfg\.[a-z_]+\.provider_ref"', html)
    assert 'x-model="%%ref%%"' in html

    # Templates the module clones, and the placeholders it must fill.
    for tpl_id in ("tpl-provider-picker", "tpl-provider-manager", "tpl-provider-key"):
        assert f'<template id="{tpl_id}">' in html, tpl_id
        assert f'"{tpl_id}"' in module, tpl_id
    assert 'customElements.define("provider-blocks", ProviderBlocks)' in module
    assert "provider-key-slot" in module

    # Template ids are getElementById targets: a duplicate silently shadows
    # the real template (and leaves the stale copy's directives in the page,
    # where Alpine tries to evaluate them).
    ids = re.findall(r'<template id="([^"]+)">', html)
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})

    # Placeholders belong to the templates only. A stray copy outside them is
    # live markup: Alpine parses %%block%% as an expression and the component
    # never renders it.
    tpl_start = html.index("provider-blocks: one markup source")
    tpl_end = html.index('<script src="/static/provider-blocks.js">')
    assert "%%" not in html[:tpl_start], "a placeholder escaped into live markup"
    assert "%%" not in html[tpl_end:], "a placeholder escaped past the templates"

    used = set(re.findall(r"%%([A-Za-z]+)%%", html))
    declared_block = re.search(r"PLACEHOLDERS = \[([^\]]+)\]", module)
    assert declared_block, "the module no longer declares its placeholders"
    declared = set(re.findall(r'"([A-Za-z]+)"', declared_block.group(1)))
    assert used, "no placeholder found — did the templates change shape?"
    assert used <= declared, sorted(used - declared)
    assert used == declared, "a declared placeholder is no longer used"


# ---------------------------------------------------------------------------
# Asset cache policy
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """TestClient with an isolated profile dir (same shape as the other
    dashboard fixtures)."""
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import AppConfig, save_profile
    save_profile(AppConfig(profile_name="default"), "default")

    from dashboard.server import DashboardState, _build_app
    from starlette.testclient import TestClient
    return TestClient(_build_app(DashboardState(), None))


def test_ui_assets_tell_the_browser_to_revalidate(client):
    """Without this, a freshly edited stylesheet/app.js keeps rendering from
    the browser's heuristic cache (that is a minutes-long lie)."""
    for url in ("/", "/static/style.css", "/static/app.js"):
        r = client.get(url)
        assert r.status_code == 200, url
        assert r.headers.get("cache-control") == "no-cache", url


def test_revalidation_is_a_304_not_a_refetch(client):
    """'no-cache' must not mean 'download again': the ETag has to cover it."""
    first = client.get("/static/style.css")
    etag = first.headers.get("etag")
    assert etag
    again = client.get("/static/style.css", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""

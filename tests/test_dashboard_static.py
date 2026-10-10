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

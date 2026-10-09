"""Kokoro install badge — detection + the status / one-click install endpoints.

The Voice tab shows a live badge and a one-click install button; these tests
cover the cheap detector and the background install job (the pip run itself is
stubbed — no network, no real install).
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from config import AppConfig, Runtime, Secrets


def _app(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from config import load_profile, save_profile
    save_profile(AppConfig(profile_name="p-kokoro"), "p-kokoro")
    monkeypatch.setattr(
        config, "get_runtime",
        lambda: Runtime(config=load_profile("p-kokoro"), secrets=Secrets()),
    )
    from dashboard.server import DashboardState, _build_app
    state = DashboardState()
    return _build_app(state, None), state


# ─────────────────────────────────────────────────────────────────────
# tts.kokoro_setup.detect
# ─────────────────────────────────────────────────────────────────────

def test_detect_when_kokoro_absent(monkeypatch):
    from tts import kokoro_setup as k
    monkeypatch.setattr(k, "_importable", lambda module: False)
    d = k.detect()
    assert d["installed"] is False
    assert d["missing"] == ["kokoro", "soundfile"]
    assert d["installer_present"] is True


def test_detect_when_installed(monkeypatch):
    from tts import kokoro_setup as k
    monkeypatch.setattr(k, "_importable", lambda module: True)
    monkeypatch.setattr(k, "_version", lambda pkg: {"kokoro": "1.0.0", "soundfile": "0.12.1"}.get(pkg))
    d = k.detect()
    assert d["installed"] is True
    assert d["missing"] == []
    assert d["kokoro_version"] == "1.0.0"


def test_detect_never_imports_heavy_packages(monkeypatch):
    """Detection must not import torch/kokoro (that alone costs seconds)."""
    import builtins
    from tts import kokoro_setup as k
    real_import = builtins.__import__

    def _guard(name, *a, **kw):
        assert name not in ("torch", "kokoro", "soundfile"), f"detect imported {name}"
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _guard)
    k.detect()  # must not raise


def test_install_argv_targets_the_installer_script():
    from tts import kokoro_setup as k
    argv = k.install_argv("p")
    assert argv[1].endswith("install_kokoro.py")
    assert argv[2:] == ["--install", "--lang", "p"]
    # A configured voice pre-downloads that exact voice...
    assert k.install_argv("p", "pf_dora")[2:] == [
        "--install", "--lang", "p", "--voice", "pf_dora"
    ]
    # ...but a crafted voice can never masquerade as an installer flag.
    assert k.install_argv("p", "--install")[2:] == ["--install", "--lang", "p"]


def test_python_ok_bounds():
    from tts import kokoro_setup as k
    assert k.python_ok((3, 10, 0)) is True
    assert k.python_ok((3, 12, 7)) is True
    assert k.python_ok((3, 9, 9)) is False
    assert k.python_ok((3, 13, 0)) is False
    assert k.python_ok((3, 14, 7)) is False


def test_detect_exposes_python_compat(monkeypatch):
    from tts import kokoro_setup as k
    d = k.detect()
    assert d["python_version"] == k.python_version_str()
    assert d["python_ok"] is k.python_ok()
    # A one-click install is only offered when the interpreter can host kokoro.
    assert d["can_install"] == (k.python_ok() and k.INSTALL_SCRIPT.is_file())


def test_installer_check_python_bounds():
    import importlib.util
    from tts import kokoro_setup as k
    spec = importlib.util.spec_from_file_location("install_kokoro", k.INSTALL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._check_python((3, 11, 0)) is True
    assert mod._check_python((3, 9, 0)) is False
    assert mod._check_python((3, 14, 7)) is False


def test_installer_exit_code_matches_python_support():
    """On an unsupported interpreter the script must refuse with exit 3 and a
    fix; on a supported one it proceeds to the package/warm path."""
    import subprocess
    import sys
    from tts import kokoro_setup as k
    r = subprocess.run(
        [sys.executable, str(k.INSTALL_SCRIPT), "--lang", "p"],
        capture_output=True, text=True, cwd=str(k.REPO_ROOT), timeout=120,
    )
    if k.python_ok():
        assert r.returncode in (0, 2), r.stdout + r.stderr
    else:
        assert r.returncode == 3
        assert "Python 3.10" in r.stdout


def test_lang_codes_match_installer():
    from tts import kokoro_setup as k
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "install_kokoro", k.INSTALL_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(k.LANG_CODES) == set(mod._LANGS)


# ─────────────────────────────────────────────────────────────────────
# endpoints
# ─────────────────────────────────────────────────────────────────────

def test_status_endpoint(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.get("/api/tts/kokoro/status")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["ok"] is True
        assert "installed" in d and "missing" in d
        assert d["install"]["running"] is False


def test_install_endpoint_runs_and_records(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    gate = threading.Event()
    seen: dict = {}

    async def _fake_exec(*argv, **kw):
        seen["argv"] = list(argv)
        seen["cwd"] = kw.get("cwd")

        class _Proc:
            returncode = 0

            async def communicate(self):
                while not gate.is_set():
                    await asyncio.sleep(0.01)
                return (b"ok: installed kokoro", b"")

            def kill(self):
                gate.set()

        return _Proc()

    monkeypatch.setattr("dashboard.server.asyncio.create_subprocess_exec", _fake_exec)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/kokoro/install", json={"lang": "p", "voice": "pf_dora"})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "started": True, "lang": "p"}
        gate.set()
        for _ in range(200):
            if not state.kokoro_install["running"]:
                break
            time.sleep(0.02)

    assert state.kokoro_install["running"] is False
    assert state.kokoro_install["lang"] == "p"
    assert state.kokoro_install["returncode"] == 0
    assert "installed kokoro" in state.kokoro_install["log"]
    assert seen["argv"][1].endswith("install_kokoro.py")
    # The configured voice rides along so the panel's voice is what gets fetched.
    assert seen["argv"][2:] == ["--install", "--lang", "p", "--voice", "pf_dora"]


def test_install_endpoint_rejects_unknown_language(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/kokoro/install", json={"lang": "zz"})
        assert r.status_code == 400


def test_install_endpoint_rejects_when_already_running(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch)
    state.kokoro_install["running"] = True
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/kokoro/install", json={"lang": "a"})
        assert r.status_code == 409


def test_install_endpoint_reports_missing_script(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    monkeypatch.setattr("tts.kokoro_setup.INSTALL_SCRIPT", Path("/nope/install_kokoro.py"))
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.post("/api/tts/kokoro/install", json={"lang": "a"})
        assert r.status_code == 400
        assert "installer script missing" in r.json()["detail"]

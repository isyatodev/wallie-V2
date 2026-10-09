"""scripts/install_kokoro.py — language table stays in sync with the dashboard.

The installer prints "configure it in the panel" guidance, so its language codes
and default voices must match the picker in dashboard/static/app.js
(KOKORO_LANGS / KOKORO_VOICES). This guards against drift between the two.

It also resolves the output device for ``--play`` from the active profile, and
that lookup runs during setup: it must degrade (never raise, never write) so a
missing or corrupt profile can't break a user's one-click install.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import pathlib
import re

_ROOT = pathlib.Path(__file__).parent.parent


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "install_kokoro", _ROOT / "scripts" / "install_kokoro.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _app_js() -> str:
    return (_ROOT / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")


def _js_lang_codes(js: str) -> list[str]:
    block = re.search(r"const KOKORO_LANGS = \[(.*?)\];", js, re.S).group(1)
    return re.findall(r'\{\s*code:\s*"([a-z])"', block)


def _js_voices(js: str) -> dict[str, list[str]]:
    block = re.search(r"const KOKORO_VOICES = \{(.*?)\n\};", js, re.S).group(1)
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"([a-z]):\s*\[(.*?)\]", block, re.S):
        out[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    return out


def test_installer_languages_match_dashboard():
    mod = _load_installer()
    assert set(mod._LANGS) == set(_js_lang_codes(_app_js()))


def test_installer_default_voices_exist_in_dashboard():
    mod = _load_installer()
    voices = _js_voices(_app_js())
    for code, (_label, voice) in mod._LANGS.items():
        assert voice in voices.get(code, []), f"{voice!r} missing for lang {code!r}"


def test_pip_requirements():
    mod = _load_installer()
    base = mod._pip_requirements("a")
    assert any(r.startswith("kokoro>=") for r in base)
    assert any(r.startswith("soundfile>=") for r in base)
    # Japanese / Mandarin pull their misaki front; others do not.
    assert "misaki[ja]" in mod._pip_requirements("j")
    assert "misaki[zh]" in mod._pip_requirements("z")
    assert "misaki[ja]" not in mod._pip_requirements("a")


def test_warm_text_covers_every_language():
    mod = _load_installer()
    assert set(mod._WARM_TEXT) == set(mod._LANGS)


def test_main_help_exits_zero(capsys):
    mod = _load_installer()
    try:
        mod.main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    else:  # pragma: no cover - argparse always raises SystemExit on --help
        raise AssertionError("--help did not exit")


# ---------------------------------------------------------------------------
# Output-device resolution for the voice check (setup must never break on it)
# ---------------------------------------------------------------------------

def _profile(root: pathlib.Path, name: str, body: str) -> None:
    (root / "profiles").mkdir(exist_ok=True)
    (root / "profiles" / f"{name}.yaml").write_text(body, encoding="utf-8")


def _active(root: pathlib.Path, name: str) -> None:
    (root / ".wallie_state.json").write_text(
        json.dumps({"active": name}), encoding="utf-8")


def test_output_device_is_system_default_without_a_profile(tmp_path):
    """Fresh machine: no state file, no profile → system default, nothing written."""
    device, source = _load_installer()._profile_output_device(tmp_path)
    assert device == ""
    assert "system default" in source
    # The check must not create the profile it went looking for: that would
    # pre-empt the app's own first-run setup.
    assert not (tmp_path / "profiles").exists()


def test_output_device_reads_the_active_profile(tmp_path):
    _active(tmp_path, "stream")
    _profile(tmp_path, "stream", "tts:\n  output_device: 'CABLE Input'\n")
    assert _load_installer()._profile_output_device(tmp_path) == (
        "CABLE Input", "saved by profile 'stream'")


def test_output_device_without_a_state_file_uses_default_profile(tmp_path):
    _profile(tmp_path, "default", "tts:\n  output_device: Speakers\n")
    assert _load_installer()._profile_output_device(tmp_path)[0] == "Speakers"


def test_output_device_empty_in_the_profile_means_system_default(tmp_path):
    _active(tmp_path, "default")
    _profile(tmp_path, "default", "tts:\n  output_device: ''\n")
    device, source = _load_installer()._profile_output_device(tmp_path)
    assert device == ""
    assert "system default" in source


def test_output_device_is_stripped(tmp_path):
    """Sounddevice matches on a name substring, so stray spaces must not leak."""
    _active(tmp_path, "default")
    _profile(tmp_path, "default", "tts:\n  output_device: '  CABLE Input  '\n")
    assert _load_installer()._profile_output_device(tmp_path)[0] == "CABLE Input"


def test_output_device_tolerates_a_non_string_device(tmp_path):
    """A hand-edited profile may hold an index (int) or null — both survive."""
    mod = _load_installer()
    _active(tmp_path, "default")
    _profile(tmp_path, "default", "tts:\n  output_device: 112\n")
    assert mod._profile_output_device(tmp_path)[0] == "112"
    _profile(tmp_path, "default", "tts:\n  output_device: null\n")
    assert mod._profile_output_device(tmp_path)[0] == ""


def test_output_device_falls_back_safely_on_broken_files(tmp_path):
    """Every unreadable profile shape degrades to the system default with a
    reason — the setup must report a fallback, never a traceback."""
    mod = _load_installer()
    good_profile = "tts:\n  output_device: CABLE Input\n"
    cases = {
        # fixture comment -> file to corrupt
        "truncated_state_json": (".wallie_state.json", '{"active": '),
        "state_is_not_an_object": (".wallie_state.json", '["default"]'),
        "truncated_profile_yaml": ("profiles/default.yaml", "tts: [unclosed"),
        "profile_is_not_a_mapping": ("profiles/default.yaml", "- just\n- a list\n"),
        "profile_tts_is_not_a_mapping": ("profiles/default.yaml", "tts: 42\n"),
    }
    for label, (relative, body) in cases.items():
        root = tmp_path / label
        root.mkdir()
        (root / "profiles").mkdir()
        if relative == ".wallie_state.json":
            (root / relative).write_text(body, encoding="utf-8")
            (root / "profiles" / "default.yaml").write_text(good_profile, encoding="utf-8")
        else:
            (root / relative).write_text(body, encoding="utf-8")

        device, source = mod._profile_output_device(root)

        assert isinstance(device, str) and isinstance(source, str), label
        assert device in ("", "CABLE Input"), label
        assert "could not read the profile" in source, label


def test_profile_scheme_matches_config(tmp_path, monkeypatch):
    """install_kokoro.py reads the profile itself (config.load_profile() CREATES
    what is missing), so its private scheme — state file, "active" key, profile
    directory, name rule — must stay equivalent to config.py's. If either side
    drifts, the voice check reads a different file than the app and reports the
    wrong device in silence."""
    import config

    # The literals install_kokoro.py hardcodes must be config.py's own constants.
    assert config.STATE_FILE.name == ".wallie_state.json"
    assert config.PROFILES_DIR.name == "profiles"

    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / ".wallie_state.json")

    mod = _load_installer()
    # "we ird/na me" is the interesting one: the app sanitises it to a file name,
    # so the private reader must sanitise it the same way or it looks elsewhere.
    for name in ("stream", "p-voices", "we ird/na me"):
        (tmp_path / "profiles").mkdir(exist_ok=True)
        (tmp_path / ".wallie_state.json").write_text(
            json.dumps({"active": name}), encoding="utf-8")
        # Write the profile where CONFIG says the active profile lives...
        target = config._profile_path(name)
        target.write_text("tts:\n  output_device: 'CABLE Input'\n", encoding="utf-8")

        # ...then the state file, the app's loader and the installer's private
        # reader must all agree it is the same profile, with the same device.
        assert config._active_profile_name() == name
        assert config.load_profile().tts.output_device == "CABLE Input"
        device, source = mod._profile_output_device(tmp_path)
        assert device == "CABLE Input"
        assert target.stem in source


def test_play_verify_reports_instead_of_raising_without_the_audio_stack(monkeypatch, capsys):
    """The check runs inside someone's one-click setup: a checkout where the app's
    audio/TTS modules can't be imported must make it FAIL cleanly, not explode."""
    mod = _load_installer()
    monkeypatch.setattr(mod, "_profile_output_device",
                        lambda *a, **k: ("CABLE Input", "test"))

    real_import = builtins.__import__

    def _no_app_modules(name, *args, **kwargs):
        if name.split(".")[0] in {"audio", "tts"}:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_app_modules)

    assert mod._play_verify("a", "af_heart") == -1
    assert "could not import" in capsys.readouterr().err

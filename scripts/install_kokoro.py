"""Install + pre-download Kokoro, the free local TTS voice for Wallie-Kansai.

Kokoro is an optional, zero-cost voice that runs entirely on your machine — no
API key, 24 kHz output, and it sounds better than Piper. It is listed as an
optional dependency in ``requirements.txt``; this script installs the exact
versions the project expects and pre-downloads the model weights and the voice
you pick, so the first line of a live stream is never stalled by a HuggingFace
download.

Usage:
    python scripts/install_kokoro.py                      # English (US), af_heart
    python scripts/install_kokoro.py --install            # also pip-install deps
    python scripts/install_kokoro.py --lang p --voice pf_dora
    python scripts/install_kokoro.py --lang j --install   # Japanese (adds misaki[ja])
    python scripts/install_kokoro.py --play               # verify + speak it out loud

With ``--play`` the script does not stop at "the model loads": it speaks one line
through the SAME provider and player a live session builds (tts.kokoro.KokoroTTS
into audio.player.AudioPlayer) and out the output device the ACTIVE profile saved.
A renamed, unplugged or never-picked device is caught here instead of on stream.

What "compatible with this project" means:
  * kokoro     >= 0.9.4   (0.9.2 works for English only; 0.9.4+ ships the
                           multilingual voice packs the Voice picker lists)
  * soundfile  >= 0.12.1
  * PyTorch    (CPU is plenty; CUDA / Apple MPS is used automatically if present)

Run it inside the project virtualenv, e.g. ``.venv\\Scripts\\python.exe``.
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from importlib import metadata
from pathlib import Path

# The status glyphs below (→ ✓ ✗ ·) crash a cp1252 Windows console, so force
# UTF-8 on stdout/stderr where the interpreter allows it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# --- versions / language metadata (kept in sync with dashboard/static/app.js) ---
_MIN_KOKORO = "0.9.4"
_MIN_SOUNDFILE = "0.12.1"
_MODEL_REPO = "hexgrad/Kokoro-82M"
# kokoro's multilingual releases (>= 0.9.4) declare Requires-Python >=3.10,<3.13.
_MIN_PY = (3, 10)
_MAX_PY = (3, 13)   # exclusive


def _check_python(version_info: tuple[int, ...] | None = None) -> bool:
    """Kokoro >= 0.9.4 has no wheels for Python 3.13+. Fail early with a fix
    instead of letting pip print 'No matching distribution found'."""
    vi = version_info or sys.version_info
    if _MIN_PY <= (vi[0], vi[1]) < _MAX_PY:
        return True
    print()
    print(
        "✗ Kokoro's multilingual releases need Python 3.10-3.12, but this "
        f"interpreter is {vi[0]}.{vi[1]}.{vi[2]}."
    )
    print("  kokoro>=0.9.4 (required for Portuguese and other non-English voices)")
    print("  is published only for Python >=3.10,<3.13, so pip cannot install it here.")
    print()
    print("  To use Kokoro:")
    print("    1. Install Python 3.12  (https://www.python.org/downloads/)")
    print("    2. Recreate this project's virtualenv with it:")
    print("         py -3.12 -m venv .venv")
    print("    3. Reinstall the project requirements, then re-run:")
    print("         python scripts/install_kokoro.py --install --lang p")
    return False

# lang_code -> (human label, default voice). Mirrors KOKORO_LANGS / KOKORO_VOICES.
_LANGS: dict[str, tuple[str, str]] = {
    "a": ("English (US)", "af_heart"),
    "b": ("English (UK)", "bf_alice"),
    "p": ("Portuguese (Brazil)", "pf_dora"),
    "e": ("Spanish", "ef_dora"),
    "f": ("French", "ff_siwis"),
    "h": ("Hindi", "hf_alpha"),
    "i": ("Italian", "if_sara"),
    "j": ("Japanese", "jf_alpha"),
    "z": ("Chinese (Mandarin)", "zf_xiaobei"),
}

# Extra packages some language fronts need on top of kokoro + soundfile.
_LANG_EXTRAS: dict[str, str] = {"j": "misaki[ja]", "z": "misaki[zh]"}

# A warm-up sentence per language — forcing one synthesis downloads the voice
# .pt file into the HuggingFace cache. Falls back to English.
_WARM_TEXT: dict[str, str] = {
    "a": "Hello, this is a test of the Kokoro voice.",
    "b": "Hello, this is a test of the Kokoro voice.",
    "p": "Olá! Esta é uma amostra da voz local do Kokoro.",
    "e": "Hola, esta es una prueba de la voz local.",
    "f": "Bonjour, ceci est un test de la voix locale.",
    "h": "नमस्ते, यह एक परीक्षण है।",
    "i": "Ciao, questa è una prova della voce locale.",
    "j": "こんにちは、これはテストです。",
    "z": "你好，这是一个测试。",
}


def _installed_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _pip_requirements(lang: str) -> list[str]:
    reqs = [f"kokoro>={_MIN_KOKORO}", f"soundfile>={_MIN_SOUNDFILE}"]
    extra = _LANG_EXTRAS.get(lang)
    if extra:
        reqs.append(extra)
    return reqs


def _install(lang: str) -> bool:
    reqs = _pip_requirements(lang)
    print(f"→ pip install {' '.join(reqs)}")
    print("  (this pulls in PyTorch, a few hundred MB on first install)")
    cmd = [sys.executable, "-m", "pip", "install", *reqs]
    try:
        rc = subprocess.call(cmd)
    except OSError as e:
        print(f"error: could not run pip: {e}", file=sys.stderr)
        return False
    if rc != 0:
        print(f"error: pip exited with {rc}", file=sys.stderr)
        return False
    return True


def _check_requirements(lang: str) -> list[str]:
    """Return the *already satisfied* packages, warning about missing ones."""
    missing = []
    if _installed_version("kokoro") is None:
        missing.append("kokoro")
    if _installed_version("soundfile") is None:
        missing.append("soundfile")
    return missing


def _describe_env() -> str:
    torch_ver = _installed_version("torch")
    if not torch_ver:
        return "PyTorch: not installed yet"
    try:
        import torch  # noqa: PLC0415 — optional, only for the device hint

        if torch.cuda.is_available():
            dev = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "Apple MPS"
        else:
            dev = "CPU"
        return f"PyTorch {torch_ver} · device: {dev}"
    except Exception:  # pragma: no cover - defensive
        return f"PyTorch {torch_ver}"


def _warm(lang: str, voice: str, device: str | None) -> int:
    """Instantiate the pipeline and synthesise one line to pull the weights."""
    try:
        from kokoro import KPipeline  # type: ignore
    except ImportError as e:
        print(f"error: kokoro import failed: {e}", file=sys.stderr)
        print("       run this script with --install first.", file=sys.stderr)
        return -1

    kwargs: dict = {"lang_code": lang, "repo_id": _MODEL_REPO}
    if device:
        kwargs["device"] = device
    print(f"→ loading KPipeline(lang_code={lang!r}, repo_id={_MODEL_REPO!r})")
    try:
        pipe = KPipeline(**kwargs)
    except Exception as e:
        print(f"error: could not initialise the pipeline: {e}", file=sys.stderr)
        return -1

    text = _WARM_TEXT.get(lang, _WARM_TEXT["a"])
    print(f"→ synthesising a warm-up line with voice {voice!r} (downloads model + voice)")
    try:
        chunks = sum(1 for _ in pipe(text, voice=voice, speed=1.0))
    except Exception as e:
        print(f"error: synthesis failed: {e}", file=sys.stderr)
        if lang in _LANG_EXTRAS:
            print(
                f"       language {lang!r} may need {_LANG_EXTRAS[lang]} + a system "
                "espeak-ng install.",
                file=sys.stderr,
            )
        return -1
    if chunks == 0:
        print("error: pipeline produced no audio.", file=sys.stderr)
        return -1
    return chunks


def _profile_output_device(root: Path | None = None) -> tuple[str, str]:
    """(device spec, human source) of the ACTIVE profile's TTS output.

    Reads the profile file directly on purpose: ``config.load_profile()`` CREATES
    the profile when it is missing, and a verification step must not have that
    side effect (it would pre-empt the app's own first-run setup).

    ``root`` defaults to the project root (tests pass a scratch directory). It
    NEVER raises and never writes: anything unreadable degrades to the system
    default, because a broken profile must not break the setup's voice check.
    """
    root = root or Path(__file__).resolve().parent.parent
    try:
        import json
        import yaml  # type: ignore

        name = "default"
        state = root / ".wallie_state.json"   # config.STATE_FILE
        if state.exists():
            name = (json.loads(state.read_text(encoding="utf-8")) or {}).get("active") or "default"
        # Mirrors config._profile_path's sanitisation, so a name the app happily
        # loads can't make this reader look at a different file (there is a drift
        # test in tests/test_install_kokoro.py guarding the whole scheme).
        safe = "".join(c for c in name if c.isalnum() or c in "-_") or "default"
        path = root / "profiles" / f"{safe}.yaml"   # config.PROFILES_DIR
        if not path.exists():
            return "", f"the system default device (no profile {safe!r} yet)"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        device = str((data.get("tts") or {}).get("output_device") or "").strip()
        if device:
            return device, f"saved by profile {safe!r}"
        return "", f"the system default device (profile {safe!r} leaves it empty)"
    except Exception as e:  # noqa: BLE001 — a broken profile must not block the check
        return "", f"the system default device (could not read the profile: {e})"


async def _speak_and_wait(tts, text: str, device: str, player_cls) -> bytes:
    """Synthesise through the provider and play it on the profile's device."""
    pcm = b"".join([chunk async for chunk in tts.synthesize(text)])
    if not pcm:
        return b""
    player = player_cls(sample_rate=tts.sample_rate, channels=tts.channels,
                        device=device or None)
    try:
        player.start()
        await player.write(pcm)
        await player.wait_drained()
    finally:
        player.close()
    return pcm


def _play_verify(lang: str, voice: str) -> int:
    """Speak one line through the project's own stack, out the configured device.

    This is the end-to-end check a warm-up cannot make: the provider class and the
    player are the exact ones a live session builds, and the device is the one the
    profile saved — so "the voice works" and "the voice comes out where you expect"
    are both proven before the first line of a session ever needs them.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))   # the script lives in scripts/, the app at the root

    device, source = _profile_output_device()
    print(f"→ speaking one line through {device!r} ({source})" if device
          else f"→ speaking one line through {source}")
    try:
        from audio.player import AudioPlayer, check_output_device  # type: ignore
        from tts.kokoro import KokoroTTS  # type: ignore
    except Exception as e:  # noqa: BLE001 — report, never traceback at the user
        print(f"error: could not import the project's TTS/audio stack: {e}", file=sys.stderr)
        return -1

    text = _WARM_TEXT.get(lang, _WARM_TEXT["a"])
    try:
        tts = KokoroTTS(voice=voice, lang_code=lang, speed=1.0)
        pcm = asyncio.run(_speak_and_wait(tts, text, device, AudioPlayer))
    except Exception as e:  # noqa: BLE001
        where = repr(device) if device else "the system default device"
        print(f"error: could not speak through {where}: {e}", file=sys.stderr)
        if device and not check_output_device(device).get("exists"):
            print(f"       the saved output device {device!r} is not in the system "
                  "any more — re-pick it on the dashboard's Voice page and save.",
                  file=sys.stderr)
        return -1
    if not pcm:
        print("error: the provider produced no audio.", file=sys.stderr)
        return -1

    seconds = len(pcm) / max(1, tts.sample_rate * tts.channels * 2)
    played_on = check_output_device(device).get("name") or "the system default device"
    print(f"✓ spoke {voice!r} ({seconds:.1f}s of audio) through {played_on}")
    return 0


def _print_panel(lang: str, voice: str) -> None:
    label = _LANGS.get(lang, ("?", voice))[0]
    print()
    print("✓ Kokoro is ready.")
    print("  In the dashboard → Voice tab:")
    print("    Provider : Kokoro (local · free)")
    print(f"    Language : {label}   (lang_code {lang!r})")
    print(f"    Voice    : {voice}")
    print("    Speed    : 1.0  (tune 0.5–2.0)")
    print()
    print("  Then press ▶ Play in the Voice test strip to hear it live.")
    print("  Voices follow the language — click “Language” to see the rest of the list.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="install_kokoro",
        description="Install and pre-download the free local Kokoro TTS voice.",
    )
    ap.add_argument("--install", action="store_true",
                    help="pip-install kokoro + soundfile (+ language extras) first")
    ap.add_argument("--lang", default="a", choices=sorted(_LANGS),
                    help="Kokoro language code (default: a = English US)")
    ap.add_argument("--voice", default=None,
                    help="voice id to download (default: the language's first voice)")
    ap.add_argument("--device", default=None,
                    help="force a torch device (cpu / cuda / mps); auto by default")
    ap.add_argument("--play", action="store_true",
                    help="speak one line through the app's own provider/player and out "
                         "the active profile's saved output device, instead of only "
                         "warming the pipeline")
    args = ap.parse_args(argv)

    lang = args.lang
    label, default_voice = _LANGS[lang]
    voice = args.voice or default_voice

    print(f"Kokoro installer — language: {label} ({lang}) · voice: {voice}")
    print(f"  {_describe_env()}")

    # The Python range only blocks us when we actually have to INSTALL kokoro;
    # an already-installed copy can still be warmed/verified on any interpreter.
    if _installed_version("kokoro") is None and not _check_python():
        return 3

    if args.install:
        if not _install(lang):
            return 1
    else:
        missing = _check_requirements(lang)
        if missing:
            print()
            print(f"✗ Missing package(s): {', '.join(missing)}")
            print("  Re-run with --install to fetch them, e.g.:")
            print(f"    python scripts/install_kokoro.py --lang {lang} --install")
            return 2

    print()
    if args.play:
        # The app-path check warms and pre-downloads too (KokoroTTS loads the model,
        # synthesising fetches the voice), so it *replaces* the warm-up rather than
        # paying for a second pipeline load.
        if _play_verify(lang, voice) < 0:
            return 1
    elif _warm(lang, voice, args.device) < 0:
        return 1

    _print_panel(lang, voice)
    print()
    print(f"  Model cache: {_MODEL_REPO} (HuggingFace cache, reused offline afterwards).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

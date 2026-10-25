#!/usr/bin/env python3
"""Attach a cloud-trained Piper model to a Wallie profile.

The cloud script writes a zip with three files:

* ``<voice>.onnx``
* ``<voice>.onnx.json``
* ``train.json`` (clips, seconds, epochs, GPU, Piper version)

This helper refuses a model whose ``train.json`` says it was trained on fewer
than 60 s of speech — that export is noise, not a voice — then copies the pair
into the Piper voices folder, registers it in the profile's library, and
optionally makes it the active voice.

Usage::

    python scripts/attach_cloud_voice.py luna-output.zip --profile luna --activate
    python scripts/attach_cloud_voice.py extracted-folder/ --profile luna
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(message: str, code: int = 2) -> None:
    print(f"✗ {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    """Extract a user-supplied zip without allowing paths outside ``dest``."""
    root = dest.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            fail(f"the zip contains an unsafe path: {member.filename!r}")
    archive.extractall(root)


def load_train_json(root: Path) -> tuple[dict, Path]:
    path = next(iter(sorted(root.rglob("train.json"))), None)
    if path is None:
        fail(
            "train.json is missing from this archive. The cloud output must "
            "come from scripts/cloud/train_luna.py — it is what records the "
            "dataset size. An older zip without it cannot be checked."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read train.json: {exc}")
    if not isinstance(data, dict):
        fail("train.json is not a JSON object")
    return data, path


def pick_model(root: Path, voice_name: str) -> Path:
    candidates = sorted(root.rglob("*.onnx"))
    if not candidates:
        fail("no .onnx file found in the archive")
    if voice_name:
        named = [p for p in candidates if p.stem == voice_name]
        if named:
            return named[0]
    if len(candidates) > 1:
        fail(
            "the archive has more than one .onnx and none matches "
            f"{voice_name or 'the recorded voice'} — use --name to pick one"
        )
    return candidates[0]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="cloud output zip, or an extracted folder")
    parser.add_argument("--profile", default="luna",
                        help="Wallie profile whose library receives the voice (default: luna)")
    parser.add_argument("--name", default="",
                        help="override the Piper voice name (must match <lang_REGION>-<voice>-<quality>)")
    parser.add_argument("--display-name", default="",
                        help="label shown in the Voice Lab (default: from train.json)")
    parser.add_argument("--activate", action="store_true",
                        help="also point the profile's active Piper voice at this model")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        fail(f"not found: {source}")

    temp: Path | None = None
    try:
        if source.is_dir():
            root = source
        elif source.is_file() and source.suffix.lower() == ".zip":
            temp = Path(tempfile.mkdtemp(prefix="wallie-cloud-voice-"))
            with zipfile.ZipFile(source) as archive:
                safe_extract(archive, temp)
            root = temp
        else:
            fail("pass the cloud output .zip or the extracted folder")

        train, train_path = load_train_json(root)
        voice_name = (
            args.name.strip()
            or str(train.get("voice_name") or "").strip()
        )
        onnx = pick_model(root, voice_name)
        voice_name = voice_name or onnx.stem
        config = onnx.parent / f"{onnx.name}.json"
        if not config.is_file():
            fail(
                f"{config.name} is missing — Piper loads the .onnx and its "
                ".onnx.json as a pair"
            )

        seconds = float(train.get("seconds") or 0.0)
        clips = int(train.get("clips") or 0)
        if seconds <= 0:
            fail(
                "train.json has no measured dataset duration — refusing to "
                "attach a model that cannot be checked"
            )
        if seconds < 60.0:
            fail(
                f"this model was trained on only {seconds:.1f}s of speech "
                f"({clips} clip(s)). That is below the 60 s floor and the "
                "export is noise, not a voice. Add more audio, train again on "
                "the cloud, and attach that zip."
            )
        if clips and clips < 3:
            fail(f"train.json reports only {clips} clip(s) — too few to train")

        print(f"archive : {source}")
        print(f"voice   : {voice_name} · {clips} clip(s) · {seconds:.1f}s")
        print(f"model   : {onnx.name} · {config.name}")

        try:
            from tts import voice_port
        except Exception as exc:  # noqa: BLE001
            fail(f"cannot import the local voice code: {exc}")

        try:
            result = voice_port.attach(
                profile=args.profile,
                target="piper",
                files=[
                    (onnx.name, onnx.read_bytes()),
                    (config.name, config.read_bytes()),
                ],
                name=voice_name,
                display_name=(
                    args.display_name.strip()
                    or str(train.get("display_name") or "").strip()
                    or "luna"
                ),
                source_provider=str(train.get("provider") or "cloud"),
            )
        except Exception as exc:  # noqa: BLE001 - show the library's own reason
            fail(str(exc), 1)

        print(f"\n✓ attached {result['name']} -> {result['path']}")
        for path in result.get("files") or []:
            print(f"  {path}")

        if args.activate:
            try:
                from config import load_profile, save_profile
            except Exception as exc:  # noqa: BLE001
                fail(f"attached, but cannot load the profile to activate it: {exc}", 1)
            cfg = load_profile(args.profile)
            cfg.tts = cfg.tts.model_copy(update={
                "provider": "piper",
                "voice_id": str(result["path"]),
                "piper_model_path": str(result["path"]),
            })
            save_profile(cfg, args.profile)
            print(f"✓ profile {args.profile!r} now uses {result['path']}")
            print("  restart the dashboard to drop its old in-memory config")
        else:
            print(
                "  saved in the library; select it in the Voice Lab or pass "
                "--activate to switch the profile now"
            )
        print(f"\ntrain.json: {train_path.name} "
              f"({train.get('epochs')} epochs, {train.get('gpu') or 'GPU'})")
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()

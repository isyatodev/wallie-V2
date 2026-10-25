#!/usr/bin/env python3
"""Build the portable cloud-training bundle for a ported Piper package.

The bundle contains only what a fresh Colab/Kaggle runtime needs:

* ``dataset/`` + ``manifest.json`` from the package,
* ``train_luna.py`` (standalone trainer/exporter),
* ``monotonic_core.py`` (the pure-Python alignment core piper's wheel omits),
* ``luna_cloud.ipynb`` + ``README.md`` + ``requirements.txt``.

Nothing from ``train/`` (checkpoints, the old noisy .onnx) is copied. The
bundle is a snapshot: after importing more audio into the package, run this
script again.

Usage::

    python scripts/build_luna_cloud_bundle.py
    python scripts/build_luna_cloud_bundle.py --package voices/ported/other-piper
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOUD = Path(__file__).resolve().parent / "cloud"
sys.path.insert(0, str(CLOUD))
import train_luna  # noqa: E402  (the standalone bundle script)

REQUIREMENTS = """\
piper-tts[train]==1.8.0
onnxscript
"""


def human(n: float) -> str:
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="voices/ported/luna-piper",
                        help="ported Piper package with dataset/ + manifest.json")
    parser.add_argument("--out", default="dist/luna-cloud",
                        help="staging folder for the bundle")
    parser.add_argument("--zip", default="dist/luna-cloud.zip",
                        help="output zip (upload this to Colab/Kaggle)")
    args = parser.parse_args()

    package = Path(args.package).expanduser()
    if not package.is_absolute():
        package = (ROOT / package).resolve()
    if not package.is_dir():
        raise SystemExit(f"package not found: {package}")
    if not (package / "dataset" / "metadata.csv").is_file():
        raise SystemExit(f"dataset/metadata.csv not found in {package}")

    manifest = train_luna.read_manifest(package)
    rows, seconds, missing = train_luna.dataset_stats(package)
    target = manifest.get("target") or {}
    voice = str(target.get("name_hint") or "") or train_luna.DEFAULT_VOICE
    lang = manifest.get("lang") or {}

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    zip_path = Path(args.zip).expanduser()
    if not zip_path.is_absolute():
        zip_path = (ROOT / zip_path).resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Refuse an --out that could wipe something we do not own. A previous
    # bundle carries BUNDLE.json; anything else is user data.
    if out == ROOT or ROOT not in out.parents:
        raise SystemExit("--out must be a folder inside the project")
    if out == package or package in out.parents:
        raise SystemExit("--out cannot be inside the training package")
    if out.exists() and not (out / "BUNDLE.json").is_file():
        raise SystemExit(
            f"{out} already exists and is not a previous bundle — "
            "choose another --out or delete it yourself"
        )

    # Build in a sibling and swap at the end, so a failed build never leaves a
    # half-written bundle that looks usable.
    stage = out.with_name(out.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        shutil.copytree(package / "dataset", stage / "dataset")
        shutil.copy2(package / "manifest.json", stage / "manifest.json")
        shutil.copy2(CLOUD / "train_luna.py", stage / "train_luna.py")
        shutil.copy2(CLOUD / "luna_cloud.ipynb", stage / "luna_cloud.ipynb")
        shutil.copy2(CLOUD / "README.md", stage / "README.md")
        shutil.copy2(ROOT / "tts" / "monotonic_core.py", stage / "monotonic_core.py")
        (stage / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
        meta = {
            "format": "wallie.cloud.bundle",
            "version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "package": str(package),
            "voice_name": voice,
            "display_name": str(manifest.get("name") or "luna"),
            "provider": str((manifest.get("source") or {}).get("provider") or "fish"),
            "espeak": str(lang.get("espeak") or "pt-br"),
            "sample_rate": int(target.get("sample_rate") or 22050),
            "clips": len(rows),
            "seconds": seconds,
            "trainable": seconds >= train_luna.MIN_SECONDS,
        }
        (stage / "BUNDLE.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if out.exists():
            shutil.rmtree(out)
        stage.rename(out)

        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    # Zip entries always use '/', even when the bundle is
                    # built on Windows; a backslash-arcname breaks unzip on
                    # Colab/Linux.
                    archive.write(
                        path,
                        "luna-cloud/" + path.relative_to(out).as_posix(),
                    )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    print(f"bundle : {out}")
    print(f"zip    : {zip_path} ({human(zip_path.stat().st_size)})")
    print(
        f"voice  : {voice} · {len(rows)} clip(s) · {seconds:.2f}s · "
        f"espeak {meta['espeak']} · {meta['sample_rate']} Hz"
    )
    if missing:
        print(f"⚠ missing WAV(s): {', '.join(missing[:6])}"
              + (" …" if len(missing) > 6 else ""))
    if seconds < train_luna.MIN_SECONDS:
        print(
            f"\n⚠ NOT TRAINABLE YET — {seconds:.1f}s is below the "
            f"{int(train_luna.MIN_SECONDS)}s floor. The cloud script and the "
            "notebook will stop with the same message. Import more Luna audio "
            "into the package (Voice Lab → Convert to a local engine), then run "
            "this builder again."
        )
    elif seconds < train_luna.RECOMMENDED_SECONDS:
        print(
            f"\n⚠ trainable, but below the recommended "
            f"{int(train_luna.RECOMMENDED_SECONDS)}s; the voice may sound rough."
        )
    else:
        print("\n✓ ready to train on a cloud GPU.")
    print(
        "\nNext:\n"
        f"  1. Upload {zip_path.name} to Colab/Kaggle.\n"
        "  2. Run luna_cloud.ipynb (or train_luna.py) with a GPU enabled.\n"
        "  3. Download the generated *-cloud.zip.\n"
        "  4. Attach it locally:\n"
        "     python scripts/attach_cloud_voice.py <downloaded.zip> "
        "--profile luna --activate"
    )


if __name__ == "__main__":
    main()

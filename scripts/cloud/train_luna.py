#!/usr/bin/env python3
"""Train a ported Piper package on a cloud GPU and export the .onnx pair.

This script is copied into the cloud bundle by
``scripts/build_luna_cloud_bundle.py``. It is intentionally standalone: the
bundle has no Wallie checkout, only the dataset, ``manifest.json``, this script
and the pure-Python alignment core.

Typical use on Colab/Kaggle::

    pip install "piper-tts[train]==1.8.0" onnxscript
    python train_luna.py --bundle luna-cloud --out luna-output

The same 60 s gate as the dashboard applies. A three-clip, eight-second package
is not trainable: Piper starts from random weights and would only learn noise.
Run ``--check`` to validate a bundle without touching a GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MIN_SECONDS = 60.0
RECOMMENDED_SECONDS = 600.0
EXPECTED_PIPER = "1.8."
DEFAULT_VOICE = "pt_BR-luna-medium"


def say(message: str) -> None:
    print(message, flush=True)


def fail(message: str, code: int = 2) -> None:
    print(f"\n✗ {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# ---------------------------------------------------------------- dataset

def read_manifest(bundle: Path) -> dict:
    path = bundle / "manifest.json"
    if not path.is_file():
        fail(f"manifest.json is missing from {bundle} — rebuild the bundle")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read manifest.json: {exc}")
    return {}


def wav_seconds(path: Path) -> float:
    """Duration from the WAV header; 0.0 when the file is missing/broken."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate) if rate else 0.0
    except (OSError, EOFError, wave.Error):
        return 0.0


def dataset_rows(bundle: Path) -> list[tuple[str, str]]:
    csv_path = bundle / "dataset" / "metadata.csv"
    if not csv_path.is_file():
        fail(f"dataset/metadata.csv is missing from {bundle}")
    rows: list[tuple[str, str]] = []
    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "|" not in line:
            continue
        name, text = line.split("|", 1)
        name = Path(name.strip()).name
        text = text.strip()
        if name and text:
            rows.append((name, text))
    return rows


def dataset_stats(bundle: Path) -> tuple[list[tuple[str, str]], float, list[str]]:
    rows = dataset_rows(bundle)
    audio = bundle / "dataset" / "wav"
    seconds = 0.0
    missing: list[str] = []
    for name, _text in rows:
        path = audio / name
        if not path.is_file():
            missing.append(name)
            continue
        seconds += wav_seconds(path)
    return rows, round(seconds, 2), missing


def normalize_csv(bundle: Path, rows: list[tuple[str, str]], dest: Path) -> None:
    """Piper joins the first column onto ``--data.audio_dir``.

    The package's CSV stores ``wav/0001.wav``; keeping that prefix would make
    the trainer look for ``wav/wav/0001.wav``.
    """
    dest.write_text(
        "\n".join(f"{name}|{text}" for name, text in rows) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------- piper fixes

def ensure_alignment_core(bundle: Path) -> None:
    """Install the pure-Python monotonic_align core the wheel forgot.

    piper-tts 1.8.0 ships ``monotonic_align/__init__.py`` but not the compiled
    ``core`` extension on any platform. Without this copy the first training
    step dies with ``ModuleNotFoundError``.
    """
    try:
        import piper
    except Exception as exc:  # noqa: BLE001 - report the real import failure
        fail(f"piper-tts is not importable: {exc}. Install 'piper-tts[train]==1.8.0'")
    base = Path(piper.__file__).parent / "train" / "vits" / "monotonic_align"
    slot = base / "monotonic_align"
    if not base.is_dir():
        fail(f"this piper install has no monotonic_align folder: {base}")
    if (
        (slot / "core.py").is_file()
        or list(slot.glob("core*.so"))
        or list(slot.glob("core*.pyd"))
    ):
        say("• monotonic_align core already present")
        return
    source = bundle / "monotonic_core.py"
    if not source.is_file():
        fail("the bundle is missing monotonic_core.py — rebuild it")
    try:
        slot.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, slot / "core.py")
    except OSError as exc:
        fail(f"could not install the alignment core in {slot}: {exc}")
    say("• installed the pure-Python monotonic_align core")


def torch_version() -> tuple[int, int]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return (0, 0)
    base = torch.__version__.split("+", 1)[0]
    parts = base.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


def patch_callbacks() -> None:
    """Drop the optional ``val_mos`` checkpoint callback.

    piper's CLI always adds a second ModelCheckpoint that monitors ``val_mos``.
    That metric is only logged when the UTMOS predictor runs over a test split,
    and this trainer deliberately uses ``--data.num_test_examples 0`` — so once
    validation runs, Lightning raises ``ModelCheckpoint(monitor='val_mos') could
    not find the monitored key`` at the end of the first epoch. The val_mel
    checkpoint is the one that matters; remove the optional one from the
    installed CLI before training.
    """
    try:
        import piper
    except Exception:  # noqa: BLE001
        return
    path = Path(piper.__file__).parent / "train" / "__main__.py"
    if not path.is_file():
        fail(f"piper's train/__main__.py is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if 'monitor="val_mos"' not in text:
        say("• no val_mos checkpoint callback to remove")
        return
    patched, count = re.subn(
        r"\n    ModelCheckpoint\(\n        monitor=\"val_mos\",.*?\n    \),?\n",
        "\n",
        text,
        count=1,
        flags=re.S,
    )
    if not count:
        fail(
            "could not remove the val_mos checkpoint callback from piper's "
            "train/__main__.py — this piper build needs a manual look"
        )
    try:
        path.write_text(patched, encoding="utf-8")
    except OSError as exc:
        fail(f"could not patch train/__main__.py: {exc}")
    say("• removed the val_mos checkpoint callback (no test split to score it)")


def patch_export() -> None:
    """Pass ``dynamo=False`` on torch >= 2.9.

    piper's exporter passes ``dynamic_axes`` at opset 15 but never sets
    ``dynamo``. torch 2.9 made the dynamo exporter the default, and it cannot
    export this model. The dashboard patches the same line locally.
    """
    try:
        import piper
    except Exception:  # noqa: BLE001
        return
    path = Path(piper.__file__).parent / "train" / "export_onnx.py"
    if not path.is_file():
        fail(f"export_onnx.py is missing from this piper install: {path}")
    text = path.read_text(encoding="utf-8")
    if "dynamo=" in text:
        say("• export_onnx.py already passes dynamo=")
        return
    if torch_version() < (2, 9):
        say("• torch < 2.9 — the legacy ONNX exporter is already the default")
        return
    patched, count = re.subn(
        r"torch\.onnx\.export\(\s*\r?\n",
        "torch.onnx.export(\n"
        "        dynamo=False,  # torch >= 2.9 defaults to dynamo, which cannot "
        "export this model\n",
        text,
        count=1,
    )
    if not count:
        fail(
            "could not patch export_onnx.py: torch.onnx.export( was not found — "
            "this piper build needs a manual look"
        )
    try:
        path.write_text(patched, encoding="utf-8")
    except OSError as exc:
        fail(f"could not patch export_onnx.py: {exc}")
    say("• patched export_onnx.py with dynamo=False")


# ---------------------------------------------------------------- training

def run(cmd: list[str], cwd: Path | None = None) -> None:
    say("\n$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except subprocess.CalledProcessError as exc:
        fail(
            f"command failed (exit {exc.returncode}) — the log above has the "
            "trainer's reason",
            1,
        )


def find_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return None
    for path in reversed(ckpts):
        if "last" in path.name.lower():
            return path
    return ckpts[-1]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - best effort on older consoles
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent),
                        help="folder with dataset/, manifest.json and monotonic_core.py")
    parser.add_argument("--out", default="luna-cloud-output",
                        help="where the run, .onnx and download zip are written")
    parser.add_argument("--epochs", type=int, default=150,
                        help="training epochs (150 is a sane cloud first run)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="requested batch (clamped to the train split)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers (Linux only; 0 disables them)")
    parser.add_argument("--voice-name", default="",
                        help=f"Piper voice name (default: {DEFAULT_VOICE})")
    parser.add_argument("--check", action="store_true",
                        help="validate the bundle and exit without training")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="smoke-test the pipeline on CPU (slow; never for a real run)")
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    if not bundle.is_dir():
        fail(f"bundle folder not found: {bundle}")
    manifest = read_manifest(bundle)
    target = manifest.get("target") or {}
    lang = manifest.get("lang") or {}
    voice = (
        args.voice_name.strip()
        or str(target.get("name_hint") or "")
        or DEFAULT_VOICE
    )
    espeak = str(lang.get("espeak") or "pt-br")
    sample_rate = int(target.get("sample_rate") or 22050)
    provider = str((manifest.get("source") or {}).get("provider") or "fish")
    display_name = str(manifest.get("name") or "luna")

    rows, seconds, missing = dataset_stats(bundle)
    say(f"bundle: {bundle}")
    say(
        f"voice: {voice} · {len(rows)} clip(s) · {seconds:.2f}s · "
        f"espeak {espeak} · {sample_rate} Hz"
    )
    if missing:
        say(f"⚠ missing WAV(s): {', '.join(missing[:6])}"
            + (" …" if len(missing) > 6 else ""))

    if len(rows) < 3:
        fail(
            f"a Piper dataset needs at least 3 clips — this bundle has {len(rows)}"
        )
    if seconds < MIN_SECONDS:
        fail(
            f"this dataset has {len(rows)} clip(s) / {seconds:.1f}s of speech. "
            f"A Piper voice needs at least {int(MIN_SECONDS)}s (10+ minutes "
            "recommended); below that the model only learns noise, which is the "
            "robotic audio an 8-second run exports. Add more audio to the "
            "package, rebuild the cloud bundle, and run this again."
        )
    if seconds < RECOMMENDED_SECONDS:
        say(
            f"⚠ {seconds:.1f}s is below the recommended "
            f"{int(RECOMMENDED_SECONDS)}s — the result may still sound rough."
        )
    if args.check:
        say("\n✓ bundle is trainable (no GPU or training was touched)")
        return

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        fail(f"torch is not importable: {exc}. Install 'piper-tts[train]==1.8.0'")
    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        fail(
            "no CUDA GPU is visible. On Colab: Runtime → Change runtime type → "
            "T4 GPU (or another GPU), then Runtime → Restart and run again."
        )
    if not cuda:
        say(
            "⚠ --allow-cpu: this is a smoke test of the pipeline, not a real "
            "training run. A CPU run takes far longer and is not the point here."
        )
    try:
        gpu = torch.cuda.get_device_name(0) if cuda else "CPU (smoke test)"
    except Exception:  # noqa: BLE001
        gpu = "CUDA device"
    accelerator = "gpu" if cuda else "cpu"
    say(f"device: {gpu} · torch {torch.__version__}")

    try:
        import importlib.metadata as metadata
        piper_version = metadata.version("piper-tts")
    except Exception:  # noqa: BLE001
        piper_version = "unknown"
    say(f"piper-tts: {piper_version}")
    if not piper_version.startswith(EXPECTED_PIPER):
        say(
            f"⚠ this bundle was built for piper-tts {EXPECTED_PIPER}x; "
            f"{piper_version} may still work, but the argv is not guaranteed."
        )

    ensure_alignment_core(bundle)
    patch_callbacks()
    patch_export()

    clips = len(rows)
    # Same rule as the dashboard: hold out at least one clip, even on a tiny
    # package, so validation and ModelCheckpoint have a real metric.
    split = min(0.5, max(0.1, 1.5 / clips))
    held_out = int(clips * split)
    train_clips = max(1, clips - held_out)
    batch = max(1, min(args.batch_size, train_clips))
    say(f"split: {train_clips} train / {held_out} validation · batch {batch}")

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "train.csv"
    normalize_csv(bundle, rows, csv_path)
    config_path = run_dir / f"{voice}.json"

    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        "--data.voice_name", voice,
        "--data.csv_path", str(csv_path),
        "--data.audio_dir", str(bundle / "dataset" / "wav"),
        "--model.sample_rate", str(sample_rate),
        "--data.espeak_voice", espeak,
        "--data.cache_dir", str(run_dir / "cache"),
        "--data.config_path", str(config_path),
        "--data.batch_size", str(batch),
        "--data.validation_split", f"{split:.4f}",
        "--data.num_test_examples", "0",
        "--data.num_workers", str(max(0, args.workers)),
        "--trainer.accelerator", accelerator,
        "--trainer.devices", "1",
        "--trainer.max_epochs", str(max(1, args.epochs)),
        "--trainer.default_root_dir", str(run_dir),
    ]
    say(f"run folder: {run_dir}")
    run(cmd, cwd=run_dir)

    checkpoint = find_checkpoint(run_dir)
    if checkpoint is None:
        fail(
            "training finished but wrote no .ckpt — there is nothing to export. "
            f"Check the logs under {run_dir}",
            1,
        )

    onnx = out / f"{voice}.onnx"
    run(
        [
            sys.executable, "-m", "piper.train.export_onnx",
            "--checkpoint", str(checkpoint),
            "--output-file", str(onnx),
        ],
        cwd=run_dir,
    )
    if not onnx.is_file() or onnx.stat().st_size == 0:
        fail(f"the exporter did not write {onnx}", 1)

    config_source = config_path if config_path.is_file() else None
    if config_source is None:
        candidates = [
            p for p in run_dir.rglob("*.json")
            if p.name not in {"run.json"} and not p.name.endswith(".onnx.json")
        ]
        config_source = candidates[0] if candidates else None
    if config_source is None:
        fail(
            "training wrote no config JSON — the .onnx needs it as .onnx.json. "
            f"Look under {run_dir}",
            1,
        )
    config_dest = out / f"{onnx.name}.json"
    shutil.copyfile(config_source, config_dest)

    train_json = out / "train.json"
    payload = {
        "format": "wallie.cloud.train",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "voice_name": voice,
        "display_name": display_name,
        "provider": provider,
        "clips": clips,
        "seconds": seconds,
        "epochs": int(args.epochs),
        "batch_size": batch,
        "sample_rate": sample_rate,
        "espeak": espeak,
        "piper_version": piper_version,
        "torch": str(torch.__version__),
        "gpu": gpu,
        "checkpoint": checkpoint.name,
    }
    train_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    zip_path = out / f"{voice}-cloud.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(onnx, onnx.name)
        archive.write(config_dest, config_dest.name)
        archive.write(train_json, train_json.name)

    say("")
    say(f"✓ done — {zip_path}")
    say(f"  {onnx.name} ({onnx.stat().st_size / 1024 / 1024:.1f} MB)")
    say(f"  {config_dest.name}")
    say("  train.json")
    say(
        "\nDownload that zip and attach it on the machine running Wallie:\n"
        f"  python scripts/attach_cloud_voice.py {zip_path.name} "
        "--profile luna --activate"
    )


if __name__ == "__main__":
    main()

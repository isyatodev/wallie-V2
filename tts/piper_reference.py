"""Turn an inference ``.onnx`` back into training weights — a Piper warm start.

A Piper ``.onnx`` carries the generator's weights, not the Lightning checkpoint
the trainer wants: no discriminator, no optimizer state, and no posterior
encoder (``enc_q``, inference-only). This module reads the graph and rebuilds a
``state_dict`` that piper's **non-strict warm start** (``--model.warmstart_ckpt``)
can consume, so a short dataset can begin from a voice that already sounds like
one instead of from noise.

Two details make it work:

* ``torch.onnx`` names initializers after the torch parameters (``enc_p.…``,
  ``dec.…``), so most tensors map straight across — they only need the
  ``model_g.`` prefix a ``VitsModel`` uses.
* Parameters behind ``weight_norm`` are folded into an anonymous constant at
  export (``onnx::Conv_8168``), which loses the name. The Conv/ConvTranspose
  node consuming that constant still carries the module path
  (``/flow/flows.6/enc/in_layers.0/Conv``), and the name is recovered from
  there. Folding is reversible: ``g`` is the row norm of the folded weight and
  ``v`` is the weight divided by it, which reconstructs the exact tensor.

Run it in the **training interpreter** — ``piper-tts[train]`` installs both
torch and onnx, and a separate venv is where that interpreter lives::

    python tts/piper_reference.py --onnx voices/pt_BR-faber-medium.onnx \\
        --out /path/to/run/reference.ckpt

Nothing here imports the ``tts`` package, so the script runs from any working
directory with only torch + onnx available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Where the recovered tensors are hung in a VitsModel's state dict.
STATE_PREFIX = "model_g."

# Modules whose weights the exporter leaves behind as anonymous "onnx::…"
# constants and whose node paths are worth trusting. Anything else in the graph
# (attention convs, biases, …) arrives with its own name.
KNOWN_ROOTS = ("enc_p", "enc_q", "flow", "dec", "dp")

# Conv ops whose weight input may be such an anonymous constant.
_CONV_OPS = ("Conv", "ConvTranspose")

# Modules in piper's VITS that carry weight_norm (so the target state dict has
# ``weight_g``/``weight_v`` instead of ``weight``). The exporter dropped it from
# ``dec`` before exporting, so those convs arrive named; ``flow`` and ``enc_q``
# keep it, and their convs arrive folded. ``dp`` convs are plain.
WEIGHT_NORM_PREFIXES = ("flow.", "dec.resblocks.", "dec.ups.", "enc_q.enc.")

# ``torch.onnx`` appends the op name to the scope path of the node it created.
_NODE_SUFFIXES = (
    "Conv", "ConvTranspose", "Squeeze", "Unsqueeze", "Transpose", "Gather",
)


class ReferenceError(RuntimeError):
    """A failure worth showing to the user verbatim (not a Piper voice, …)."""


# ---------------------------------------------------------------------------
# Reading the graph
# ---------------------------------------------------------------------------

def _module_scope(node_name: str) -> str:
    """``/flow/flows.6/enc/in_layers.0/Conv`` -> ``flow.flows.6.enc.in_layers.0``."""
    parts = [p for p in (node_name or "").strip("/").split("/") if p]
    if parts and parts[-1] in _NODE_SUFFIXES:
        parts = parts[:-1]
    return ".".join(parts)


def _split_weight_norm(weight: Any) -> tuple[Any, Any]:
    """Undo ``weight_norm``: a per-output-row gain and the normalised weight.

    ``weight_norm`` computes ``w = g * v / ||v||``; setting ``g = ||w||`` and
    ``v = w / ||w||`` reproduces ``w`` exactly, whatever the shape of ``g`` in
    the target module (``(out,)``, ``(out, 1, 1)``, …) — the caller reshapes.
    """
    import torch  # noqa: PLC0415 — torch is only needed when converting

    flat = weight.reshape(weight.shape[0], -1)
    norm = torch.linalg.vector_norm(flat, dim=1).clamp_min(1e-12)
    gain = norm.reshape((weight.shape[0],) + (1,) * (weight.dim() - 1))
    return gain, weight / gain


def _ensure_piper_graph(graph: Any) -> None:
    """Refuse a graph that clearly is not a Piper voice, with a reason."""
    inputs = {i.name for i in graph.input}
    if not {"input", "scales"} <= inputs:
        raise ReferenceError(
            "this .onnx is not a Piper voice model (its inputs are "
            f"{sorted(inputs) or ['none']} — a voice takes 'input' and 'scales'); "
            "pick the .onnx Piper exported for a voice, not another engine's model"
        )
    names = [i.name for i in graph.initializer]
    if not any(n.startswith("enc_p.") for n in names) or not any(
        n.startswith("dec.") for n in names
    ):
        raise ReferenceError(
            "this .onnx has no Piper text encoder/decoder weights — it may be a "
            "different model that happens to take 'input'/'scales'"
        )


def collect_state_dict(onnx_path: str | Path, *, prefix: str = STATE_PREFIX) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    """Rebuild a ``model_g`` state dict (and a summary) from a Piper ``.onnx``."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover — only a broken train venv
        raise ReferenceError(
            "reading a Piper .onnx needs the 'onnx' package in this interpreter "
            "(piper-tts[train] installs it) — pip install onnx"
        ) from e
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover — a broken train venv
        raise ReferenceError(
            "reading a Piper .onnx needs torch and numpy in this interpreter "
            "(piper-tts[train] installs them)"
        ) from e

    path = Path(onnx_path)
    if not path.is_file():
        raise ReferenceError(f"reference not found: {path}")
    try:
        model = onnx.load(str(path))
    except Exception as e:  # noqa: BLE001 — a truncated/corrupt download
        raise ReferenceError(f"could not read {path.name}: {str(e)[:200]}") from e
    graph = model.graph
    _ensure_piper_graph(graph)

    arrays = {
        init.name: np.asarray(onnx.numpy_helper.to_array(init))
        for init in graph.initializer
        if not init.name.startswith("/")
    }
    state: dict[str, Any] = {}
    named = folded = 0

    def put(key: str, array: Any, *, fold: bool = False) -> bool:
        """Add one weight under ``key``, un-folding weight_norm when folded.

        The trainer's warm start copies by name AND shape, so the weight is
        offered in both forms and whichever the target module has is the one
        that lands; the spare key is skipped. ``fold`` is only set where a
        weight_norm module is certain, which keeps the checkpoint from doubling
        every convolution.
        """
        nonlocal named, folded
        tensor = torch.from_numpy(np.array(array, copy=True)).float()
        if tensor.numel() == 0 or tensor.dim() == 0:
            return False
        state[prefix + key] = tensor
        named += 1
        if fold and key.endswith(".weight") and tensor.dim() >= 2:
            gain, value = _split_weight_norm(tensor)
            stem = prefix + key[: -len(".weight")]
            state[stem + ".weight_g"] = gain
            state[stem + ".weight_v"] = value
            folded += 1
        return True

    # 1. Everything the exporter named after its torch parameter.
    for name, array in arrays.items():
        if name.startswith("onnx::") or "." not in name or name == "sid":
            continue
        put(name, array, fold=name.startswith(WEIGHT_NORM_PREFIXES))

    # 2. The folded weight_norm convolutions, named by the node using them.
    recovered = 0
    for node in graph.node:
        if node.op_type not in _CONV_OPS:
            continue
        for candidate in list(node.input)[:2]:
            weight = arrays.get(candidate)
            if weight is None or not candidate.startswith("onnx::"):
                continue
            scope = _module_scope(node.name)
            if scope and scope.split(".")[0] in KNOWN_ROOTS:
                # Anonymous constants are exactly the folded weight_norm ones.
                recovered += 1 if put(f"{scope}.weight", weight, fold=True) else 0
            break

    # 3. The phoneme embedding: a single-speaker export names it ``sid`` and
    #    gathers it by the phoneme ids.
    for node in graph.node:
        if node.op_type != "Gather" or not node.input:
            continue
        table = arrays.get(node.input[0])
        scope = _module_scope(node.name)
        if table is not None and scope.endswith("enc_p.emb") and table.ndim == 2:
            put(f"{scope}.weight", table)
            break

    if not state:
        raise ReferenceError(
            f"{path.name} carries no weights this trainer recognises — export the "
            "voice again with piper's export_onnx"
        )
    emb = state.get(f"{prefix}enc_p.emb.weight")
    summary = {
        "onnx": str(path),
        "name": path.name,
        "tensors": len(state),
        "named": named,
        "folded": folded,
        "recovered": recovered,
        "vocab": int(emb.shape[0]) if emb is not None else 0,
        "hidden": int(emb.shape[1]) if emb is not None and emb.dim() == 2 else 0,
    }
    return state, summary


def write_checkpoint(onnx_path: str | Path, out_path: str | Path,
                     *, prefix: str = STATE_PREFIX) -> dict[str, Any]:
    """Convert ``onnx_path`` into a warm-start checkpoint at ``out_path``."""
    import torch  # noqa: PLC0415

    state, summary = collect_state_dict(onnx_path, prefix=prefix)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    payload = {
        "format": "wallie.onnx-reference",
        "version": 1,
        "source": str(Path(onnx_path)),
        "stats": summary,
        "state_dict": state,
    }
    try:
        torch.save(payload, tmp)
        tmp.replace(out)
    except Exception as e:  # noqa: BLE001 — disk, permissions or serialization
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ReferenceError(f"could not write {out}: {str(e)[:200]}") from e
    summary["checkpoint"] = str(out)
    return summary


# ---------------------------------------------------------------------------
# CLI — run by the training supervisor with the training interpreter
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="piper_reference",
        description="Convert a Piper .onnx voice into a warm-start checkpoint.",
    )
    parser.add_argument("--onnx", required=True, help="the inference .onnx to reuse")
    parser.add_argument("--out", required=True, help="checkpoint to write (.ckpt)")
    args = parser.parse_args(argv)

    # Plain ASCII: this runs on a Windows console whose codepage may be cp1252.
    print(f"[reference] reading {Path(args.onnx).name} ...")
    try:
        summary = write_checkpoint(args.onnx, args.out)
    except ReferenceError as e:
        print(f"[reference] ERROR: {e}")
        return 1
    print(
        f"[reference] recovered {summary['tensors']} weight tensor(s) "
        f"({summary['recovered']} from weight_norm convolutions)"
    )
    if summary["vocab"]:
        print(f"[reference] phoneme vocabulary {summary['vocab']} / hidden {summary['hidden']}")
    print(f"[reference] wrote {summary['checkpoint']}")
    print(
        "[reference] the trainer copies every tensor whose name and shape match "
        "its own model (a different quality or language transfers only what "
        "lines up); the posterior encoder and the discriminators start fresh"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

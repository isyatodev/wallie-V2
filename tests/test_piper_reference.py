"""Piper training from an inference .onnx — the warm-start converter.

A ``.onnx`` is not a checkpoint: the exporter only writes the generator, folds
``weight_norm`` into anonymous constants and drops the posterior encoder. These
tests build a tiny graph with exactly those traits (named weights, a folded
convolution consumed by a named Conv node, the ``sid`` phoneme embedding) and
check that the converter rebuilds the same tensors, un-folds the weight_norm
pair exactly, and refuses a model that is not a Piper voice with a message the
panel can show verbatim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tts import piper_reference as pr  # noqa: E402


def _voice(path: Path) -> dict[str, np.ndarray]:
    """A minimal Piper-shaped graph with the three ways weights arrive."""
    rng = np.random.default_rng(7)
    arrays = {
        "emb": rng.normal(size=(6, 8)).astype("float32"),
        "proj": rng.normal(size=(8, 8)).astype("float32"),
        "pre": rng.normal(size=(8, 8, 1)).astype("float32"),
        "folded": rng.normal(size=(8, 8, 3)).astype("float32"),
    }
    initializers = [
        numpy_helper.from_array(arrays["emb"], "sid"),
        numpy_helper.from_array(arrays["proj"], "enc_p.proj.weight"),
        numpy_helper.from_array(arrays["pre"], "dec.conv_pre.weight"),
        # torch.onnx folds weight_norm into a constant with no module name…
        numpy_helper.from_array(arrays["folded"], "onnx::Conv_9001"),
    ]
    nodes = [
        # …the phoneme embedding travels as ``sid`` through this Gather.
        helper.make_node("Gather", ["sid", "input"], ["emb_out"], name="/enc_p/emb/Gather"),
        helper.make_node("Add", ["emb_out", "scales"], ["joined"], name="/enc_p/Add"),
        # …and the module path survives on the node that consumes the weight.
        helper.make_node(
            "Conv", ["joined", "onnx::Conv_9001"], ["conv_out"],
            name="/dec/resblocks.0/convs.0/Conv",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny-voice",
        [
            helper.make_tensor_value_info("input", TensorProto.INT64, ["n"]),
            helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3]),
        ],
        [helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, str(path))
    return arrays


def _unfold(state: dict, stem: str) -> np.ndarray:
    weight_g = state[stem + ".weight_g"]
    weight_v = state[stem + ".weight_v"]
    folded = weight_g * weight_v / weight_v.reshape(weight_v.shape[0], -1).norm(
        dim=1
    ).reshape(weight_g.shape)
    return folded.detach().numpy()


def test_named_and_folded_weights_are_both_recovered(tmp_path):
    arrays = _voice(tmp_path / "tiny.onnx")
    state, summary = pr.collect_state_dict(tmp_path / "tiny.onnx")

    np.testing.assert_allclose(state["model_g.enc_p.proj.weight"].numpy(), arrays["proj"], rtol=0, atol=0)
    np.testing.assert_allclose(state["model_g.dec.conv_pre.weight"].numpy(), arrays["pre"], rtol=0, atol=0)
    # The folded convolution comes back under its module name…
    assert "model_g.dec.resblocks.0.convs.0.weight" in state
    # …and as the weight_norm pair the target module actually holds.
    np.testing.assert_allclose(
        _unfold(state, "model_g.dec.resblocks.0.convs.0"), arrays["folded"], rtol=0, atol=1e-6
    )
    # A plain convolution must not grow a weight_norm pair.
    assert "model_g.enc_p.proj.weight_g" not in state
    # The single-speaker export names the phoneme embedding ``sid``.
    np.testing.assert_allclose(
        state["model_g.enc_p.emb.weight"].numpy(), arrays["emb"], rtol=0, atol=0
    )
    assert summary["recovered"] == 1
    assert summary["vocab"] == 6 and summary["hidden"] == 8


def test_write_checkpoint_persists_a_warm_start(tmp_path):
    _voice(tmp_path / "tiny.onnx")
    out = tmp_path / "run" / "reference.ckpt"

    summary = pr.write_checkpoint(tmp_path / "tiny.onnx", out)

    assert out.is_file()
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["format"] == "wallie.onnx-reference"
    assert payload["stats"]["tensors"] == summary["tensors"]
    assert "model_g.enc_p.proj.weight" in payload["state_dict"]
    assert payload["state_dict"]["model_g.enc_p.proj.weight"].dtype == torch.float32


def test_a_model_that_is_not_a_piper_voice_is_refused(tmp_path):
    path = tmp_path / "other.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="/y")],
        "other",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    onnx.save(helper.make_model(graph), str(path))

    with pytest.raises(pr.ReferenceError) as e:
        pr.collect_state_dict(path)
    assert "not a Piper voice" in str(e.value)


def test_a_truncated_onnx_names_the_file(tmp_path):
    path = tmp_path / "broken.onnx"
    path.write_bytes(b"not a protobuf")

    with pytest.raises(pr.ReferenceError) as e:
        pr.collect_state_dict(path)
    assert "broken.onnx" in str(e.value)


def test_the_cli_reports_a_bad_reference_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"nope")

    code = pr.main(["--onnx", str(bad), "--out", str(tmp_path / "ref.ckpt")])

    assert code == 1
    assert "[reference] ERROR:" in capsys.readouterr().out

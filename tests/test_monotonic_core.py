"""The numpy alignment core piper's wheel forgets to ship.

``tts.monotonic_core`` is dropped into the trainer as ``core.py`` (see
:func:`tts.piper_train.ensure_monotonic_align`), so it has to produce exactly
the alignment the compiled Cython core would: the best-scoring monotonic
path per item. Correctness is pinned against a brute-force enumeration of
every valid path on small random grids — the dynamic program may only ever
pick the optimum, and the traceback must land on a valid shape.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tts import monotonic_core as mc

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Reference — every valid path, enumerated
# ---------------------------------------------------------------------------


def brute_force_best(value: np.ndarray, t_y: int, t_x: int) -> float:
    """Best sum over all valid alignments, by enumeration.

    A valid alignment has one column per row, starts at column 0, only ever
    steps right by at most one between rows, and ends at the last column —
    exactly the family the traceback can produce.
    """
    best = -np.inf

    def walk(y: int, x: int, acc: float) -> None:
        nonlocal best
        if y == t_y:
            if x == t_x - 1:
                best = max(best, acc)
            return
        for nx in (x, x + 1):
            if nx < t_x:
                walk(y + 1, nx, acc + float(value[y, nx]))

    # Row 0 can only be column 0 (the origin); rows 1.. follow from there.
    walk(1, 0, float(value[0, 0]))
    return best


def align(value: np.ndarray, t_ys: list[int], t_xs: list[int]) -> np.ndarray:
    """Run the fallback over a batch and return the int32 paths."""
    b = value.shape[0]
    path = np.zeros_like(value, dtype=np.int32)
    mc.maximum_path_c(
        path, value.copy(), np.array(t_ys, dtype=np.int32), np.array(t_xs, dtype=np.int32)
    )
    return path


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_the_traceback_always_finds_the_best_path(seed):
    rng = np.random.default_rng(seed)
    t_y = int(rng.integers(1, 7))
    t_x = int(rng.integers(1, t_y + 1))
    value = rng.normal(size=(1, t_y, t_x)).astype(np.float32)
    path = align(value, [t_y], [t_x])

    item = path[0]
    assert set(np.unique(item)) <= {0, 1}
    assert np.array_equal(item.sum(axis=1), np.ones(t_y)), "one alignment per row"
    cols = np.argmax(item, axis=1)
    assert cols[0] == 0 and cols[-1] == t_x - 1
    assert np.all(np.diff(cols) >= 0), "monotonic — never steps back"
    assert np.all(np.diff(cols) <= 1), "at most one step per row"

    scored = float(value[0][item.astype(bool)].sum())
    assert scored == pytest.approx(brute_force_best(value[0], t_y, t_x), abs=1e-4)


def test_a_single_column_aligns_every_frame_to_it():
    path = align(np.array([[[-2.0], [9.0], [9.0]]], dtype=np.float32), [3], [1])
    assert np.array_equal(path[0], np.array([[1], [1], [1]]))


def test_the_diagonal_wins_when_scores_point_straight_down():
    # Score the band's own cells: the only path that can visit them all is
    # the forced one — no room to step right.
    t = 4
    value = np.full((1, t, t), -5.0, dtype=np.float32)
    for y in range(t):
        value[0, y, y] = 10.0
    assert np.array_equal(align(value, [t], [t])[0], np.eye(t, dtype=np.int32))


def test_item_lengths_are_respected():
    # Two items in one batch: the second is shorter, and the junk packed
    # beyond its lengths must not bleed into its alignment (or the first's).
    value = np.full((2, 4, 3), 1e8, dtype=np.float32)
    value[0, :4, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]
    )
    value[1, :2, :1] = np.array([[5.0], [6.0]])
    path = align(value, [4, 2], [3, 1])
    # The buffer stays full-size; only the first two rows/one column are live.
    expected = np.zeros((4, 3), dtype=np.int32)
    expected[:2, 0] = 1
    assert np.array_equal(path[1], expected)
    assert np.array_equal(path[0], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))


def test_empty_items_are_left_blank():
    path = align(np.zeros((1, 0, 0), dtype=np.float32), [0], [0])
    assert path.shape == (1, 0, 0)


# ---------------------------------------------------------------------------
# The file as the trainer imports it — copied into site-packages as core.py
# ---------------------------------------------------------------------------


def test_the_dropped_in_core_py_is_selfcontained_and_runs(tmp_path):
    target = tmp_path / "core.py"
    target.write_bytes((REPO / "tts" / "monotonic_core.py").read_bytes())

    spec = importlib.util.spec_from_file_location("core", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # must import with numpy alone

    value = np.array([[[3.0, 0.0], [0.0, 3.0]]], dtype=np.float32)
    path = np.zeros((1, 2, 2), dtype=np.int32)
    module.maximum_path_c(path, value, np.array([2], dtype=np.int32), np.array([2], dtype=np.int32))
    assert np.array_equal(path[0], np.eye(2, dtype=np.int32))

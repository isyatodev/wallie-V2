"""Monotonic alignment search (MAS) in pure numpy — the trainer's fallback core.

``piper-tts`` (<=1.8.0, every platform) installs ``train/vits/monotonic_align``
with only an ``__init__.py`` and a ``setup.py``: the compiled Cython core is
missing from both the wheels and the sdist, so the first training step dies
with ``ModuleNotFoundError: …monotonic_align.monotonic_align`` and no pip
reinstall can fix it.

This module implements the same published alignment search — the monotonic
dynamic program of VITS (Kim et al., 2021), a forward recurrence over a
monotonic band followed by a backtrace — as a self-contained numpy module
exposing the call signature the trainer expects. It is copied into the empty
core slot by :func:`tts.piper_train.ensure_monotonic_align` as ``core.py``;
a locally compiled ``core`` extension would still be found first, because
extension modules load before source files.

Layout of the arrays, as ``piper.train.vits.models`` calls them:

* ``values`` — ``[b, spec, text]`` float32 scores (one row per audio frame,
  one column per phoneme);
* ``paths`` — ``[b, spec, text]`` int32, written in place: exactly one ``1``
  per row, its column non-decreasing down the rows, so ``paths.sum(2)`` is
  the per-phoneme duration the model trains on;
* ``t_ys`` / ``t_xs`` — per-item valid lengths (spec frames, phonemes);
  ``spec >= phonemes`` always holds for real speech.
"""
from __future__ import annotations

import numpy as np

# Below any realistic score, but finite — the band constraints are enforced
# by poisoning the disallowed transitions with it (scores are O(1)).
MAX_NEG = -1e9


def _align_one(path: np.ndarray, value: np.ndarray, t_y: int, t_x: int) -> None:
    """Best monotonic path for one item, accumulated into ``value`` in place.

    ``value`` is ``[t_y, t_x]`` float32: each visited cell ends holding the
    best score of any valid path reaching it. ``path`` is a ``[t_y, t_x]``
    int32 buffer; the backtrace writes one ``1`` per row.
    """
    if t_y <= 0 or t_x <= 0:
        return
    for y in range(t_y):
        # Band: a cell may only be reached if the rows above could still
        # cover the remaining columns (lower edge) and the token index never
        # runs ahead of the frame that may attend to it (upper edge).
        lo = max(0, t_x + y - t_y)
        hi = min(t_x, y + 1)
        if y == 0:
            # Row 0's only in-band cell is (0, 0), reached from the virtual
            # origin at cost 0 — nothing to accumulate.
            continue
        if hi <= lo:
            continue
        prev = value[y - 1]
        xs = np.arange(lo, hi)
        # Continue straight down from (y-1, x) …
        cur = prev[xs].copy()
        cur[xs == y] = MAX_NEG  # … unless that would place token y before frame y.
        # Or step right from (y-1, x-1); column 0 has no left neighbour.
        left = prev[xs - 1].copy()
        left[xs == 0] = MAX_NEG
        value[y, lo:hi] += np.maximum(cur, left)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        # Walk down-left when the left predecessor scores higher; index == y
        # is the band edge and forces the step. y > 0 guards the reads — the
        # invariant keeps index inside the band, where row y-1 was updated.
        if index != 0 and y > 0 and (
            index == y or value[y - 1, index] < value[y - 1, index - 1]
        ):
            index -= 1


def maximum_path_c(
    path: np.ndarray, value: np.ndarray, t_ys: np.ndarray, t_xs: np.ndarray
) -> None:
    """In-place drop-in for the compiled core's ``maximum_path_c``."""
    for i in range(int(path.shape[0])):
        _align_one(path[i], value[i], int(t_ys[i]), int(t_xs[i]))

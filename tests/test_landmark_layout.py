"""Tests for pipeline/landmark_layout.py — the 66-point selection that keeps
extraction and the live code from drifting (risks.md #1).

The trick: give each source landmark a value equal to its own index, so after
selection we can assert *exactly which* dots were picked and in what order.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from landmark_layout import (
    select_66, LIPS_IDXS, HAND_IDXS, POSE_IDXS,
    N_POINTS, LIPS_SLICE, HAND_SLICE, POSE_SLICE,
)


def _indexed(n):
    """(n, 3) where row i is [i, i, i] — so a picked row reveals its source index."""
    return np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 3))


def test_layout_counts():
    assert len(LIPS_IDXS) == 40
    assert len(HAND_IDXS) == 21
    assert len(POSE_IDXS) == 5
    assert N_POINTS == 66
    # No duplicate lip indices (a duplicate would silently corrupt the block).
    assert len(set(LIPS_IDXS)) == 40


def test_selects_correct_dots_in_order():
    face = _indexed(478)
    hand = _indexed(21)
    pose = _indexed(33)

    out = select_66(face, hand, pose)

    assert out.shape == (66, 3)
    # Each selected row's value == its source index, in the fixed [lips,hand,pose] order.
    assert np.array_equal(out[LIPS_SLICE, 0], np.array(LIPS_IDXS, dtype=np.float32))
    assert np.array_equal(out[HAND_SLICE, 0], np.array(HAND_IDXS, dtype=np.float32))
    assert np.array_equal(out[POSE_SLICE, 0], np.array(POSE_IDXS, dtype=np.float32))


def test_missing_parts_become_zeros():
    face = _indexed(478)
    out = select_66(face, np.empty((0, 3)), np.empty((0, 3)))
    assert out.shape == (66, 3)
    assert np.all(out[HAND_SLICE] == 0)   # no hand detected -> zero block
    assert np.all(out[POSE_SLICE] == 0)   # no pose detected -> zero block
    assert np.any(out[LIPS_SLICE] != 0)   # lips still populated

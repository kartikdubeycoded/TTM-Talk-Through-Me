"""Tests for pipeline/word_features.py (Phase 5, T12).

A word sequence's per-frame features are:
  [0:63]  the 21 hand landmarks, normalized per frame (wrist at 0, lm9 unit)
  [63:66] wrist trajectory: raw wrist position minus its position in frame 0
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from word_features import sequence_features, resample


@pytest.fixture
def hand_seq():
    """A fake sequence: 12 frames of 21 hand landmarks, hand drifting right."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(21, 3)).astype(np.float32)
    frames = []
    for t in range(12):
        f = base.copy()
        f[:, 0] += 0.05 * t  # whole hand moves right over time
        frames.append(f)
    return np.stack(frames)  # (12, 21, 3)


def test_features_shape(hand_seq):
    out = sequence_features(hand_seq)
    assert out.shape == (12, 66)


def test_per_frame_normalization_invariants(hand_seq):
    out = sequence_features(hand_seq)
    lms = out[:, :63].reshape(-1, 21, 3)
    assert np.allclose(lms[:, 0, :], 0.0, atol=1e-6)           # wrist anchored
    assert np.allclose(np.linalg.norm(lms[:, 9, :], axis=1), 1.0, atol=1e-4)


def test_trajectory_captures_motion(hand_seq):
    out = sequence_features(hand_seq)
    traj = out[:, 63:66]
    assert np.allclose(traj[0], 0.0, atol=1e-6)        # starts at zero
    assert traj[-1, 0] == pytest.approx(0.55, abs=1e-4)  # drifted right
    # Shape features identical across frames (same handshape all along)
    assert np.allclose(out[0, :63], out[-1, :63], atol=1e-4)


def test_resample_to_fixed_length(hand_seq):
    feats = sequence_features(hand_seq)
    out = resample(feats, 30)
    assert out.shape == (30, 66)
    # endpoints preserved
    assert np.allclose(out[0], feats[0], atol=1e-5)
    assert np.allclose(out[-1], feats[-1], atol=1e-5)


def test_resample_downsamples_too(hand_seq):
    feats = sequence_features(hand_seq)
    out = resample(feats, 5)
    assert out.shape == (5, 66)

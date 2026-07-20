"""Tests for pipeline/word_features.py (Phase 5, T12).

A word sequence's per-frame features are:
  [0:63]  the 21 hand landmarks, normalized per frame (wrist at 0, lm9 unit)
  [63:66] wrist trajectory: raw wrist position minus its position in frame 0
  [66:68] wrist location: absolute wrist x,y in the camera frame
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from word_features import sequence_features, resample, whole_body_features

WHOLE_BODY_WIDTH = 81  # 68 hand block + 2 hand-rel-face + 10 pose-rel-face + 1 lip openness


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
    assert out.shape == (12, 68)


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


def test_location_captures_absolute_position(hand_seq):
    out = sequence_features(hand_seq)
    loc = out[:, 66:68]
    # y unchanged (hand only drifts in x), so absolute y is constant
    assert np.allclose(loc[:, 1], loc[0, 1], atol=1e-6)
    # absolute x drifts the same 0.55 as the trajectory, but does NOT start at 0
    assert loc[-1, 0] - loc[0, 0] == pytest.approx(0.55, abs=1e-4)
    assert loc[0, 0] != pytest.approx(0.0, abs=1e-3)  # absolute, not relative


def test_resample_to_fixed_length(hand_seq):
    feats = sequence_features(hand_seq)
    out = resample(feats, 30)
    assert out.shape == (30, 68)
    # endpoints preserved
    assert np.allclose(out[0], feats[0], atol=1e-5)
    assert np.allclose(out[-1], feats[-1], atol=1e-5)


def test_resample_downsamples_too(hand_seq):
    feats = sequence_features(hand_seq)
    out = resample(feats, 5)
    assert out.shape == (5, 68)


# --- whole-body features (Phase A / T23): hand + lips + pose ----------------

@pytest.fixture
def body_seq():
    """A fake 10-frame sequence with hand (21), lips (40) and pose (5)."""
    rng = np.random.default_rng(7)
    hand = rng.normal(0.5, 0.10, size=(10, 21, 3)).astype(np.float32)
    lips = rng.normal(0.5, 0.05, size=(10, 40, 3)).astype(np.float32)
    pose = rng.normal(0.5, 0.10, size=(10, 5, 3)).astype(np.float32)
    return hand, lips, pose


def test_whole_body_shape_and_reuses_hand_block(body_seq):
    hand, lips, pose = body_seq
    out = whole_body_features(hand, lips, pose)
    assert out.shape == (10, WHOLE_BODY_WIDTH)
    # the first 68 columns must be byte-for-byte the proven hand recipe
    assert np.allclose(out[:, :68], sequence_features(hand), atol=1e-6)


def test_whole_body_block_is_face_anchored_and_scale_invariant(body_seq):
    """Cols [68:81] anchor to the lips and scale by lip width, so shifting AND
    resizing the whole signer in frame must not change them (signer-invariance
    is the entire justification for the feature)."""
    hand, lips, pose = body_seq
    base = whole_body_features(hand, lips, pose)[:, 68:]

    a, b = 1.7, np.array([0.3, -0.2, 0.0], dtype=np.float32)  # scale + shift x,y
    moved = whole_body_features(hand * a + b, lips * a + b, pose * a + b)[:, 68:]
    assert np.allclose(base, moved, atol=1e-4)


def test_whole_body_missing_lips_frame_is_zeroed(body_seq):
    """A frame with no detected lips (all-zero, the dataset's missing convention)
    can't be face-anchored, so its whole-body block must be zero, not garbage."""
    hand, lips, pose = body_seq
    lips = lips.copy()
    lips[3] = 0.0
    out = whole_body_features(hand, lips, pose)
    assert np.allclose(out[3, 68:], 0.0, atol=1e-6)
    assert not np.allclose(out[0, 68:], 0.0, atol=1e-6)  # other frames still populated

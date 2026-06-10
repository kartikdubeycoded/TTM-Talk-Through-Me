"""Tests for pipeline/augment.py (Task 4).

Invariants the live pipeline guarantees, which augmented training data
must also satisfy after re-normalization:
  - wrist (landmark 0) at exactly (0, 0, 0)
  - landmark 9 at exactly unit distance from the wrist
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from augment import (
    mirror,
    rotate,
    jitter,
    renormalize,
    augment_training_set,
)


@pytest.fixture
def samples():
    """A small batch of valid normalized samples: wrist 0, landmark 9 unit."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(10, 21, 3)).astype(np.float32)
    x[:, 0, :] = 0.0
    norms = np.linalg.norm(x[:, 9, :], axis=1, keepdims=True)
    x = x / norms[:, None, :]
    return x.reshape(10, 63)


def lm(arr, i):
    """Landmark i of every sample as (N, 3)."""
    return arr.reshape(-1, 21, 3)[:, i, :]


def test_mirror_negates_x_only(samples):
    out = mirror(samples)
    reshaped_in = samples.reshape(-1, 21, 3)
    reshaped_out = out.reshape(-1, 21, 3)
    assert np.allclose(reshaped_out[:, :, 0], -reshaped_in[:, :, 0])
    assert np.allclose(reshaped_out[:, :, 1:], reshaped_in[:, :, 1:])


def test_mirror_twice_is_identity(samples):
    assert np.allclose(mirror(mirror(samples)), samples)


def test_rotation_preserves_distances(samples):
    rng = np.random.default_rng(0)
    out = rotate(samples, max_degrees=15, rng=rng)
    # Rigid rotation: distance of every landmark from the wrist is unchanged
    assert np.allclose(
        np.linalg.norm(out.reshape(-1, 21, 3), axis=2),
        np.linalg.norm(samples.reshape(-1, 21, 3), axis=2),
        atol=1e-5,
    )
    # ... but the points actually moved
    assert not np.allclose(out, samples)


def test_jitter_changes_data_but_stays_small(samples):
    rng = np.random.default_rng(0)
    out = jitter(samples, sigma=0.02, rng=rng)
    assert not np.allclose(out, samples)
    assert np.max(np.abs(out - samples)) < 0.2  # noise, not destruction


def test_renormalize_restores_invariants(samples):
    rng = np.random.default_rng(0)
    messy = jitter(samples, sigma=0.05, rng=rng)
    out = renormalize(messy)
    assert np.allclose(lm(out, 0), 0.0, atol=1e-6)          # wrist anchored
    assert np.allclose(np.linalg.norm(lm(out, 9), axis=1), 1.0, atol=1e-5)


def test_augment_at_least_triples_data(samples):
    y = np.arange(10) % 2
    X_aug, y_aug = augment_training_set(samples, y, seed=42)
    assert X_aug.shape[0] >= 3 * samples.shape[0]
    assert X_aug.shape[1] == 63
    assert y_aug.shape[0] == X_aug.shape[0]


def test_augment_preserves_class_balance(samples):
    y = np.arange(10) % 2
    _, y_aug = augment_training_set(samples, y, seed=42)
    counts = np.bincount(y_aug)
    assert counts[0] == counts[1]


def test_augment_output_satisfies_invariants(samples):
    y = np.zeros(10, dtype=np.int64)
    X_aug, _ = augment_training_set(samples, y, seed=42)
    assert np.allclose(lm(X_aug, 0), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(lm(X_aug, 9), axis=1), 1.0, atol=1e-5)


def test_augment_is_reproducible(samples):
    y = np.zeros(10, dtype=np.int64)
    a, _ = augment_training_set(samples, y, seed=42)
    b, _ = augment_training_set(samples, y, seed=42)
    assert np.array_equal(a, b)

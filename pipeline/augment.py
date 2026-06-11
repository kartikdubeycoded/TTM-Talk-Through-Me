"""Task 4: honest data augmentation for normalized landmark samples.

All functions take and return (N, 63) float32 arrays. Apply ONLY to the
training split, never before the train/val split — augmenting first leaks
near-duplicates into validation and inflates the accuracy.

After any geometric noise the samples are re-normalized, because the live
pipeline guarantees wrist == (0,0,0) and |landmark 9| == 1 on every frame;
training data must match that distribution exactly.
"""
import numpy as np


def mirror(samples):
    """Right hand -> left hand: negate every x coordinate."""
    out = samples.reshape(-1, 21, 3).copy()
    out[:, :, 0] *= -1
    return out.reshape(-1, 63)


def rotate(samples, max_degrees, rng):
    """Rotate each sample by a random angle in the camera (xy) plane.

    Simulates a tilted hand/camera. Rotation is around the wrist, which is
    already the origin in normalized space.
    """
    n = samples.shape[0]
    angles = np.radians(rng.uniform(-max_degrees, max_degrees, size=n))
    cos, sin = np.cos(angles), np.sin(angles)

    pts = samples.reshape(n, 21, 3).copy()
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    pts[:, :, 0] = cos[:, None] * x - sin[:, None] * y
    pts[:, :, 1] = sin[:, None] * x + cos[:, None] * y
    return pts.reshape(n, 63)


def rotate_3d(samples, max_degrees_y, max_degrees_x, rng):
    """Rotate each sample out of the camera plane — around the vertical (y)
    axis and the horizontal (x) axis. Simulates the camera seeing the same
    handshape from the side / above. This is the main weapon against a
    single-viewpoint training set.
    """
    n = samples.shape[0]
    pts = samples.reshape(n, 21, 3).copy()

    # Around y axis: x/z mix (camera shifted sideways)
    ay = np.radians(rng.uniform(-max_degrees_y, max_degrees_y, size=n))
    cos_y, sin_y = np.cos(ay), np.sin(ay)
    x, z = pts[:, :, 0].copy(), pts[:, :, 2].copy()
    pts[:, :, 0] = cos_y[:, None] * x + sin_y[:, None] * z
    pts[:, :, 2] = -sin_y[:, None] * x + cos_y[:, None] * z

    # Around x axis: y/z mix (camera shifted up/down)
    ax = np.radians(rng.uniform(-max_degrees_x, max_degrees_x, size=n))
    cos_x, sin_x = np.cos(ax), np.sin(ax)
    y, z = pts[:, :, 1].copy(), pts[:, :, 2].copy()
    pts[:, :, 1] = cos_x[:, None] * y - sin_x[:, None] * z
    pts[:, :, 2] = sin_x[:, None] * y + cos_x[:, None] * z

    return pts.reshape(n, 63)


def jitter(samples, sigma, rng):
    """Add small gaussian noise — simulates landmark detection wobble."""
    return samples + rng.normal(0, sigma, size=samples.shape).astype(np.float32)


def renormalize(samples):
    """Re-apply the pipeline invariants: wrist at origin, landmark 9 unit."""
    pts = samples.reshape(-1, 21, 3)
    pts = pts - pts[:, 0:1, :]
    scale = np.linalg.norm(pts[:, 9, :], axis=1, keepdims=True)
    scale = np.where(scale == 0, 1e-6, scale)
    pts = pts / scale[:, None, :]
    return pts.reshape(-1, 63).astype(np.float32)


def augment_training_set(X, y, seed):
    """Original + mirrored + rotated + jittered -> >= 4x data, balanced.

    Every augmented variant keeps its source sample's label, so per-class
    counts scale uniformly and balance is preserved.
    """
    rng = np.random.default_rng(seed)

    variants = [
        X,
        mirror(X),
        renormalize(rotate(X, max_degrees=15, rng=rng)),
        renormalize(jitter(X, sigma=0.02, rng=rng)),
        rotate_3d(X, max_degrees_y=30, max_degrees_x=20, rng=rng),
        rotate_3d(mirror(X), max_degrees_y=30, max_degrees_x=20, rng=rng),
    ]
    X_aug = np.vstack(variants).astype(np.float32)
    y_aug = np.concatenate([y] * len(variants))
    return X_aug, y_aug

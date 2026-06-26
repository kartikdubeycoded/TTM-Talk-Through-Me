"""Per-frame features for word-level (temporal) sign recognition.

Each frame of a word sequence becomes 68 numbers:
  [0:63]  the 21 hand landmarks normalized per frame — same wrist-anchor +
          knuckle-scale convention as the alphabet pipeline (normalize.py),
          so the live extension computes them identically
  [63:66] wrist trajectory: this frame's raw wrist position minus the first
          frame's — per-frame normalization erases WHERE the hand is, but
          signs are movements; this puts the motion back.
  [66:68] wrist LOCATION: this frame's absolute wrist x,y in the camera frame.
          Trajectory says how the hand MOVED; location says WHERE it is. Signs
          like "dad" (hand at forehead) vs "mom" (hand at chin) share a
          handshape and barely move — only the height in frame separates them,
          which shape+trajectory alone cannot see. z is omitted: MediaPipe hand
          z is wrist-relative, so the wrist's own z is ~0 (a dead column).
"""
import numpy as np


def sequence_features(frames):
    """(T, 21, 3) raw hand landmarks -> (T, 68) features."""
    frames = np.asarray(frames, dtype=np.float32)
    wrist = frames[:, 0:1, :]                      # (T, 1, 3)
    centered = frames - wrist
    scale = np.linalg.norm(centered[:, 9, :], axis=1, keepdims=True)  # (T, 1)
    scale = np.where(scale == 0, 1e-6, scale)
    normalized = centered / scale[:, None, :]      # (T, 21, 3)

    trajectory = (wrist[:, 0, :] - wrist[0, 0, :])  # (T, 3)
    location = wrist[:, 0, 0:2]                      # (T, 2) absolute x,y
    return np.concatenate(
        [normalized.reshape(len(frames), 63), trajectory, location], axis=1
    ).astype(np.float32)


def resample(seq, n_frames):
    """Linearly resample (T, F) to (n_frames, F) along the time axis."""
    seq = np.asarray(seq, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, len(seq))
    t_out = np.linspace(0.0, 1.0, n_frames)
    out = np.empty((n_frames, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(t_out, t_in, seq[:, f])
    return out

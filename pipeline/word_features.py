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
    wrist = frames[:, 0:1, :]                      # (T, 1, 3) for broadcasting
    centered = frames - wrist
    scale = np.linalg.norm(centered[:, 9, :], axis=1, keepdims=True)  # (T, 1)
    scale = np.where(scale == 0, 1e-6, scale)
    normalized = centered / scale[:, None, :]      # (T, 21, 3)

    wrist_pos = wrist[:, 0, :]                      # (T, 3) per-frame wrist
    trajectory = wrist_pos - wrist_pos[0]          # (T, 3) motion since frame 0
    location = wrist_pos[:, 0:2]                    # (T, 2) absolute x,y
    return np.concatenate(
        [normalized.reshape(len(frames), 63), trajectory, location], axis=1
    ).astype(np.float32)


def whole_body_features(hand, lips, pose):
    """(T,21,3) hand + (T,40,3) lips + (T,5,3) pose -> (T, 81) features.

    The first 68 columns are exactly sequence_features(hand) — the proven
    hand-only recipe, reused unchanged. The next 13 are a whole-body block that
    tells the model WHERE the hand is on the face/body — the signal a hand-only
    model is blind to (why "fine"/"can"/"finish" sit near the accuracy floor,
    and why "dad" at the forehead vs "mom" at the chin are confusable):

      [68:70] wrist relative to the lips centroid, scaled by lip width
      [70:80] the 5 pose points relative to the lips centroid, same scale
      [80:81] lip openness (lip height / lip width) — a mouthing cue

    Everything is anchored to the lips centroid and scaled by lip width, so it
    is invariant to where the signer sits and how large they appear in frame
    (same math idea as the hand's wrist-anchor + knuckle-scale normalization).
    Frames with no detected lips (all-zero, the dataset's missing convention)
    get a zero whole-body block — they can't be face-anchored.
    """
    hand = np.asarray(hand, dtype=np.float32)
    lips = np.asarray(lips, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    T = len(hand)

    base = sequence_features(hand)                     # (T, 68) proven hand block

    lips_xy = lips[:, :, 0:2]                           # (T, 40, 2)
    face_c = lips_xy.mean(axis=1)                       # (T, 2) lips centroid
    lip_w = lips_xy[:, :, 0].max(axis=1) - lips_xy[:, :, 0].min(axis=1)  # (T,)
    lip_h = lips_xy[:, :, 1].max(axis=1) - lips_xy[:, :, 1].min(axis=1)  # (T,)
    safe_w = np.where(lip_w == 0, 1e-6, lip_w)          # (T,)

    wrist_xy = hand[:, 0, 0:2]                          # (T, 2)
    hand_rel = (wrist_xy - face_c) / safe_w[:, None]    # (T, 2)
    pose_rel = ((pose[:, :, 0:2] - face_c[:, None, :])
                / safe_w[:, None, None]).reshape(T, 10)  # (T, 10)
    openness = (lip_h / safe_w)[:, None]                # (T, 1)

    body = np.concatenate([hand_rel, pose_rel, openness], axis=1)  # (T, 13)

    # Zero the block for frames whose lips weren't detected (all-zero input).
    lips_present = np.abs(lips).sum(axis=(1, 2)) > 0    # (T,)
    body[~lips_present] = 0.0

    return np.concatenate([base, body], axis=1).astype(np.float32)


def resample(seq, n_frames):
    """Linearly resample (T, F) to (n_frames, F) along the time axis."""
    seq = np.asarray(seq, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, len(seq))
    t_out = np.linspace(0.0, 1.0, n_frames)
    out = np.empty((n_frames, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(t_out, t_in, seq[:, f])
    return out

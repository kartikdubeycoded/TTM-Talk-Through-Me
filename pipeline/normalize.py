"""Single source of truth for landmark normalization.

The model never sees raw coordinates. Every frame is:
  1. Anchored  — wrist (landmark 0) moved to (0, 0, 0), so position on
     screen doesn't matter.
  2. Scaled    — every coordinate divided by the wrist -> middle-finger-knuckle
     (landmark 9) distance, so hand size / camera distance doesn't matter.

This MUST stay identical to getNormalizedLandmarks() in extension/content.js.
"""
import math


def get_normalized_landmarks(hand_landmarks):
    """21 MediaPipe landmarks -> flat list of 63 normalized floats."""
    wrist = hand_landmarks[0]
    mcp_9 = hand_landmarks[9]

    scale = math.sqrt(
        (mcp_9.x - wrist.x) ** 2 +
        (mcp_9.y - wrist.y) ** 2 +
        (mcp_9.z - wrist.z) ** 2
    )
    if scale == 0:
        scale = 1e-6

    normalized = []
    for lm in hand_landmarks:
        normalized.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return normalized

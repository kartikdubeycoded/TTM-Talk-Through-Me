"""The single source of truth for the 66-point landmark layout.

The word model's input is 66 landmarks/frame in a fixed order:
    [lips 0:40, hand 40:61, pose 61:66]
(same order as ingest_words.py's LIPS/HAND/POSE slices and the GISLR training
data). Both the ASL Citizen extractor (video -> landmarks) AND the live browser
code must select the *same* dots in the *same* order, or the model silently
receives garbage (see risks.md #1, landmark-map.md). Defining the selection ONCE
here — and testing it — is what stops that divergence.

Indices are into the MediaPipe Tasks landmarkers:
  * face -> FaceLandmarker (468/478 pts; all lip indices are < 468, so they are
    identical in the old Holistic FaceMesh and the new FaceLandmarker)
  * hand -> HandLandmarker (21 pts, native order)
  * pose -> PoseLandmarker (BlazePose 33 pts)
"""
import numpy as np

# MediaPipe FACEMESH_LIPS — 40 mouth points (authoritative list; see landmark-map.md).
LIPS_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 42, 183, 78,
]

# HandLandmarker's native 21 points, in native order.
HAND_IDXS = list(range(21))

# BlazePose subset for face-anchored signs: nose + shoulders + elbows. This is a
# design choice (we own both ends) — the ONLY rule is that extraction and live
# use this exact same 5. If you change it, change it in one place: here.
POSE_IDXS = [0, 11, 12, 13, 14]  # nose, L/R shoulder, L/R elbow

N_LIPS, N_HAND, N_POSE = len(LIPS_IDXS), len(HAND_IDXS), len(POSE_IDXS)
N_POINTS = N_LIPS + N_HAND + N_POSE  # 66

# Slices into the assembled (66, 3) array — mirror ingest_words.py exactly.
LIPS_SLICE = slice(0, N_LIPS)                       # 0:40
HAND_SLICE = slice(N_LIPS, N_LIPS + N_HAND)         # 40:61
POSE_SLICE = slice(N_LIPS + N_HAND, N_POINTS)       # 61:66


def select_66(face, hand, pose):
    """Assemble live landmarks into the model's fixed (66, 3) layout.

    face: (>=468, 3)  hand: (21, 3)  pose: (>=15, 3)  ->  (66, 3)
    Order is [lips(40), hand(21), pose(5)] — the layout the model was built on.
    Missing parts (empty arrays) become zeros, the dataset's missing convention.
    """
    def take(arr, idxs, n):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((n, 3), dtype=np.float32)
        return arr[idxs]

    lips = take(face, LIPS_IDXS, N_LIPS)
    hand_sel = take(hand, HAND_IDXS, N_HAND)
    pose_sel = take(pose, POSE_IDXS, N_POSE)
    return np.concatenate([lips, hand_sel, pose_sel], axis=0).astype(np.float32)

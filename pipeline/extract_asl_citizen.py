"""Stream ASL Citizen videos out of the zip -> MediaPipe landmarks -> training
tensors, WITHOUT ever fully unzipping (we don't have the disk for that).

For each wanted video: pull just that one file from the zip to a temp path, run
MediaPipe Face + Hand + Pose over its frames, select the fixed 66 dots
(landmark_layout.select_66), turn them into the word model's 81-D whole-body
features (word_features.whole_body_features), resample to 30 frames, then delete
the temp file. Output matches pipeline/ingest_words.py exactly, so
training/train_words.py can train on it unchanged:

    <out>/X.npy            (M, 30, 81) float32
    <out>/y.npy            (M,)        int64  (index into labels.json)
    <out>/participants.npy (M,)        int64  (signer id, for honest splits)
    <out>/labels.json      the kept vocabulary, alphabetical

Usage (verify first!):
    python pipeline/extract_asl_citizen.py --top-n 50 --limit 5     # 5-video smoke
    python pipeline/extract_asl_citizen.py --top-n 50               # full pilot

Known pilot-level simplifications (revisit before scaling up):
  * num_hands=1 — takes the single most-prominent hand (the word model is
    dominant-hand, 21 points). Two-handed signs lose the second hand.
  * no left/right-hand mirroring — a left-handed signer is not flipped.
"""
import argparse
import json
import os
import sys
import tempfile
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from landmark_layout import select_66, LIPS_SLICE, HAND_SLICE, POSE_SLICE
from word_features import whole_body_features, resample
from asl_citizen import load_split, select_vocab

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

N_FRAMES = 30          # sequence length fed to the model (matches ingest_words)
MIN_HAND_FRAMES = 8    # sequences with fewer hand-present frames are skipped
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "extension", "model")


def _landmarks_to_array(landmark_list, n_points):
    """A MediaPipe landmark list -> (n_points, 3); zeros if nothing detected."""
    if not landmark_list:
        return np.zeros((n_points, 3), dtype=np.float32)
    lm = landmark_list[0]  # first detected face/hand/pose
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)


def build_landmarkers():
    """Create the three MediaPipe VIDEO-mode landmarkers (face, hand, pose)."""
    def opts(cls, name, **extra):
        base = mp_python.BaseOptions(
            model_asset_path=os.path.join(MODEL_DIR, name))
        return cls(base_options=base, running_mode=vision.RunningMode.VIDEO, **extra)

    face = vision.FaceLandmarker.create_from_options(
        opts(vision.FaceLandmarkerOptions, "face_landmarker.task", num_faces=1))
    hand = vision.HandLandmarker.create_from_options(
        opts(vision.HandLandmarkerOptions, "hand_landmarker.task", num_hands=1))
    pose = vision.PoseLandmarker.create_from_options(
        opts(vision.PoseLandmarkerOptions, "pose_landmarker.task", num_poses=1))
    return face, hand, pose


def video_to_features(video_path, face, hand, pose, clock):
    """One video file -> (30, 81) whole-body features, or None if too few hands.

    `clock` is a one-element list holding a strictly-increasing millisecond
    counter that PERSISTS across videos — MediaPipe VIDEO mode rejects any
    timestamp that isn't larger than the previous one, even across files.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames66 = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            clock[0] += 33  # ~30fps step; only strict monotonicity matters here
            ts = clock[0]
            f = _landmarks_to_array(face.detect_for_video(mp_img, ts).face_landmarks, 478)
            h = _landmarks_to_array(hand.detect_for_video(mp_img, ts).hand_landmarks, 21)
            p = _landmarks_to_array(pose.detect_for_video(mp_img, ts).pose_landmarks, 33)
            frames66.append(select_66(f, h, p))
    finally:
        cap.release()  # always release, so the temp file can be deleted
    if not frames66:
        return None

    seq = np.stack(frames66)                                  # (T, 66, 3)
    has_hand = np.abs(seq[:, HAND_SLICE, :]).sum(axis=(1, 2)) > 0
    if has_hand.sum() < MIN_HAND_FRAMES:
        return None
    seq = seq[has_hand]                                       # keep hand frames
    feats = whole_body_features(seq[:, HAND_SLICE, :],
                                seq[:, LIPS_SLICE, :],
                                seq[:, POSE_SLICE, :])         # (T, 81)
    return resample(feats, N_FRAMES)                          # (30, 81)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.expanduser("~/asl_citizen/ASL_Citizen.zip"))
    ap.add_argument("--split", default="data/asl_citizen/train.csv")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--out", default="data/asl_citizen_words")
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all); for a smoke run")
    args = ap.parse_args()

    records = load_split(args.split)
    vocab, kept = select_vocab(records, args.top_n)
    label_index = {g: i for i, g in enumerate(vocab)}
    if args.limit:
        kept = kept[:args.limit]
    print(f"{len(kept)} videos across {len(vocab)} signs (top-{args.top_n})")

    face, hand, pose = build_landmarkers()
    clock = [0]  # strictly-increasing ms clock shared across all videos
    X, y, participants, skipped, missing = [], [], [], 0, 0

    # Map signer ids (e.g. "P1") to stable int codes for participants.npy.
    part_codes = {}
    def signer_code(p):
        if p not in part_codes:
            part_codes[p] = len(part_codes)
        return part_codes[p]

    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        for n, rec in enumerate(kept):
            entry = f"ASL_Citizen/videos/{rec['video']}"
            if entry not in names:
                missing += 1
                continue
            # Stream this one video out to a temp file, process, delete.
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(zf.read(entry))
                tmp_path = tmp.name
            try:
                feats = video_to_features(tmp_path, face, hand, pose, clock)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Windows may hold the handle briefly; temp dir cleans up

            if feats is None:
                skipped += 1
                continue
            X.append(feats)
            y.append(label_index[rec["gloss"]])
            participants.append(signer_code(rec.get("participant")))
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(kept)} processed ({skipped} skipped, {missing} missing)")

    if not X:
        print("No sequences extracted — nothing saved.")
        return
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "X.npy"), np.stack(X))
    np.save(os.path.join(args.out, "y.npy"), np.array(y, dtype=np.int64))
    np.save(os.path.join(args.out, "participants.npy"),
            np.array(participants, dtype=np.int64))
    with open(os.path.join(args.out, "labels.json"), "w") as fh:
        json.dump(vocab, fh)
    print(f"\nSaved {len(X)} sequences to {args.out} "
          f"({skipped} skipped, {missing} missing videos)")


if __name__ == "__main__":
    main()

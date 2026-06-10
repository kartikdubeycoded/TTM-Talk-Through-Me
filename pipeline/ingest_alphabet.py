"""Task 3: Kaggle ASL Alphabet images -> normalized landmark arrays.

Walks data/raw/asl_alphabet_train/<CLASS>/ folders, runs MediaPipe
HandLandmarker on each image, normalizes (see normalize.py), and saves
one (N, 63) float32 array per class to data/alphabet/<CLASS>.npy.

Images where MediaPipe finds no hand are counted and reported, not
silently dropped. Folder names are mapped to our 28 training classes:
space -> SPACE, del -> DELETE; the 'nothing' folder is skipped (no hand
to find — the live pipeline already treats "no detection" as nothing).

Usage:
    python pipeline/ingest_alphabet.py            # 1000 images per class
    python pipeline/ingest_alphabet.py --limit 0  # all (~3000 per class)
"""
import argparse
import os
import sys

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(__file__))
from normalize import get_normalized_landmarks

MODEL_PATH = "hand_landmarker.task"
RAW_DIR = os.path.join("data", "raw", "asl_alphabet_train", "asl_alphabet_train")
OUT_DIR = os.path.join("data", "alphabet")

FOLDER_TO_CLASS = {chr(i): chr(i) for i in range(ord("A"), ord("Z") + 1)}
FOLDER_TO_CLASS["space"] = "SPACE"
FOLDER_TO_CLASS["del"] = "DELETE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000,
                        help="max images per class (0 = all)")
    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        print(f"Error: dataset not found at {RAW_DIR}")
        print("Download + unzip the Kaggle ASL Alphabet dataset first.")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = []

    for folder, cls_name in sorted(FOLDER_TO_CLASS.items()):
        folder_path = os.path.join(RAW_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: missing folder {folder_path}, skipping.")
            continue

        files = sorted(os.listdir(folder_path))
        if args.limit > 0:
            files = files[: args.limit]

        samples, skipped = [], 0
        for fname in files:
            image = mp.Image.create_from_file(os.path.join(folder_path, fname))
            result = detector.detect(image)
            if result.hand_landmarks:
                samples.append(get_normalized_landmarks(result.hand_landmarks[0]))
            else:
                skipped += 1

        arr = np.array(samples, dtype=np.float32)
        np.save(os.path.join(OUT_DIR, f"{cls_name}.npy"), arr)
        report.append((cls_name, len(samples), skipped))
        print(f"{cls_name}: saved {len(samples)} samples, "
              f"skipped {skipped} (no hand detected)")

    print("\n--- Ingest summary ---")
    total_ok = sum(n for _, n, _ in report)
    total_skip = sum(s for _, _, s in report)
    print(f"Classes: {len(report)}  Samples: {total_ok}  Skipped: {total_skip}")
    low = [c for c, n, _ in report if n < 200]
    if low:
        print(f"WARNING — thin classes (<200 samples): {', '.join(low)}")


if __name__ == "__main__":
    main()

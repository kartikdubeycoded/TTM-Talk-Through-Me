"""Honest evaluation: score the model on landmarks from a signer it has
NEVER seen (data/test_signer, from the danrasband/asl-alphabet-test set).

This number is the one that predicts live performance — unlike the
validation split, which shares a signer with the training data.

Usage:
    python training/evaluate.py                      # eval data/test_signer
    python training/evaluate.py --data-dir <dir>
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "alphabet.h5")
LABELS_PATH = os.path.join("models", "labels.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("data", "test_signer"))
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        classes = json.load(f)

    rows = []
    total_correct, total_n = 0, 0
    for idx, cls_name in enumerate(classes):
        path = os.path.join(args.data_dir, f"{cls_name}.npy")
        if not os.path.exists(path):
            continue
        x = np.load(path)
        if len(x) == 0:
            continue
        pred = np.argmax(model.predict(x, verbose=0), axis=1)
        correct = int((pred == idx).sum())
        total_correct += correct
        total_n += len(x)

        # Most common wrong answer, to see what it's confused with
        wrong = pred[pred != idx]
        confused_with = classes[np.bincount(wrong).argmax()] if len(wrong) else "-"
        rows.append((cls_name, correct, len(x), confused_with))

    print(f"\n=== Held-out signer accuracy: {total_correct}/{total_n} "
          f"= {100 * total_correct / total_n:.1f}% ===\n")
    print(f"{'class':>6} {'acc':>9}   mistaken-for")
    for cls_name, correct, n, confused in sorted(rows, key=lambda r: r[1] / r[2]):
        print(f"{cls_name:>6} {correct:>4}/{n:<4}   {confused if correct < n else ''}")


if __name__ == "__main__":
    main()

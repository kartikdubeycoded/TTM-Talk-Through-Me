"""Phase 5 / T12: GISLR word sequences -> training tensors for the starter
vocabulary.

Input:  data/raw/gislr-words/  (X.npy (N,64,66,3), y.npy, frame indexes)
        data/raw/gislr-meta/extended_train.csv  (sign names, participants)
Output: data/words/X.npy  (M, 30, 66)  per-frame features (word_features.py)
        data/words/y.npy  (M,)          label index into labels.json
        data/words/participants.npy     for honest signer-level splits
        data/words/labels.json

The 66 input landmarks per frame are [lips 0:40, hand 40:61, pose 61:66];
we use only the hand block. Frames with no hand are dropped; sequences with
fewer than 8 hand frames are skipped (reported).
"""
import json
import os

import numpy as np
import pandas as pd

from word_features import sequence_features, resample

RAW_DIR = os.path.join("data", "raw", "gislr-words")
META_CSV = os.path.join("data", "raw", "gislr-meta", "extended_train.csv")
OUT_DIR = os.path.join("data", "words")

HAND_SLICE = slice(40, 61)
N_FRAMES = 30
MIN_HAND_FRAMES = 8

VOCAB = [
    "hello", "bye", "yes", "no", "please", "thankyou", "fine", "bad",
    "happy", "sad", "mad", "hungry", "thirsty", "sick", "sleepy", "water",
    "food", "drink", "milk", "mom", "dad", "who", "where", "why", "now",
    "later", "tomorrow", "night", "morning", "home", "finish", "can", "go",
    "wait", "look", "listen", "talk", "think", "like", "have",
]


def main():
    X_all = np.load(os.path.join(RAW_DIR, "X.npy"), mmap_mode="r")
    y_all = np.load(os.path.join(RAW_DIR, "y.npy"))
    frame_idxs = np.load(os.path.join(RAW_DIR, "NON_EMPTY_FRAME_IDXS.npy"),
                         mmap_mode="r")
    meta = pd.read_csv(META_CSV)
    assert len(meta) == len(X_all), "metadata/X row count mismatch"

    sign_of_label = (
        pd.DataFrame({"y": y_all, "sign": meta["sign"]})
        .drop_duplicates().set_index("y")["sign"].to_dict()
    )
    wanted_labels = {lbl for lbl, s in sign_of_label.items() if s in VOCAB}
    label_index = {s: i for i, s in enumerate(VOCAB)}

    rows = np.where(np.isin(y_all, list(wanted_labels)))[0]
    print(f"{len(rows)} sequences for {len(VOCAB)} words")

    X_out, y_out, p_out, skipped = [], [], [], 0
    for n, r in enumerate(rows):
        slots = np.where(frame_idxs[r] >= 0)[0]
        frames = np.array(X_all[r])[slots]            # (T, 66, 3)
        hand = frames[:, HAND_SLICE, :]               # (T, 21, 3)
        has_hand = np.abs(hand).sum(axis=(1, 2)) > 0
        hand = hand[has_hand]
        if len(hand) < MIN_HAND_FRAMES:
            skipped += 1
            continue
        X_out.append(resample(sequence_features(hand), N_FRAMES))
        y_out.append(label_index[sign_of_label[y_all[r]]])
        p_out.append(meta["participant_id"].iloc[r])
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(rows)} processed...")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X.npy"), np.stack(X_out))
    np.save(os.path.join(OUT_DIR, "y.npy"), np.array(y_out, dtype=np.int64))
    np.save(os.path.join(OUT_DIR, "participants.npy"),
            np.array(p_out, dtype=np.int64))
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(VOCAB, f)

    counts = np.bincount(y_out, minlength=len(VOCAB))
    print(f"\nSaved {len(X_out)} sequences ({skipped} skipped, too few hand frames)")
    print(f"Participants: {len(set(p_out))}")
    thin = [VOCAB[i] for i in range(len(VOCAB)) if counts[i] < 150]
    print(f"Per-word min/median/max: {counts.min()}/{int(np.median(counts))}/{counts.max()}")
    if thin:
        print(f"WARNING — thin words (<150): {', '.join(thin)}")


if __name__ == "__main__":
    main()

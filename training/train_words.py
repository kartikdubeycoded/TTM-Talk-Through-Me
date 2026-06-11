"""Phase 5 / T13: train the word-level temporal model.

Reads data/words/ (from pipeline/ingest_words.py). The validation split is
by PARTICIPANT — entire signers held out — because random-row splits flatter
the score (today's capacity experiment proved it on the alphabet model).
"""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = os.path.join("data", "words")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "words.h5")


def main():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    groups = np.load(os.path.join(DATA_DIR, "participants.npy"))
    with open(os.path.join(DATA_DIR, "labels.json")) as f:
        vocab = json.load(f)

    print(f"{X.shape[0]} sequences, {X.shape[1]} frames x {X.shape[2]} features, "
          f"{len(vocab)} words, {len(set(groups))} signers")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train: {len(X_train)} ({len(set(groups[train_idx]))} signers)  "
          f"Val: {len(X_val)} ({len(set(groups[val_idx]))} held-out signers)")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(vocab), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6, restore_best_weights=True),
        ],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n=== Held-out signer word accuracy: {val_acc:.3f} ===")

    # Per-word accuracy, worst first — shows which words to fix or drop
    pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(f"\n{'word':>10}  acc")
    rows = []
    for i, w in enumerate(vocab):
        mask = y_val == i
        if mask.sum() == 0:
            continue
        rows.append((w, (pred[mask] == i).mean(), int(mask.sum())))
    for w, acc, n in sorted(rows, key=lambda r: r[1]):
        print(f"{w:>10}  {acc:.2f}  (n={n})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

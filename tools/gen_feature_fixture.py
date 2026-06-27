"""Generate a JS<->Python parity fixture for the word-feature pipeline.

Writes tests/fixtures/word_features_io.json: random raw hand sequences and
their expected (T, 68) features from pipeline/word_features.sequence_features.

tests/smoke_features.mjs replays the same raw frames through
extension/features.js and asserts the live JS math matches Python. The LSTM
smoke test only proves the *model* forward pass matches Keras on a shared
input — it cannot see a divergence in how the live extension *builds* that
input. This fixture closes that gap.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from word_features import sequence_features

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                   "word_features_io.json")


def main():
    rng = np.random.default_rng(7)
    frames = rng.normal(size=(30, 21, 3)).astype(np.float32)  # raw hand landmarks
    expected = sequence_features(frames)                       # (30, 68)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"frames": frames.tolist(), "expected": expected.tolist()}, f)
    print(f"Wrote {OUT}: frames {frames.shape} -> expected {expected.shape}")


if __name__ == "__main__":
    main()

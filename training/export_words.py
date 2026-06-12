"""Export the word LSTM to plain-JSON weights for the extension.

Same philosophy as export_weights.py (no TF.js in MV3): dump raw weights,
re-implement the forward pass (including the LSTM cell) in NumPy here, and
assert it reproduces model.predict() before writing anything. The JS port
in extension/inference.js is then verified against the same fixture by
tests/smoke_words.mjs.

Keras LSTM cell, gate order i, f, c, o (columns of the fused matrices):
  z = x @ kernel + h @ recurrent_kernel + bias        (size 4*units)
  i, f, c~, o = split(z); i,f,o -> sigmoid; c~ -> tanh
  c = f * c_prev + i * c~ ;  h = o * tanh(c)
"""
import json
import os

import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "words.h5")
LABELS_PATH = os.path.join("data", "words", "labels.json")
EXPORT_PATH = os.path.join("extension", "model", "words.json")
FIXTURE_PATH = os.path.join("tests", "fixtures", "words_io.json")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_forward(layer, seq):
    """(T, in) -> (T, units) hidden states, NumPy reimplementation."""
    kernel = np.array(layer["kernel"])
    recurrent = np.array(layer["recurrent"])
    bias = np.array(layer["bias"])
    units = layer["units"]

    h = np.zeros(units)
    c = np.zeros(units)
    out = []
    for x in seq:
        z = x @ kernel + h @ recurrent + bias
        i = sigmoid(z[0:units])
        f = sigmoid(z[units:2 * units])
        c_t = np.tanh(z[2 * units:3 * units])
        o = sigmoid(z[3 * units:4 * units])
        c = f * c + i * c_t
        h = o * np.tanh(c)
        out.append(h.copy())
    return np.stack(out)


def numpy_forward(layers, seq):
    x = seq
    for layer in layers:
        if layer["type"] == "lstm":
            x = lstm_forward(layer, x)
            if not layer["returnSequences"]:
                x = x[-1]
        else:  # dense
            x = x @ np.array(layer["kernel"]) + np.array(layer["bias"])
            if layer["activation"] == "relu":
                x = np.maximum(x, 0)
            elif layer["activation"] == "softmax":
                e = np.exp(x - x.max())
                x = e / e.sum()
    return x


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            kernel, recurrent, bias = layer.get_weights()
            layers.append({
                "type": "lstm",
                "units": int(recurrent.shape[0]),
                "returnSequences": bool(layer.return_sequences),
                "kernel": kernel.tolist(),
                "recurrent": recurrent.tolist(),
                "bias": bias.tolist(),
            })
        elif isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            layers.append({
                "type": "dense",
                "units": int(kernel.shape[1]),
                "activation": layer.get_config()["activation"],
                "kernel": kernel.tolist(),
                "bias": bias.tolist(),
            })

    # Self-verify on 3 random sequences before writing anything
    rng = np.random.default_rng(0)
    seqs = rng.normal(size=(3, 30, 66)).astype(np.float32)
    expected = model.predict(seqs, verbose=0)
    for s in range(3):
        actual = numpy_forward(layers, seqs[s])
        if not np.allclose(expected[s], actual, atol=1e-4):
            raise AssertionError(
                f"NumPy LSTM forward diverges from Keras (seq {s}, "
                f"max diff {np.abs(expected[s] - actual).max()})")
    print("Smoke test passed: NumPy LSTM forward reproduces model.predict().")

    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump({"frames": 30, "features": 66, "labels": labels,
                   "layers": layers}, f)
    size_mb = os.path.getsize(EXPORT_PATH) / 1024 / 1024
    print(f"Exported {len(layers)} layers + {len(labels)} words "
          f"to {EXPORT_PATH} ({size_mb:.1f} MB)")

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump({"inputs": seqs.tolist(), "expected": expected.tolist()}, f)
    print(f"JS smoke-test fixture written to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()

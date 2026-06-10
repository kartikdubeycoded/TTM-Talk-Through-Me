"""Export the alphabet model to a plain-JSON weights file for the extension.

Why not tensorflowjs_converter: tensorflowjs 3.18 (the last version that
installs on Windows) is incompatible with modern NumPy and cannot run.
The model is 3 small Dense layers, so we export the raw weights ourselves
and rebuild the same network in TF.js inside content.js.

Self-verifies: re-runs the forward pass in NumPy from the exported arrays
and asserts it matches model.predict() before writing anything.
"""
import json
import os

import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "alphabet.h5")
LABELS_PATH = os.path.join("models", "labels.json")
EXPORT_PATH = os.path.join("extension", "model", "weights.json")


def numpy_forward(layers, x):
    """Re-implement the dense stack: x @ kernel + bias, relu/softmax."""
    for layer in layers:
        x = x @ np.array(layer["kernel"]) + np.array(layer["bias"])
        if layer["activation"] == "relu":
            x = np.maximum(x, 0)
        elif layer["activation"] == "softmax":
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            x = e / e.sum(axis=-1, keepdims=True)
    return x


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            layers.append({
                "units": int(kernel.shape[1]),
                "activation": layer.get_config()["activation"],
                "kernel": kernel.tolist(),
                "bias": bias.tolist(),
            })

    # Smoke test: exported arrays must reproduce the real model's output
    x = np.random.default_rng(0).normal(size=(5, 63)).astype(np.float32)
    expected = model.predict(x, verbose=0)
    actual = numpy_forward(layers, x)
    if not np.allclose(expected, actual, atol=1e-5):
        raise AssertionError("Exported weights do NOT reproduce model output")
    print("Smoke test passed: exported weights reproduce model.predict().")

    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump({"inputSize": 63, "labels": labels, "layers": layers}, f)

    size_kb = os.path.getsize(EXPORT_PATH) / 1024
    print(f"Exported {len(layers)} dense layers + {len(labels)} labels "
          f"to {EXPORT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

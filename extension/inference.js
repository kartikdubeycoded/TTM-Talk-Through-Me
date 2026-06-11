// inference.js — plain-JS forward pass for the exported classifier.
//
// Why not TensorFlow.js: tf.js calls new Function() (eval) during startup,
// which Manifest V3 forbids in extension pages with no opt-in. Our model is
// 3 dense layers (~18k weights), so we run the matrix math directly.
// Verified against Keras outputs in tests/smoke_model.mjs.

export function createClassifier(spec) {
  const layers = spec.layers;

  return function predict(input) {
    let x = input;
    for (const layer of layers) {
      const out = new Float32Array(layer.units);
      for (let j = 0; j < layer.units; j++) {
        let sum = layer.bias[j];
        for (let i = 0; i < x.length; i++) {
          sum += x[i] * layer.kernel[i][j];
        }
        out[j] = sum;
      }

      if (layer.activation === "relu") {
        for (let j = 0; j < out.length; j++) out[j] = Math.max(0, out[j]);
      } else if (layer.activation === "softmax") {
        let max = -Infinity;
        for (const v of out) max = Math.max(max, v);
        let total = 0;
        for (let j = 0; j < out.length; j++) {
          out[j] = Math.exp(out[j] - max);
          total += out[j];
        }
        for (let j = 0; j < out.length; j++) out[j] /= total;
      }
      x = out;
    }
    return x; // softmax scores, one per class
  };
}

export function argmax(scores) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

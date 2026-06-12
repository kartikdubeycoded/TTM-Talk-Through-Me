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

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// One LSTM layer over a whole sequence. Keras gate order: i, f, c, o.
function lstmForward(layer, seq) {
  const units = layer.units;
  const { kernel, recurrent, bias } = layer;
  let h = new Float32Array(units);
  let c = new Float32Array(units);
  const out = [];

  for (const x of seq) {
    const z = Float32Array.from(bias);
    for (let i = 0; i < x.length; i++) {
      const xi = x[i], row = kernel[i];
      if (xi === 0) continue;
      for (let j = 0; j < z.length; j++) z[j] += xi * row[j];
    }
    for (let i = 0; i < units; i++) {
      const hi = h[i], row = recurrent[i];
      if (hi === 0) continue;
      for (let j = 0; j < z.length; j++) z[j] += hi * row[j];
    }

    const hNext = new Float32Array(units);
    const cNext = new Float32Array(units);
    for (let j = 0; j < units; j++) {
      const i_g = sigmoid(z[j]);
      const f_g = sigmoid(z[units + j]);
      const c_t = Math.tanh(z[2 * units + j]);
      const o_g = sigmoid(z[3 * units + j]);
      cNext[j] = f_g * c[j] + i_g * c_t;
      hNext[j] = o_g * Math.tanh(cNext[j]);
    }
    h = hNext;
    c = cNext;
    out.push(h);
  }
  return out;
}

// Sequence model: LSTM layers then dense layers, mirroring
// training/export_words.py's verified NumPy implementation.
export function createSequenceClassifier(spec) {
  const layers = spec.layers;

  return function predict(seq) {
    let x = seq; // (T, features)
    for (const layer of layers) {
      if (layer.type === "lstm") {
        const states = lstmForward(layer, x);
        x = layer.returnSequences ? states : states[states.length - 1];
      } else {
        const out = new Float32Array(layer.units);
        for (let j = 0; j < layer.units; j++) {
          let sum = layer.bias[j];
          for (let i = 0; i < x.length; i++) sum += x[i] * layer.kernel[i][j];
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
    }
    return x;
  };
}

export function argmax(scores) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

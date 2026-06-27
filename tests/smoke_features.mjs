// Parity test: the live extension's word-feature construction
// (extension/features.js) must produce byte-identical features to the Python
// training pipeline (pipeline/word_features.sequence_features). If they drift,
// the model is fed different numbers live than it trained on — and no other
// test would catch it. Fixture: tools/gen_feature_fixture.py.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { normalizeLandmarks, buildWordSequence } from "../extension/features.js";

const dir = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(
  readFileSync(join(dir, "fixtures", "word_features_io.json"), "utf8")
);

// Rebuild the live rolling buffer from the raw frames exactly as offscreen.js
// does: normalize each frame's landmarks, keep the raw wrist for trajectory +
// location.
const buffer = fx.frames.map((frame) => {
  const lms = frame.map(([x, y, z]) => ({ x, y, z }));
  return {
    norm: normalizeLandmarks(lms),
    wrist: { x: lms[0].x, y: lms[0].y, z: lms[0].z },
  };
});
const actual = buildWordSequence(buffer);

let maxDiff = 0;
for (let t = 0; t < fx.expected.length; t++) {
  for (let i = 0; i < fx.expected[t].length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(fx.expected[t][i] - actual[t][i]));
  }
}

if (actual.length !== fx.expected.length || actual[0].length !== fx.expected[0].length) {
  console.error(
    `FAIL: shape mismatch — JS ${actual.length}x${actual[0].length}, ` +
    `Python ${fx.expected.length}x${fx.expected[0].length}`
  );
  process.exit(1);
}
if (maxDiff > 1e-5) {
  console.error(`FAIL: JS features diverge from Python (max diff ${maxDiff.toExponential(2)})`);
  process.exit(1);
}
console.log(
  `PASS: JS features match Python on ${fx.expected.length} frames ` +
  `x ${fx.expected[0].length} features (max diff ${maxDiff.toExponential(2)})`
);

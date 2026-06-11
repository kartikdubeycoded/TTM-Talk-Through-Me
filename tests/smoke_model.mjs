// T6 smoke test: the extension's actual inference module (plain-JS forward
// pass) must reproduce the Keras model's outputs on the fixture generated
// by training/export_weights.py.
//
// Run:  node tests/smoke_model.mjs
import { readFileSync } from "fs";
import { createClassifier } from "../extension/inference.js";

const spec = JSON.parse(
  readFileSync(new URL("../extension/model/weights.json", import.meta.url)));
const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/model_io.json", import.meta.url)));

const predict = createClassifier(spec);

let maxDiff = 0;
fixture.inputs.forEach((input, r) => {
  const actual = predict(input);
  fixture.expected[r].forEach((exp, c) => {
    maxDiff = Math.max(maxDiff, Math.abs(actual[c] - exp));
  });
});

if (maxDiff > 1e-4) {
  console.error(`FAIL: JS predictions diverge from Keras (max diff ${maxDiff})`);
  process.exit(1);
}
console.log(`PASS: extension inference matches Keras on ${fixture.inputs.length} ` +
            `samples (max diff ${maxDiff.toExponential(2)}, ${spec.labels.length} classes)`);

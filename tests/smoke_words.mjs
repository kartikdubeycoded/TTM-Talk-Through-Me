// T13 smoke test: the extension's plain-JS LSTM forward pass must reproduce
// the Keras word model's outputs on the fixture from training/export_words.py.
//
// Run:  node tests/smoke_words.mjs
import { readFileSync } from "fs";
import { createSequenceClassifier } from "../extension/inference.js";

const spec = JSON.parse(
  readFileSync(new URL("../extension/model/words.json", import.meta.url)));
const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/words_io.json", import.meta.url)));

const predict = createSequenceClassifier(spec);

let maxDiff = 0;
fixture.inputs.forEach((seq, r) => {
  const actual = predict(seq);
  fixture.expected[r].forEach((exp, c) => {
    maxDiff = Math.max(maxDiff, Math.abs(actual[c] - exp));
  });
});

if (maxDiff > 1e-4) {
  console.error(`FAIL: JS LSTM diverges from Keras (max diff ${maxDiff})`);
  process.exit(1);
}
console.log(`PASS: JS LSTM matches Keras on ${fixture.inputs.length} sequences ` +
            `(max diff ${maxDiff.toExponential(2)}, ${spec.labels.length} words)`);

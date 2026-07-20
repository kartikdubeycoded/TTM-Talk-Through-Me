// Unit tests for the "writer" layer (extension/writer.js) — the context-aware
// sentence layer of the eyes+writer pipeline. It wraps the pure rule-based
// assembler with a rolling conversation memory and an OPTIONAL on-device LLM
// refiner (Chrome Prompt API / Gemini Nano). The rule-based path must always
// work; the LLM is a best-effort polish that can never break captions.
//
// DOM-free + LLM-free: the refiner is dependency-injected so this runs in Node.
import { createWriter } from "../extension/writer.js";

let failures = 0;
function check(name, actual, expected) {
  if (actual !== expected) {
    console.error(
      `FAIL: ${name}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`
    );
    failures++;
  }
}

function fakeRefiner(fn) {
  return {
    available: true,
    calls: [],
    async refine(raw, ctx) {
      this.calls.push({ raw, ctx });
      return fn(raw, ctx);
    },
  };
}

// --- with no refiner, refined() falls back to the rule-based text ---
{
  const w = createWriter();
  w.addSignedWord("hello");
  w.addSpelledWord("world");
  check("rule-based text is built like the assembler", w.text, "Hello world");
  check("refined() with no LLM === rule text", await w.refined(), "Hello world");
}

// --- a refiner polishes the raw text and receives conversation context ---
{
  const r = fakeRefiner((raw) => `POLISHED: ${raw}`);
  const w = createWriter({ refiner: r });
  w.addSignedWord("hello");
  w.endSentence();               // "Hello." -> remembered
  w.addSignedWord("why");
  check("refined() returns the LLM output", await w.refined(), "POLISHED: Hello. Why");
  check("refiner saw the raw rule text", r.calls[0].raw, "Hello. Why");
  check("refiner got the user's history as context", r.calls[0].ctx.history[0], "Hello.");
}

// --- memory is a rolling window capped at memorySize ---
{
  const w = createWriter({ memorySize: 2 });
  for (const s of ["a", "b", "c"]) { w.addSpelledWord(s); w.endSentence(); }
  check("memory keeps only the last N finished sentences", w.memory.length, 2);
  check("memory dropped the oldest", w.memory[0], "B.");
  check("memory kept the newest", w.memory[1], "C.");
}

// --- a refiner that throws must fall back to rule text, not crash captions ---
{
  const r = { available: true, async refine() { throw new Error("nano exploded"); } };
  const w = createWriter({ refiner: r });
  w.addSignedWord("water");
  check("refiner error falls back to rule text", await w.refined(), "Water");
}

// --- a refiner returning empty/whitespace also falls back ---
{
  const r = fakeRefiner(() => "   ");
  const w = createWriter({ refiner: r });
  w.addSignedWord("food");
  check("empty LLM output falls back to rule text", await w.refined(), "Food");
}

// --- the other party's captions are passed as external context ---
{
  const r = fakeRefiner((raw) => raw);
  const w = createWriter({ refiner: r });
  w.setExternalContext(["Are you coming?"]);
  w.addSignedWord("yes");
  await w.refined();
  check("external context reaches the refiner", r.calls[0].ctx.otherParty[0], "Are you coming?");
}

// --- clear() resets both the sentence and the memory ---
{
  const w = createWriter();
  w.addSpelledWord("hi"); w.endSentence();
  w.clear();
  check("clear resets text", w.text, "");
  check("clear resets memory", w.memory.length, 0);
}

if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log("PASS: writer layer — rule fallback, LLM refine, rolling memory, error safety");

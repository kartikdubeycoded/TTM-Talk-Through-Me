// Unit tests for the sentence assembler (extension/assembler.js) — the pure
// logic that turns recognized tokens into readable sentences. DOM-free so it
// runs in Node; content.js owns the timing and rendering.
import { createAssembler, autocorrect } from "../extension/assembler.js";

let failures = 0;
function check(name, actual, expected) {
  if (actual !== expected) {
    console.error(
      `FAIL: ${name}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`
    );
    failures++;
  }
}

// --- capitalization + terminal punctuation ---
{
  const a = createAssembler();
  a.addSpelledWord("hello");
  a.addSpelledWord("world");
  a.endSentence();
  check("spelled words -> capitalized sentence with period", a.text, "Hello world.");
}

// --- question words end with '?' ---
{
  const a = createAssembler();
  a.addSignedWord("why");
  a.addSpelledWord("not");
  a.endSentence();
  check("sentence starting with a question word ends in '?'", a.text, "Why not?");
}

// --- in-progress sentence: capitalized, no terminal punctuation yet ---
{
  const a = createAssembler();
  a.addSpelledWord("hello");
  check("in-progress sentence is capitalized but unterminated", a.text, "Hello");
}

// --- multiple sentences are joined with a space ---
{
  const a = createAssembler();
  a.addSpelledWord("hi");
  a.endSentence();
  a.addSpelledWord("bye");
  a.endSentence();
  check("two finished sentences join cleanly", a.text, "Hi. Bye.");
}

// --- autocorrect fixes a single-edit fingerspell slip, leaves valid words ---
check("autocorrect fixes a one-edit slip ('watar' -> 'water')", autocorrect("watar"), "water");
check("autocorrect leaves a valid word unchanged", autocorrect("water"), "water");
check("autocorrect leaves an unknown word unchanged", autocorrect("zzzzz"), "zzzzz");
check("autocorrect won't touch very short words", autocorrect("go"), "go");

// --- signed (model-vocab) words are trusted and never autocorrected ---
{
  const a = createAssembler();
  a.addSignedWord("dad"); // near bad/mad/sad — must NOT be rewritten
  a.endSentence();
  check("signed word is not autocorrected", a.text, "Dad.");
}

// --- backspace removes the last word; clear resets everything ---
{
  const a = createAssembler();
  a.addSpelledWord("hello");
  a.addSpelledWord("world");
  a.backspace();
  check("backspace removes the last word", a.text, "Hello");
  a.clear();
  check("clear resets to empty", a.text, "");
}

if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log("PASS: sentence assembler — capitalization, punctuation, autocorrect, backspace");

// Unit tests for the fusion referee (extension/fusion.js) — the T18 precedence
// rule that stops the word LSTM from injecting spurious whole words while the
// user is deliberately fingerspelling. Pure + DOM-free, so it runs in Node.
import {
  wordSignAllowed,
  FINGERSPELL_LOCKOUT_MS,
  WORD_OVERRIDE_CONFIDENCE,
} from "../extension/fusion.js";

let failures = 0;
function check(name, actual, expected) {
  if (actual !== expected) {
    console.error(
      `FAIL: ${name}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`
    );
    failures++;
  }
}

// --- not fingerspelling: any word above the normal threshold is allowed ---
{
  check(
    "empty buffer + long quiet → word allowed",
    wordSignAllowed({ bufferLength: 0, msSinceLastLetter: 9999, wordConfidence: 0.86 }),
    true
  );
}

// --- open letter buffer: the word model is muzzled unless near-certain ---
{
  check(
    "mid-spell (buffer open) + ordinary word conf → BLOCKED",
    wordSignAllowed({ bufferLength: 3, msSinceLastLetter: 100, wordConfidence: 0.9 }),
    false
  );
  check(
    "mid-spell (buffer open) + overwhelming word conf → allowed to interrupt",
    wordSignAllowed({ bufferLength: 3, msSinceLastLetter: 100, wordConfidence: WORD_OVERRIDE_CONFIDENCE }),
    true
  );
}

// --- just finished a letter (buffer cleared but still in the lockout window) ---
{
  check(
    "buffer empty but a letter locked <lockout ago → still BLOCKED",
    wordSignAllowed({ bufferLength: 0, msSinceLastLetter: FINGERSPELL_LOCKOUT_MS - 1, wordConfidence: 0.9 }),
    false
  );
  check(
    "buffer empty and lockout elapsed → word allowed again",
    wordSignAllowed({ bufferLength: 0, msSinceLastLetter: FINGERSPELL_LOCKOUT_MS, wordConfidence: 0.86 }),
    true
  );
}

if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log("PASS: fusion referee — word signs muzzled during active fingerspelling (T18)");

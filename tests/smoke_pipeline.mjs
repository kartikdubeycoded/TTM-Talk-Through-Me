// End-to-end proof of the caption pipeline WITHOUT a browser: it replays a
// scripted recognition stream through the exact same writer + fusion logic that
// content.js now uses live, and asserts the final caption. This is the concrete
// demonstration that A (writer wiring) and B (fusion referee) work together:
//   raw model tokens  ->  "Hi. How are you?"
// and a spurious word offered mid-spell is refused.
import { createWriter } from "../extension/writer.js";
import { wordSignAllowed } from "../extension/fusion.js";

let failures = 0;
function check(name, actual, expected) {
  if (actual !== expected) {
    console.error(`FAIL: ${name}\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(actual)}`);
    failures++;
  }
}

// A tiny stand-in for content.js's controller: same decisions, no DOM/timers.
function runStream(events) {
  const writer = createWriter();
  let buffer = "";
  let msSinceLastLetter = 99999; // "not spelling" until the first letter
  const log = [];

  for (const ev of events) {
    if (ev.letter) {                 // a locked fingerspelling letter
      buffer += ev.letter;
      msSinceLastLetter = 0;
    } else if (ev.word) {            // the word LSTM offers a whole word
      const allowed = wordSignAllowed({
        bufferLength: buffer.length,
        msSinceLastLetter,
        wordConfidence: ev.conf,
      });
      log.push(`word "${ev.word}" (${ev.conf}) -> ${allowed ? "COMMIT" : "blocked"}`);
      if (allowed) { buffer = ""; writer.addSignedWord(ev.word); }
    } else if (ev.pauseWord) {        // a pause: the buffer commits as a spelled word
      if (buffer) { writer.addSpelledWord(buffer); buffer = ""; }
      msSinceLastLetter = 99999;
    } else if (ev.endSentence) {      // a long pause: close + punctuate
      if (buffer) { writer.addSpelledWord(buffer); buffer = ""; }
      writer.endSentence();
      msSinceLastLetter = 99999;
    }
  }
  return { text: writer.text, log };
}

// Scenario: user fingerspells "hi", then signs the words "how are you".
// While spelling H-I, the word model wrongly offers "kite" at 0.90 — it must be
// refused (mid-spell, below the 0.97 override). After the spell commits, the
// three signed words assemble into a question with a "?".
const { text, log } = runStream([
  { letter: "h" },
  { word: "kite", conf: 0.90 },   // spurious, mid-spell -> must be blocked
  { letter: "i" },
  { endSentence: true },          // "hi" commits, long pause closes it -> "Hi."
  { word: "how", conf: 0.95 },   // now allowed (not spelling)
  { word: "are", conf: 0.95 },
  { word: "you", conf: 0.95 },
  { endSentence: true },
]);

check("spurious 'kite' was refused mid-spell", log[0], 'word "kite" (0.9) -> blocked');
check("final caption reads as a clean, punctuated sentence", text, "Hi. How are you?");

if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log("PASS: end-to-end pipeline — tokens -> 'Hi. How are you?', fingerspell not hijacked");
console.log("  fusion decisions:", log.join(" | "));

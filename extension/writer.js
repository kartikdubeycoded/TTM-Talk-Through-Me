// writer.js — the "writer" half of the eyes+writer pipeline.
//
// The recognition model (the "eyes") produces a stream of words. The writer
// turns that stream into readable, context-aware English. It has two backends:
//
//   1. Rule-based (extension/assembler.js) — capitalization, punctuation,
//      conservative fingerspell autocorrect. ALWAYS available, synchronous,
//      never wrong-but-confident. This is the floor.
//   2. On-device LLM (Chrome Prompt API / Gemini Nano) — an OPTIONAL refiner
//      that rephrases the rule text into more natural English using the
//      conversation so far. Best-effort only.
//
// Design = strategy + dependency injection, so the orchestration (memory,
// fallback, error-safety) is testable in Node without any LLM. createWriter
// takes an injected `refiner`; the browser supplies the real one via
// createGeminiNanoRefiner(), tests inject a fake.
//
// Hard rule: the LLM can NEVER break or block captions. If it is absent, errors,
// or returns nothing, refined() falls back to the rule-based text. (A smooth but
// wrong sentence is worse than an honest rough one — see log.md 20 Jul.)

import { createAssembler } from "./assembler.js";

export function createWriter({ refiner = null, memorySize = 6 } = {}) {
  const assembler = createAssembler();
  const history = [];          // this user's recent finished sentences (memory)
  let otherParty = [];         // the other side's recent captions (call context)

  function remember(sentence) {
    history.push(sentence);
    while (history.length > memorySize) history.shift();
  }

  return {
    addSpelledWord(w) { assembler.addSpelledWord(w); },
    addSignedWord(w) { assembler.addSignedWord(w); },
    backspace() { assembler.backspace(); },

    endSentence() {
      const finished = assembler.endSentence();
      if (finished) remember(finished);
      return finished;
    },

    // The other participant's captions, so the refiner can make sense of the
    // ongoing conversation (kept to the same rolling window).
    setExternalContext(lines) {
      otherParty = (lines || []).slice(-memorySize);
    },

    // Rule-based text — synchronous, always available.
    get text() { return assembler.text; },

    // Best-effort LLM polish; falls back to rule text on absence/empty/error.
    async refined() {
      const raw = assembler.text;
      if (!refiner || !refiner.available || raw.trim() === "") return raw;
      try {
        const out = await refiner.refine(raw, {
          history: history.slice(),
          otherParty: otherParty.slice(),
        });
        return out && out.trim() ? out : raw;
      } catch {
        return raw; // an LLM hiccup must never break captions
      }
    },

    clear() {
      assembler.clear();
      history.length = 0;
    },

    get memory() { return history.slice(); },
  };
}

// --- On-device LLM adapter (browser only) --------------------------------
// Feature-detects Chrome's Prompt API (Gemini Nano) and returns a refiner, or
// null when it isn't available — which is the common case, INCLUDING Brave.
// Callers must handle null (createWriter does, via the rule-based fallback).
//
// NOTE: this cannot be unit-tested in Node (there is no on-device model), so it
// is kept tiny, guarded, and null-returning on any failure. It needs live
// verification on a Gemini-Nano-capable Chrome before we trust it.
const SYSTEM_PROMPT =
  "You clean up sign-language recognition output into one natural English " +
  "sentence. Rules: stay faithful — never add facts, names, or meaning that " +
  "aren't in the words given. Fix grammar, order, and punctuation only. Use the " +
  "conversation context only to disambiguate, not to invent. Reply with just the " +
  "sentence.";

export async function createGeminiNanoRefiner() {
  try {
    const LM = (typeof self !== "undefined" && self.LanguageModel) || null;
    if (!LM) return null;
    if (LM.availability) {
      const status = await LM.availability();
      if (status === "unavailable") return null;
    }
    const session = await LM.create({
      initialPrompts: [{ role: "system", content: SYSTEM_PROMPT }],
    });
    return {
      available: true,
      async refine(raw, ctx) {
        const context = [
          ...(ctx.otherParty || []).map((l) => `Them: ${l}`),
          ...(ctx.history || []).map((l) => `Me: ${l}`),
        ].join("\n");
        const prompt =
          (context ? `Conversation so far:\n${context}\n\n` : "") +
          `Rough signed words: ${raw}\nClean sentence:`;
        const out = await session.prompt(prompt);
        return (out || "").trim();
      },
    };
  } catch {
    return null; // any failure -> no LLM, rule-based fallback takes over
  }
}

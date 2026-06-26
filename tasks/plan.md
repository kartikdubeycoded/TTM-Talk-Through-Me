---
name: plan
description: Implementation plan for Sign-to-Text (TTM — Talk Through Me). Phases 1–5 shipped; this refresh targets the reading experience — assembling fingerspelled letters AND whole-word signs into clean sentences/paragraphs, best-effort.
type: reference
status: refreshed 2026-06-26 — Phases 1–5 done, planning Phase 5b→6
last-updated: 2026-06-26
---

# Implementation Plan: Sign-to-Text / TTM

## North star (reaffirmed 2026-06-26)

Watch a signer's webcam → read **both** fingerspelled letters and whole-word
signs → assemble them into clean, readable **sentences and paragraphs**,
best-effort ("whatever we get"). Everything client-side, no server. Repo is
already public as **TTM — Talk Through Me**. The v1 finish line (Chrome Web Store
vs. polished load-unpacked repo) is **deliberately left open** — resolved at the
Phase 6 checkpoint.

Source material: [[../firstread]], [[../in-depth-study]]. Honest numbers and the
debugging history live in [[../log]].

---

## Current state — what is actually DONE (grounded in git, not the old plan)

The original 16-task plan (T1–T16) is **complete through T14**. Reconciliation:

| Phase | Tasks | Status | Honest result |
|-------|-------|--------|---------------|
| 1 Foundation | T1 sandbox+repo, T2 see-the-dots | ✅ done | Public repo `kartikdubeycoded/TTM-Talk-Through-Me` |
| 2 Alphabet model | T3 ingest, T4 augment, T5 train, T6 export | ✅ done | **93.6%** held-out signer; real Kaggle + Lexset synthetic; 3D-rotation aug |
| 3 Extension MVP | T7 MV3, T8 webcam+MediaPipe, T9 predict | ✅ done | Offscreen-document architecture (WASM/CSP solved); plain-JS inference (no TF.js), verified to 1e-7 |
| 4 On the call | T10 overlay+health dot, T11 sentence builder | ✅ done | Live captions on real Meet; lock-in gate, debounce, velocity spacing, backspace/clear |
| 5 Word signs | T12 ingest, T13 train+export, T14 integrate | ✅ done | **76.1%** held-out; 40-word vocab; plain-JS LSTM verified to 2e-6; rolling 30-frame buffer wired live |

**What this leaves unfinished toward the north star** (the real gaps):

1. **Words aren't trustworthy yet.** `word_features.py` uses hand-only features and
   *discards* the lips/pose landmarks GISLR provides. Face-anchored signs therefore
   fail badly — `go` 0.23, `fine` 0.26, `dad` 0.48 (`dad`=forehead vs `mom`=chin
   differ only in *where* the hand is, which hand-only features can't see). A word
   model this weak corrupts the sentence stream rather than helping it.
2. **The assembly layer is raw.** `content.js` concatenates uppercase tokens
   (`HELLO HOW ARE YOU`) — no capitalization, punctuation, sentence boundaries, or
   autocorrect. Not yet "sentences and paragraphs."
3. **Fusion is untuned.** The word LSTM runs continuously and can fire spuriously
   during fingerspelling; precedence rules exist but aren't validated.
4. **No honest combined live test.** Neither letters nor words have a reported
   end-to-end live hit-rate on a real call since the word model landed.

---

## Architecture decisions (unchanged unless noted)

- Two models: static alphabet classifier + temporal word LSTM. ✅ both shipped.
- Skeletal compression — all ML runs on landmarks, never pixels. ✅
- Client-side only; plain-JS inference (TF.js dropped — MV3 forbids its `eval`). ✅
- MediaPipe Tasks `HandLandmarker`, MV3, own webcam via `getUserMedia()`. ✅
- Offscreen-document architecture: camera+MediaPipe+inference in an invisible page;
  content script is UI-only; background relays frames. ✅
- **The Python feature pipeline and the live JS pipeline MUST stay byte-identical.**
  This is the dominant constraint on the word-accuracy work below: any feature we
  add in training must be computable live in `offscreen.js`.
- **NEW DECISION (Phase 5b) — how to fix face-anchored signs:**
  - **Option A (recommended, default): hand-only absolute-position proxy.** Add the
    hand's position within the camera frame (e.g. initial/mean wrist x,y) as extra
    features. "Hand near top of frame" ≈ forehead. *Live-computable today* — no
    architecture change, `HandLandmarker` already gives frame coords.
  - **Option B (escalation, only if A is insufficient): real face landmarks.** Switch
    the live pipeline to `HolisticLandmarker` / add `FaceLandmarker`, use true
    wrist-relative-to-face features. More accurate, but heavier and a real
    architecture change (second model live, latency budget at risk).
  - Plan: **try A first, measure on `go`/`fine`/`dad`, escalate to B only if A fails.**

---

## Dependency graph (remaining work)

```
[DONE: T1–T14 — letters 93.6% live, words 76.1% wired in]
        │
        ├─ Phase 5b: trustworthy words ──────────────────────┐
        │     T15 positional feature (Python→retrain→JS→live) │
        │            │                                        │
        │            ▼                                        │
        │     T16 vocabulary honesty pass (drop weak words)   │
        │                                                     │
        ├─ Phase 5c: readable output (parallel to 5b) ────────┤
        │     T17 sentence assembler (cap/punct/autocorrect)  │
        │            │                                        │
        │            ▼                                        │
        │     T18 fusion tuning (letters ↔ words) ◄───────────┘
        │            │   (needs T15 conf meaningful + T17 assembler)
        │            ▼
        │     Phase 5d checkpoint:
        │     T19 honest combined LIVE test on real Meet  ◄── Katti runs + reports
        │            │
        │            ▼
        └─ Phase 6: polish + publish (finish line TBD)
              T20 settings/options UI  →  T21 chat-injection hardening
                                          →  T22 README/TTM + privacy + license + ship decision
```

Each task is a **vertical slice** that leaves the extension runnable/demoable.

---

## Remaining tasks

### Phase 5b — Make words trustworthy

## T15: Positional feature for face-anchored signs

**Description:** Fix the documented failure (`go`/`fine`/`dad`) by giving the word
model a sense of *where* the hand is, not just its shape and motion. Start with
**Option A** (hand-only absolute frame position — live-computable, no architecture
change). Full vertical slice through every layer that touches word features:
`word_features.py` → `ingest_words.py` → retrain `train_words.py` → re-export
`export_words.py` → mirror the feature in `offscreen.js` → live.

**Acceptance criteria:**
- [ ] `go`, `fine`, `dad` each improve materially over baseline (0.23 / 0.26 / 0.48).
- [ ] Overall held-out-signer accuracy ≥ 76.1% (no net regression).
- [ ] `offscreen.js` computes the new feature identically to Python (same units, same order).
- [ ] `tests/smoke_words.mjs` still passes; JS output matches Keras to < 1e-4.
- [ ] If Option A does NOT lift the face-anchored words, the result is recorded and
  T15b (Option B / face landmarks) is opened — **measured, not assumed.**

**Verification:**
- [ ] Re-run `train_words.py`; inspect the per-word table (worst-first) — face-anchored words moved.
- [ ] Run `node tests/smoke_words.mjs` — passes against the new `words.json`.
- [ ] Live: sign `dad` vs `mom`; they no longer collapse to the same prediction.

**Dependencies:** none (builds on shipped T12–T14)
**Files likely touched:** `pipeline/word_features.py`, `pipeline/ingest_words.py`,
`training/train_words.py`, `training/export_words.py`, `tests/test_word_features.py`,
`extension/offscreen.js`, `extension/model/words.json`
**Scope:** M

## T16: Vocabulary honesty pass

**Description:** After T15, some words may still sit below a usable bar. Don't ship
signs the model can't read — they only corrupt sentences. Set a quality floor; drop
or flag words below it; regenerate `labels.json` and retrain on the kept vocab.

**Acceptance criteria:**
- [ ] A documented quality floor (e.g. per-word held-out acc ≥ 0.5) is applied.
- [ ] Shipped `labels.json` contains only words at/above the floor.
- [ ] The dropped words are listed in the log with their scores (honest record).

**Verification:**
- [ ] Retrained model's per-word table shows no shipped word below the floor.
- [ ] Live: every word in the shipped vocab is demonstrably recognizable.

**Dependencies:** T15
**Files likely touched:** `pipeline/ingest_words.py` (VOCAB), `models/words.h5`,
`extension/model/words.json`
**Scope:** S

### Checkpoint: Words trustworthy
- [ ] Honest per-word accuracy known; shipped vocab is all above the floor.
- [ ] **Teach-back:** Katti explains *why* hand-only features can't tell `dad` from
  `mom`, and what feature fixed it. **Human review before merging into the stream.**

---

### Phase 5c — Readable sentences (the assembly layer)

## T17: Sentence assembler

**Description:** Turn the token firehose into readable prose. In `content.js`
(pure JS, no model change): sentence-case capitalization, sentence-boundary +
end punctuation heuristic (long pause / explicit terminator sign), word-level
de-duplication, and a light dictionary autocorrect for fingerspelled words.
Output should read `Hello, how are you?` not `HELLO HOW ARE YOU`.

**Acceptance criteria:**
- [ ] Output is sentence-cased with terminal punctuation, not all-caps tokens.
- [ ] A long pause closes a sentence; the next token starts capitalized.
- [ ] Fingerspelled near-misses are softened by a small dictionary autocorrect.
- [ ] Backspace/clear/correction still work on the assembled text.

**Verification:**
- [ ] New JS unit test (extend the `smoke_*.mjs` pattern): feed a scripted prediction
  stream → assert the exact assembled sentence string.
- [ ] Manual: spell two words + sign one word with pauses → reads as a clean sentence.

**Dependencies:** none (operates on the existing T14 stream)
**Files likely touched:** `extension/content.js`, a new `tests/smoke_assembler.mjs`
**Scope:** M

## T18: Fusion tuning (letters ↔ words)

**Description:** Make the two recognizers cooperate instead of fighting. Define and
tune clear precedence: the word LSTM must not inject spurious words during
deliberate fingerspelling, and a recognized word sign must cleanly replace the
stray letters its motion produces. Uses the now-meaningful word confidence from T15.

**Acceptance criteria:**
- [ ] Fingerspelling a name (e.g. K-A-T-T-I) produces no spurious whole-word injection.
- [ ] Signing a known word leaves no letter litter in the buffer.
- [ ] Precedence rules are explicit in code comments and tunable via the existing thresholds.

**Verification:**
- [ ] Live: fingerspell a word the LSTM doesn't know → letters only, no false word.
- [ ] Live: sign a known word mid-sentence → word appears, buffer clears.

**Dependencies:** T15 (meaningful word confidence), T17 (assembler to write into)
**Files likely touched:** `extension/content.js`, `extension/offscreen.js`
**Scope:** M

### Checkpoint: Readable, fused output
- [ ] Letters + words assemble into clean sentences without fighting each other.
- [ ] **Teach-back:** Katti explains the precedence rule (when a word beats letters)
  and the pause→punctuation rule. **Human review before the live test.**

---

### Phase 5d — Honest combined live test

## T19: End-to-end live validation on a real Meet

**Description:** The honest test. **Katti runs it himself** (teaching mandate —
operate + report, not watch): join a real Meet, enable TTM, fingerspell + sign,
and record per-letter and per-word hit rates plus how readable the assembled
paragraph is. This gates Phase 6.

**Acceptance criteria:**
- [ ] A short recorded run exists (notes or capture) with honest hit-rate numbers.
- [ ] Known failure modes are listed (which letters/words/lighting hurt).
- [ ] Latency stays inside the live budget (no visible lag).

**Verification:**
- [ ] The run is logged in [[../log]] with the real numbers, not estimates.

**Dependencies:** T15, T16, T17, T18
**Files likely touched:** `log.md` (results); fixes spun out as needed
**Scope:** S

### Checkpoint: Usable reading experience
- [ ] Live, on a real call, TTM produces readable sentences from letters + words.
- [ ] **Human review: is this good enough to be the v1 demo?** Decide here whether
  more accuracy work is needed before Phase 6, or we polish and ship.

---

### Phase 6 — Polish + publish (finish line TBD)

## T20: Settings / options UI

**Description:** An options page wiring the settings already read from
`chrome.storage` (confidence threshold, spacing delay, vocab on/off, chat-injection
toggle). Persist across sessions.

**Acceptance criteria:**
- [ ] Options page edits threshold, spacing, vocab on/off; values persist.
- [ ] Chat-injection toggle is present and **default-off**.

**Verification:**
- [ ] Change a setting, reload, confirm it persisted and took effect live.

**Dependencies:** T19 sign-off
**Files likely touched:** `extension/options.html`, `extension/options.js`,
`extension/popup.js`, `extension/manifest.json`
**Scope:** M

## T21: Chat-injection hardening

**Description:** The injection code exists (`injectIntoChat` in `content.js`) but is
untested. Make it default-off, fire only on explicit user action, and fail
gracefully when the Meet/Teams selector isn't found (no crash).

**Acceptance criteria:**
- [ ] Off by default; only fires on a user click.
- [ ] Missing chat selector → silent no-op, no exception.

**Verification:**
- [ ] Live: enable + click → text in Meet chat; disable → nothing auto-posts.

**Dependencies:** T20
**Files likely touched:** `extension/content.js`, `extension/options.js`
**Scope:** S

## T22: README rename to TTM + privacy + license + ship decision

**Description:** Rename the README from "Sign-to-Text" to TTM; write an honest
README (what it does, install, **honest accuracy limits**, respectful language about
the Deaf community), a privacy note (webcam never leaves the device), resolve the
**shipped-model license** question (Kaggle ASL Alphabet is GPL-2.0; Lexset synthetic
+ GISLR terms must be checked before any store listing), and make the **finish-line
decision**: polished load-unpacked repo vs. Chrome Web Store submission.

**Acceptance criteria:**
- [ ] README/TTM + PRIVACY complete and honest about limits.
- [ ] Shipped-model license resolved and documented.
- [ ] Finish-line decided and recorded; if "store," listing submitted.

**Verification:**
- [ ] Fresh load-unpacked from a clean checkout → core flow works.

**Dependencies:** T21
**Files likely touched:** `README.md`, `PRIVACY.md`, `datasets.md` (license note)
**Scope:** M

### Checkpoint: Complete
- [ ] v1 finish line met (per the T22 decision); demo reproducible from a clean clone.
- [ ] Final teach-back: Katti walks the whole pipeline end-to-end unaided.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Option A (hand-only position) doesn't fix face-anchored signs | Med | Measure on `go`/`fine`/`dad` first; escalate to face landmarks (Option B) only if proven needed — don't pre-build the heavy path. |
| Adding face landmarks (Option B) blows the latency budget | Med | Word LSTM already runs at ~2.5 Hz, not 20; profile before committing; keep `HandLandmarker`-only as the fallback. |
| Autocorrect "fixes" correct fingerspelling into wrong words | Med | Keep the dictionary small + high-confidence-only; always allow manual backspace/correction. |
| Re-planning becomes avoidance (constitution anti-patterns #1/#7) | High | This refresh is scoped to **6 remaining tasks**, not a re-architecture. T15 is concrete and starts immediately on confirmation. |
| Shipped-model license blocks a store listing | Med | Resolve in T22 before any submission; load-unpacked repo (already public) is the safe fallback finish line. |

## Open questions

- **T15 outcome:** does the cheap hand-only positional feature fix face-anchored
  signs, or do we need real face landmarks? (Resolved by measurement in T15.)
- **Final vocabulary** after the T16 honesty pass — which of the 40 words survive the floor?
- **v1 finish line** — store vs. polished unpacked repo. (Resolved at the Phase 5d / T22 checkpoint.)

## Parallelization

- **5b (T15→T16)** and **5c T17** are independent — word-accuracy work (Python +
  retrain) can run alongside the pure-JS sentence assembler.
- **T18 fusion** is the join point: it needs T15's meaningful confidence AND T17's assembler.
- **Coordination constraint (still dominant):** any feature added in `word_features.py`
  must be mirrored exactly in `offscreen.js`. Define once, port carefully.

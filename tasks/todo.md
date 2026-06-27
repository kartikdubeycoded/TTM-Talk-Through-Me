---
name: todo
description: Flat, checkable task list for Sign-to-Text / TTM. Phases 1–5 shipped; remaining work targets readable sentences from trustworthy letters + words. Full detail in [[plan]].
type: reference
status: refreshed 2026-06-26 — Phases 1–5 done, working Phase 5b→6
last-updated: 2026-06-26
---

# Sign-to-Text / TTM — TODO

Smallest working thing first. Don't start a phase until the previous one runs.
Full acceptance criteria + verification: [[plan]].

## DONE — Phases 1–5 (T1–T14)
- [x] **Phase 1** Foundation — public repo `TTM-Talk-Through-Me`, live landmark tracking
- [x] **Phase 2** Alphabet model — **93.6%** held-out signer; real + synthetic data; 3D-rotation aug
- [x] **Phase 3** Extension MVP — offscreen-document architecture; plain-JS inference (no TF.js)
- [x] **Phase 4** On the call — live overlay + health dot; lock-in gate, debounce, velocity spacing
- [x] **Phase 5** Word signs — **76.1%** held-out; 40-word vocab; LSTM verified to 2e-6; wired live

## Phase 5b: Make words trustworthy
- [x] **T15** Positional feature (abs. wrist x,y) — dad 0.48→0.89, fine 0.26→0.64, held-out 76.1→78.6%. Option A worked; B (face landmarks) not needed. Commit `053be30`.
- [x] **T16** Vocabulary honesty pass — floor 0.5, dropped `go` (0.15), 39 words → **79.4%** held-out. Lowest survivor `fine` 0.51. Commit `dec7f1a`.
- [~] **Checkpoint:** shipped vocab all above floor ✅; **teach-back + human review PENDING** before Phase 5c.

## Phase 5c: Readable sentences
- [~] **T17** Sentence assembler — slice 1 ✅ pure tested module `extension/assembler.js` (cap, punctuation, `?`, autocorrect, backspace; commit `135858b`). **Slice 2 = wire into content.js + pause→sentence timing — needs live reload, held for next session.**
- [ ] **T18** Fusion tuning — letters ↔ words don't fight; precedence explicit + tunable — M
- [ ] **Checkpoint:** clean fused sentences; teach-back on precedence + pause→punctuation rules. **Human review.**

> **Next session starts with LIVE TEST RUNS (T19 pulled forward):** reload the extension on a real Meet, sign letters + the 39 words, honestly record per-letter/per-word hit rates. The live numbers decide what to fix before finishing 5c/5d. Also added this session: `tests/smoke_features.mjs` (JS↔Python feature parity) + `extension/features.js` (shared module).

## Phase 5d: Honest combined live test
- [ ] **T19** End-to-end live test on real Meet — Katti runs + logs honest per-letter/per-word hit rates — S
- [ ] **Checkpoint:** readable reading experience live. **Human review: good enough for v1 demo?**

## Phase 6: Polish + publish (finish line TBD)
- [ ] **T20** Settings / options UI (threshold, spacing, vocab on/off, chat toggle default-off) — M
- [ ] **T21** Chat-injection hardening — default-off, click-only, graceful failure — S
- [ ] **T22** README→TTM + privacy + license resolution + finish-line decision — M
- [ ] **Checkpoint:** v1 finish line met; final full-pipeline teach-back.

---

## Decisions locked
- [x] v1 language = **ASL** (multi-language expansion later — see [[../datasets]])
- [x] **Public datasets only** — no self-capture
- [x] Browser = **Chrome / Brave** (both Chromium)
- [x] Two models, client-side, offscreen-document architecture, plain-JS inference
- [x] North star = readable sentences/paragraphs from letters **and** words (2026-06-26)

## Still open
- [ ] T15: does hand-only position fix face-anchored signs, or do we need face landmarks? (measure)
- [ ] Final vocabulary after the T16 honesty floor
- [ ] **v1 finish line** — Chrome Web Store vs. polished load-unpacked repo (decide at Phase 5d/T22)
- [ ] Shipped-model license (Kaggle ASL Alphabet is GPL-2.0; check Lexset + GISLR terms before any store listing)

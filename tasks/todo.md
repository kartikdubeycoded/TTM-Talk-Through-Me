---
name: todo
description: Flat, checkable task list for Sign-to-Text. Full detail (acceptance criteria, verification, files) lives in [[plan]].
type: reference
status: approved — in progress (T1)
last-updated: 2026-06-08
---

# Sign-to-Text — TODO

Smallest working thing first. Don't start a phase until the previous one runs.
Full acceptance criteria + verification for each task: [[plan]].

## Phase 1: Foundation
- [~] **T1** Sandbox + public repo — venv ✅, deps ✅, scaffold ✅ · **git init + commit + GitHub remote remaining** — S
- [ ] **T2** "See the dots": webcam → MediaPipe → 21 landmarks on screen (Python) — S
- [ ] **Checkpoint:** repo backed up, live tracking works, teach-back on landmarks. **Human review.**

## Phase 2: Alphabet data + model (easy ML)
- [ ] **T3** Ingest Kaggle ASL Alphabet → MediaPipe landmarks + normalise (A–Z) — M
- [ ] **T4** Augmentation pipeline (jitter, mirror, rotate) — S
- [ ] **T5** Train alphabet classifier; validate on held-out hands — M
- [ ] **T6** Export model to TensorFlow.js — S
- [ ] **Checkpoint:** honest held-out accuracy, model loads in JS, teach-back on overfitting. **Human review.**

## Phase 3: Extension MVP (alphabet)
- [ ] **T7** Manifest V3 scaffold (popup, background, bundled assets) — M
- [ ] **T8** Webcam + MediaPipe tracking inside the extension — M
- [ ] **T9** Load TFJS model + predict letters in the browser — M
- [ ] **Checkpoint:** first shippable demo (spells from webcam), teach-back on Python→browser. **Human review.**

## Phase 4: On the call
- [ ] **T10** Overlay panel on Google Meet + tracking-health dot — M
- [ ] **T11** Sentence builder (lock-in gate, pause spacing, autocorrect, correction) — M
- [ ] **Checkpoint:** live spaced correctable captions on a real Meet, teach-back on debouncing. **Human review.**

## Phase 5: Word-level signs (hard ML)
- [ ] **T12** Ingest Google `asl-signs` landmark data (250 words) → ~20–50 word subset — M
- [ ] **T13** Train + export the word-level LSTM — M
- [ ] **T14** Integrate word recognition into the extension — M
- [ ] **Checkpoint:** whole-word signing works on a real call, teach-back on temporal models. **Human review.**

## Phase 6: Polish + publish
- [ ] **T15** Settings + optional (default-off) chat-injection toggle — M
- [ ] **T16** README + privacy note + Chrome Web Store publish — M
- [ ] **Checkpoint:** public extension installable; final full-pipeline teach-back.

---

## Decisions locked (2026-06-07)
- [x] v1 language = **ASL** (big multi-language/dataset expansion later — see [[../datasets]])
- [x] **Public datasets only** — no self-capture
- [x] **Parallel** to get-your-knowledge-right
- [x] Browser = **Chrome / Brave** (both Chromium)

## Still open
- [ ] License for the **shipped** model — resolve before Phase 6 (publish). See [[../datasets]] "License reality".
- [ ] Which ~20–50 words from `asl-signs` to ship first?

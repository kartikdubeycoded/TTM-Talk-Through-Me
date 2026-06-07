---
name: plan
description: Implementation plan for Sign-to-Text — staged, vertically-sliced task breakdown to ship a public Chrome extension that translates sign language to live text on Meet/Zoom/Teams.
type: reference
status: approved — in progress (T1)
last-updated: 2026-06-08
---

# Implementation Plan: Sign-to-Text

## Overview

Build a browser-based system that watches a Deaf user's webcam, recognises sign
language, and shows English text live on a video call — first as an on-screen
overlay, later (optionally) injected into the meeting chat. Everything runs
client-side (no server). We ship in stages: a working **fingerspelling alphabet**
extension first (the easy, static ML), then add **word-level signs** (the hard,
temporal ML), then polish and publish.

Source material: [[../firstread]] (original blueprint) and [[../in-depth-study]]
(the corrected, taught version). This plan operationalises the staged build in
§11 of the study.

## Architecture Decisions

- **Two models, not one.** Fingerspelling = a per-frame *static* classifier (easy).
  Word signs = a *temporal* LSTM over 30-frame windows (hard). We build and ship
  the static one first. Rationale: smallest working thing first; learn the full
  pipeline on the easy ML before adding sequence modelling.
- **Skeletal compression is the core.** MediaPipe reduces each frame to 21 hand
  landmarks (63 numbers). All ML runs on those 63 numbers, never raw pixels.
  Rationale: this is the only way to stay fast enough (<100 ms) for a live call.
- **Client-side only.** Train offline in Python; convert to TensorFlow.js; run in
  the browser on the GPU (WebGL/WebGPU). Rationale: free, private (webcam never
  leaves the machine), no backend to maintain.
- **MediaPipe Tasks – Vision** (`HandLandmarker`), **not** the deprecated legacy
  "MediaPipe Hands" solution. **Manifest V3**, not V2.
- **Own webcam via `getUserMedia()`**, not scraping the call's `<video>` elements.
  We translate the *signer themselves*.
- **Overlay is the product; chat-injection is a fragile, toggle-able bonus** built
  last (ToS-grey + breaks on every Meet HTML change).
- **Target sign language for v1: ASL** (confirmed). Big multi-language + multi-dataset
  expansion planned afterward (ISL next, then global) so the data layer scales — see
  [[../datasets]].
- **No self-capture — public datasets only** (confirmed). Phase 2 alphabet from Kaggle
  ASL Alphabet; Phase 5 words from Google `asl-signs` (already MediaPipe landmarks).
- **Parallel to get-your-knowledge-right** (confirmed — deliberate, not a switch).
- **Target browser: Chrome / Brave** (both Chromium — same build).

## Dependency Graph

```
Stage 0: env + git repo (FOUNDATION)
   │
   ├── Python pipeline ───────────────────────────────┐
   │     │                                             │
   │     ▼                                             │
   │   T2 landmark extraction (webcam → 21 dots)       │
   │     │                                             │
   │     ▼                                             │
   │   T3 data collection (alphabet) ──► T4 augment ──► T5 train alphabet model
   │                                                          │
   │                                                          ▼
   │                                                   T6 export to TFJS ──┐
   │                                                                       │
   └── Extension track ──► T7 MV3 scaffold ──► T8 webcam+MediaPipe in ext  │
                                                       │                   │
                                                       ▼                   │
                                            T9 load TFJS model + predict ◄─┘
                                                       │
                                                       ▼
                                            T10 overlay on Meet + health dot
                                                       │
                                                       ▼
                                            T11 sentence builder
                                                       │
                              ┌────────────────────────┤
                              ▼                         ▼
   T12 sequence data (words) ─► T13 train LSTM ─► T14 integrate words
                                                       │
                                                       ▼
                                            T15 settings + chat injection
                                                       │
                                                       ▼
                                            T16 README + publish
```

Order follows the graph bottom-up: foundation → data/model → extension MVP →
on-call overlay → word-level → polish/ship. Each task is a **vertical slice** that
leaves the system in a runnable/demoable state.

---

## Task List

### Phase 1: Foundation

## Task 1: Sandbox + public repo  ⏳ IN PROGRESS

**Description:** Create the project scaffold — a Python virtual environment (venv,
never global), a `.gitignore`, and an initialised git repo pushed to a **public**
GitHub remote (this counts toward the 200-commit target). Note: TensorFlow /
tensorflowjs are deliberately deferred to Phase 2 (avoid heavy/conflicting installs
before they're needed — smallest working thing first).

**Acceptance criteria:**
- [x] `venv/` exists and activates; `pip list` shows **mediapipe, opencv-python, numpy** (TF deferred to Phase 2).
- [x] `.gitignore` excludes `venv/`, `__pycache__/`, large data, model binaries.
- [ ] `git status` is clean on `main`; repo pushed to a **public GitHub remote**. ← *remaining*

**Verification:**
- [x] `python -c "import mediapipe, cv2, numpy"` runs without error (deps installed clean).
- [ ] `git log` shows the initial commit; the repo is visible on GitHub. ← *remaining*

**Progress note (2026-06-08):** venv + core deps done; scaffold files written
(`requirements.txt`, `.gitignore`, `README.md`, plus study/dataset/plan docs).
**Still to do:** `git init -b main` → first commit → create public GitHub repo
(install `gh` OR create on github.com + `git remote add`). `gh` is not installed yet.

**Dependencies:** None
**Files likely touched:** `.gitignore`, `requirements.txt`, `README.md` (stub)
**Estimated scope:** S

## Task 2: "See the dots" — webcam landmark extraction (Python)

**Description:** A Python script that opens the webcam, runs MediaPipe HandLandmarker,
and draws the 21 hand landmarks live on the video window. The first motivating win
and the foundation for all data work. A complete vertical path: webcam → landmarks →
screen.

**Acceptance criteria:**
- [ ] Running the script shows the live webcam with 21 dots overlaid on the hand.
- [ ] Script prints/exposes the 63-number landmark array per frame.
- [ ] Runs at a usable frame rate (≥15 fps) on the dev machine.

**Verification:**
- [ ] Manual: wave hand, dots track fingers in real time.
- [ ] Manual: covering the hand removes the dots (no false detections).

**Dependencies:** Task 1
**Files likely touched:** `pipeline/track_hands.py`
**Estimated scope:** S

### Checkpoint: Foundation
- [ ] venv + public repo working and backed up.
- [ ] Live landmark tracking visible on screen.
- [ ] **Teach-back:** Katti explains, in his words, what a "landmark" is and why we
  use 63 numbers instead of pixels. **Human review before Phase 2.**

---

### Phase 2: Alphabet data + model (the easy ML)

## Task 3: Ingest the public alphabet dataset → landmarks (no self-capture)

**Description:** Download the **Kaggle ASL Alphabet** dataset (permissive license,
safe to ship — see [[../datasets]]), run MediaPipe over the images once to extract
the 21 hand landmarks per image, apply normalisation (wrist-anchor + knuckle-scale),
and save as `.npy` arrays per class (A–Z + space/del/nothing). No webcam capture.
(Sign Language MNIST as an optional faster warm-up.)

**Acceptance criteria:**
- [ ] `data/alphabet/` contains normalised landmark arrays for each class, shape `(N, 63)`.
- [ ] Normalisation verified: same letter at different image scales yields near-identical numbers.
- [ ] Images that MediaPipe can't find a hand in are logged and skipped (not silently dropped).

**Verification:**
- [ ] Manual: re-plot a few landmark arrays; they match the source images.
- [ ] Array shapes consistent; per-class counts reported.

**Dependencies:** Task 2
**Files likely touched:** `pipeline/ingest_alphabet.py`, `pipeline/normalize.py`, `data/alphabet/`
**Estimated scope:** M

## Task 4: Augmentation pipeline

**Description:** A NumPy script that multiplies the dataset via spatial jitter
(±0.01 noise), X-axis mirroring (right→left hand), and small rotation/scale jitter.
Honest expansion of existing data, not fabrication.

**Acceptance criteria:**
- [ ] Augmented dataset is ≥3× the raw size.
- [ ] Mirroring produces valid left-hand variants (verified by re-plotting).
- [ ] Augmentation is reproducible (seeded).

**Verification:**
- [ ] Manual: plot a mirrored sample; it's a clean left-hand mirror of the original.
- [ ] Class balance preserved after augmentation.

**Dependencies:** Task 3
**Files likely touched:** `pipeline/augment.py`
**Estimated scope:** S

## Task 5: Train the alphabet classifier

**Description:** A Keras per-frame classifier (Dense layers + dropout + softmax over
26 classes) on the normalised + augmented landmarks. **Validate on held-out hands**,
not just a random split, so the accuracy number is honest.

**Acceptance criteria:**
- [ ] Model trains and saves `models/alphabet.h5`.
- [ ] Validation accuracy on **held-out signer(s)** reported (the honest number, not train accuracy).
- [ ] Confusion matrix saved so weak letters are visible.

**Verification:**
- [ ] Run training; inspect the held-out accuracy and confusion matrix.
- [ ] Manual: feed a live frame from `track_hands.py`; predicted letter is plausible.

**Dependencies:** Task 4
**Files likely touched:** `training/train_alphabet.py`, `models/alphabet.h5`
**Estimated scope:** M

## Task 6: Export model to TensorFlow.js

**Description:** Convert `alphabet.h5` to browser-ready `model.json` + weight shards
with `tensorflowjs_converter`, placed inside the extension's asset folder. Save the
label map (index → letter).

**Acceptance criteria:**
- [ ] `extension/model/model.json` + weight shards exist.
- [ ] `extension/model/labels.json` maps output index → letter.
- [ ] A tiny Node/TFJS smoke test loads the model and runs one dummy prediction.

**Verification:**
- [ ] Smoke test loads the model without shape/format errors.

**Dependencies:** Task 5
**Files likely touched:** `training/export_tfjs.py`, `extension/model/*`
**Estimated scope:** S

### Checkpoint: Alphabet model
- [ ] Honest held-out accuracy known and acceptable (re-collect data if poor).
- [ ] Model converted and loadable in JS.
- [ ] **Teach-back:** Katti explains why we validate on unseen hands and what
  overfitting is. **Human review before Phase 3.**

---

### Phase 3: Extension MVP (alphabet, on its own page)

## Task 7: Manifest V3 extension scaffold

**Description:** Minimal MV3 extension: `manifest.json` (permissions, web-accessible
resources for WASM/model), a background service worker, a popup with on/off + status.
Loads in `chrome://extensions` in developer mode.

**Acceptance criteria:**
- [ ] Extension loads unpacked with no manifest errors.
- [ ] Popup shows a status line and an enable toggle.
- [ ] MediaPipe WASM + TFJS assets are bundled and declared (no CDN fetch).

**Verification:**
- [ ] Manual: load unpacked; popup opens; no console errors in the service worker.

**Dependencies:** Task 1
**Files likely touched:** `extension/manifest.json`, `extension/background.js`, `extension/popup.html`, `extension/popup.js`
**Estimated scope:** M

## Task 8: Webcam + MediaPipe tracking inside the extension

**Description:** In the extension, open the user's webcam with `getUserMedia()`, run
MediaPipe Tasks `HandLandmarker`, and render the 21 dots in a small preview in the
popup/panel. The browser equivalent of Task 2.

**Acceptance criteria:**
- [ ] Webcam permission is requested and the preview shows live landmarks.
- [ ] Landmarks are normalised in JS identically to the Python pipeline.
- [ ] Loop runs at ≥15 fps without freezing the page.

**Verification:**
- [ ] Manual: dots track the hand live in the extension preview.
- [ ] Manual: JS-normalised numbers match Python output for the same pose (spot check).

**Dependencies:** Task 7
**Files likely touched:** `extension/content.js` (or a module), `extension/track.js`
**Estimated scope:** M

## Task 9: Load TFJS model + predict letters in the browser

**Description:** Load the converted alphabet model, feed normalised landmarks, and
display the predicted letter live in the popup. First end-to-end in-browser ML.

**Acceptance criteria:**
- [ ] Model loads in the extension without error.
- [ ] Holding a letter handshape displays the correct letter most of the time.
- [ ] Confidence score is shown alongside the prediction.

**Verification:**
- [ ] Manual: spell 3–4 letters; correct letters appear with sensible confidence.

**Dependencies:** Task 6, Task 8
**Files likely touched:** `extension/predict.js`, `extension/popup.js`
**Estimated scope:** M

### Checkpoint: First shippable demo
- [ ] The extension reads fingerspelling from the webcam in a real browser.
- [ ] **Teach-back:** Katti explains how a model trained in Python ends up running
  in the browser (export → TFJS → WebGL). **Human review before Phase 4.**

---

### Phase 4: On the call (overlay + sentences)

## Task 10: Overlay panel on Google Meet + tracking-health dot

**Description:** A content script that injects a clean, absolutely-positioned panel
onto a Google Meet page, showing the live predicted letters and a green/red dot for
tracking health (green = hand detected clearly). Reliable core product — no chat
injection yet.

**Acceptance criteria:**
- [ ] Joining a Meet and enabling the extension shows the overlay panel.
- [ ] The health dot turns green when the hand is tracked, red when lost.
- [ ] Predicted text appears live in the panel during a real call.

**Verification:**
- [ ] Manual: in a live Meet, sign letters; panel updates; health dot reflects tracking.

**Dependencies:** Task 9
**Files likely touched:** `extension/content.js`, `extension/overlay.css`
**Estimated scope:** M

## Task 11: Sentence builder (lock-in, spacing, autocorrect)

**Description:** Turn the prediction firehose into clean text: a confidence lock-in
gate (accept a unit only after ~5 consecutive high-confidence frames), pause-based
word spacing (wrist velocity ≈ 0 for ~750 ms → space), and a light n-gram/dictionary
autocorrect + capitalisation. Includes a manual correction affordance (backspace /
edit).

**Acceptance criteria:**
- [ ] Holding one sign emits one letter/word, not a repeated stream.
- [ ] A natural pause inserts a space between words.
- [ ] The user can correct/backspace the output.

**Verification:**
- [ ] Manual: spell two words with a pause between; output is `word1 word2`, not `wwwooorrrd`.

**Dependencies:** Task 10
**Files likely touched:** `extension/sentence.js`, `extension/overlay.js`
**Estimated scope:** M

### Checkpoint: Usable on a real call
- [ ] Live, spaced, correctable captions appear on a real Meet.
- [ ] **Teach-back:** Katti explains debouncing and the pause-based spacing rule.
  **Human review before Phase 5.**

---

### Phase 5: Word-level signs (the hard ML)

## Task 12: Ingest Google `asl-signs` landmark data → word sequences

**Description:** Download Google's **Isolated Sign Language Recognition (`asl-signs`)**
dataset — 250 words, **already MediaPipe Holistic landmarks (parquet)**, so no video
processing. Select the **hand landmarks** out of the 543 Holistic points, re-normalise
to match our pipeline, resample each sequence to 30 frames, and save shaped
`(N, 30, 63)`. Pick a starter subset (~20–50 common words) for v1. Apply temporal
augmentation (frame drop/dup). No self-capture.

**Acceptance criteria:**
- [ ] `data/words/` holds sequences shaped `(N, 30, 63)` for the chosen subset.
- [ ] Hand-landmark selection from the 543-point Holistic format is correct (re-plotted).
- [ ] Temporal augmentation produces fast/slow variants; ≥20 word classes.

**Verification:**
- [ ] Array shapes verified; replay a sequence and confirm it's the intended sign.
- [ ] Normalisation output matches the alphabet pipeline's convention.

**Dependencies:** Task 4 (augmentation patterns), Task 11
**Files likely touched:** `pipeline/ingest_words.py`, `data/words/`
**Estimated scope:** M

## Task 13: Train + export the word-level LSTM

**Description:** Stacked LSTM (+ dropout, softmax over the vocabulary) on the
30-frame sequences; validate on held-out signers; export to TFJS.

**Acceptance criteria:**
- [ ] `models/words.h5` trained; held-out accuracy reported.
- [ ] "GO" vs "COME"-type motion pairs are distinguished.
- [ ] Exported to `extension/model-words/`.

**Verification:**
- [ ] Held-out accuracy + confusion matrix inspected; TFJS smoke test passes.

**Dependencies:** Task 12
**Files likely touched:** `training/train_words.py`, `training/export_tfjs.py`, `extension/model-words/*`
**Estimated scope:** M

## Task 14: Integrate word recognition into the extension

**Description:** Run the rolling 30-frame buffer through the word model in the
browser; merge word predictions with the existing letter/sentence pipeline (mode
switch or combined output).

**Acceptance criteria:**
- [ ] Signing a known word produces that word in the overlay.
- [ ] Fingerspelling still works alongside word signs.
- [ ] Latency stays under the live-call budget (no visible lag).

**Verification:**
- [ ] Manual: sign 3 known words on a real call; correct words appear.

**Dependencies:** Task 13
**Files likely touched:** `extension/predict.js`, `extension/sentence.js`
**Estimated scope:** M

### Checkpoint: Words working
- [ ] Whole-word signing works for the starter vocabulary on a real call.
- [ ] **Teach-back:** Katti explains why words need a temporal model (LSTM) and
  letters don't. **Human review before Phase 6.**

---

### Phase 6: Polish + publish

## Task 15: Settings + chat-injection toggle

**Description:** Settings (vocabulary on/off, sensitivity, language) and the
**optional, default-off** chat-injection feature: find the Meet chat box and post
the translated text on user action. Documented as fragile/best-effort.

**Acceptance criteria:**
- [ ] Settings persist across sessions.
- [ ] Chat injection is off by default and only fires on a user click.
- [ ] Graceful failure if the chat selector isn't found (no crash).

**Verification:**
- [ ] Manual: enable injection, click send; text appears in Meet chat. Disable it; nothing auto-posts.

**Dependencies:** Task 14
**Files likely touched:** `extension/options.html`, `extension/options.js`, `extension/inject.js`
**Estimated scope:** M

## Task 16: README, privacy note, and Chrome Web Store publish

**Description:** Write a real README (what it does, install, limits, respectful
language about the Deaf community), a privacy statement (webcam never leaves the
device), package the extension, and submit to the Chrome Web Store.

**Acceptance criteria:**
- [ ] README + privacy note complete and honest about accuracy limits.
- [ ] Packaged build loads cleanly from a fresh unpack.
- [ ] Store listing submitted.

**Verification:**
- [ ] Fresh-machine install test; core flow works.

**Dependencies:** Task 15
**Files likely touched:** `README.md`, `PRIVACY.md`, store assets
**Estimated scope:** M

### Checkpoint: Complete
- [ ] All acceptance criteria met; public extension installable.
- [ ] Final teach-back: Katti walks the whole pipeline end to end unaided.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Real-world accuracy far below lab accuracy | High | Validate on held-out signers from the start; lean on the visual health dot + manual correction rather than chasing a magic number. |
| Chat-injection breaks on Meet HTML changes | Med | Build it last, default-off, behind a click; fail gracefully; treat overlay as the real product. |
| CSP blocks MediaPipe WASM / TFJS on Meet | High | Bundle all assets in the extension; declare `web_accessible_resources`; never load from CDN. |
| Latency too high for a live call | Med | Skeletal compression (63 numbers), small models, WebGL/WebGPU backend; profile against a <100 ms budget. |
| Scope creep / project-stacking (constitution anti-pattern #2) | High | Strict stage gates + human review checkpoints; do not start a phase until the prior runs. |
| ASL vs ISL ambiguity | Med | Decide v1 language before Phase 2 (open question below). |

## Open Questions

**Resolved (2026-06-07):** v1 language = **ASL**; **public datasets only** (no
self-capture); **parallel** to get-your-knowledge-right; browser = **Chrome/Brave**.

**Still open:**
- **License for the shipped model (resolve before Phase 6 / publish).** Research
  datasets (WLASL, MS-ASL, How2Sign-NC, etc.) are fine to *learn/prototype* on, but
  the *distributed* product should be trained on permissive data (Sign Language MNIST
  CC0, Kaggle ASL Alphabet, Google competition data per its rules). See the "License
  reality" section in [[../datasets]]. Decide before submitting to the Web Store.
- **Word vocabulary for v1** — which ~20–50 of the 250 `asl-signs` words to ship
  first? (Pick high-frequency conversational signs.)

## Parallelization Opportunities

- **Sequential (must):** Phases follow the dependency graph; data → model → export →
  integrate cannot be reordered.
- **Safe to parallelize:** The Python pipeline (Tasks 3–6) and the extension scaffold
  (Task 7) can proceed independently until they meet at Task 9. Documentation
  (Task 16 README) can be drafted incrementally.
- **Needs coordination:** The normalisation logic must be *identical* in Python
  (Task 3) and JS (Task 8) — define it once, port it carefully.

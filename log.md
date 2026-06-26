---
name: log
description: Session log for the Sign-to-Text project. Append-only, newest entry at the bottom.
type: log
status: active
last-updated: 2026-06-07
---

# Sign-to-Text — Session Log

Newest entry at the bottom.

---

Session: 2026-06-07
- What we worked on: Project kickoff — turned firstread.txt into a full study, a dataset catalog, a staged build plan, and started T1 (sandbox setup).
- Decisions:
  - v1 sign language = **ASL**; big multi-language/dataset expansion planned later (ISL next). See [[datasets]].
  - **Public datasets only** — no self-capture.
  - Build is **parallel** to get-your-knowledge-right (deliberate, not a switch).
  - Browser target = **Chrome / Brave** (both Chromium).
  - Two-model approach: static alphabet classifier first (easy), temporal LSTM for words later (hard).
  - Key shortcut found: Google `asl-signs` Kaggle data is **already MediaPipe landmarks** → skips video processing for the word model (T12 updated to use it).
  - Open: license of the **shipped** model must be permissive before publishing (Phase 6); which ~20–50 words to ship first.
- Files changed (created):
  - `in-depth-study.md` — full ground-up explanation (with corrections to firstread.txt).
  - `datasets.md` — comprehensive dataset catalog (Tier 0 landmark-native → Tier 6 tooling).
  - `tasks/plan.md`, `tasks/todo.md` — 16-task staged plan, 6 phases, checkpoints.
  - `requirements.txt`, `.gitignore`, `README.md` — project scaffold.
- T1 progress (sandbox):
  - [x] venv created + activated (Katti confirmed `(venv)` prompt).
  - [x] Core deps installed into venv: mediapipe 0.10.35, opencv-python 4.13, numpy 2.4.6 (clean, no conflicts).
  - [x] Installed: Python 3.12.4, git 2.52, node 22.11. **Missing: `gh` (GitHub CLI)** — needed for the GitHub push.
  - [ ] **git init + first commit — NOT done** (command was interrupted/rejected; repo is still un-initialized).
  - [ ] GitHub public remote — not done (needs `gh` install or create repo via website).
- Next: **Finish T1** — run `git init -b main`, first commit, then create the public GitHub repo (install `gh` OR make it on github.com and `git remote add`). Then move to **T2: "see the dots"** — a Python script that draws the 21 hand landmarks live from the webcam.
- Open: Confirm whether to install `gh` or create the GitHub repo manually on the website.

---
Session: 10 June 2026, evening
- What we worked on: Audited Antigravity's generated code, then made the fake parts real — Tasks 3, 5, 6 done properly end-to-end.
- Decisions:
  - Antigravity's `data/alphabet/` was **synthetic random noise** (headless fallback in collect_data.py) — model was a placebo. Proven mathematically (landmark-9 norm 0.5 vs real 1.0). Replaced wholesale.
  - Kaggle ASL Alphabet license is **GPL-2.0** (not "CC0-ish" as datasets.md guessed) — feeds the Phase 6 shipped-model license question. Sign Language MNIST (CC0) is the clean fallback.
  - tensorflowjs 3.18 (last Windows-installable) cannot import with NumPy 2.x → ditched the converter; custom plain-JSON weight export + rebuild the 3-layer MLP in TF.js code. Self-verifying smoke test (NumPy forward pass == model.predict).
  - Third-party binaries (hand_landmarker.task, extension lib/wasm) gitignored — re-fetchable by script; our weights.json IS committed (project artifact, demoable repo).
- Files changed:
  - `pipeline/normalize.py` (new) — single source of truth for landmark normalization.
  - `pipeline/ingest_alphabet.py` (new) — real Task 3: Kaggle images → MediaPipe → normalized .npy; skip-counts reported. Result: 25,173 samples / 28 classes; M (480 skipped) and N (430) weakest — thumb-tucked fists confuse hand detection.
  - `models/alphabet.h5` retrained on real data — 99.1% val accuracy (same-signer caveat: dataset is one signer/room; live webcam is the honest test).
  - `training/export_weights.py` (new) — exports weights.json (381 KB) with smoke test.
  - `extension/content.js` — fixed `result.handLandmarks` → `result.landmarks` (Python/JS API naming bug; overlay would never detect hands); model loading rebuilt from weights.json.
  - `extension/manifest.json` — web_accessible_resources updated; `.gitignore` — binary excludes.
  - Dataset (1.1 GB) downloaded via resumable curl loop after Kaggle CLI timed out twice (flaky ISP route to Kaggle; debugged auth → DNS → TCP layer by layer).
- Next: **Katti runs the git ceremony himself** (status → add → commit, commands already given in-session), creates the public GitHub repo on github.com, `git remote add` + push — finishes T1. Then load the extension unpacked in Brave and live-test on a Meet.
- Open: GPL-2.0 vs shipped model (decide before Phase 6); live browser test not yet done; teach-backs done: synthetic-accuracy trap (answered correctly), weights-are-just-numbers (delivered, not yet tested).

---
Session: 11 June 2026 (marathon — full day)
- What we worked on: **First live success + the great extension debugging arc + accuracy campaign + repo went public as TTM.**
- The debugging arc (Katti hit "Init Failed" live; I drove Brave via puppeteer-core with a fake camera — tools/debug_extension.cjs — and peeled 3 real bugs):
  1. **Brave WASM bug**: Brave hides CPU features (fingerprint protection) → MediaPipe requests the no-SIMD WASM variant Antigravity never bundled. Fixed: both variants downloaded + whitelisted.
  2. **Broadcast never worked**: background.js filtered tabs by tab.url, which is always undefined without the "tabs" permission → the popup toggle never reached open tabs, ever. Fixed: broadcast to all tabs, catch rejections.
  3. **Architecture DOA**: MediaPipe WASM cannot load in content scripts (isolated world + page CSP, "ModuleFactory not set"), and TF.js cannot run in MV3 extension pages at all (needs eval, MV3 forbids it). Fixed: **offscreen document architecture** — camera+MediaPipe+inference in an invisible extension page, content script is UI-only, background relays frames. TF.js deleted: replaced with ~40 lines of plain-JS matmul (extension/inference.js), verified vs Keras to 1e-7 (tests/smoke_model.mjs). One-time camera grant via permission.html.
- **IT WORKS LIVE** — letters appeared on a real Meet call. ~3-4/12 letters accurate initially → accuracy campaign:
  - Held-out signer benchmark built (danrasband/asl-alphabet-test → data/test_signer, training/evaluate.py). Baseline: **89.8%** unseen signer.
  - 3D rotation augmentation (rotate_3d, y±30°/x±20°, TDD, 12 tests) — viewpoint robustness for laptop-camera angles.
  - Full re-ingest (84k imgs → 63,589 samples) + Lexset Synthetic ASL (7 GB, 23.4k renders → 20,879 samples; diverse skin/light/angle). Retrain → **91.4%** (SPACE 30%→100%; N & X regressed — synthetic style mismatch, noted).
  - Capacity experiment 256/128: val went UP (98.3) but held-out went DOWN (90.7) → **measured no, reverted** — textbook overfit catch by the honest benchmark.
  - Restoration retrain landed **93.6%** (lucky init). Exported, smoke-tested, shipped.
  - "Model sees (live)" line added to overlay — per-frame guess + confidence, makes live testing surgical.
- **Repo PUBLIC**: github.com/kartikdubeycoded/TTM-Talk-Through-Me ("TTM — Talk Through Me", Katti's name). 8 commits pushed. T1 closed after 4 days.
- Files changed: extension/* (offscreen.html/js, inference.js, permission.html/js, background/content/manifest rewrites), pipeline/augment.py (+rotate_3d), pipeline/ingest_alphabet.py (--raw-dir/--out-dir/--append), training/evaluate.py (new), tools/debug_extension.cjs (new), tests (12 pytest + smoke_model.mjs).
- Next: **Katti live-tests the 93.6% model** (reload extension; read "Model sees" per letter, esp. S/G/M/N) and reports per-letter results. Then Phase 5 words: Katti must join kaggle.com/competitions/asl-signs (accept rules) so the 250-word landmark dataset can download. Optional: README rename to TTM.
- Open: N/X regression from synthetic style mismatch (lever: real-data weighting); J/Z impossible until temporal model (physics); README still says "Sign-to-Text"; chat-injection still untested live.

---
Session: 11 June 2026, late evening (continuation) — **Phase 5 opened: the word model exists. 76.1% on held-out signers, 40 words.**
- Research: best word data = Google asl-signs landmarks; found **markwijkhuizen/gislr-dataset-public** (2.8 GB republish, no competition join needed — Katti's blocker dissolved). Bonus find: swaptr ISL MediaPipe landmarks (2.7 GB, May 2026) — Tier 4 ISL expansion now has a landmark-native dataset waiting.
- Reverse-engineered the format: X (94477, 64, 66, 3) = [lips 0:40, dominant hand 40:61, pose 61:66], zero-padded frames, NON_EMPTY_FRAME_IDXS = original video frame numbers (mask is >=0, not the values). Label↔sign mapping verified via dschettler extended_train.csv.
- Built (TDD, 17 tests total now): pipeline/word_features.py — per-frame 66 features = 63 normalized landmarks (same convention as alphabet) + 3 wrist-trajectory (per-frame normalization erases motion; trajectory puts it back) + linear resample to 30 frames. pipeline/ingest_words.py — 40-word conversational starter vocab → 11,420 sequences, 21 signers (3,820 skipped, <8 hand frames). training/train_words.py — 2-layer LSTM(128), split by PARTICIPANT (GroupShuffleSplit).
- Result: **76.1%** held-out-signer accuracy (vs ~80% competition-winner transformer on all 250). Worst words (go 0.23, fine 0.26, dad 0.48) are face/body-anchored signs — hand-only features can't see them. **Lever for next session: add wrist-relative-to-face features from the lips/pose blocks we discarded.**
- 9 commits public on TTM-Talk-Through-Me.
- Next: (1) Katti's live letter-test report (S/G/M/N vs "Model sees") — still pending; (2) word-model browser integration: plain-JS LSTM forward pass + rolling 30-frame buffer in offscreen.js; (3) face-relative features experiment.
- Open: words.h5 not yet exported/integrated (browser LSTM inference unwritten); letter-model N/X synthetic-style regression; README still says Sign-to-Text not TTM (Katti hasn't confirmed rename).

---
Session: 12 June 2026 — word model into the browser (Phase 5 / T13–T14 finish)
- What we worked on: exported the word LSTM to plain JS and wired it into the live pipeline.
- Decisions:
  - No TF.js (MV3 forbids the eval it needs) — extend the alphabet model's plain-JS forward pass with an LSTM cell (gate order i,f,c,o). Verified against Keras to 2e-6 before shipping.
  - Whole-word signs take precedence over fingerspelling; a word commits on 2 agreeing inferences ≥0.85 then a cooldown, so one sign = one word.
- Files changed: `extension/inference.js` (+LSTM forward), `training/export_words.py` (new), `extension/model/words.json`, `tests/smoke_words.mjs` + fixtures; `extension/content.js` + `extension/offscreen.js` (rolling 30-frame buffer, word-over-letter commit logic, dual live-guess display). Commits 82ff227, b0f3be1.
- Next: improve word accuracy (face-anchored signs go/fine/dad); live-test the combined letter+word pipeline.
- Open: words still weak on face-anchored signs.

---
Session: 26 June 2026 — plan refresh + Phase 5b (trustworthy words)
- What we worked on: refreshed the stale `tasks/plan` around the real north star (readable sentences from letters AND words), then shipped Phase 5b to make the word model trustworthy.
- Decisions:
  - Plan reframed: the old 16-task plan was done through T14. Remaining work = word quality (5b) → readable assembly (5c) → honest live test (5d) → publish (6). v1 finish line (store vs polished unpacked repo) left open.
  - **T15** — fixed face-anchored signs the cheap way (Option A): added the absolute wrist x,y as features (location, not just shape + motion). "dad"=forehead vs "mom"=chin are now separable. Mirrored exactly in `offscreen.js` so live == training. Option B (live face landmarks via Holistic) NOT needed — measured, not assumed.
  - **T16** — vocabulary honesty floor = 0.5 held-out per-word accuracy; dropped "go" (0.15), a directional point whose meaning is the pointing direction, not the handshape.
- Results: dad 0.48→0.89, fine 0.26→0.64; overall held-out **76.1% → 78.6%** (T15) → **79.4%** (T16, 39 words). Lowest survivor "fine" 0.51 — the fine/can/finish/look cluster near the floor is a watch-item. 18 pytest + JS smoke (JS↔Keras 1.94e-7) all green.
- Files changed: `tasks/plan.md` + `tasks/todo.md` (refresh), `pipeline/word_features.py` (+location, 66→68), `pipeline/ingest_words.py` (39-word vocab + floor doc), `training/export_words.py` (derive feature width from model), `extension/offscreen.js`, `extension/model/words.json` + fixtures (re-export). Commits 053be30, dec7f1a.
- Next: Phase 5c / **T17** sentence assembler (capitalization, punctuation, sentence boundaries, autocorrect) — gated behind the Phase 5b human-review checkpoint + teach-back.
- Open: is 79.4% good enough to build the assembler on, or push accuracy first? (awaiting Katti's call); borderline cluster fine/can/finish/look ~0.5; teach-back on the wrist-location fix still pending.

---

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

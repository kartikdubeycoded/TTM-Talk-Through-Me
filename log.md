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

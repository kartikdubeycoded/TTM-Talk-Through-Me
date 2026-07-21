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
Session: 26–27 June 2026 (continued) — review-driven hardening + sentence assembler started
- What we worked on: ran the changed code through a simplify pass and a five-axis review, hardened the weakest spot, then started the sentence assembler (Phase 5c / T17).
- Decisions:
  - **Simplify pass:** the Phase 5b code was already minimal — applied one honest win (named the thrice-reindexed wrist as `wrist_pos`), left two apparent-duplications alone (the `offscreen.js` wrist copy is *defensive* against MediaPipe object reuse; the `export_words.py` LSTM/Dense branches extract genuinely different weights). Commit 84dd585.
  - **Code review** flagged the real gap: the LSTM smoke test proved the JS *model* matches Keras, but nothing proved the JS *feature construction* matches Python — a silent divergence would feed the model different numbers live than in training and pass every test. Fixed (TDD): extracted `normalizeLandmarks` + `buildWordSequence` into `extension/features.js`, which `offscreen.js` now imports (one implementation, not a copy); `tests/smoke_features.mjs` replays a Python-generated fixture and asserts JS↔Python parity (3.95e-7). Commit 859256a. Empirically confirmed `traj_z` is a dead column (std=0) — deferred drop to next retrain.
  - **T17 slice 1 (of 2):** built `extension/assembler.js` — pure, DOM-free: capitalizes sentences, ends with `.`/`?` (question words), conservative one-edit autocorrect for *fingerspelled* words only (signed vocab trusted, never rewritten). Tested via `tests/smoke_assembler.mjs` (RED→GREEN). Commit 135858b.
  - **Slice 2 (content.js wiring) deliberately held back** — it changes live behavior and can't be verified in-harness; wants a real browser reload. Doing it next to the live test.
- Results: 18 pytest + 4 JS smokes all green. 6 commits today (053be30, dec7f1a, 9aad987, 84dd585, 859256a, 135858b).
- Next session: **actual live test runs** — reload the extension on a real Meet, sign letters + the 39 words, and honestly record per-letter/per-word hit rates (this is T19 territory and tells us if 79.4% survives a real webcam). Then wire the assembler into content.js (T17 slice 2) and tune letter↔word fusion (T18).
- Open: Phase 5b teach-back still owed (one line: why the model couldn't tell dad from mom, and what wrist-location fixed); is 79.4% real on a live cam? (the test answers it); borderline word cluster ~0.5; README still says Sign-to-Text not TTM.

---
Session: 20 July 2026 — first live-pipeline run + whole-body feature upgrade (Phase A)
- What we worked on: (1) built the honest test harness and ran the WHOLE pipeline end-to-end on a real ASL video for the first time; (2) on Katti's insight, added face+pose to the word features and retrained on the full 250-word vocab.
- The live-pipeline milestone: fed a real "25 ASL signs" video through Chrome's fake camera (`tools/debug_extension.cjs`, `--use-file-for-fake-video-capture`). First-ever end-to-end run — hands tracked, both models fired. Word model recognized real signs in the wild (tomorrow/sad/dad/morning/happy/thirsty/look/listen 90–100%). Letters spat junk (letter firehose during word-signing = T18, known).
- The upgrade (Katti's call, and he was right): the word model read only the HAND; face + upper-body position is the signal that separates confusable/face-anchored signs. The data was ALREADY in the GISLR files (`lips 0:40, hand 40:61, pose 61:66`) — `ingest_words.py` was discarding lips+pose. No new video needed.
- Decisions:
  - `whole_body_features(hand, lips, pose)` — keep the proven 68-col hand recipe, append a 13-col block anchored to the lips centroid, scaled by lip width (signer-invariant). TDD: RED→GREEN, 3 new tests, 6 old hand tests still green.
  - Ingest the FULL 250-word vocab (was 39) through the new recipe → `(70249, 30, 81)`.
  - **RESULT: 185 / 250 words clear the 0.50 honesty floor** (was 39 shipped), 89 ≥0.70. Headline 61.9% (all-250 average, dragged by 65 sub-floor words). **~5× usable vocabulary, same architecture, zero new data — the whole-body idea worked.** Full table in `training/results_wholebody_250.md`.
  - Scale-up decision (gate passed): go for Microsoft **ASL Citizen** (2,700 signs, 43 GB raw MP4) next, 500-word subset first. Katti started the 43 GB download this session. GPU reality: RTX 4050 present but TF 2.21 can't use it on native Windows → plan is WSL2+PyTorch (or free Kaggle) for the big job; extraction (~1 day) is CPU-bound, GPU only speeds training.
- Files changed: `pipeline/word_features.py` (+whole_body_features), `pipeline/ingest_words.py` (full vocab + lips/pose slices), `tests/test_word_features.py` (+3 tests), `training/results_wholebody_250.md` (new), `tools/debug_extension.cjs` + `.gitignore` (test harness). Commits on branch `whole-body-features`: 1102afe, + this session.
- Next: (1) commit/verify extension functionality on Meet (harness audit); (2) once ASL Citizen downloads → WSL2+PyTorch setup + extraction/training scripts (I draft against microsoft/ASL-citizen-code); (3) still no runtime to WATCH the new 185-word model — Katti chose to defer the offline tester; his phone-camera test still exercises the OLD 39-word extension.
- Open: no A/B proving whole-body caused the lift vs just more words (clean 39-word hand-vs-body run would nail it); pass-2 final shipped model on the 185 survivors not yet retrained/exported; live integration of the new model (face+pose trackers in-browser, real-time budget) = the hard Phase B, undecided (desktop demo vs live extension); teach-back still owed.

---
Session: 20–21 July 2026 (continued) — rename to TTM, UX fixes, strategy pivot, scale-up front-end
- What we worked on: shipped product-facing fixes (rename + one-click camera + caption UI), researched the competitive/patent reality, and locked a strategic direction. All while the ASL Citizen 40 GB downloads.
- Live test frustration → real fix: Katti's phone-camera test on Meet hit camera-permission hell (init-fail, second tab, toggle off/on dance) and poor tracking. Key diagnosis: the broken permission flow meant the engine often never got the camera, so it wasn't the model failing — it was never running. Fixed the flow: one "Allow camera & start" click now grants → restarts the offscreen engine (auto-retry) → closes its own tab; storage-throttled to stop duplicate tabs. NEEDS his manual re-test (fake-camera harness can't exercise the deny path).
- Shipped to `main` (public repo): renamed SignToText → **Talk Through Me** across all user-visible surfaces; the permission fix. Then, on this branch (`asl-citizen-scaleup`): **Deaf-friendly caption UI** — one big 26px caption as the hero, dev readouts (word buffer / "model sees") collapsed into <details>; all element ids preserved so the render loop is untouched.
- Strategy (researched, decided): sign→text is **patent-crowded** — live patents (US20220327961A1 etc.), **Google SignGemma** (on-device ASL, Q4 2025) + **SignAll** shipping. So NO patent, and no head-to-head on ASL. Katti's call: **don't narrow yet — get one working end-to-end proof (on ASL, already in motion), then expand to ISL** (same pipeline, the niche giants ignore; landmark-native dataset = small/fast). Primary value = **proof he can build a scalable end-to-end system** (portfolio), not market capture. Extension form = UX/distribution differentiator, not IP. Architecture locked: **eyes** (recognition model) + **writer** (SLM sentence layer, e.g. Chrome Gemini Nano with assembler.js fallback).
- Scale-up front-end built (TDD, 25 pytest green): `pipeline/asl_citizen.py` — parses ASL Citizen split CSVs → (video, gloss), `select_vocab()` keeps top-N. Extraction core DEFERRED until the videos are on disk (landmark-index selection must match the model's 66-pt input; guessing blind = the silent-divergence trap).
- Dataset discipline: talked Katti off downloading How2Sign (33 GB, continuous = a different, parked architecture) and the `asl-citizen-processed-200` HF set (200 words < our 250, a downgrade). Locked: finish the 40 GB ASL Citizen video, extract ourselves.
- Approved plan for the surrounding system saved (productization: caption UI ✓, model-swap seam, SLM writer, Phase B live model, ASL training, ISL). See the plan file / project-brain for the staged triggers.
- Next session: (1) when the 40 GB finishes → build the real extractor against the actual files, run on 500-word subset, train; (2) model-swap seam in inference.js/offscreen.js; (3) then the SLM writer + Phase B live face+pose integration. Also: Katti to re-test the one-click permission + new caption UI (reload the extension first).
- Open: permission fix + caption UI un-verified live (need Katti's reload+retest); download was slow (~2 MB/s, ~4 h ETA, resumable); RTX 4050 present but TF can't use it on native Windows (WSL2+PyTorch or Kaggle for the big training); pass-2 185-word model still not exported into the extension.

---
Session: 21 July 2026 (late) — writer layer built + honest remaining-work map
- What we worked on: built the "writer" half of the eyes+writer pipeline (the SLM sentence layer with memory), while the 43 GB ASL Citizen download runs.
- Built (all committed + pushed to main):
  - `extension/writer.js` (`createWriter`) — wraps the rule-based `assembler.js` with (a) a rolling conversation MEMORY (user's finished sentences + the other party's captions) and (b) an OPTIONAL on-device LLM refiner. Strategy + dependency injection so the core is testable in Node.
  - Guaranteed fallback (tested): if the LLM is absent/errors/returns empty, `refined()` returns the rule-based text — the LLM can never break or block captions. Conservative system prompt (fix grammar only, never invent facts/names).
  - `createGeminiNanoRefiner()` — feature-detected Chrome Prompt API adapter; returns null when unavailable (the common case, incl. BRAVE). Isolated, not unit-testable, needs live verification on a Nano-capable Chrome.
  - `assembler.endSentence()` now returns the finished sentence (so writer can remember it; existing tests unaffected).
  - TDD: `tests/smoke_writer.mjs` (7 cases) RED→GREEN; all 5 JS smokes + 25 pytest green.
- Honest status reality-check delivered: the SURROUNDING system is largely built (writer, caption UI, one-click permission, 185-word model trained, ASL Citizen front-end). But "just get the data and we're done" is FALSE. Still ahead: (a) data+train (real work, not just a download), (b) wire the writer into content.js (needs a live reload), (c) **the hard one — Phase B: run the better model LIVE (Face+Pose landmarkers in real-time in the browser, latency budget). The extension STILL runs the OLD 39-word hand-only model.** (d) muzzle the letter firehose (T18).
- NEXT SESSION (Katti returns with the download done):
  1. Build the real ASL Citizen extractor against the actual files → run on 500-word subset → train the bigger model.
  2. Wire `writer.js` into `content.js` (replace the inline sentence logic) — verify with a live reload.
  3. Muzzle the letter firehose (T18) + the 3→5 quality push.
  4. (Bigger, staged) Phase B: live Face+Pose model integration.
  5. Also pending: Katti to reload the extension + Meet TAB and confirm the one-click permission + new caption UI actually show (harness proved the code renders — the fix is: reload the page after reloading the extension).
- Goal framing locked: tomorrow = 3/10 → 5/10 (cleaner output, more words, + a scalability/system-design doc). North star = beat SignGemma/SignAll on PRODUCT + REACH + ISL, not on raw ASL model accuracy (unwinnable solo). No patent (crowded).
- Open: Gemini Nano won't run in Brave → writer uses rule-based fallback there until tested on a Nano Chrome; writer not yet wired live; download completion + speed unknown.

---
Session: 21 July 2026 (later) — download rescue + writer wired live + fusion referee + scale-up de-risking
- What we worked on: recovered the two lost dataset downloads and resumed them, then used the download window to (a) actually wire the writer into the live UI, (b) add the letter↔word referee, and (c) hunt + document the landmines waiting on the other side of the download. All data-independent, all tested at the logic level.
- Download rescue: laptop was closed mid-download, session lost. Recovered both commands from PSReadline history + found the partial files on disk, so BOTH resume (no restart):
  - ASL Citizen (~45 GB): `curl.exe -L -C -` → `C:\Users\katti\asl_citizen\ASL_Citizen.zip` (was ~18 GB, resuming from Microsoft download.microsoft.com/.../ASL_Citizen.zip).
  - How2Sign (33 GB): `huggingface_hub.snapshot_download('PSewmuthu/How2Sign_Holistic')` → `how2sign_continuous_data/` (was ~31 GB; HF `.cache` = the resume state). Had to `pip install huggingface_hub` into venv first.
  - Katti chose to download BOTH incl. How2Sign — flagged it's continuous/parked (risks.md #2), his call. Will finish both (~2 h ETA) then we plan extraction together.
- Built (all committed? NO — everything below is UNCOMMITTED; Katti hasn't asked to commit yet):
  - **A — writer wired into the live UI.** `content.js` no longer builds captions by hand; it dynamically `import()`s `writer.js`/`fusion.js` (MV3-correct: added them to manifest `web_accessible_resources`; classic content scripts can't static-import). Captions now assemble to `Hi. How are you?` (cap + punctuation + long-pause sentence boundary), routed through writer→assembler. Gemini-Nano seam attached but out of the live path (untestable w/o Nano Chrome).
  - **B — T18 fusion referee.** New `extension/fusion.js`: `wordSignAllowed()` muzzles the word LSTM while the user is actively fingerspelling (buffer open OR letter locked < `FINGERSPELL_LOCKOUT_MS`=1500ms) unless word conf ≥ `WORD_OVERRIDE_CONFIDENCE`=0.97. Gated the word-commit in `content.js`. `tests/smoke_fusion.mjs` (green).
  - **End-to-end proof:** `tests/smoke_pipeline.mjs` replays a scripted stream through the real writer+fusion → asserts `"Hi. How are you?"` AND that a spurious "kite" is refused mid-spell. Green.
  - **Risk register:** `risks.md` — 4 landmines (dot-numbering silent-divergence, How2Sign unusable, no GPU on native Win, tight disk).
  - **Dot-numbering map:** `landmark-map.md` — traced training-data provenance to `markwijkhuizen/gislr-dataset-public`; KEY REALIZATION: for the ASL Citizen model we own BOTH extraction and live code, so we define our own consistent 66-layout — no need to reverse-engineer his exact indices (that's only needed to ship the OLD GISLR model live; deferred). Confirmed lips-40 = FACEMESH_LIPS (authoritative index list saved).
  - **Anti-divergence anchor (tested):** `pipeline/landmark_layout.py` — the single source of truth for the 66-dot selection (`select_66`, LIPS/HAND/POSE idxs, slices mirroring ingest_words). `tests/test_landmark_layout.py` — 3 pytest green (picks right dots in right order, zero-fills missing parts).
- Verified: all JS smokes (writer, assembler, fusion, words, pipeline) + `node --check content.js` + 3 pytest landmark_layout. NOT verified live — A+B need Katti to reload the extension + Meet tab.
- Decisions: ASL Citizen (Goal B) uses our own 66-layout {lips FACEMESH_LIPS 40, hand 21 native, pose [0,11,12,13,14] = nose+shoulders+elbows}; exact markwijkhuizen match reserved for shipping the old model live (optional). Training path (Kaggle recommended vs WSL2+PyTorch vs CPU) — undecided (risks.md #3).
- Files changed: `extension/content.js`, `extension/manifest.json`, `extension/fusion.js` (new), `pipeline/landmark_layout.py` (new), `risks.md` (new), `landmark-map.md` (new), `tests/smoke_fusion.mjs` + `tests/smoke_pipeline.mjs` + `tests/test_landmark_layout.py` (new). All UNCOMMITTED.
- Next: when both downloads finish → extract 3–5 ASL Citizen test videos via `landmark_layout.select_66` → eyeball `dad`(forehead) vs `mom`(chin) → then full 84k extraction → train (Kaggle). ALSO pending: Katti reload extension to confirm A+B captions render live; and commit this session's work (walk Katti through git).
- Open: downloads incomplete (~2 h ETA); disk tight (~96 GB free vs 78 GB downloading + ~45 GB to unzip — delete zip/MP4s after extract); GPU training path undecided; A+B unverified live; writer LLM refiner still untested (Brave has no Nano).

---
Session: 21 July 2026 (later still) — downloads DONE, ASL Citizen streaming extractor built + verified + pilot running
- What we worked on: both datasets finished; built and verified the ASL Citizen video→landmarks extractor; kicked off the 50-sign pilot extraction. This is the piece asl_citizen.py had deliberately deferred until real videos existed.
- Downloads DONE:
  - How2Sign 32.6 GB → `how2sign_continuous_data/` (still PARKED — continuous, wrong architecture for the word model; deletable for disk headroom if needed).
  - ASL Citizen zip 42.77 GB → `C:\Users\katti\asl_citizen\ASL_Citizen.zip` (curl `-C -` auto-retry loop survived repeated ISP resets).
- ASL Citizen reality (inspected from inside the zip, 83,406 entries):
  - Layout: `ASL_Citizen/splits/{train,val,test}.csv` + `ASL_Citizen/videos/<id>-<GLOSS>.mp4`.
  - train.csv = **40,154 videos, 2,731 signs** → **~15 videos/sign (flat, shallow)**. Headers: `Participant ID, Video file, Gloss, ASL-LEX Code` — parsed fine by asl_citizen.py. Split CSVs extracted to `data/asl_citizen/`.
  - **Honest expectation set:** ~15/sign is little; ASL Citizen's value is BREADTH (vocab size for the "scalable system" proof), not high accuracy. Don't expect GISLR-level numbers.
- 🚨 DISK CRISIS + fix: only **42.9 GB free vs a 42.77 GB zip → cannot `Expand-Archive`** (would fail/corrupt mid-way). Fix baked into the design: the extractor **streams one video at a time out of the zip → processes → deletes the temp** — never unzips, disk stays flat.
- Infra fixed BEFORE the long op: mediapipe 0.10.35 + opencv 4.13 already in venv; downloaded `face_landmarker.task` (3.6 MB) + `pose_landmarker.task` (9 MB, "full") into `extension/model/` (hand model already there). Added `*.task` to .gitignore.
- Built `pipeline/extract_asl_citizen.py` (the streaming extractor): CSV → `select_vocab(top_n)` → per video: zip→temp → MediaPipe Face(478)+Hand(21)+Pose(33) VIDEO mode → `landmark_layout.select_66` → `whole_body_features` → `resample(30)` → (30,81). Saves X/y/participants/labels.json in the SAME format `train_words.py` reads. Pilot simplifications flagged: `num_hands=1` (dominant hand only), no L/R mirroring.
- Two bugs the 5-video verify caught + fixed (exactly why we verify first): (1) MediaPipe VIDEO timestamps must be STRICTLY increasing across ALL videos → a global `clock[0]` ms counter that never resets; (2) Windows temp-file lock → `cap.release()` in a `finally` + best-effort `os.unlink`.
- VERIFIED: 5-video run → `X (5,30,81)`, **face detected in 100% of frames**, 0 skipped/0 missing, sane feature range. The whole chain works end-to-end on real video.
- Signer IDs wired (needed for honest signer-disjoint splits): `asl_citizen.load_split` now returns `participant` (added `_pick_optional`, `PARTICIPANT_COLS`); 2 tests updated; extractor maps signer→int codes for `participants.npy`. **16 pytest green.**
- `training/train_words.py`: added `--data-dir` / `--model-path` args (backward-compatible) so it trains the ASL Citizen pilot with no code duplication.
- PILOT RESULT (extraction + train DONE): extracted **945 sequences** from 950 videos (5 skipped, 0 missing) → `data/asl_citizen_words/`. Trained locally in seconds (`models/asl_citizen_pilot.h5`). **Held-out-SIGNER accuracy = 55.2% on 50 signs** (random = 2%). 10 signs at 1.00 (BELT/EAT/FINE/LUNCH/MAPLE/SQUEEZE/HALLOWEEN/NOON/GUESS/DINNER), ~20 at ≥0.75, a few at 0.00 (DEVELOP/ELEVATOR/TWINS). **The whole scale-up pipeline is PROVEN end-to-end on a new dataset** — that was the session's question; answer = yes. 55% is exactly what the thin ~15/sign data predicts; the win is the working scalable system (north star), not the number.
- Files changed (ALL UNCOMMITTED): `pipeline/extract_asl_citizen.py` (new), `pipeline/landmark_layout.py` (new, prior), `pipeline/asl_citizen.py` (+participant), `training/train_words.py` (+args), `tests/test_asl_citizen.py` (+participant asserts), `tests/test_landmark_layout.py` (new), `.gitignore` (+*.task), `data/asl_citizen/*.csv` (extracted), `extension/model/{face,pose}_landmarker.task` (downloaded, gitignored).
- NEXT SESSION (start here):
  1. **If pilot extraction finished:** `python training/train_words.py --data-dir data/asl_citizen_words --model-path models/asl_citizen_pilot.h5` — trains LOCALLY in minutes (small data, no Kaggle needed for pilot). Read held-out accuracy + per-sign table → **decide if the 500-sign scale-up is worth it.**
  2. Scale command when ready: `python pipeline/extract_asl_citizen.py --top-n 500 --out data/asl_citizen_500` (≈8,237 videos, slow; Kaggle for the big TRAIN, extraction stays local/CPU).
  3. Still owed from earlier: Katti reloads the extension to confirm A+B (writer sentences + fusion muzzle) render live; commit this whole session's work (walk Katti through git — counts toward 200).
- Open: pilot accuracy unknown (thin data); ASL Citizen model is a FRESH standalone model (Tasks landmarkers + our own 66-layout ≠ the old GISLR/Holistic 185-word model — not interchangeable); live Phase B must mirror `select_66`+`whole_body_features` in `offscreen.js`; everything this continuation uncommitted.

---

---
name: project-brain
description: The complete current-state map of Sign-to-Text / TTM — what it is, what is actually built today (file by file, with real accuracy numbers), how every piece connects, what is left, the tech stack, honest limits, and how it scales. Read this to understand the whole project; read in-depth-study.md for the "why" behind the design.
type: reference
status: active — current as of 2026-07-21
last-updated: 2026-07-21
---

# Sign-to-Text / TTM — The Whole Project, As It Actually Is

> **Update 2026-07-21 (read this first — the deltas since the body below):**
> - **Renamed to Talk Through Me (TTM)** across the extension (name, overlay,
>   permission page). Shipped to `main`.
> - **Word model: 39 → 185 words.** Adding whole-body features (lips + pose to the
>   hand, `whole_body_features` in `pipeline/word_features.py`) and retraining on
>   the full 250-word GISLR vocab put 185 words above the 0.50 honesty floor.
>   Details + per-word table: `training/results_wholebody_250.md`. The new model
>   is trained but **not yet wired into the extension** (that's Phase B, needs
>   live face+pose trackers).
> - **First end-to-end live-pipeline run** happened, via a fake-camera video harness
>   (`tools/debug_extension.cjs`). The word model recognized real signs in the wild.
> - **UX fixes shipped:** one-click camera permission (no more toggle dance) and a
>   Deaf-friendly caption UI (big caption, debug readouts collapsed).
> - **Strategy locked (patent research done):** space is patent-crowded (Google
>   SignGemma, SignAll) → no patent; the extension form is a UX/distribution edge,
>   not IP. Goal = prove a scalable end-to-end system; ASL is the first proof, **ISL
>   is the differentiated expansion** (same pipeline). Architecture = **eyes**
>   (recognition) + **writer** (on-device SLM sentence layer, e.g. Gemini Nano with
>   `assembler.js` fallback).
> - **In progress:** downloading ASL Citizen (43 GB, ~2,700 signs) to scale the
>   vocabulary; front-end (`pipeline/asl_citizen.py`) built, extraction pending the
>   download. **How2Sign (continuous) is explicitly parked** — different architecture.
>
> The status board in §2 below is the 2026-06-29 snapshot; trust this block + `log.md`
> for anything newer.

> **What this document is.** A single place that tells you *exactly what you have
> built*, today, with nothing hand-waved. It maps every real file, the real
> accuracy numbers, what's done, what's left, and how it all connects.
>
> **Two companion docs (don't duplicate them, read them):**
> - [`in-depth-study.md`](in-depth-study.md) — the *concept* doc. The "why" behind
>   every design choice, the math explained for an interview, the staged thinking.
>   Frozen at kickoff (7 Jun) on purpose — it's the theory, and the theory held.
> - [`datasets.md`](datasets.md) — the full dataset catalog (every sign-language
>   dataset on the web, by language and license). The "where the data comes from."
> - This file (`project-brain.md`) — the *current reality*. The "what exists now."
>
> If those three ever disagree, **this file wins on current state**, the study
> wins on concept, datasets wins on data. The honest blow-by-blow history lives in
> [`log.md`](log.md).

---

## 1. The one-paragraph truth

TTM ("Talk Through Me") is a **Chrome/Brave extension** that watches *your own*
webcam during a video call (Google Meet / Zoom / Teams), recognises the
sign language you're making, and shows it as **live English text** in a floating
overlay on the call — and optionally types it into the meeting chat. It runs
**100% in your browser**: the webcam video never leaves your machine, there is no
server, no cloud, no account, no cost to run. It is built **for and with respect
for the Deaf and hard-of-hearing community**. The repo is **public** at
`github.com/kartikdubeycoded/TTM-Talk-Through-Me`.

It does this with **two trained models working together**: a fast **alphabet
classifier** (reads fingerspelled letters, one frame at a time) and a slower
**word LSTM** (reads ~39 whole-word signs from a 1-second window of motion). Both
were trained offline in Python on public datasets, then exported to **plain
JavaScript** (no TensorFlow.js — Manifest V3 forbids the `eval()` it needs) and
run live in the browser.

---

## 2. Where it stands TODAY — the honest status board

This is the single most important table in the document. Green = it runs and is
verified. Yellow = partly done. Red = not started.

| Area | Status | The honest truth |
|------|:------:|------------------|
| **Public repo + git** | ✅ | Live and public as `TTM-Talk-Through-Me`. |
| **Hand tracking** | ✅ | MediaPipe `HandLandmarker` draws/extracts 21 landmarks live. |
| **Alphabet model** | ✅ | **93.6%** accuracy on signers the model *never saw*. Real Kaggle data + Lexset synthetic + 3D-rotation augmentation. |
| **Word model** | ✅ | **79.4%** held-out-signer accuracy on **39 words**. Plain-JS LSTM, verified to match Keras to ~2e-6. |
| **Extension runs live** | ✅ | **It worked on a real Meet call** (11 Jun) — letters appeared on screen. Offscreen-document architecture solved the hard WASM/CSP bugs. |
| **Overlay UI** | ✅ | Floating panel: live captions, word buffer, "Model sees" live guess, health dot, Clear/Backspace. |
| **Sentence builder (basic)** | ✅ | Lock-in gate, debounce, velocity-based spacing, backspace/clear — in `content.js`. |
| **Sentence *assembler* (pretty)** | 🟡 | `assembler.js` is **built and tested** (capitalization, punctuation, `?`, autocorrect) but **NOT yet wired into `content.js`**. This is T17 slice 2 — needs a live reload to verify. |
| **Letter↔word fusion tuning** | 🔴 | Precedence rules exist but are **untuned/unvalidated** (T18). |
| **Honest combined live test** | 🔴 | **No end-to-end hit-rate has been recorded since the word model landed.** This is the next thing — T19 — and it's what we'll do together. |
| **Settings/options page** | 🔴 | Settings are read from storage and controlled by the popup, but there's no full options page (T20). |
| **Chat injection** | 🟡 | Code exists (`injectIntoChat`) but is **untested live** and fragile (T21). |
| **README/privacy/license** | 🟡 | README still says "Sign-to-Text" not TTM; shipped-model **license is unresolved** (Kaggle alphabet data is GPL-2.0) — must be settled before any Web Store listing (T22). |
| **Published to Web Store** | 🔴 | Not decided yet — finish line is "polished load-unpacked repo" vs. "store listing". Open question. |

**The single honest gap that matters most right now:** the models work in the
lab, the extension worked live once for letters, but **nobody has sat down and
measured how good the combined letter+word system actually is on a real webcam,
in real light, since the word model was added.** Everything in Phase 6 (polish,
publish) is gated on that number. **That is exactly what the testing session is
for.**

---

## 3. The pipeline — station by station, mapped to real files

The whole system is an assembly line: a webcam frame goes in, English text comes
out. Five stations. Here's each one *and the actual file that does it*:

```
 Webcam frame                                                        English text
      │                                                                   ▲
      ▼                                                                   │
┌──────────┐  ┌────────────┐  ┌─────────────┐  ┌────────────┐  ┌──────────────┐
│ 1. GRAB  │─▶│ 2. TRACK   │─▶│ 3. NORMALISE│─▶│ 4. PREDICT │─▶│ 5. ASSEMBLE  │
│ a frame  │  │ the hand   │  │ the numbers │  │ the sign   │  │ the sentence │
└──────────┘  └────────────┘  └─────────────┘  └────────────┘  └──────────────┘
 getUserMedia   MediaPipe       features.js      inference.js     content.js +
 (offscreen.js) HandLandmarker  normalizeLand-   createClassifier assembler.js
                (offscreen.js)  marks /          + createSequence-
                                buildWordSequence Classifier
```

1. **GRAB** — `offscreen.js` opens the webcam with `getUserMedia()` and reads a
   frame every 50 ms (~20 fps).
2. **TRACK** — `offscreen.js` runs MediaPipe `HandLandmarker` on the frame →
   gets **21 landmarks** (wrist → fingertips), each an (x, y, z). This is the
   magic compression step: a 640×480 image becomes ~63 numbers.
3. **NORMALISE** — `features.js` (`normalizeLandmarks`, `buildWordSequence`)
   anchors to the wrist and scales by hand size, so any hand at any distance
   gives the same numbers for the same sign.
4. **PREDICT** — `inference.js` runs the two models by hand in plain JS:
   - `createClassifier` → the alphabet model (3 dense layers) → a letter +
     confidence, *every frame*.
   - `createSequenceClassifier` → the word LSTM → a word + confidence, every 8th
     frame, fed the rolling 30-frame buffer.
5. **ASSEMBLE** — `content.js` turns the firehose of guesses into clean output
   (lock-in gate, spacing, backspace). `assembler.js` (once wired) makes it
   *pretty* (capitals, punctuation, autocorrect).

**The one idea:** the project's core trick is **compression** — collapse a fat
video frame into 63–68 numbers *before* any ML runs. That's what makes it fast
enough to feel live.

---

## 4. The two models (this is the heart of the project)

The single most important conceptual fact, and the one most people miss: **this
is two different ML problems, not one.** (See in-depth-study §4 for the full why.)

### 4a. The alphabet model — STATIC handshapes

- **Problem type:** look at *one frozen frame* and say which letter it is. "B"
  looks like "B" in a single photo. So a per-frame classifier works.
- **Architecture:** a small **3-layer dense (MLP)** network, ~18k weights.
- **Input:** 63 normalised numbers (21 landmarks × xyz).
- **Output:** a probability for each of 28 classes (A–Z + SPACE + DELETE).
- **Trained on:** real Kaggle ASL Alphabet images (run once through MediaPipe to
  get landmarks) **+ Lexset synthetic ASL renders** (diverse skin/light/angle) →
  ~84k images → ~63.6k usable landmark samples. Plus **3D-rotation augmentation**
  (simulates laptop-camera angles), mirroring, jitter.
- **Honest accuracy: 93.6%** on *held-out signers* — people and rooms the model
  never trained on. That's the number that matters; same-signer accuracy was 99%
  and is a vanity number.
- **Known weak spots:** J and Z are **impossible** for this model — they involve
  *motion*, and a static classifier can't see motion (a physics limit, not a
  bug). N and X regressed slightly from the synthetic-data style mismatch.
- **Files:** trained by `training/train_alphabet.py`, evaluated by
  `training/evaluate.py`, exported by `training/export_weights.py` →
  `extension/model/weights.json` (the numbers the browser actually loads).

### 4b. The word model — DYNAMIC motion over time

- **Problem type:** "GO" vs "COME" can be the *same handshape* moving in opposite
  directions. A single frame can't tell them apart — you need the *sequence*. So
  you feed a **window of 30 frames (~1 second)** into a model that understands
  order: an **LSTM**.
- **Architecture:** a **2-layer LSTM (128 units)** → dense → softmax.
- **Input:** a (30, 68) block — 30 time-steps, each with **68 features**:
  - `[0:63]` the 63 normalised landmarks (same convention as the alphabet model)
  - `[63:66]` **wrist trajectory** — this frame's wrist minus frame 0's. (Per-frame
    normalisation *erases* where the hand is; trajectory puts the motion back.)
  - `[66:68]` **wrist location** — absolute x,y in the camera frame. This is the
    clever fix: "dad" (hand at forehead) vs "mom" (hand at chin) share a handshape
    and barely move — only the *height in frame* separates them. Shape + motion
    alone can't see that; location can. (z is dropped — it's a dead column.)
- **Trained on:** Google's `asl-signs` landmark dataset (via the
  `gislr-dataset-public` republish — no Kaggle competition sign-up needed),
  ~94k sequences, 100+ signers. Split **by signer** (`GroupShuffleSplit`) so the
  test is honest.
- **Honest accuracy: 79.4%** on held-out signers, **39 words**.
- **The vocab (39 words):** hello, bye, yes, no, please, thankyou, fine, bad,
  happy, sad, mad, hungry, thirsty, sick, sleepy, water, food, drink, milk, mom,
  dad, who, where, why, now, later, tomorrow, night, morning, home, finish, can,
  wait, look, listen, talk, think, like, have.
- **The honesty floor:** any word below **0.5** held-out accuracy was **dropped**
  — "go" (0.15, a directional point whose meaning *is* the pointing direction) was
  cut. Shipping a word the model can't read just corrupts sentences.
- **Watch-item cluster:** fine/can/finish/look all sit near the 0.5 floor — the
  weakest survivors. The live test will tell us if they're usable.
- **Files:** features in `pipeline/word_features.py`, ingested by
  `pipeline/ingest_words.py`, trained by `training/train_words.py`, exported by
  `training/export_words.py` → `extension/model/words.json`.

### 4c. The math you should be able to say out loud

**Normalisation (makes *space* fair):**
```
P_normalised(i) = ( P(i) − P(wrist) ) / distance(P(wrist), P(knuckle_9))
```
Plain English: *move the hand to the origin (subtract the wrist), then shrink or
grow it to a standard size (divide by hand size).* This is why the model works on
hands it never trained on.

**Temporal window (makes *time* part of the input):** a word sign is a movie, not
a photo, so the input shape is `[30 time-steps, 68 features]` and the LSTM reads
those 30 frames *in order*.

---

## 5. The extension architecture — why it's shaped the way it is

This is the part that took the most blood (the 11 Jun "great debugging arc"). The
naïve design — run the camera + MediaPipe + model inside the content script —
**does not work**, for three real reasons we hit live:

1. **MediaPipe WASM can't load in a content script.** Content scripts run in an
   "isolated world" under the *page's* Content Security Policy → "ModuleFactory
   not set".
2. **TensorFlow.js can't run in an MV3 extension page at all** — it calls
   `eval()`/`new Function()` at startup, which Manifest V3 forbids.
3. **Brave hides CPU features** (fingerprint protection) → MediaPipe asks for a
   *no-SIMD* WASM variant that has to be bundled or you get instant "Init Failed".

**The solution = the offscreen-document architecture.** Here's the real wiring:

```
┌─────────────────────────── TTM EXTENSION (Manifest V3) ───────────────────────────┐
│                                                                                    │
│  popup.html / popup.js ──► toggles: enable, chat-injection, threshold, spacing     │
│         │  writes settings to chrome.storage.local                                 │
│         ▼                                                                           │
│  background.js (service worker)                                                     │
│    • watches storage → creates/destroys the offscreen document                     │
│    • RELAYS messages: offscreen ⇄ content scripts (they can't talk directly)       │
│    • opens permission.html once if the camera is denied                            │
│         │                                                                           │
│    ┌────┴───────────────────────────┐         ┌──────────────────────────────┐    │
│    ▼                                 ▼         ▼                              │    │
│  offscreen.html / offscreen.js              content.js (runs IN the Meet tab) │    │
│   (invisible page — does ALL the work)       (UI ONLY — no ML)                │    │
│    • getUserMedia() — your webcam              • draws the floating overlay    │    │
│    • MediaPipe HandLandmarker (WASM)           • lock-in gate, spacing, buffer │    │
│    • features.js → normalise                   • Clear / Backspace buttons     │    │
│    • inference.js → both models                • (soon) assembler.js for prose │    │
│    • sends one msg per frame ──────────────────► renders it                    │    │
│                                                • injectIntoChat() (optional)   │    │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Key point:** `offscreen.js` and `content.js` never talk directly —
`background.js` is the switchboard between them. The offscreen page is **invisible
and never rendered**, which is why `offscreen.js` uses `setInterval` instead of
`requestAnimationFrame` (rAF doesn't fire on a page that's never drawn).

### File-by-file map of `extension/`

| File | Role |
|------|------|
| `manifest.json` | The rulebook. MV3, permissions (`storage`, `offscreen`), which sites to inject into, the CSP allowing `wasm-unsafe-eval`. |
| `background.js` | Service worker / switchboard. Lifecycle + message relay. |
| `offscreen.js` | **The engine.** Camera + MediaPipe + both models. Streams one message per frame. |
| `content.js` | **The UI brain.** Renders overlay, turns the prediction stream into letters→words→sentence. |
| `features.js` | Shared feature math (normalise + build word sequence). **One implementation, imported by both the live code and the parity test.** |
| `inference.js` | Plain-JS forward pass for both the dense classifier and the LSTM. The "no TF.js" replacement. |
| `assembler.js` | Pure prose-builder (capitals, punctuation, autocorrect). Built + tested, not yet wired. |
| `popup.html/js` | The toolbar popup: on/off, sliders, chat toggle. |
| `permission.html/js` | One-time camera-permission grant page. |
| `overlay.css` / `style.css` | The overlay's look. |
| `model/weights.json` | The alphabet model's numbers (379 KB). |
| `model/words.json` | The word LSTM's numbers (5 MB). |
| `model/hand_landmarker.task` | MediaPipe's hand-tracking model (7.5 MB). |
| `lib/vision_bundle.js` | MediaPipe Tasks-Vision JS. |
| `lib/tf.min.js` | **Dead leftover** — TF.js, no longer used by anything. Safe to delete (see §10). |
| `wasm/*` | MediaPipe WASM, both SIMD and no-SIMD variants (the Brave fix). |

---

## 6. The dominant constraint: Python and JS must compute IDENTICAL numbers

This is the rule that governs all model work, and it's worth understanding deeply
because it's the kind of subtle bug that *passes every test and still fails live*.

The model is trained in Python on features computed by `pipeline/word_features.py`
and `pipeline/normalize.py`. Live, the features are computed by
`extension/features.js`. **If those two ever compute the numbers even slightly
differently, the model gets fed different inputs live than in training — and
silently gets worse, with no error anywhere.**

The defence (built 26–27 Jun):
- `features.js` is **one shared module** — `offscreen.js` imports it, it isn't
  copy-pasted.
- `tests/smoke_features.mjs` replays a **Python-generated fixture** and asserts
  the JS output matches Python to **3.95e-7**. That test is the contract.

So there are really **two layers of parity tests**, both green:
- `smoke_model.mjs` / `smoke_words.mjs` — prove the JS *model math* matches Keras.
- `smoke_features.mjs` — proves the JS *feature math* matches Python.

**The one idea:** "the model matches Keras" was not enough — you also have to
prove "the features match Python." Both are now locked behind tests.

---

## 7. Tech stack (what's actually installed and used)

**Offline (training — Python, in `venv/`):**
- Python 3.12, MediaPipe, OpenCV, NumPy — landmark extraction + augmentation.
- TensorFlow / Keras — train both models (`models/alphabet.h5`, `models/words.h5`).
- **No `tensorflowjs` converter** — it's incompatible with NumPy 2.x on Windows.
  We wrote our own JSON weight-exporters instead (`export_weights.py`,
  `export_words.py`), each with a NumPy-vs-Keras smoke test baked in.

**Online (the extension — runs in Chrome/Brave):**
- **Manifest V3** — current Chrome extension format.
- **MediaPipe Tasks-Vision** (`HandLandmarker`) — current JS hand-tracking API.
- **Plain JavaScript inference** (`inference.js`) — NOT TF.js. Hand-written matrix
  math + LSTM cell, verified against Keras to ~1e-7. This was forced by MV3's
  `eval` ban and turned out to be a *strength* (tiny, auditable, no dependency).
- Plain JS + `<canvas>`/DOM overlay + CSS for the UI.

**Testing:**
- `pytest` (Python) — 18 tests (augmentation, word features).
- Node `.mjs` smoke tests — 4 (model parity, word parity, feature parity, assembler).
- `puppeteer-core` — drove Brave with a fake camera to debug the extension headless
  (`tools/debug_extension.cjs`).

**Why no server is a feature, not a gap:** free to run, private (webcam never
leaves the machine), nothing for you to maintain or pay for. Lean on this in the
README.

---

## 8. The data story (short — full catalog in datasets.md)

- **Decision locked:** v1 = **ASL**. **Public datasets only**, no self-capture.
- **Alphabet:** Kaggle ASL Alphabet (GPL-2.0!) + Lexset synthetic.
- **Words:** Google `asl-signs` Holistic landmarks (the huge shortcut — already
  landmarks, no video processing) via the `gislr-dataset-public` republish.
- **The big honest catch — licensing.** Most academic sign datasets are
  **research-only**. The Kaggle alphabet data is **GPL-2.0**. *Training to learn
  is fine; distributing a product trained on them is a legal gray zone.* Before
  any Chrome Web Store listing, the **shipped** model's training data must be on
  permissive licenses (or replaced). This is **Open Question #1** and it gates
  publishing. The safe fallback finish line — a polished public "load-unpacked"
  repo — sidesteps it.
- **Expansion already scouted:** ISL has a landmark-native dataset waiting (swaptr
  ISL MediaPipe set), so the path to an Indian Sign Language version is real, not
  hypothetical.

---

## 9. What's left — the road to a finished v1

Pulled straight from [`tasks/plan.md`](tasks/plan.md) and
[`tasks/todo.md`](tasks/todo.md). Six tasks remain:

```
[DONE: T1–T16 — letters 93.6% live, words 79.4%, assembler built]
      │
      ├─ T17 slice 2: wire assembler.js into content.js  (needs live reload)
      ├─ T18: fusion tuning — letters and words stop fighting
      │
      ▼
   T19: HONEST COMBINED LIVE TEST on a real Meet  ◄── YOU run it (this is next)
      │   record per-letter + per-word hit rates, list failure modes
      ▼
   Phase 6 — polish + publish:
   T20 settings/options UI → T21 chat-injection hardening →
   T22 README→TTM + privacy + license + finish-line decision
```

**The next concrete step is T19 — the live test — and it's deliberately pulled to
the front** because every later decision depends on its numbers. We do that
together in the testing session.

---

## 10. Honest limitations & loose ends (so nothing surprises you)

1. **No honest combined accuracy number exists yet.** The headline gap.
2. **J and Z can't work** on the alphabet model — they need motion. Known physics
   limit, not a bug. (A future temporal letter model could fix it.)
3. **Word vocab is 39 words.** Useful for a demo, not a conversation. The
   fine/can/finish/look cluster is shaky (~0.5).
4. **The assembler isn't wired in.** Output is still `HELLO HOW ARE YOU`, not
   `Hello, how are you?` — the pretty layer exists but isn't connected (T17.2).
5. **Chat injection is fragile and untested** — Meet/Teams change their HTML; the
   selectors will rot. It's a behind-a-toggle bonus, never the core.
6. **Licensing is unresolved** for shipping (§8). Blocks the Web Store path.
7. **`extension/lib/tf.min.js` is dead weight** (1.4 MB) — nothing imports it
   anymore. A clean-up commit could drop it.
8. **`traj_z` is a dead feature column** (always ~0) — flagged to drop on the next
   word-model retrain.
9. **README still says "Sign-to-Text"** while the repo and product are "TTM".
10. **Real-world accuracy is humbling.** Lab light + one signer ≫ your bedroom at
    night. The green/red health dot and manual Backspace/Clear are what make it
    usable *despite* imperfect accuracy — the feedback loop matters more than
    chasing a perfect number.

---

## 11. How this becomes bigger — scalability & helping others

You asked specifically: *how do we make this scalable, how can it help others?*
Here's the honest map, cheapest-first.

**Make it help more people (product reach):**
- **More words.** The word pipeline already ingests Google's *250*-word dataset;
  you ship 39 because of the honesty floor. Better features (face landmarks) lift
  the floor → more words pass → bigger vocab, same code.
- **Indian Sign Language (ISL).** Biggest "help others" lever for *your* context —
  a landmark-native ISL dataset is already scouted. ISL ≠ ASL, so it's a second
  model, not a tweak, but the entire pipeline (ingest → features → LSTM → export →
  plain-JS) is reusable. This is the headline expansion in the constitution's "so
  it isn't a waste" sense.
- **Other sign languages.** `datasets.md` Tier 5 lists German, BSL, Chinese,
  Turkish, Arabic, etc. The architecture is language-agnostic — each new language
  = ingest a dataset + retrain + export. OpenHands pretrained pose models can
  bootstrap each one instead of training from zero.

**Make it scale technically:**
- **It already scales by design** — all compute is on the user's machine, so
  100 users or 100k users cost *you* nothing. There's no server to fall over.
- **Latency budget** is the real constraint, not throughput: grab→predict→render
  must stay well under ~100 ms. That's why models are small and the LSTM runs at
  ~2.5 Hz, not 20. Adding a face-landmark model (Option B) risks this budget — to
  be measured, not assumed.
- **A continuous-sentence model** (full grammar, not isolated words) is the
  research-grade ceiling — out of scope for v1, listed for honesty.

**Make it credible / shareable:**
- **The finish line decision (T22):** a polished public repo people can
  load-unpacked is the safe, license-clean shipping path; the Chrome Web Store is
  the high-visibility path but needs the license question solved first.
- **The honest README** — what it does, install steps, *honest accuracy limits*,
  respectful language about the Deaf community, and the privacy guarantee (webcam
  never leaves the device). That honesty is itself a differentiator.

---

## 12. How to run & test it (the short version — full walkthrough is the next session)

**Run the Python tests:**
```powershell
.\venv\Scripts\Activate.ps1
python -m pytest -q
```

**Run the JS parity/smoke tests (proves models + features are correct):**
```powershell
node tests/smoke_model.mjs
node tests/smoke_words.mjs
node tests/smoke_features.mjs
node tests/smoke_assembler.mjs
```

**Load the extension live (the real test):**
1. Open `brave://extensions` (or `chrome://extensions`).
2. Turn on **Developer mode** (top-right).
3. Click **Load unpacked** → select the `extension/` folder.
4. Join a Google Meet, click the TTM toolbar icon, toggle it **on**, grant the
   camera once (a permission tab opens the first time).
5. Sign letters and words; watch the overlay's **"Model sees (live)"** line — that
   per-frame guess is your surgical debugging tool.

All the model binaries are **already present** in `extension/model/` and
`extension/wasm/`, so it loads with no extra downloads. (If they were ever
missing, `pipeline/download_extension_assets.py` re-fetches them.)

**This whole section gets done *with you, hands-on*, in the testing session — this
is just the map.**

---

## 13. The teach-back checklist (the constitution's test)

The mandate's bar: *could you explain this in an interview?* Tick these off out
loud. If you can't, that part isn't learned yet — and that's what we fix.

- [ ] Why is this **two ML problems**, not one? (static letters vs. dynamic words)
- [ ] What does **normalisation** do and why does it let the model work on a
      stranger's hand? (anchor to wrist, scale by hand size)
- [ ] Why a **63-number** input and not the raw image? (compression = speed)
- [ ] Why an **LSTM** for words but a plain **dense net** for letters? (time/order)
- [ ] Why couldn't the model tell **dad from mom**, and what feature fixed it?
      (same shape + barely moves → needed absolute wrist *location*)
- [ ] Why is there **no TensorFlow.js** in the extension? (MV3 forbids `eval`)
- [ ] Why the **offscreen document**? (content scripts can't load MediaPipe WASM)
- [ ] Why does **`background.js`** exist? (offscreen and content can't talk directly)
- [ ] Why does **"matches Keras" not prove correctness** on its own? (features must
      also match Python — the silent-divergence trap)
- [ ] Why is **93.6% on held-out signers** the real number, not 99%? (overfitting
      vs. generalisation)

---

## 14. File index (the whole project at a glance)

```
sign-to-text/
├─ README.md               ← public front page (still says "Sign-to-Text" — T22)
├─ in-depth-study.md       ← the CONCEPT doc (the "why")
├─ project-brain.md        ← THIS FILE (the "what exists now")
├─ datasets.md             ← every sign dataset on the web, by language + license
├─ log.md                  ← append-only session history (the honest blow-by-blow)
├─ tasks/plan.md, todo.md  ← the remaining-work plan
│
├─ extension/              ← THE PRODUCT (load this unpacked in the browser)
│   ├─ manifest.json, background.js, content.js, offscreen.js/html
│   ├─ features.js, inference.js, assembler.js
│   ├─ popup.*, permission.*, *.css
│   ├─ model/  (weights.json, words.json, hand_landmarker.task)
│   ├─ lib/    (vision_bundle.js, tf.min.js←dead)
│   └─ wasm/   (MediaPipe WASM, SIMD + no-SIMD)
│
├─ pipeline/               ← OFFLINE data prep (Python)
│   ├─ normalize.py        ← the one true normalisation (alphabet)
│   ├─ word_features.py    ← the 68-feature word recipe (MUST match features.js)
│   ├─ ingest_alphabet.py, ingest_words.py
│   ├─ augment.py          ← mirror / rotate / jitter / 3D-rotation
│   └─ track_hands.py, collect_data.py, download_extension_assets.py
│
├─ training/               ← OFFLINE model training (Python)
│   ├─ train_alphabet.py, train_words.py
│   ├─ evaluate.py         ← held-out-signer benchmark (the honest number)
│   └─ export_weights.py, export_words.py, export_tfjs.py
│
├─ models/                 ← trained Keras models (alphabet.h5, words.h5) + labels
├─ data/                   ← ingested landmarks (alphabet/*.npy, words/labels.json)
└─ tests/                  ← pytest (augment, word_features) + .mjs smoke tests
```

---

*Maintain this file.* When a phase ships or a number changes, update §2 (the
status board) and the relevant section, and add the one-line session entry to
`log.md`. This is the document you re-read when you forget what you built.

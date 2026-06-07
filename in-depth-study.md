---
name: in-depth-study
description: Full ground-up study of the Sign-to-Text project — problem, daily-life walkthrough, ML + hand-tracking pipeline, data strategy, and a staged build plan to ship a public Chrome extension for Meet/Zoom/Teams.
type: reference
status: draft
last-updated: 2026-06-07
---

# Sign-to-Text — In-Depth Study

> Goal of *this* document: not to hand you code, but to make you able to **explain
> every part of this project out loud**, the way the constitution's teaching
> mandate demands. Read it slowly. Each section ends with the one idea that matters.

---

## 0. Honest reality check first (read this before getting excited)

`firstread.txt` is a good blueprint, but it is written in "confident demo" voice.
A few claims in it are wrong or much harder than they sound. I'd be doing you a
disservice (and breaking the anti-sycophancy rule) if I just re-typed it prettier.
Here is the truth up front:

1. **This is actually two different ML problems, not one.** Spelling letters
   (H-E-L-L-O) and signing whole words ("Task", "Finished") are recognised by
   *different* kinds of models. `firstread.txt` mixes them in one breath. We will
   build them as two separate stages.
2. **The dataset names in `firstread.txt` are partly wrong.** It says *"For
   Fingerspelling (Alphabet A-Z): WLASL."* WLASL literally stands for **W**ord-
   **L**evel **ASL** — it is *not* a fingerspelling alphabet set. I'll give you the
   correct mapping in Section 6.
3. **"Train until 92% accuracy" is a trap number.** It's easy to hit 92% on *your
   own hand* and then get 40% on a stranger's hand in bad lighting. Real success is
   measured on signers and conditions the model has *never seen*. That's the whole
   reason you want a large, varied corpus.
4. **Auto-injecting text into the Meet/Zoom/Teams chat box is the most fragile,
   ToS-risky part.** It will be the *last* thing we build, behind a toggle, and it
   may break every time Google changes their HTML. The reliable core is the overlay
   that *shows* the text on your own screen.

None of this kills the project. It just means we build it in **stages**, smallest
working thing first. That also matches your anti-pattern list: no architecting v9
while nothing runs.

**The one idea:** It's two ML problems wearing one coat, and the "send to chat"
feature is a fragile bonus, not the foundation.

---

## 1. The problem & who it's for

A Deaf or hard-of-hearing person whose first language is a sign language (Indian
Sign Language / American Sign Language) is at a disadvantage on a normal video
call. Hearing colleagues talk; the Deaf person signs — but the hearing side often
can't read sign language. Today they fall back to typing in chat, which is slow and
breaks the flow of a real conversation.

**What we build:** software that watches the signer's webcam, recognises the signs,
turns them into English text in real time, and shows that text on the call — first
as an on-screen overlay, later (optionally) injected into the meeting chat so
hearing people just read it.

**Note on language ("dumb/deaf"):** the respectful term is **Deaf** (or
*hard-of-hearing*). "Dumb" is considered offensive now — worth knowing because this
project is *for* that community, and the language you use in the README and UI
matters to them. Small thing, real thing.

**The one idea:** We translate *sign → English text live on a call*, so a Deaf
person can converse at conversation speed instead of typing.

---

## 2. Daily-life walkthrough (the requirement walkthrough)

This is the "how a real person uses it" view you asked for. I've made it concrete.

**Meet Aravind** — a backend engineer who is Deaf and signs in ASL.

**08:55 — joining.** Aravind opens the Google Meet link for standup in Chrome. In
his browser toolbar he clicks the **SignToText** icon. A small, semi-transparent
panel slides in on the right edge of the Meet window. It says
`[ Camera: ON · Model: loaded · Status: waiting for a sign… ]`.

**08:56 — calibration.** The panel shows a tiny live thumbnail of his own webcam
with a green skeleton drawn over his hand — 21 glowing dots from wrist to
fingertips. Green = "I can see your hand clearly." If the dots flicker red, he
knows to move into better light or center his hand. This feedback is *critical* —
without it the user has no idea why translation is failing.

**09:00 — his turn to talk.** The manager asks for his update.

- He **fingerspells** his ticket id: `J-I-R-A` `dash` `4` `0` `2`. As each letter's
  handshape holds steady for a beat, a letter pops into the buffer. The panel groups
  them: `JIRA-402`.
- He switches to **whole-word signs**: he signs *FINISH*, then *DEPLOY*, then
  *TODAY*.
- Between signs his hands drop and relax for about ¾ of a second. The system reads
  that pause as a word boundary and inserts spaces.
- The overlay now reads: **"JIRA-402 finished, deploy today."**

**09:01 — sharing it.** Aravind taps **Send to chat** (a button he chose to enable).
The text drops into the Meet chat; his hearing teammates get the chat ping and read
it. He never touched the keyboard.

**09:02 — a correction.** The model misreads one sign and writes "deploy *Friday*".
Aravind clicks the word in the overlay and picks "today" from a small dropdown, or
just backspaces and re-signs. **Design rule: the human is always able to correct
the machine.** A translation tool that can't be corrected is worse than useless in
a real meeting.

**The one idea of the walkthrough:** the product is *live captioning of the
signer*, with constant visual feedback and an always-available correction path —
not a magic black box.

---

## 3. How the whole thing works — the pipeline, in plain words

Think of it as an assembly line. A video frame goes in one end; a word comes out
the other. Five stations:

```
 Webcam frame                                                  English text
      │                                                             ▲
      ▼                                                             │
┌───────────┐   ┌────────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐
│ 1. GRAB   │──▶│ 2. TRACK   │──▶│ 3. NORMAL- │──▶│ 4. PREDICT│──▶│ 5. BUILD │
│ a frame   │   │ the hand   │   │ ise the    │   │ the sign  │   │ sentence │
│ from cam  │   │ (21 dots)  │   │ numbers    │   │ (the ML)  │   │ + spaces │
└───────────┘   └────────────┘   └────────────┘   └───────────┘   └──────────┘
```

1. **Grab.** Take the current picture from the webcam, many times a second.
2. **Track.** A library called **MediaPipe** finds the hand and returns **21
   landmark points** — wrist, knuckles, fingertips — each as an (x, y, z) number.
   This is the magic step: it turns a heavy 1280×720 *image* into a tiny list of 63
   numbers (21 points × 3 coordinates). Models love small lists; they choke on raw
   pixels.
3. **Normalise.** Make those numbers fair, so a big hand close to the camera and a
   small hand far away produce the *same* numbers for the same sign. (Math in §5.)
4. **Predict.** Feed the numbers to the trained model. It outputs a probability for
   each known sign and we take the highest — "I'm 94% sure that's the sign FINISH."
5. **Build sentence.** Decide when a word is "locked in", when to insert a space,
   fix obvious typos, and stream the growing sentence to the screen.

**The one idea:** the project's real trick is **compression** — collapse a fat
video frame into 63 numbers *before* any ML runs. That's what makes it fast enough
for a live call. `firstread.txt` calls this "Skeletal Data Compression" and it's
the single most important design choice.

---

## 4. The key insight `firstread.txt` glosses over: two recognition problems

This deserves its own section because it shapes the whole build.

### 4a. Fingerspelling = mostly **static** handshapes
Spelling A, B, C is mostly about the *shape* of a still hand. The letter "B" looks
like "B" in a single freeze-frame. So you can recognise it with a **per-frame
classifier**: look at one frame's 63 numbers → output a letter. Simpler model,
simpler data, easier to ship. (A few letters like J and Z involve motion — handle
those as small exceptions.)

### 4b. Word signs = **dynamic** motion over time
"GO" vs "COME" can be the *same handshape* moving in opposite directions. A single
frozen frame can't tell them apart — you need the *sequence over time*. So you feed
a **window of frames** (e.g. 30 frames ≈ 1 second) into a model that understands
order: an **LSTM** or a small **Transformer**. Harder model, much harder data.

### Why this matters for *you*
- **Build 4a first.** A working alphabet-fingerspelling extension is a real,
  demoable, shippable thing in weeks. It teaches you the whole pipeline end to end
  on the *easy* version of the ML.
- **Then graduate to 4b** for a starter vocabulary of common whole-word signs
  (hello, yes, no, thanks, finish, help…).

**The one idea:** static handshapes (letters) and moving signs (words) are
different ML problems; ship the static one first to learn the pipeline, then add
motion.

---

## 5. The math, explained so you could say it in an interview

Two pieces of math, both easy once you see *why*.

### 5a. Coordinate anchoring + scaling (normalisation)
**Problem:** if you sit closer to the camera, every landmark number gets bigger,
even though your hand made the *same* shape. The model would think it's a different
sign. Bad.

**Fix, in two moves:**
1. **Anchor.** Pick the wrist (landmark 0) and call it the origin (0,0,0). Subtract
   the wrist's position from every other point. Now the numbers describe the hand's
   shape *relative to its own wrist*, no matter where the hand is on screen.
2. **Scale.** Divide every point by the **hand size**, measured as the distance
   from the wrist (point 0) to the middle-finger knuckle (point 9). Now a big hand
   and a small hand making the same shape give the same numbers.

In symbols (this is all `firstread.txt`'s formula means):

```
P_normalised(i) = ( P(i) − P(wrist) ) / distance(P(wrist), P(knuckle_9))
```

Plain English: *"move the hand to the origin, then shrink/grow it to a standard
size."* That single step is what lets your model work on hands it never trained on.

### 5b. Temporal windows (the "time" math for word signs)
A word sign is a *movie*, not a *photo*. So instead of feeding one frame, you feed a
stack: the last 30 frames. The shape of that data block is:

```
[ batch_size , 30 time-steps , 63 features ]
                  ▲                ▲
            1 second of      21 points × (x,y,z)
            video frames
```

The LSTM reads those 30 frames *in order* and outputs "this 1-second clip is the
sign FINISH (92%)." The word "temporal" just means "across time." That's it.

**The one idea:** normalisation makes the *space* fair (any hand, any distance);
temporal windows make *time* part of the input (so motion-based signs are
distinguishable).

---

## 6. Data strategy — the large corpus you asked for (with the names corrected)

You said you want a *large, varied* dataset for accuracy. Correct instinct — variety
beats raw size. A model trained only on your hand fails on everyone else. Here is
the corrected dataset map (`firstread.txt` mislabeled these):

| You want to recognise | Use these public datasets | Notes |
|---|---|---|
| **Fingerspelling (A–Z)** | **ChicagoFSWild / ChicagoFSWild+**, **ASL Alphabet (Kaggle)** | ChicagoFSWild is real fingerspelling "in the wild." Kaggle ASL Alphabet is huge and beginner-friendly. |
| **Word-level ASL signs** | **WLASL** (2,000 words), **MS-ASL** | WLASL = *Word-Level ASL* — this is words, **not** the alphabet, despite what `firstread.txt` says. |
| **Continuous sentence translation** | **RWTH-PHOENIX-Weather 2014T**, **How2Sign** | PHOENIX is **German** Sign Language, weather-domain only — great for research, *not* a general English chat model. Know that before you lean on it. |
| **Indian Sign Language (ISL)** | **INCLUDE / INCLUDE-50**, **ISL-CSLTR** | If your real users are Indian, ISL ≠ ASL. Different signs entirely. Decide your target language early. |

**Decision you must make early:** *which sign language?* ASL has the most data; ISL
is closer to home for India. They are not interchangeable. Pick one for v1. (I'd
start ASL fingerspelling because the data is enormous and clean, then add ISL.)

### Building a bigger corpus *without* filming thousands of videos — augmentation
Once you've converted videos to landmark numbers, you can **multiply** your data
with a small NumPy script. This is honest data expansion, not faking:

- **Spatial jitter:** add tiny random noise (±0.01) to coordinates → simulates
  shaky hands / imperfect tracking.
- **Temporal stretch:** drop or duplicate every Nth frame → simulates fast/slow
  signers.
- **Mirroring:** flip the X-axis (`x_new = 1.0 − x`) → instantly turns every
  right-handed example into a left-handed one. Doubles your data and your model
  stops being right-hand-only.
- **Rotation/scale jitter:** small rotations and size changes → camera angle
  robustness.

**The one idea:** variety > volume. Use the *correct* public datasets for the
*specific* problem (alphabet vs words vs sentences vs ISL), then multiply with
landmark-level augmentation so the model generalises to strangers' hands.

---

## 7. Tech stack — the modern, correct versions

`firstread.txt` is mostly right but names some deprecated tools. Current stack:

**Offline (training — Python, on your machine):**
- **Python + OpenCV** — read dataset videos frame by frame.
- **MediaPipe (Python)** — extract the 21 landmarks per frame.
- **NumPy** — store landmarks as `.npy` arrays; run augmentation.
- **TensorFlow / Keras** — build and train the model.
- **tensorflowjs (converter)** — export the trained model to browser format.

**Online (the extension — runs in Chrome):**
- **Manifest V3** — the current Chrome extension format (V2 is dead).
- **MediaPipe Tasks – Vision** (`HandLandmarker` / `GestureRecognizer`) — the
  *current* JS hand-tracking API. The old "MediaPipe Hands solution" in
  `firstread.txt` is deprecated; use Tasks.
- **TensorFlow.js (TFJS)** with the **WebGL** (or WebGPU) backend — runs your
  trained model on the user's GPU, right in the browser, no server.
- **Plain JS + a `<canvas>` overlay + CSS** for the UI panel.

**Why no server?** Everything runs locally in the browser. That means: free to run,
private (the webcam video never leaves the user's machine), and no backend for you
to maintain. This is a genuine strength of the design — lean into it in your README.

**The one idea:** train in Python offline, *convert* the model, and run it
100% client-side in the browser with TFJS + MediaPipe Tasks — no server, private by
default.

---

## 8. The architecture, as a picture

```
┌──────────────────────── CHROME EXTENSION (Manifest V3) ────────────────────────┐
│                                                                                  │
│  popup.html / panel  ──►  user toggles, status, the live text box (overlay UI)   │
│                                                                                  │
│  content script (runs inside the Meet/Zoom/Teams page)                           │
│     │                                                                            │
│     ├─ getUserMedia() ──► your OWN webcam stream                                  │
│     │                                                                            │
│     ├─ MediaPipe HandLandmarker (WASM) ──► 21 landmarks / frame                  │
│     │                                                                            │
│     ├─ normalise (anchor + scale)  ──► 63 clean numbers                          │
│     │                                                                            │
│     ├─ rolling buffer of last 30 frames                                          │
│     │                                                                            │
│     ├─ TFJS model.predict() ──► sign + confidence                                │
│     │                                                                            │
│     ├─ sentence builder (lock-in, spacing, autocorrect)                          │
│     │                                                                            │
│     └─ render to overlay  ──► (optional) inject into chat box                     │
│                                                                                  │
│  background service worker ──► loads model assets, manages on/off state          │
└──────────────────────────────────────────────────────────────────────────────┘
```

One correction to `firstread.txt`'s diagram: for translating **your own** signing,
you read **your own webcam** via `getUserMedia()`. You do **not** scrape the call's
`<video>` elements — those are the *other* participants. (Reading call videos would
only matter if you wanted to translate *someone else* on the call.)

---

## 9. Sentence-building logic (Phase 4 in `firstread.txt`, explained)

Two problems to solve so the text doesn't turn to mush:

**Problem A — the model fires the same word 30× while you hold a sign.**
Fix = **lock-in gate**: only accept a word if its confidence stays above ~90% for
several frames in a row (say 5). One confident streak = one word. This is called
*debouncing*.

**Problem B — when is one word finished and the next begun?**
Fix = **pause detection**: watch the wrist's movement. When the hand stops moving
(velocity ≈ 0) and relaxes for ~750 ms, that's a word boundary → push the current
word into the sentence and add a space.

Then a light **autocorrect / n-gram** pass fixes obvious nonsense ("helo" → "hello")
and capitalisation.

**The one idea:** raw predictions are a firehose; the sentence builder is the
*filter* that turns a stream of guesses into clean, spaced, corrected words.

---

## 10. The hard / honest parts (so you're not surprised)

1. **Chat injection breaks.** Meet/Zoom/Teams are built with React and obfuscated,
   ever-changing CSS class names. The `document.querySelector(...)` trick in
   `firstread.txt` *will* break when Google ships an update. Treat it as a
   best-effort, behind-a-toggle bonus. The overlay is the reliable product.
2. **Terms of Service.** Some platforms frown on extensions that automate their UI.
   Showing an overlay on your own screen is fine; auto-posting messages is grayer.
   Keep it user-initiated (a button press), not silent.
3. **Content Security Policy (CSP).** Strict pages can block scripts/WASM you try to
   load. MediaPipe Tasks + TFJS in MV3 need their assets bundled *inside* the
   extension and declared as `web_accessible_resources`, not fetched from a CDN.
4. **Accuracy in the wild is humbling.** Lab accuracy (clean light, one signer) ≫
   real accuracy (your bedroom at night, a new signer). The green/red feedback dot
   and the manual-correction path are what make it usable *despite* imperfect
   accuracy. Don't chase a magic number; chase a good feedback loop.
5. **Latency budget.** To feel "live," the grab→predict→render loop should finish
   well under ~100 ms. That's why we compress to 63 numbers and run a *small* model,
   not a giant one.

**The one idea:** the overlay is the product; chat-injection is a fragile extra;
and a good *correction/feedback* loop beats chasing a perfect model.

---

## 11. Staged build plan (smallest working thing first)

This is sequenced so you always have something that *runs*, and so it feeds your
200-commit target honestly. Each stage = real commits.

**Stage 0 — Sandbox setup.**
Make the folder, a Python virtual environment (venv, *never* global per the
constitution), install deps, `git init`, push an empty public repo. You learn: git
basics, venv. *Deliverable: empty repo that's backed up.*

**Stage 1 — See the dots (no ML yet).**
A tiny Python script: open webcam, run MediaPipe, draw the 21 landmarks live. You
*see* the skeleton on your hand. *Deliverable: proof the tracking works on your
machine.* This is the single most motivating first win.

**Stage 2 — Collect & normalise data (alphabet).**
Script to record landmark `.npy` files for letters; apply normalisation + the
augmentation tricks. *Deliverable: a clean dataset folder.*

**Stage 3 — Train the alphabet model (the easy ML).**
A per-frame classifier on the static letters. Validate on hands it *didn't* train
on. *Deliverable: `model.h5` that reads fingerspelling.*

**Stage 4 — Get it into the browser.**
Convert with `tensorflowjs_converter`. Build a minimal Manifest V3 extension that
opens your webcam, runs MediaPipe Tasks + the TFJS model, and prints the predicted
letter in a popup. *Deliverable: the first real extension — spells words from your
hand.* **This is your first shippable demo.**

**Stage 5 — The overlay on a real call.**
Inject the panel + live text box onto a Google Meet page. Add the green/red
tracking-health dot. *Deliverable: captions on your own screen during a real
Meet.*

**Stage 6 — Sentence logic.**
Lock-in gate, pause-based spacing, autocorrect. *Deliverable: clean sentences, not
letter soup.*

**Stage 7 — Word-level signs (the hard ML).**
LSTM on 30-frame windows for a starter vocabulary (~20–50 common signs). *
Deliverable: whole-word signing works for a small dictionary.*

**Stage 8 — Polish + publish.**
Settings, the (toggle-able, fragile) chat injection, a real README, and submit to
the Chrome Web Store. *Deliverable: a public extension people can install.*

**The one idea:** never build Stage N+1 until Stage N runs. That's the antidote to
"design-to-avoid-execution" — your own anti-pattern #1.

---

## 12. What you'll actually learn (the teaching map)

Per the constitution, building this *and being able to explain it* is the job. Here's
the skill you bank at each stage:

- **Stages 0–1:** git, venv, webcam I/O, what a "landmark" is. (Backend/tooling.)
- **Stages 2–3:** the ML loop — data → train → validate, overfitting, why you test
  on unseen hands. (Core ML literacy — exactly what interviews probe.)
- **Stage 4:** model export/conversion, how a browser runs ML (TFJS/WASM/WebGL),
  Manifest V3 basics. (Deployment.)
- **Stages 5–6:** DOM manipulation, real-time loops, debouncing, UX of feedback.
  (Frontend + systems thinking.)
- **Stage 7:** sequence models (LSTM/Transformer), temporal data shapes. (Deeper
  DL — the part that impresses.)
- **Stage 8:** shipping, README, store submission, versioning. (The "actually
  finished it" credential.)

After each stage we **stop and you explain it back** — that's the active-recall rule
from the teaching mandate, not optional.

---

## 13. Mini-glossary (so the jargon stops being scary)

- **Landmark** — one tracked point on the hand (e.g. a fingertip), as (x, y, z).
- **MediaPipe** — Google's library that finds those 21 points in an image, fast.
- **Normalisation** — rescaling numbers so they're comparable across hands/distances.
- **LSTM** — a neural network that remembers order, good for sequences over time.
- **Softmax** — turns raw model scores into probabilities that add up to 1.
- **TFJS (TensorFlow.js)** — runs trained models in the browser, on the GPU.
- **Manifest V3** — Chrome's current rulebook/format for extensions.
- **Content script** — extension JS that runs *inside* a webpage (the Meet tab).
- **Service worker (background)** — the extension's always-on coordinator.
- **Debounce** — ignore rapid repeats; accept one event when input settles.
- **Augmentation** — make more training data by mutating existing data.
- **Gloss** — a written label for a sign (how sign datasets are annotated).

---

## Where we are & the next concrete step

Right now the folder holds `firstread.txt` and this study. Nothing runs yet — and
that's fine, because the rule is *smallest working thing first*.

**The single next step is Stage 0 → Stage 1:** set up the sandbox (venv + git) and
get MediaPipe drawing the 21 dots on your hand from your webcam. That one script is
the motivating first win and the foundation everything else sits on.

When you're ready, say the word and we'll do Stage 0 together — you running the git
and venv commands yourself, me explaining each line as you go.

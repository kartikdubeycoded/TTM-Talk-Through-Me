---
name: landmark-map
description: The "dot-numbering" map behind risk #1 — exactly which landmarks the word model expects, where the 66-point training layout comes from, why the old→new MediaPipe mapping is portable, and the verify-first plan for ASL Citizen extraction and Phase B live. Prep work done during the 2026-07-21 download.
type: reference
status: prep — indices to be pinned + verified against real videos
last-updated: 2026-07-21
---

# The landmark ("dot-numbering") map

This is the detail behind [[risks#1. The dot-numbering mismatch]]. Goal: when the
videos land, extraction is *checked against a known source*, not guessed.

## What the model expects (ground truth)

The word model's input is **66 landmarks per frame**, laid out as:

| Slice | Points | Meaning | Source in `word_features.py` |
|-------|--------|---------|------------------------------|
| `[0:40]`  | 40 | **lips** | `LIPS_SLICE` → `whole_body_features(lips=...)` |
| `[40:61]` | 21 | **dominant hand** | `HAND_SLICE` → the proven 68-col hand recipe |
| `[61:66]` | 5  | **pose** (arms/shoulders) | `POSE_SLICE` → face-relative body block |

Confirmed from `pipeline/ingest_words.py` (`LIPS_SLICE=0:40`, `HAND_SLICE=40:61`,
`POSE_SLICE=61:66`) and `log.md:80`.

## Where that layout comes from (provenance)

The training tensor `data/raw/gislr-words/X.npy` is `(94477, 64, 66, 3)`, from the
Kaggle republish **`markwijkhuizen/gislr-dataset-public`** (a preprocessed subset
of Google's `asl-signs` / GISLR, which itself is **MediaPipe Holistic** = 543
points: 468 face + 21 + 21 hands + 33 pose).

**So the 66-point selection (which 40 face indices = "lips", which 5 pose indices)
is defined by markwijkhuizen's preprocessing — NOT by anything in this repo.**
That external index list is the authoritative key. It must be recovered from his
public notebook/dataset config, not invented.

- **Strong hypothesis (verify, don't trust):** the 40 lips = MediaPipe's standard
  `FACEMESH_LIPS` vertex set (which has exactly 40 unique points — the count
  matches). The 5 pose points are a small arm/shoulder subset of BlazePose's 33.
- **Must pin exactly:** the precise index list AND their order for both blocks.

## Why the old→new MediaPipe mapping is portable (the reassuring part)

Live in the browser we won't use Holistic (deprecated); we'll use the MediaPipe
**Tasks** API: `FaceLandmarker` (478), `PoseLandmarker` (33), `HandLandmarker` (21).

- **Face:** Tasks FaceLandmarker's first 468 points are the **same FaceMesh
  topology** as Holistic's 468 (the extra 10 are iris, indices ≥ 468). All
  `FACEMESH_LIPS` indices are < 468 → **they map 1:1**. ✅
- **Pose:** Tasks PoseLandmarker uses the **same BlazePose 33-point** topology as
  Holistic pose → **same indices, same points**. ✅
- **Hand:** HandLandmarker's 21 points match Holistic's hand 21 order. ✅

Meaning: the same index refers to the same anatomical dot in both worlds. The risk
collapses to (a) getting markwijkhuizen's exact selection right, and (b) matching
coordinate normalization (all should be image-normalized x,y in [0,1]; z differs
and we already drop hand-z as a dead column).

## Two places this map gets used

1. **ASL Citizen extraction (post-download):** run MediaPipe on each video →
   select these 66 dots in this order → feed `whole_body_features` → train.
2. **Phase B (live):** add FaceLandmarker + PoseLandmarker in `offscreen.js`,
   select the same lips/pose dots, compute `whole_body_features` identically to
   Python. This is the parity constraint the repo keeps stressing.

## BIG REALIZATION (2026-07-21): we mostly don't need markwijkhuizen's exact list

There are two *different* goals, with very different risk:

- **Goal A — reuse the EXISTING 185-word GISLR model live (Phase B).** This model
  was trained on markwijkhuizen's exact 66-point selection, so going live with *it*
  requires matching his precise 40-lip + 5-pose index list AND order. Hard exact
  match. **Defer this** — it's optional.
- **Goal B — train a NEW model on ASL Citizen (the whole point of the download).**
  Here **we control both ends** — we extract the videos AND we write the live code.
  So we can *define our own* consistent 66-point layout; there is no external index
  list to reverse-engineer. As long as extraction and `offscreen.js` use the **same**
  selection, the model is correct. **Risk #1 largely dissolves for Goal B.**

Conclusion: for the scale-up we're actually doing (Goal B), pick one clean layout,
use it for both extraction and live, done. Recovering markwijkhuizen's exact indices
only matters if we later decide to ship the old GISLR model live without retraining.

### The layout we'll define for Goal B
- **Lips (40):** MediaPipe `FACEMESH_LIPS` — the authoritative index list is:
  `[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
  14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312,
  13, 82, 81, 42, 183, 78]` (40 points, all < 468 → identical in Holistic and the
  Tasks FaceLandmarker). Source: MediaPipe FaceMesh lips set.
- **Hand (21):** HandLandmarker's native 21, in native order.
- **Pose (5):** our choice from BlazePose's 33 — the natural pick for face-anchored
  signs is `[nose(0), left_shoulder(11), right_shoulder(12), left_elbow(13),
  right_elbow(14)]`. **Decision to lock at build time**; whatever we pick, use the
  same 5 in extraction and live.

## Verify-first plan (do this BEFORE mass extraction)

1. Recover markwijkhuizen's exact lips(40) + pose(5) index lists from his public
   GISLR preprocessing. Write them into a single shared constants file.
2. Extract **3–5** ASL Citizen videos with those indices.
3. Sanity-check the feature values on a known face-anchored contrast:
   `dad` (hand near forehead) vs `mom` (hand near chin) — the wrist-relative-to-lips
   value must differ in the expected direction.
4. Only after that check passes, run the full 84k extraction.
5. Mirror the exact same selection + math in `offscreen.js` and re-verify JS vs
   Python to < 1e-4 (same bar as the existing `smoke_words.mjs`).

**Open items:** markwijkhuizen's precise index lists (pending); confirm ASL Citizen
extraction should use Tasks (not Holistic) so training data matches the live path.

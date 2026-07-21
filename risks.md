---
name: risks
description: The known landmines ahead for TTM's scale-up (ASL Citizen training + Phase B live model). Plain-language risk register so no session walks into an avoidable trap. Read alongside project-brain.md (current state) and landmark-map.md (the dot-numbering detail behind risk #1).
type: reference
status: active — opened 2026-07-21 (during the ASL Citizen / How2Sign download)
last-updated: 2026-07-21
---

# Risks & landmines — read before the downloads finish

Four problems that could waste days if hit blind. Ranked by damage.

---

## 1. The dot-numbering mismatch (HIGHEST — silent failure)

**Plain version.** The model watches ~66 tracked dots (hand + lips + body), not
video. The training data numbers those dots one specific way (markwijkhuizen's
GISLR preprocessing). ASL Citizen ships **raw video**, so we must run MediaPipe
ourselves and select the **exact same 66 dots in the exact same order**. If we
pick the wrong 40 lip points or the wrong 5 pose points, the model gets garbage
with **no error message** — captions just quietly go wrong. This project already
fought a silent Python↔JS divergence once; this is the same class of bug.

**Why it also blocks Phase B (live).** The 185-word whole-body model has **never
run live** — the live extension still runs the OLD 39-word hand-only model. Going
live means adding Face + Pose landmarkers in the browser and selecting the *same*
lips/pose dots there too. Same mapping problem, second front.

**The good news (found 2026-07-21).** Old MediaPipe Holistic and the new MediaPipe
Tasks landmarkers share the underlying FaceMesh (468) and BlazePose (33)
topologies — the same index means the same anatomical point. So the mapping is
*portable*, not a from-scratch guess. The remaining unknown is markwijkhuizen's
exact 40-lip + 5-pose index list and order. See **landmark-map.md**.

**Mitigation (non-negotiable): verify on 3–5 videos BEFORE extracting 84,000.**
Extract a handful, print the feature values for a known face-anchored sign
(`dad` = forehead vs `mom` = chin), and confirm they look right. Fail fast.

**Status (updated 2026-07-21):** largely DE-RISKED for the scale-up. Because we
control both the ASL Citizen extraction and the live code, we define our own clean
66-point layout (lips = FACEMESH_LIPS 40, confirmed authoritative list in
landmark-map.md; hand 21; pose = 5 chosen BlazePose points) and use it on both
sides — no need to reverse-engineer markwijkhuizen's exact indices. Exact-match is
only needed to ship the *old* GISLR model live (optional, deferred). Verify-on-5-
videos-first still applies before mass extraction.

---

## 2. How2Sign (33 GB) is probably an unusable download

**Plain version.** How2Sign is **continuous** sign language (full sentences), a
different model architecture than the word-by-word model you've built. Your own
notes already parked it. What you actually downloaded is
`PSewmuthu/How2Sign_Holistic` — landmark features in `.rar` files, still
continuous. **Expect: "we have 33 GB we can't use with the current pipeline."**
Not fatal, not wasted forever (it's the seed for a future continuous track), but
do not count it toward the ASL scale-up.

**Mitigation:** leave it parked; don't build against it now. Revisit only if/when
a continuous-sign track is deliberately opened.

**Status:** confirmed continuous; parked.

---

## 3. The graphics card can't train on native Windows

**Plain version.** The RTX 4050 exists, but the current training stack (TF 2.21)
can't use the GPU on native Windows. Training a 500–2,700-word model on ~84k
sequences on CPU is painfully slow.

**Options (decide before the data lands):**
- **Kaggle (free GPU):** upload the extracted landmarks (small — landmarks, not
  video) and train there. Least local friction. Recommended default.
- **WSL2 + PyTorch:** GPU works under Linux, but it's a **stack rewrite** (Keras →
  PyTorch) *and* the JS export/verify pipeline (currently Keras→JSON, verified to
  1e-7) must be rebuilt and re-verified. Heavy.
- **CPU only:** simplest, slowest. Fine for the 500-word subset maybe; not for 2,700.

**Mitigation:** extract locally (CPU, ~1 day, unavoidable), then train the small
landmark files on Kaggle. Keep Keras so the existing JS export path still works.

**Status:** undecided — needs Katti's call.

---

## 4. Disk space is tight

**Plain version.** ~96 GB free at the start. Downloads ~78 GB (45 + 33). Unzipping
ASL Citizen needs ~45 GB more on top of the zip. That can run the disk dry mid-unzip.

**Mitigation:** unzip ASL Citizen, extract landmarks, then **delete the 45 GB zip
and the raw MP4s** — we only need the landmark files afterward. Check free space
before unzipping.

**Status:** monitor; act at unzip time.

---

## Cross-cutting note (not a blocker, but don't forget)

ASL Citizen uses a **different vocabulary** than GISLR. Training on it means a new
label set, a new `words.json`, and a new model — the writer/assembler autocorrect
dictionary and UX are currently tuned to the small vocab. Expected, just flagged.

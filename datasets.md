---
name: datasets
description: Comprehensive catalog of public sign-language datasets usable for Sign-to-Text — organised by type (landmark-native, fingerspelling, isolated words, continuous) and language, with size, format, license status, and a recommended ingestion path for v1 (ASL) and the multi-language expansion later.
type: reference
status: active
last-updated: 2026-06-07
---

# Sign-Language Datasets — The Full Catalog

> You asked for "every dataset top to bottom of the web." This is the working
> catalog. It's organised so the **most directly usable** sets are at the top, and
> the global expansion (ISL + every other language) is at the bottom — the data you
> add later so the project "isn't a waste."
>
> **Read the license column.** Most academic sign datasets are **research /
> non-commercial only**. We are *publishing* an extension, which is distribution.
> See the "License reality" section — it's a real decision, not paperwork.

---

## ⭐ Tier 0 — Landmark-native (start here, no video processing)

These are the single best fit for our project because they are **already MediaPipe
landmark coordinates** (the 63-numbers idea), not raw video. We skip the slow step
of running MediaPipe over thousands of clips for the bulk of training.

| Dataset | Language | Content | Size | Format | License (check!) |
|---|---|---|---|---|---|
| **Google – Isolated Sign Language Recognition** (`asl-signs`, PopSign) | ASL | **250 isolated words** | ~94k sequences, 100+ signers | MediaPipe **Holistic** landmarks in **parquet** (543 pts/frame: face+pose+both hands) | Kaggle competition rules — research; commercial unclear |
| **Google – ASL Fingerspelling Recognition** (`asl-fingerspelling`) | ASL | **Fingerspelled phrases** (addresses, names, URLs) — letter *sequences* | very large, many signers | MediaPipe landmark sequences (parquet) | Kaggle competition rules — research; commercial unclear |

**Why these matter:**
- For **word-level signs (our Phase 5)** → `asl-signs` is near-perfect: 250 words,
  already landmarks. We only need to *select the hand landmarks* out of the 543
  Holistic points and re-normalise to match our pipeline.
- For **fingerspelling** → `asl-fingerspelling` is sequence-level (phrases), which
  is *harder* than a simple A–Z classifier. For our easy Phase-2 alphabet we'll
  still prefer the simple image sets below, then graduate to this.

Sources: [Kaggle asl-signs](https://www.kaggle.com/competitions/asl-signs) ·
[Kaggle asl-fingerspelling](https://www.kaggle.com/competitions/asl-fingerspelling)

---

## Tier 1 — ASL fingerspelling / alphabet (Phase 2, the easy static ML)

| Dataset | Content | Size | Format | License |
|---|---|---|---|---|
| **Kaggle ASL Alphabet** (akash nagaraj) | A–Z + space/del/nothing (29 classes) | ~87,000 images | RGB images | Generally permissive (CC0-ish) — verify |
| **Sign Language MNIST** | A–Z static handshapes (no J/Z) | 34,627 28×28 images | CSV pixels | CC0 |
| **ChicagoFSWild** | Fingerspelling "in the wild" (YouTube/aslized/deafvideo) | 7,304 sequences, 160 signers | Video clips + annotations | Research use |
| **ChicagoFSWild+** | Larger wild fingerspelling | 55,232 sequences, 260 signers | Video + annotations | Research use |
| **ASL Fingerspelling A** (RWTH/Surrey) | Letters | ~131,000 samples, 5 signers, 2.1 GB | Images/depth | Free download |
| **ASL Fingerspelling B** | Letters | 317 MB, 9 signers | Images/depth | Free download |

**v1 starter:** Kaggle ASL Alphabet (volume + clean) → run MediaPipe over the images
once to get landmarks → train the static classifier. Sign Language MNIST is an even
lighter "hello world" if we want a fast first pass.

Sources:
[ChicagoFSWild](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) ·
[Sign Language MNIST (Kaggle)](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

---

## Tier 2 — ASL isolated words (Phase 5, the temporal ML)

| Dataset | Content | Size | Format | License |
|---|---|---|---|---|
| **WLASL** (Word-Level ASL) | 2,000-word vocabulary | 21,083 samples (subsets WLASL100/300) | Video (from web) | Research / non-commercial; YouTube-sourced |
| **MS-ASL** (Microsoft) | 1,000-word vocabulary | 25,513 samples, 200+ signers (subsets MSASL100/200) | Video links | Microsoft research license (non-commercial) |
| **ASLLVD** (Boston ASL Lexicon Video Dataset) | 3,300+ signs | ~9,800 samples, 6 signers | Video, multi-angle | Research |
| **"ASL Citizen" / recent 108k set** | 2,208 words, ≥30 videos each | 108,618 videos | Video | Check source (some CC) |

**Note:** WLASL and MS-ASL distribute *links* to web videos, not the videos —
some links rot over time, and they're explicitly research-use. Plan for that.

Sources:
[WLASL](https://dxli94.github.io/WLASL/) ·
[arXiv survey of SLR datasets](https://arxiv.org/pdf/2007.12530)

---

## Tier 3 — ASL continuous / sentence translation (future, advanced)

| Dataset | Content | Size | License |
|---|---|---|---|
| **How2Sign** | Continuous ASL, instructional, multimodal (RGB+depth+pose) | 80+ hours, largest public ASL | **CC BY-NC 4.0** (non-commercial) |
| **OpenASL** | Open-domain ASL from online video (spontaneous + interpreted) | ~98k+ clips | Research |

Continuous translation (full sentences, grammar) is a *research-grade* problem —
out of scope until words work. Listed for completeness/expansion.

Sources:
[How2Sign](https://how2sign.github.io/) ·
[OpenASL (arXiv)](https://arxiv.org/pdf/2205.12870)

---

## Tier 4 — Indian Sign Language (your "lots more data later")

If real users are Indian, ISL ≠ ASL — different signs entirely. This is the priority
expansion after ASL v1 works.

| Dataset | Content | Size | Format | License |
|---|---|---|---|---|
| **INCLUDE / INCLUDE-50** | Isolated ISL words | INCLUDE: 263 words, 4,287 videos; INCLUDE-50: 50-word subset | Video | Research (open) |
| **ISL-CSLTR** | **Sentence-level**, fully labeled, for continuous translation | sentences + words + frames | Video + annotations | Mendeley Data (open) |
| **ISLTranslate** | ISL → English translation pairs | translation corpus | Video + text | Research (open) |
| **IIITA-ROBITA** | Isolated ISL | 284 MB | Video | Contact author |
| **Indian Kinect** | Hand gestures | 5,041 samples, 18 signers, 2 GB | RGB-D | Free download |
| **ISL pose/MediaPipe sets (various Kaggle)** | A–Z + 0–9, landmark NumPy | ~1,000 imgs/class | Landmark .npy / images | Mixed |

**ISLRTC** (islrtc.nic.in) — the Indian government body. Not a download dataset, but
the authoritative ISL dictionary/standard. Useful for label vocabularies + validation.

Sources:
[ISL-CSLTR (Mendeley)](https://data.mendeley.com/datasets/kcmpdxky7p/1) ·
[ISLTranslate (arXiv)](https://arxiv.org/html/2307.05440) ·
[ISLRTC](http://islrtc.nic.in/)

---

## Tier 5 — Every other language (the global expansion)

| Language | Dataset | Content | Size | Type | License |
|---|---|---|---|---|---|
| German (DGS) | **RWTH-PHOENIX-Weather 2014T** | Weather broadcasts | 45,760 samples, 9 signers, 52 GB | Continuous | Research |
| German | **SIGNUM** | Isolated + continuous | 33,210 samples, 25 signers, 920 GB | Both | Contact author |
| German | **DGS Kinect 40** | Isolated words | 3,000 samples, 15 signers | Isolated | Public |
| British (BSL) | **BOBSL** | BBC broadcasts | **1,467 hours**, 39 signers | Continuous | Research, strict (BBC) |
| Chinese (CSL) | **CSL (USTC)** | 500 words | 125,000 videos, 50 signers, 108.8 h | Isolated | Research |
| Chinese | **CSL-Daily** | Daily-life sentences | continuous corpus | Continuous | Research |
| Chinese | **DEVISIGN-G/D/L** | up to 24,000 samples | 8 signers | Isolated | Contact author |
| Turkish | **AUTSL** | 226 signs | ~38k clips, 43 signers | Isolated | Research (open) |
| Turkish | **BosphorusSign22k** | 744 words | 22,542 videos, 6 signers, 19 h | Isolated | Research |
| Greek (GSL) | **GSL** | 310 classes | studio-captured | Isolated/continuous | Research |
| Argentine (LSA) | **LSA64** | 64 signs | 3,200 samples, 10 signers, 20 GB | Isolated | Public |
| Argentine | **LSA16** | handshapes | 800 samples, 10 signers | Handshapes | Public |
| Arabic | **ArASL** | alphabet | 54,049 samples | Fingerspelling | Free download |
| Arabic/Saudi | **Isharah** | multi-scene continuous | large | Continuous | Research (2025) |
| Korean (KSL) | **KSL / AI-Hub KSL** | large gov corpus | varies | Both | AI-Hub terms |
| Japanese | **JSL Fingerspelling** | kana fingerspelling | 8,055 samples, 10 signers | Fingerspelling | Free download |
| Polish (PSL) | **PSL Kinect/ToF** | isolated | up to 1,680 samples | Isolated | Public |
| Irish (ISL) | **Irish SL handshapes** | handshapes | 58,114 frames, 6 signers | Handshapes | Free download |
| Romanian | **RoCoISLR** | isolated corpus | new (2025) | Isolated | Research |
| Multilingual | **JWSign** | Bible translations across many sign languages | very large, highly multilingual | Continuous | Research |

Sources:
[How2Sign survey & dataset table (arXiv 2308.12419)](https://arxiv.org/pdf/2308.12419) ·
[Multilingual datasets aggregator (GitHub: yayayru/sign-language-datasets)](https://github.com/yayayru/sign-lanuage-datasets) ·
[Isharah (arXiv 2506.03615)](https://arxiv.org/html/2506.03615v1) ·
[JWSign (arXiv 2311.10174)](https://arxiv.org/pdf/2311.10174)

---

## Tier 6 — Tooling & pretrained models (don't reinvent)

Not datasets, but they save weeks:

- **OpenHands** — standardised **pose** datasets + **pretrained** isolated-SLR models
  across 6 languages (ASL, Argentine, Chinese, Greek, Indian, Turkish). Pose-based,
  same philosophy as us. ([arXiv 2110.05877](https://arxiv.org/pdf/2110.05877))
- **research.sign.mt** — Google's sign-language research hub; pipelines & links.
- **MiCT-RANet ASL Fingerspelling** — open code hitting 92.7% letter accuracy on
  FSBoard; a reference for what "good" looks like.
  ([GitHub](https://github.com/fmahoudeau/MiCT-RANet-ASL-FingerSpelling))
- **yayayru/sign-language-datasets** — the big community aggregator list (where much
  of Tier 5 comes from).

---

## ⚠️ License reality — read before publishing

This is the part that bites people who skip it. We intend to **publish** a Chrome
extension. That's distribution, possibly commercial-adjacent. But:

- **How2Sign is CC BY-NC** (non-commercial). **WLASL, MS-ASL, PHOENIX, BOBSL,
  ChicagoFSWild** are **research-use** and often built from third-party video
  (YouTube, BBC). Training a model on them is fine for research; *distributing a
  product* built on them is a legal gray zone.
- **Safest for a public product:** datasets that are **CC0 / CC-BY / explicitly
  permissive** — e.g. **Sign Language MNIST (CC0)**, **Kaggle ASL Alphabet**, and
  the **Google Kaggle competition data** (read the specific competition rules; the
  PopSign data was released for public model-building).
- **Practical stance for v1:** build and *learn* on the research sets freely
  (nobody objects to learning); before the Web Store launch, make sure the *shipped*
  model is trained on permissively-licensed data, or get explicit permission. This
  is **Open Question #1 to resolve before Phase 6 (publish)** — flagged in [[tasks/plan]].

---

## Recommended ingestion path (ties to [[tasks/plan]])

1. **Phase 2 (alphabet, static):** Kaggle **ASL Alphabet** (+ Sign Language MNIST as
   warm-up) → run MediaPipe once → landmarks → train static classifier. *Permissive
   licenses — safe to ship.*
2. **Phase 5 (words, temporal):** Google **`asl-signs`** (250 words, *already
   landmarks*) → select hand landmarks → re-normalise → train LSTM. *Massive
   shortcut; check competition terms before shipping.*
3. **Expansion (later):** add **WLASL/MS-ASL** for vocabulary breadth (research/learning),
   then **INCLUDE/ISL-CSLTR** for the ISL version, then Tier 5 languages. Use
   **OpenHands** pretrained pose models to bootstrap each new language instead of
   training from zero.

**The one idea:** prefer **landmark-native, permissively-licensed** data for what we
*ship*; use the big research video corpora to *learn and prototype*. Variety of
signers > raw count for real-world accuracy.

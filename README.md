# Talk Through Me (TTM)

Live sign-language → English text on video calls (Google Meet / Zoom / Teams),
delivered as a Chrome/Brave extension. Runs fully in the browser — your webcam
video never leaves your machine.

> Respectful note: this project is built for the **Deaf and hard-of-hearing**
> community.

## How it works (short)
Webcam → MediaPipe extracts 21 hand landmarks (63 numbers) → a model trained on
those numbers predicts the sign → text appears live on the call.

## Status
The live extension runs a fingerspelling model at 93.6% accuracy alongside a
word-sign model. Teaching that word model to read the face and upper body, not
just the hand, lifted it from 39 to 185 signs above the honesty floor — the point
below which a sign is guessed too rarely to be worth showing.

The current work is scaling the vocabulary well past that. The recognition
pipeline now takes Microsoft's ASL Citizen dataset — 2,731 signs across roughly
40,000 clips — and processes it end to end on-device: each video is streamed
straight out of the 43 GB archive (there isn't disk to unpack it), its hand, face
and pose landmarks are tracked, and those become training sequences in the exact
shape the model expects. A first fifty-sign pilot reads 55% of signs correctly
from signers it has never seen, which proves the pipeline holds together at scale;
ten of those signs are read perfectly. What remains is the broader vocabulary, and
then the harder step of running the larger model live inside the extension. The
captions themselves are now assembled into real sentences rather than a stream of
capitalised tokens, and a referee keeps the word model from interrupting a name
being fingerspelled letter by letter. Indian Sign Language is the planned
expansion — the same pipeline, a niche the larger players ignore.

More detail:
- [`project-brain.md`](project-brain.md) — **start here**: the complete current-state map (what exists, the numbers, what's left)
- [`in-depth-study.md`](in-depth-study.md) — the concept doc (the "why" behind the design)
- [`datasets.md`](datasets.md) — the dataset catalog
- [`tasks/plan.md`](tasks/plan.md) · [`tasks/todo.md`](tasks/todo.md) — the build plan
- [`log.md`](log.md) — the honest session-by-session history

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

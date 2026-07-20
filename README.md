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
Live extension shipped: 93.6% alphabet model + word model. Whole-body features
(face + pose added to the hand) took the word model from 39 → **185 words** above
the honest floor. Next: scale the vocabulary (ASL Citizen), wire the bigger model
in live, add the on-device sentence layer; ISL is the differentiated expansion. See:
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

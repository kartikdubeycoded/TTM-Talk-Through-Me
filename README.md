# Sign-to-Text

Live sign-language → English text on video calls (Google Meet / Zoom / Teams),
delivered as a Chrome/Brave extension. Runs fully in the browser — your webcam
video never leaves your machine.

> Respectful note: this project is built for the **Deaf and hard-of-hearing**
> community.

## How it works (short)
Webcam → MediaPipe extracts 21 hand landmarks (63 numbers) → a model trained on
those numbers predicts the sign → text appears live on the call.

## Status
Phase 1 — foundation (hand tracking). See:
- [`in-depth-study.md`](in-depth-study.md) — the full ground-up explanation
- [`datasets.md`](datasets.md) — the dataset catalog
- [`tasks/plan.md`](tasks/plan.md) · [`tasks/todo.md`](tasks/todo.md) — the build plan

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

"""ASL Citizen manifest loading + vocab selection (the scale-up front-end).

ASL Citizen (Microsoft, ~2,700 signs / ~84k videos) ships raw MP4s plus split
CSVs (train/val/test). Each row names a video file and its gloss (sign label).
This module turns those CSVs into (video, gloss) records and picks the top-N
most frequent glosses, so the heavy MediaPipe landmark extraction only runs on
the subset we're actually training (the 500-word subset first).

Deliberately NOT here yet: the MediaPipe extraction itself. Its landmark-index
selection has to reproduce the trained model's exact input (the GISLR 66-point
layout: lips 0:40, hand 40:61, pose 61:66), and that can only be verified
against real videos + real MediaPipe output — guessing it blind would recreate
the silent JS/Python divergence bug this project already fought. So extraction
lands once the download finishes and we can check numbers, not before.
"""
from collections import Counter

import pandas as pd

# ASL Citizen headers vary across releases; match by name, not position.
VIDEO_COLS = ("video file", "video", "filename", "file")
GLOSS_COLS = ("gloss", "label", "sign")


def _pick_column(columns, candidates):
    lower = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"none of {candidates} found in columns {list(columns)}")


def load_split(csv_path):
    """Parse an ASL Citizen split CSV -> list of {'video','gloss'} records."""
    df = pd.read_csv(csv_path)
    vcol = _pick_column(df.columns, VIDEO_COLS)
    gcol = _pick_column(df.columns, GLOSS_COLS)
    return [{"video": str(v), "gloss": str(g)}
            for v, g in zip(df[vcol], df[gcol])]


def select_vocab(records, top_n):
    """Keep only the top_n most frequent glosses.

    Returns (vocab, filtered_records). vocab is sorted alphabetically for a
    stable label order (same convention as pipeline/ingest_words.py).
    """
    counts = Counter(r["gloss"] for r in records)
    keep = {g for g, _ in counts.most_common(top_n)}
    filtered = [r for r in records if r["gloss"] in keep]
    return sorted(keep), filtered

"""Tests for pipeline/asl_citizen.py — ASL Citizen manifest loading + vocab
selection (the scale-up front-end, before MediaPipe extraction)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from asl_citizen import load_split, select_vocab


@pytest.fixture
def split_csv(tmp_path):
    """A tiny stand-in for an ASL Citizen split CSV."""
    df = pd.DataFrame({
        "Participant ID": [1, 1, 2, 2, 3, 3, 4],
        "Video file": ["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4", "f.mp4", "g.mp4"],
        "Gloss": ["HELLO", "HELLO", "HELLO", "DOG", "DOG", "CAT", "HELLO"],
    })
    p = tmp_path / "train.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_load_split_parses_video_and_gloss(split_csv):
    recs = load_split(split_csv)
    assert len(recs) == 7
    assert recs[0] == {"video": "a.mp4", "gloss": "HELLO"}


def test_load_split_detects_alternate_column_names(tmp_path):
    # Real ASL Citizen headers vary; the loader must find the video + gloss
    # columns by name rather than by fixed position.
    df = pd.DataFrame({"filename": ["x.mp4"], "label": ["YES"]})
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    assert load_split(str(p)) == [{"video": "x.mp4", "gloss": "YES"}]


def test_load_split_raises_on_missing_columns(tmp_path):
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(KeyError):
        load_split(str(p))


def test_select_vocab_keeps_top_n_most_frequent(split_csv):
    recs = load_split(split_csv)
    vocab, filtered = select_vocab(recs, top_n=2)
    # HELLO (4) and DOG (2) are the two most frequent; CAT (1) is dropped.
    assert set(vocab) == {"HELLO", "DOG"}
    assert vocab == sorted(vocab)                       # stable label order
    assert all(r["gloss"] in {"HELLO", "DOG"} for r in filtered)
    assert len(filtered) == 6                           # 4 HELLO + 2 DOG

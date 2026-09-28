"""Concatenate teacher score files, dropping duplicate ids (first file wins).

    python -m reflex_sentry.teacher.merge_scores a.parquet b.parquet --out a.parquet

Used to add the scores of `data/processed/easy_benign_for_teachers.parquet`
onto an existing `teacher_scores_*.parquet`. `--out` may equal the first input.
"""
from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd


def merge_scores(paths: Sequence[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    for p, f in zip(paths, frames):
        if "id" not in f.columns:
            raise ValueError(f"{p} has no id column")
    frames = [f for f in frames if not f.empty] or frames[:1]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="id", keep="first").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = merge_scores(a.inputs)
    df.to_parquet(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")


if __name__ == "__main__":
    main()

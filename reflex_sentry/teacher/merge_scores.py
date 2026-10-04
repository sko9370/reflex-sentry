"""Merge compatible teacher score files, reconciling overlapping ids.

    python -m reflex_sentry.teacher.merge_scores a.parquet b.parquet --out a.parquet

Overlapping ids must have the same model and unsafe probability. Missing
optional details may be filled from another file; conflicting present details
are rejected. `--out` may equal the first input.
"""
from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from .artifact import missing_optional, validate_score_frame


def merge_scores(paths: Sequence[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    models = {model for p, f in zip(paths, frames)
              if (model := validate_score_frame(f, str(p))) is not None}
    if len(models) > 1:
        raise ValueError(f"score files contain mixed teacher models: {sorted(models)}")
    if not frames:
        raise ValueError("at least one score file is required")
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return merged
    # A missing legacy optional column is unknown, while a present value may
    # enrich it. Two present values must agree for an overlapping id.
    rows = {}
    for record in merged.to_dict("records"):
        id_ = record["id"]
        if id_ not in rows:
            rows[id_] = record
            continue
        prior = rows[id_]
        if prior["p_unsafe_teacher"] != record["p_unsafe_teacher"]:
            raise ValueError(f"conflicting p_unsafe_teacher for id {id_!r}")
        for column in merged.columns:
            if column in {"id", "p_unsafe_teacher", "teacher_model"}:
                continue
            old, new = prior.get(column), record.get(column)
            if missing_optional(old):
                if not missing_optional(new):
                    prior[column] = new
            elif not missing_optional(new) and old != new:
                raise ValueError(f"conflicting {column} for id {id_!r}")
    return pd.DataFrame(list(rows.values()), columns=merged.columns)


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

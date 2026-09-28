"""Turn one or two teacher score files into a harness prediction CSV
(README 5.1) for a hand-labeled eval set, so the teacher can be scored as
the upper-reference row of README 5.4.

Single teacher
--------------
    p_dangerous = p_unsafe_teacher
    p_unsure    = p_controversial if present else 0
    p_safe      = 1 - p_dangerous - p_unsure

Two teachers
------------
`p_dangerous` is the mean of the two teachers' `p_unsafe_teacher`.
`p_unsure` blends two signals, each in [0, 1], with configurable weights
(`--w-controversial`, `--w-disagreement`, default 0.5 / 0.5, summing to 1):

    controversial_signal = nanmean(p_controversial_1, p_controversial_2)  # 0 if both NaN
    disagreement_signal  = |p_unsafe_1 - p_unsafe_2|
    p_unsure = w_controversial * controversial_signal + w_disagreement * disagreement_signal

`p_safe = max(0, 1 - p_dangerous - p_unsure)`, then all three are
renormalized (divided by their sum) so they sum to exactly 1 -- the blend
above is not guaranteed to leave room for p_safe on its own (e.g. two
teachers both near 1.0 on p_unsafe but disagreeing with each other), and
renormalizing rather than clipping keeps the *relative* weight of the three
classes intact.

CLI
---
    python -m reflex_sentry.teacher.as_predictor \\
        --scores data/interim/teacher_scores_llama_guard_3_8b.parquet \\
        --eval data/processed/val.parquet \\
        --out preds/teacher_llama_guard_3_8b_val.csv

    python -m reflex_sentry.teacher.as_predictor \\
        --scores data/interim/teacher_scores_llama_guard_3_8b.parquet \\
                 data/interim/teacher_scores_qwen3guard_gen_8b.parquet \\
        --eval data/processed/val.parquet \\
        --out preds/teacher_both_val.csv

`latency_ms` is left empty: teacher latency is not the point of the
comparison table (README 5.4 records that separately if wanted).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_W_CONTROVERSIAL = 0.5
DEFAULT_W_DISAGREEMENT = 0.5


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def single_teacher_probs(scores: pd.DataFrame) -> pd.DataFrame:
    p_dangerous = scores["p_unsafe_teacher"].astype(float)
    p_unsure = scores["p_controversial"].astype(float).fillna(0.0) if "p_controversial" in scores.columns else pd.Series(0.0, index=scores.index)
    p_safe = 1.0 - p_dangerous - p_unsure
    return pd.DataFrame({"id": scores["id"], "p_safe": p_safe, "p_dangerous": p_dangerous, "p_unsure": p_unsure})


def two_teacher_probs(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    w_controversial: float = DEFAULT_W_CONTROVERSIAL,
    w_disagreement: float = DEFAULT_W_DISAGREEMENT,
) -> pd.DataFrame:
    merged = scores_a.merge(scores_b, on="id", suffixes=("_a", "_b"))
    p1 = merged["p_unsafe_teacher_a"].astype(float)
    p2 = merged["p_unsafe_teacher_b"].astype(float)
    c1 = merged["p_controversial_a"].astype(float) if "p_controversial_a" in merged.columns else pd.Series(np.nan, index=merged.index)
    c2 = merged["p_controversial_b"].astype(float) if "p_controversial_b" in merged.columns else pd.Series(np.nan, index=merged.index)

    p_dangerous = (p1 + p2) / 2.0
    controversial_signal = pd.concat([c1, c2], axis=1).mean(axis=1, skipna=True).fillna(0.0)
    disagreement_signal = (p1 - p2).abs()
    p_unsure = w_controversial * controversial_signal + w_disagreement * disagreement_signal
    p_safe = (1.0 - p_dangerous - p_unsure).clip(lower=0.0)

    total = p_safe + p_dangerous + p_unsure
    return pd.DataFrame({
        "id": merged["id"],
        "p_safe": p_safe / total,
        "p_dangerous": p_dangerous / total,
        "p_unsure": p_unsure / total,
    })


def run(
    score_paths: list[str],
    eval_path: str,
    out_path: str,
    w_controversial: float = DEFAULT_W_CONTROVERSIAL,
    w_disagreement: float = DEFAULT_W_DISAGREEMENT,
) -> pd.DataFrame:
    if len(score_paths) not in (1, 2):
        raise ValueError(f"expected 1 or 2 --scores files, got {len(score_paths)}")

    eval_df = read_any(eval_path)
    missing = {"id", "gold"} - set(eval_df.columns)
    if missing:
        raise ValueError(f"{eval_path} missing required column(s): {sorted(missing)}")

    if len(score_paths) == 1:
        probs = single_teacher_probs(read_any(score_paths[0]))
    else:
        probs = two_teacher_probs(
            read_any(score_paths[0]), read_any(score_paths[1]),
            w_controversial=w_controversial, w_disagreement=w_disagreement,
        )

    merged = eval_df.merge(probs, on="id", how="inner")
    n_missing = len(eval_df) - len(merged)
    if n_missing:
        print(f"[reflex_sentry.teacher.as_predictor] WARNING: {n_missing} eval rows had no teacher score and were dropped")

    out_df = pd.DataFrame({
        "id": merged["id"],
        "gold": merged["gold"],
        "p_safe": merged["p_safe"],
        "p_dangerous": merged["p_dangerous"],
        "p_unsure": merged["p_unsure"],
        "source": merged["source"] if "source" in merged.columns else "",
        "tags": merged["tags"] if "tags" in merged.columns else "",
        "latency_ms": "",
    })

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True, help="1 or 2 teacher_scores_*.parquet files")
    ap.add_argument("--eval", required=True, help="hand-labeled eval parquet/csv, e.g. data/processed/val.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-controversial", type=float, default=DEFAULT_W_CONTROVERSIAL)
    ap.add_argument("--w-disagreement", type=float, default=DEFAULT_W_DISAGREEMENT)
    a = ap.parse_args()
    out_df = run(a.scores, a.eval, a.out, w_controversial=a.w_controversial, w_disagreement=a.w_disagreement)
    print(f"wrote {len(out_df)} teacher-as-predictions to {a.out}")


if __name__ == "__main__":
    main()

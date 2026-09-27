"""Temperature scaling for the 3-way student.

Input CSVs hold raw logits: id, gold, logit_safe, logit_dangerous, logit_unsure,
plus any passthrough columns (source, tags, latency_ms).

Gold labels map to classes as benign -> safe, dangerous -> dangerous,
ambiguous -> unsure. A single temperature T is fit on the validation set by
minimizing negative log-likelihood, then applied to another logits file to
produce a prediction CSV in the report schema.

    python -m reflex_sentry.eval.calibrate --val-logits val_logits.csv \
        --apply test_logits.csv --out test_preds.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

LOGIT_COLS = ["logit_safe", "logit_dangerous", "logit_unsure"]
GOLD_TO_CLASS = {"benign": 0, "dangerous": 1, "ambiguous": 2}


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    return float(-log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda lt: nll(logits, y, float(np.exp(lt))),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def to_predictions(df: pd.DataFrame, T: float) -> pd.DataFrame:
    p = softmax(df[LOGIT_COLS].to_numpy(dtype=float) / T, axis=1)
    out = df.drop(columns=LOGIT_COLS).copy()
    out["p_safe"], out["p_dangerous"], out["p_unsure"] = p[:, 0], p[:, 1], p[:, 2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-logits", required=True)
    ap.add_argument("--apply", required=True, help="logits CSV to convert with the fitted T")
    ap.add_argument("--out", required=True, help="prediction CSV to write")
    ap.add_argument("--val-out", help="optionally also write calibrated val predictions")
    a = ap.parse_args()

    val = pd.read_csv(a.val_logits)
    y = val["gold"].map(GOLD_TO_CLASS).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ValueError("val gold labels must be benign, dangerous, or ambiguous")
    logits = val[LOGIT_COLS].to_numpy(dtype=float)
    T = fit_temperature(logits, y.astype(int))
    print(f"T = {T:.4f}  val NLL {nll(logits, y.astype(int), 1.0):.4f} -> {nll(logits, y.astype(int), T):.4f}")

    to_predictions(pd.read_csv(a.apply), T).to_csv(a.out, index=False)
    if a.val_out:
        to_predictions(val, T).to_csv(a.val_out, index=False)


if __name__ == "__main__":
    main()

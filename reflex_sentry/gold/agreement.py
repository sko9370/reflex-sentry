"""Pass-1 vs pass-2 self-agreement: relabel yourself a week later (docs/WORKFLOWS.md).

`compare` reads two data/gold/<split>.csv-shaped files (the same ids labeled
twice, e.g. by labeler_pass) and reports Cohen's kappa overall and per class,
a 3x3 confusion matrix (out_of_scope rows are excluded from these numbers and
counted separately), the rate of flips that involve `ambiguous`, and writes a
CSV of every disagreement for adjudication.

`make-pass2` builds the pass-2 sheet from a finished pass-1 gold CSV: same
ids and text, shuffled into a different row order with a different seed, and
with gold/tags/notes blanked out, so you relabel blind to your own past
answers (mirrors `sample.py`'s blind-by-default sample).

    python -m reflex_sentry.gold.agreement compare --pass1 data/gold/val_pass1.csv \
        --pass2 data/gold/val_pass2.csv --out-dir reports/agreement_val

    python -m reflex_sentry.gold.agreement make-pass2 --pass1 data/gold/val_pass1.csv \
        --out data/gold/samples/val_pass2_sheet.csv --seed 4242
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from . import GOLD_LABELS
from .export import to_spreadsheet


def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    for c in ("gold", "tags", "notes"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def compare(pass1: pd.DataFrame, pass2: pd.DataFrame) -> pd.DataFrame:
    """Inner-join pass1/pass2 on id; columns suffixed _1/_2."""
    return pass1.merge(pass2, on="id", how="inner", suffixes=("_1", "_2"))


def _in_scope_pair(merged: pd.DataFrame) -> pd.DataFrame:
    m = merged["gold_1"].isin(GOLD_LABELS) & merged["gold_2"].isin(GOLD_LABELS)
    return merged[m]


def kappa_overall(merged: pd.DataFrame) -> float:
    m = _in_scope_pair(merged)
    if m.empty:
        return float("nan")
    return float(cohen_kappa_score(m["gold_1"], m["gold_2"], labels=list(GOLD_LABELS)))


def kappa_per_class(merged: pd.DataFrame) -> dict[str, float]:
    m = _in_scope_pair(merged)
    out = {}
    for label in GOLD_LABELS:
        if m.empty:
            out[label] = float("nan")
            continue
        y1 = (m["gold_1"] == label).astype(int)
        y2 = (m["gold_2"] == label).astype(int)
        out[label] = float(cohen_kappa_score(y1, y2, labels=[0, 1])) if (y1.nunique() > 1 or y2.nunique() > 1) \
            else (1.0 if (y1 == y2).all() else 0.0)
    return out


def confusion(merged: pd.DataFrame) -> pd.DataFrame:
    m = _in_scope_pair(merged)
    cm = confusion_matrix(m["gold_1"], m["gold_2"], labels=list(GOLD_LABELS))
    return pd.DataFrame(cm, index=[f"pass1_{g}" for g in GOLD_LABELS], columns=[f"pass2_{g}" for g in GOLD_LABELS])


def ambiguous_flip_rate(merged: pd.DataFrame) -> dict:
    m = _in_scope_pair(merged)
    disagree = m["gold_1"] != m["gold_2"]
    involves_amb = disagree & ((m["gold_1"] == "ambiguous") | (m["gold_2"] == "ambiguous"))
    n = len(m)
    n_dis = int(disagree.sum())
    return {
        "n_compared": n,
        "n_disagreements": n_dis,
        "n_ambiguous_flips": int(involves_amb.sum()),
        "rate_of_total": float(involves_amb.sum() / n) if n else float("nan"),
        "rate_of_disagreements": float(involves_amb.sum() / n_dis) if n_dis else float("nan"),
    }


def disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    m = merged[merged["gold_1"] != merged["gold_2"]].copy()
    cols = ["id"] + [c for c in ("text_1", "source_1") if c in m.columns] + \
        ["gold_1", "tags_1", "notes_1", "gold_2", "tags_2", "notes_2"]
    return m[[c for c in cols if c in m.columns]].sort_values("id").reset_index(drop=True)


def make_pass2_sheet(pass1: pd.DataFrame, seed: int, blind: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pass1))
    shuffled = pass1.iloc[order].reset_index(drop=True)
    sample = pd.DataFrame({
        "id": shuffled["id"],
        "text": shuffled["text"],
        "split": shuffled.get("split", ""),
        "sample_pass": 2,
    })
    if not blind and "source" in shuffled.columns:
        sample["source"] = shuffled["source"]
    return to_spreadsheet(sample)


def run_compare(pass1_path: str, pass2_path: str, out_dir: str) -> dict:
    p1, p2 = load_gold(pass1_path), load_gold(pass2_path)
    merged = compare(p1, p2)
    result = {
        "n_pass1": len(p1), "n_pass2": len(p2), "n_matched": len(merged),
        "kappa_overall": kappa_overall(merged),
        "kappa_per_class": kappa_per_class(merged),
        "confusion_matrix": confusion(merged),
        "ambiguous_flips": ambiguous_flip_rate(merged),
    }
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    result["confusion_matrix"].to_csv(outdir / "confusion_matrix.csv")
    disagreements(merged).to_csv(outdir / "disagreements.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="compare two labeled passes")
    c.add_argument("--pass1", required=True)
    c.add_argument("--pass2", required=True)
    c.add_argument("--out-dir", required=True)

    p = sub.add_parser("make-pass2", help="build a blind, shuffled pass-2 sheet from pass-1")
    p.add_argument("--pass1", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--no-blind", action="store_true")

    a = ap.parse_args()
    if a.cmd == "compare":
        r = run_compare(a.pass1, a.pass2, a.out_dir)
        print(f"n_matched={r['n_matched']}  kappa_overall={r['kappa_overall']:.3f}")
        for label, k in r["kappa_per_class"].items():
            print(f"  kappa[{label}]={k:.3f}")
        af = r["ambiguous_flips"]
        print(f"ambiguous flips: {af['n_ambiguous_flips']}/{af['n_disagreements']} disagreements "
              f"({af['rate_of_disagreements']:.2%} of disagreements, {af['rate_of_total']:.2%} of all compared)")
        print(f"confusion matrix and disagreements written to {a.out_dir}")
    else:
        p1 = load_gold(a.pass1)
        sheet = make_pass2_sheet(p1, a.seed, blind=not a.no_blind)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(a.out, index=False)
        print(f"wrote blind pass-2 sheet ({len(sheet)} rows, seed={a.seed}) -> {a.out}")


if __name__ == "__main__":
    main()

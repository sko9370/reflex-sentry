"""Synthetic predictions for exercising the eval harness end to end.

The numbers mean nothing about any real model. They are shaped so that every
report section has something in it: hard negatives score riskier than easy
benign items, evasion-wrapped items are easier to miss, and one source is
shifted to mimic out-of-distribution data.

    python tests/make_synthetic.py --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

CATS = ["cat:malware_dev", "cat:evasion", "cat:exploit_weaponization", "cat:phishing_se",
        "cat:credential_theft", "cat:intrusion_ops", "cat:ddos"]
HNS = ["hn:detection", "hn:analysis", "hn:vuln_explain", "hn:secure_dev", "hn:ir", "hn:training", "hn:cti"]
EVS = ["evasion:roleplay", "evasion:fiction", "evasion:base64", "evasion:leetspeak"]
SOURCES = ["src_a", "src_b", "handwritten"]


def make(n: int, seed: int, ood: bool = False, temperature: float = 0.6) -> pd.DataFrame:
    """Return a logits frame. temperature < 1 makes the model overconfident."""
    rng = np.random.default_rng(seed)
    gold = rng.choice(["benign", "dangerous", "ambiguous"], size=n, p=[0.6, 0.3, 0.1])
    rows = []
    for i, g in enumerate(gold):
        tags, shift = [], 0.0
        if g == "dangerous":
            tags.append(rng.choice(CATS))
            if rng.random() < 0.3:
                tags.append(rng.choice(EVS))
                shift = -1.6
            mu = np.array([0.0, 2.2 + shift, 0.5])
        elif g == "benign":
            if rng.random() < 0.4:
                tags += ["hard_negative", rng.choice(HNS)]
                mu = np.array([1.4, 0.3, 0.4])
            else:
                mu = np.array([3.0, -1.0, -0.5])
        else:
            mu = np.array([0.7, 0.8, 1.2])
        if ood:
            mu = mu + np.array([0.9, -0.9, 0.0])  # model looks safer on unseen data
        logits = (mu + rng.normal(0, 1.1, 3)) / temperature
        rows.append({
            "id": f"{'ood' if ood else 'x'}{seed}_{i:05d}",
            "gold": g,
            "source": "held_out_src" if ood else rng.choice(SOURCES),
            "tags": ";".join(tags),
            "logit_safe": logits[0], "logit_dangerous": logits[1], "logit_unsure": logits[2],
            "latency_ms": float(rng.gamma(4.0, 1.2)),
        })
    return pd.DataFrame(rows)


def logits_to_preds(df: pd.DataFrame) -> pd.DataFrame:
    p = softmax(df[["logit_safe", "logit_dangerous", "logit_unsure"]].to_numpy(), axis=1)
    out = df.drop(columns=["logit_safe", "logit_dangerous", "logit_unsure"]).copy()
    out["p_safe"], out["p_dangerous"], out["p_unsure"] = p[:, 0], p[:, 1], p[:, 2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {"val": make(a.n, 1), "test": make(a.n, 2), "test_ood": make(a.n // 2, 3, ood=True)}.items():
        df.to_csv(out / f"{name}_logits.csv", index=False)
        logits_to_preds(df).to_csv(out / f"{name}_preds.csv", index=False)
    print(f"wrote synthetic val/test/test_ood logits and predictions to {out}")


if __name__ == "__main__":
    main()

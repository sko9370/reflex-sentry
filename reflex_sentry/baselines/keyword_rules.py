"""Transparent keyword-rule baseline for reflex-sentry (milestone 2).

Zero learned parameters. Loads weighted regex rules from a YAML config
(default `configs/keyword_rules.yaml`) in three phrasing-cue groups --
offensive intent, defensive/educational intent, and target/authorization
gaps -- and combines the matched weights into
`[p_safe, p_dangerous, p_unsure]` with a documented, deterministic function.
Every rule is a vocabulary or phrasing cue only; none matches operational
content.

Scoring
-------
For a prompt, let::

    offensive = sum of matched `offensive` weights + matched `target` weights
    defensive = sum of matched `defensive` weights
    s = offensive - defensive      # net danger signal, can be negative
    m = offensive + defensive      # total matched signal, regardless of sign

    logits = [-s, s, BASE_UNSURE + UNSURE_GAIN * m - abs(s)]
    p_safe, p_dangerous, p_unsure = softmax(logits)

A prompt with no offensive or defensive rule hits has ``s = m = 0`` and gets
a constant, unremarkable split (there is no basis for a stronger claim). A
prompt with only offensive (or only defensive) signal drives ``|s|`` up and
the unsure logit down, so one class dominates. A prompt with comparable
offensive and defensive signal has ``s`` near zero but ``m`` large, which
pushes the unsure logit *up* relative to safe and dangerous: mixed phrasing
reads as "unsure", not as confidently safe.

CLI::

    python -m reflex_sentry.baselines.keyword_rules \\
        --in data/processed/test.parquet --out preds/keyword_test.csv \\
        [--rules configs/keyword_rules.yaml]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Deterministic constants for the unsure logit. See the module docstring.
BASE_UNSURE = -1.5
UNSURE_GAIN = 0.6

GROUPS = ("offensive", "defensive", "target")

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "keyword_rules.yaml"

CompiledRule = tuple  # (re.Pattern, float weight, str note)
RuleSet = dict  # {"offensive": [CompiledRule, ...], "defensive": [...], "target": [...]}


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> RuleSet:
    """Load and compile the weighted regex rules from a YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    rules: RuleSet = {}
    for group in GROUPS:
        rules[group] = [
            (re.compile(r["pattern"], re.IGNORECASE), float(r["weight"]), r.get("note", ""))
            for r in raw.get(group, [])
        ]
    return rules


def _matched_weight(text: str, group_rules: list[CompiledRule]) -> float:
    """Sum of weights for rules that match at least once. Each rule counts
    once per prompt (a keyword repeated in the same prompt is not double
    counted -- this is a vocabulary cue, not a word-frequency score)."""
    return sum(weight for pattern, weight, _ in group_rules if pattern.search(text))


def score_one(text: str, rules: RuleSet) -> np.ndarray:
    """Score a single prompt. Returns a length-3 array
    [p_safe, p_dangerous, p_unsure] that sums to 1."""
    text = "" if text is None else str(text)
    offensive = _matched_weight(text, rules["offensive"]) + _matched_weight(text, rules["target"])
    defensive = _matched_weight(text, rules["defensive"])
    s = offensive - defensive
    m = offensive + defensive
    unsure_logit = BASE_UNSURE + UNSURE_GAIN * m - abs(s)
    logits = np.array([-s, s, unsure_logit], dtype=float)
    logits = logits - logits.max()  # numerical stability, does not change softmax
    ex = np.exp(logits)
    return ex / ex.sum()


def score_texts(texts, rules: RuleSet | None = None) -> np.ndarray:
    """Score a list (or Series) of prompts.

    Returns an (n, 3) array of [p_safe, p_dangerous, p_unsure]. Purely a
    function of `texts` and the loaded rules: same input always gives the
    same output.
    """
    rules = rules if rules is not None else load_rules()
    return np.stack([score_one(t, rules) for t in texts]) if len(texts) else np.zeros((0, 3))


def score_with_latency(texts, rules: RuleSet | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Like `score_texts`, but also returns per-prompt latency_ms, each
    timed individually with time.perf_counter (what the CLI records)."""
    rules = rules if rules is not None else load_rules()
    n = len(texts)
    probs = np.empty((n, 3), dtype=float)
    latency_ms = np.empty(n, dtype=float)
    for i, t in enumerate(texts):
        start = time.perf_counter()
        probs[i] = score_one(t, rules)
        latency_ms[i] = (time.perf_counter() - start) * 1000.0
    return probs, latency_ms


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def run(inp: str, out: str, rules_path: str | Path = DEFAULT_RULES_PATH) -> pd.DataFrame:
    """Score a labeled eval file (README 5.1 input contract: id, text, gold,
    optional source/tags) and write a harness prediction CSV (README 5.1
    output contract) to `out`. Returns the written DataFrame."""
    df = read_any(inp)
    missing = {"id", "text", "gold"} - set(df.columns)
    if missing:
        raise ValueError(f"input file missing required column(s): {sorted(missing)}")

    rules = load_rules(rules_path)
    probs, latency_ms = score_with_latency(df["text"].tolist(), rules)

    out_df = pd.DataFrame({
        "id": df["id"],
        "gold": df["gold"],
        "p_safe": probs[:, 0],
        "p_dangerous": probs[:, 1],
        "p_unsure": probs[:, 2],
        "source": df["source"] if "source" in df.columns else "",
        "tags": df["tags"] if "tags" in df.columns else "",
        "latency_ms": latency_ms,
    })

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="labeled eval file (csv or parquet), README 5.1")
    ap.add_argument("--out", required=True, help="prediction CSV to write")
    ap.add_argument("--rules", default=str(DEFAULT_RULES_PATH), help="keyword_rules.yaml path")
    a = ap.parse_args()
    out_df = run(a.inp, a.out, a.rules)
    n_esc_ready = (out_df["p_safe"] < 0.5).mean()
    print(f"wrote {len(out_df)} predictions to {a.out} "
          f"({n_esc_ready:.1%} would escalate at p_safe < 0.5)")


if __name__ == "__main__":
    main()

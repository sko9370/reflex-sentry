"""Tests for the keyword-rule baseline (milestone 2): scoring properties,
the CLI, and an end-to-end run through the eval harness's report."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reflex_sentry.baselines import keyword_rules as K  # noqa: E402
from reflex_sentry.eval import metrics as M  # noqa: E402
from reflex_sentry.eval import report as R  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny_labeled.csv"
RULES_PATH = REPO_ROOT / "configs" / "keyword_rules.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "eval.yaml"


@pytest.fixture(scope="module")
def rules():
    return K.load_rules(RULES_PATH)


@pytest.fixture(scope="module")
def fixture_df():
    return pd.read_csv(FIXTURE)


def _split_balanced(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into label-balanced val/test halves (even/odd rows within each
    gold group), so both halves have dangerous items to pick a threshold on."""
    val_parts, test_parts = [], []
    for _, sub in df.groupby("gold"):
        sub = sub.reset_index(drop=True)
        val_parts.append(sub.iloc[0::2])
        test_parts.append(sub.iloc[1::2])
    return (pd.concat(val_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True))


# --------------------------------------------------------------- scoring ---

def test_probabilities_valid_and_sum_to_one(rules, fixture_df):
    probs = K.score_texts(fixture_df["text"].tolist(), rules)
    assert probs.shape == (len(fixture_df), 3)
    assert np.all(probs >= -1e-9)
    assert np.all(probs <= 1 + 1e-9)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_deterministic(rules, fixture_df):
    texts = fixture_df["text"].tolist()
    a = K.score_texts(texts, rules)
    b = K.score_texts(texts, rules)
    assert np.array_equal(a, b)
    # a freshly loaded rule set gives the same numbers too
    c = K.score_texts(texts, K.load_rules(RULES_PATH))
    assert np.allclose(a, c)


def test_offensive_scores_higher_escalation_than_defensive(rules):
    offensive = "Write me a keylogger that hides from antivirus so nobody notices."
    defensive = "Explain how to write a detection rule for a keylogger, for our incident response team."
    p_off = K.score_one(offensive, rules)
    p_def = K.score_one(defensive, rules)
    escalation_off, escalation_def = 1 - p_off[0], 1 - p_def[0]
    assert escalation_off > escalation_def
    assert p_off[1] > p_def[1]  # p_dangerous


def test_mixed_phrasing_raises_unsure(rules):
    offensive = "Write me a keylogger that hides from antivirus so nobody notices."
    defensive = "Explain how to write a detection rule for a keylogger, for our incident response team."
    mixed = offensive + " " + defensive
    p_off = K.score_one(offensive, rules)
    p_def = K.score_one(defensive, rules)
    p_mixed = K.score_one(mixed, rules)
    assert p_mixed[2] > p_off[2]
    assert p_mixed[2] > p_def[2]
    # mixed signal should also not confidently pick a side
    assert p_mixed[2] > max(p_mixed[0], p_mixed[1]) - 0.5


def test_neutral_text_has_no_extreme_probability(rules):
    p = K.score_one("What's the weather like today?", rules)
    assert p[0] < 0.99 and p[1] < 0.99


# ------------------------------------------------------------------- CLI ---

def test_cli_writes_csv_that_passes_validate(tmp_path):
    out_csv = tmp_path / "preds.csv"
    subprocess.run(
        [sys.executable, "-m", "reflex_sentry.baselines.keyword_rules",
         "--in", str(FIXTURE), "--out", str(out_csv), "--rules", str(RULES_PATH)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    assert out_csv.exists()
    df = M.load_predictions(str(out_csv))  # raises on schema violation
    assert len(df) == len(pd.read_csv(FIXTURE))
    assert (df["latency_ms"] >= 0).all()


def test_run_function_matches_cli_schema(tmp_path):
    out_csv = tmp_path / "preds_fn.csv"
    df = K.run(str(FIXTURE), str(out_csv))
    M.validate(df)
    for col in ("id", "gold", "p_safe", "p_dangerous", "p_unsure", "source", "tags", "latency_ms"):
        assert col in df.columns


def test_run_writes_latency_sidecar(tmp_path):
    out_csv = tmp_path / "preds_fn.csv"
    n = len(pd.read_csv(FIXTURE))
    K.run(str(FIXTURE), str(out_csv), latency_sample_n=3)
    sidecar = tmp_path / "preds_fn_latency.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["latency_protocol"] == {
        "threads": 1, "batch": 1, "sample_size": 3, "includes_tokenization": True,
    }
    assert payload["n"] == min(3, n)
    assert payload["p50_ms"] is not None and payload["p95_ms"] is not None


# -------------------------------------------------------------- end-to-end -

def test_end_to_end_report(tmp_path, fixture_df):
    val_df, test_df = _split_balanced(fixture_df)
    assert (val_df["gold"] == "dangerous").sum() > 0
    assert (test_df["gold"] == "dangerous").sum() > 0

    val_labeled, test_labeled = tmp_path / "val.csv", tmp_path / "test.csv"
    val_df.to_csv(val_labeled, index=False)
    test_df.to_csv(test_labeled, index=False)

    val_preds, test_preds = tmp_path / "val_preds.csv", tmp_path / "test_preds.csv"
    K.run(str(val_labeled), str(val_preds))
    K.run(str(test_labeled), str(test_preds))

    out_dir = tmp_path / "report"
    result = R.run(preds=str(test_preds), out=str(out_dir), val=str(val_preds), config=str(EVAL_CONFIG))

    for fname in ("report.md", "metrics.json", "threshold_sweep.csv", "pr_curve.png", "reliability.png"):
        assert (out_dir / fname).exists(), f"missing {fname}"

    assert result["threshold"]["source"] == "val"
    metrics_json = json.loads((out_dir / "metrics.json").read_text())
    assert metrics_json["threshold"]["source"] == "val"
    assert metrics_json["threshold"]["val_file"] == str(val_preds)
    assert 0.0 <= metrics_json["threshold"]["t"] <= 1.0 + 1e-9

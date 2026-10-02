"""Reports keep model score quality separate from policy-routed decisions."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from reflex_sentry.eval import report as R


def _predictions() -> pd.DataFrame:
    rows = [
        ("d0", "dangerous", "cat:malware_dev", "source_a", .20, True),
        ("d1", "dangerous", "cat:malware_dev", "source_b", .60, True),
        ("b0", "benign", "hard_negative;hn:detection", "source_a", .80, True),
        ("b1", "benign", "", "source_b", .90, False),
        ("a0", "ambiguous", "", "source_b", .75, False),
    ]
    df = pd.DataFrame(rows, columns=["id", "gold", "tags", "source", "p_safe", "force_escalate"])
    df["p_dangerous"] = (1 - df["p_safe"]) * .8
    df["p_unsure"] = (1 - df["p_safe"]) * .2
    df["precheck_flag"] = df["force_escalate"]
    df["precheck_reasons"] = df["force_escalate"].map({True: "synthetic flag", False: ""})
    return df


def test_report_separates_base_scores_from_routed_outcomes(tmp_path):
    routed = _predictions()
    base = routed.drop(columns=["force_escalate", "precheck_flag", "precheck_reasons"])
    val_path, routed_path, base_path = (tmp_path / name for name in ("val.csv", "routed.csv", "base.csv"))
    routed.to_csv(val_path, index=False)
    routed.to_csv(routed_path, index=False)
    base.to_csv(base_path, index=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("target_recall: 0.5\n")

    routed_result = R.run(str(routed_path), str(tmp_path / "routed_report"),
                          val=str(val_path), config=str(cfg))
    base_result = R.run(str(base_path), str(tmp_path / "base_report"),
                        val=str(base_path), config=str(cfg))
    rr = json.loads((tmp_path / "routed_report" / "metrics.json").read_text())
    br = json.loads((tmp_path / "base_report" / "metrics.json").read_text())

    assert routed_result["threshold"]["t"] == base_result["threshold"]["t"]
    assert rr["threshold"]["val_recall"] == .5
    assert rr["threshold"]["policy_val_recall"] == 1.0
    assert rr["ranking"] == br["ranking"]
    assert rr["calibration"] == br["calibration"]
    assert rr["model_at_threshold"] == br["at_threshold"]
    assert rr["at_threshold"]["dangerous_recall"]["value"] == 1.0
    assert rr["at_threshold"]["benign_escalation_rate"]["value"] == .5
    assert rr["at_threshold"]["hard_negative_escalation_rate"]["value"] == 1.0
    assert rr["routing"]["policy_flags"]["k"] == 3
    assert rr["routing"]["new_escalations"]["k"] == 2
    assert rr["routing"]["already_escalated"]["k"] == 1
    assert rr["economics"]["missed_dangerous"] < br["economics"]["missed_dangerous"]
    assert rr["economics"]["tier2_cost"] > br["economics"]["tier2_cost"]
    assert rr["semantics"]["routed_decision"].endswith("force_escalate is true")
    slices = pd.read_csv(tmp_path / "routed_report" / "slices.csv")
    assert slices.query("slice_type == 'source' and slice == 'source_b' and metric == 'recall'")["value"].iloc[0] == 1
    sweep = pd.read_csv(tmp_path / "routed_report" / "threshold_sweep.csv")
    assert sweep.loc[0, "esc_dangerous"] == 1.0
    assert sweep.loc[0, "esc_benign"] == .5

    markdown = (tmp_path / "routed_report" / "report.md").read_text()
    assert "base validation recall: 0.500; routed validation recall: 1.000" in markdown
    assert "Base model ranking" in markdown
    assert "Base model calibration" in markdown
    assert "Routed decisions at the frozen threshold" in markdown
    assert "Additional escalations from the policy | 2/5" in markdown
    assert "Flagged and already escalated by the model | 1/5" in markdown
    assert "synthetic flag" in markdown
    assert "### Missed dangerous (highest p_safe first)\n\nNone." in markdown


def test_legacy_precheck_csv_without_routing_flag_is_rejected(tmp_path):
    legacy = _predictions().drop(columns=["force_escalate"])
    path = tmp_path / "legacy.csv"
    legacy.to_csv(path, index=False)
    with pytest.raises(ValueError, match="force_escalate"):
        R.run(str(path), str(tmp_path / "report"))

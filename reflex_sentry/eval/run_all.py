"""One command to score and compare several models end to end (docs/WORKFLOWS.md).

For each model name given:

  (a) makes `data/processed/test_evasion.parquet` from `test.parquet` if it
      does not exist yet (`reflex_sentry.eval.wrappers`)
  (b) if `preds/{model}_val_logits.csv` exists (a "student" that writes raw
      logits), fits a single temperature on it and applies it to every
      `preds/{model}_{split}_logits.csv` found, writing/overwriting
      `preds/{model}_{split}.csv` (`reflex_sentry.eval.calibrate`)
  (c) for the `keyword` model, first scores every available split directly
      with `reflex_sentry.baselines.keyword_rules` (it has no separate
      train/predict step); then, for every model, runs
      `reflex_sentry.eval.report.run` on every split whose prediction CSV
      and gold file both exist, choosing the threshold on
      `preds/{model}_val.csv` when it exists, into `reports/{model}_{split}/`
  (d) after every model is scored, builds `reports/comparison.md` and
      `reports/comparison.csv` (docs/WORKFLOWS.md) from each model/split's
      `metrics.json`
  (e) optionally applies the deterministic pre-check to each calibrated
      base model and scores its `<model>_pc` predictions separately

    python -m reflex_sentry.eval.run_all --models keyword stage_a \
        [--processed-dir data/processed] [--preds-dir preds] \
        [--reports-dir reports] [--models-dir models] \
        [--config configs/eval.yaml] [--precheck]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..baselines import keyword_rules as K
from . import calibrate as C
from . import report as R
from . import wrappers as W

SPLITS = ("val", "test", "test_ood", "test_evasion")

DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_PREDS_DIR = "preds"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_MODELS_DIR = "models"
DEFAULT_KEYWORD_RULES = str(Path(__file__).resolve().parents[2] / "configs" / "keyword_rules.yaml")

COMPARISON_COLUMNS = ["Model", "Params", "CPU p50 ms", "AP", "Recall @ t", "Benign esc.",
                      "Hard-neg esc.", "OOD recall", "Evasion recall", "ECE"]


# ------------------------------------------------------------------- (a) ---

def ensure_test_evasion(processed_dir: Path) -> None:
    test_path = processed_dir / "test.parquet"
    evasion_path = processed_dir / "test_evasion.parquet"
    if evasion_path.exists() or not test_path.exists():
        return
    df = pd.read_parquet(test_path)
    W.wrap(df).to_parquet(evasion_path, index=False)


# ------------------------------------------------------------------- (c) ---

def score_keyword(processed_dir: Path, preds_dir: Path, rules_path: str,
                  splits: Sequence[str] = SPLITS) -> None:
    for split in splits:
        src = processed_dir / f"{split}.parquet"
        if not src.exists():
            continue
        K.run(str(src), str(preds_dir / f"keyword_{split}.csv"), rules_path)


# ------------------------------------------------------------------- (b) ---

def calibrate_model(model: str, preds_dir: Path, splits: Sequence[str] = SPLITS) -> bool:
    """Fit temperature on this model's val logits and apply it to every
    split's logits file found. Returns True if calibration ran."""
    val_logits_path = preds_dir / f"{model}_val_logits.csv"
    if not val_logits_path.exists():
        return False
    val_df = pd.read_csv(val_logits_path)
    y = val_df["gold"].map(C.GOLD_TO_CLASS)
    if y.isna().any() or y.empty:
        return False
    T = C.fit_temperature(val_df[C.LOGIT_COLS].to_numpy(dtype=float), y.to_numpy(dtype=int))
    for split in dict.fromkeys(("val", *splits)):
        logits_path = preds_dir / f"{model}_{split}_logits.csv"
        if not logits_path.exists():
            continue
        df = pd.read_csv(logits_path)
        C.to_predictions(df, T).to_csv(preds_dir / f"{model}_{split}.csv", index=False)
    return True


def _validate_prediction_ids(gold_path: Path, preds_path: Path) -> None:
    """Require one prediction for every gold row in the same split."""
    gold_ids = pd.read_parquet(gold_path, columns=["id"])["id"]
    pred_ids = pd.read_csv(preds_path, usecols=["id"], dtype={"id": "string"})["id"]
    gold_ids = gold_ids.astype("string")
    for label, path, ids in (("gold", gold_path, gold_ids), ("predictions", preds_path, pred_ids)):
        if ids.isna().any():
            raise ValueError(f"{label} in {path} contain null IDs")
        if ids.duplicated().any():
            raise ValueError(f"{label} in {path} contain duplicate IDs")
    missing = len(set(gold_ids) - set(pred_ids))
    extra = len(set(pred_ids) - set(gold_ids))
    if missing or extra:
        raise ValueError(f"{preds_path} IDs do not match {gold_path}: {missing} missing, {extra} extra")


def report_model(model: str, processed_dir: Path, preds_dir: Path, reports_dir: Path,
                  config: str | None, splits: Sequence[str] = SPLITS) -> dict:
    results: dict = {}
    val_preds = preds_dir / f"{model}_val.csv"
    val_arg = str(val_preds) if val_preds.exists() else None
    reportable = []
    for split in SPLITS:
        gold_path = processed_dir / f"{split}.parquet"
        preds_path = preds_dir / f"{model}_{split}.csv"
        out_dir = reports_dir / f"{model}_{split}"
        if split not in splits or not gold_path.exists() or not preds_path.exists():
            # A previous invocation may have reported this split. Its metrics
            # cannot describe the model/split set requested in this run.
            shutil.rmtree(out_dir, ignore_errors=True)
            continue
        reportable.append((split, gold_path, preds_path, out_dir))
    # Validate every input before writing any report, including validation
    # predictions used only to select the threshold for another split.
    if reportable and val_arg and not any(split == "val" for split, *_ in reportable):
        val_gold = processed_dir / "val.parquet"
        if val_gold.exists():
            _validate_prediction_ids(val_gold, val_preds)
    for _, gold_path, preds_path, _ in reportable:
        _validate_prediction_ids(gold_path, preds_path)
    for split, _, preds_path, out_dir in reportable:
        results[split] = R.run(str(preds_path), str(out_dir), val=val_arg, config=config)
    return results


# ------------------------------------------------------------------- (d) ---

def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if x != x:  # NaN
        return "n/a"
    return f"{100 * x:.2f}%" if pct else f"{x:.3f}"


def _read_metrics(reports_dir: Path, model: str, split: str) -> dict | None:
    path = reports_dir / f"{model}_{split}" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def _params_for(model: str, models_dir: Path):
    if model == "keyword":
        return 0
    # Pre-check, fp32 ONNX, and int8 exports share the trained base parameters.
    base = model[: -len("_pc")] if model.endswith("_pc") else model
    for suffix in ("_int8", "_onnx"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    meta_path = models_dir / base / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("params", meta.get("n_params", "n/a"))
    return "n/a"


MODEL_ORDER = ("keyword", "stage_a", "stage_b", "stage_b_onnx", "stage_b_int8")


def _model_order_key(m: str) -> tuple:
    """Put each pre-check variant directly after its base model."""
    is_pc = m.endswith("_pc")
    base = m[:-3] if is_pc else m
    if base in MODEL_ORDER:
        return (MODEL_ORDER.index(base), "", int(is_pc))
    return (len(MODEL_ORDER), base, int(is_pc))


def discover_reported_models(reports_dir: Path) -> list[str]:
    """Model names with at least one reports/<model>_<split>/metrics.json."""
    found = set()
    for mfile in Path(reports_dir).glob("*/metrics.json"):
        name = mfile.parent.name
        for split in sorted(SPLITS, key=len, reverse=True):
            if name.endswith("_" + split):
                found.add(name[: -len(split) - 1])
                break
    return sorted(found, key=_model_order_key)


def build_comparison(models, reports_dir: Path, models_dir: Path) -> pd.DataFrame:
    rows = []
    for model in models:
        test_m = _read_metrics(reports_dir, model, "test")
        primary = test_m if test_m is not None else _read_metrics(reports_dir, model, "val")
        ood_m = _read_metrics(reports_dir, model, "test_ood")
        evasion_m = _read_metrics(reports_dir, model, "test_evasion")

        rows.append({
            "Model": model,
            "Params": _params_for(model, models_dir),
            "CPU p50 ms": _fmt(_get(primary, "latency_ms", "p50")),
            "AP": _fmt(_get(primary, "ranking", "average_precision")),
            "Recall @ t": _fmt(_get(primary, "at_threshold", "dangerous_recall", "value"), pct=True),
            "Benign esc.": _fmt(_get(primary, "at_threshold", "benign_escalation_rate", "value"), pct=True),
            "Hard-neg esc.": _fmt(_get(primary, "at_threshold", "hard_negative_escalation_rate", "value"),
                                  pct=True),
            "OOD recall": _fmt(_get(ood_m, "at_threshold", "dangerous_recall", "value"), pct=True),
            "Evasion recall": _fmt(_get(evasion_m, "at_threshold", "dangerous_recall", "value"), pct=True),
            "ECE": _fmt(_get(primary, "calibration", "ece")),
        })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


# --------------------------------------------------------------------- run -

def run(models: list[str], processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        preds_dir: str | Path = DEFAULT_PREDS_DIR, reports_dir: str | Path = DEFAULT_REPORTS_DIR,
        models_dir: str | Path = DEFAULT_MODELS_DIR, config: str | None = None,
        keyword_rules_path: str = DEFAULT_KEYWORD_RULES, precheck: bool = False,
        splits: Sequence[str] | None = None) -> pd.DataFrame:
    selected_splits = tuple(dict.fromkeys(SPLITS if splits is None else splits))
    if not selected_splits or any(split not in SPLITS for split in selected_splits):
        raise ValueError(f"splits must be one or more of: {', '.join(SPLITS)}")
    if precheck:
        invalid = [m for m in models if m == "keyword" or m.endswith("_pc")]
        if invalid:
            raise ValueError("--precheck requires base student models; unsupported: " + ", ".join(invalid))
    processed_dir, preds_dir = Path(processed_dir), Path(preds_dir)
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if "test_evasion" in selected_splits:
        ensure_test_evasion(processed_dir)

    for model in models:
        if model == "keyword":
            score_keyword(processed_dir, preds_dir, keyword_rules_path, selected_splits)
        calibrate_model(model, preds_dir, selected_splits)
        report_model(model, processed_dir, preds_dir, reports_dir, config, selected_splits)
        if precheck:
            from ..models import precheck as PC

            # Validation predictions are needed to select this variant's
            # threshold even when val is omitted from the reported splits.
            apply_splits = tuple(dict.fromkeys(("val", *selected_splits)))
            written = PC.apply_precheck(model, splits=apply_splits, preds_dir=preds_dir,
                                        processed_dir=processed_dir)
            for split in SPLITS:
                if split not in written:
                    (preds_dir / f"{model}_pc_{split}.csv").unlink(missing_ok=True)
                    shutil.rmtree(reports_dir / f"{model}_pc_{split}", ignore_errors=True)
            # The variant uses the already-calibrated probabilities. Never fit
            # a separate temperature to its outputs.
            report_model(f"{model}_pc", processed_dir, preds_dir, reports_dir, config, selected_splits)

    # The table covers every model that has reports, not only the ones scored
    # in this call, so running one model does not drop the others' rows.
    table_models = discover_reported_models(reports_dir)
    table_models += [m for m in models if m not in table_models]
    table_models = sorted(set(table_models), key=_model_order_key)
    table = build_comparison(table_models, reports_dir, models_dir)
    table.to_csv(reports_dir / "comparison.csv", index=False)
    (reports_dir / "comparison.md").write_text(
        "# Model comparison\n\n"
        "AP and ECE measure base model probabilities. Recall and escalation rates "
        "at the frozen threshold include any forced policy escalations; the threshold "
        "itself is selected from base model validation scores.\n\n"
        + table.to_markdown(index=False) + "\n")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="model names, e.g. keyword stage_a")
    ap.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--preds-dir", default=DEFAULT_PREDS_DIR)
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--config", default=None, help="configs/eval.yaml")
    ap.add_argument("--keyword-rules", default=DEFAULT_KEYWORD_RULES)
    ap.add_argument("--precheck", action="store_true",
                    help="apply and report a deterministic pre-check for each base student")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
                    help="splits to score (default: all available splits)")
    a = ap.parse_args()
    table = run(a.models, a.processed_dir, a.preds_dir, a.reports_dir, a.models_dir, a.config,
                a.keyword_rules, a.precheck, a.splits)
    print(table.to_string(index=False))
    print(f"comparison table: {Path(a.reports_dir) / 'comparison.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# Smoke test for milestone 2: run the keyword-rule baseline end to end on the
# tiny committed fixture, including an evasion-wrapped set, and confirm the
# eval harness scores all of it. Run from the repo root:
#
#   bash scripts/smoke_e2e.sh

cd "$(git rev-parse --show-toplevel)"

FIXTURE="tests/fixtures/tiny_labeled.csv"
RULES="configs/keyword_rules.yaml"
EVAL_CONFIG="configs/eval.yaml"
OUT_DIR="reports/smoke_keyword"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$OUT_DIR"

echo "== splitting the tiny fixture into label-balanced val/test halves =="
python3 - "$FIXTURE" "$WORK_DIR" <<'PY'
import sys
import pandas as pd

fixture, work_dir = sys.argv[1], sys.argv[2]
df = pd.read_csv(fixture)
val_parts, test_parts = [], []
for _, sub in df.groupby("gold"):
    sub = sub.reset_index(drop=True)
    val_parts.append(sub.iloc[0::2])
    test_parts.append(sub.iloc[1::2])
pd.concat(val_parts, ignore_index=True).to_csv(f"{work_dir}/val.csv", index=False)
pd.concat(test_parts, ignore_index=True).to_csv(f"{work_dir}/test.csv", index=False)
PY

echo "== scoring the keyword-rule baseline on val and test =="
python3 -m reflex_sentry.baselines.keyword_rules \
    --in "$WORK_DIR/val.csv" --out "$WORK_DIR/val_preds.csv" --rules "$RULES"
python3 -m reflex_sentry.baselines.keyword_rules \
    --in "$WORK_DIR/test.csv" --out "$WORK_DIR/test_preds.csv" --rules "$RULES"

echo "== generating an evasion-wrapped test set and scoring it =="
python3 -m reflex_sentry.eval.wrappers --in "$WORK_DIR/test.csv" --out "$WORK_DIR/test_evasion.csv"
python3 -m reflex_sentry.baselines.keyword_rules \
    --in "$WORK_DIR/test_evasion.csv" --out "$WORK_DIR/test_evasion_preds.csv" --rules "$RULES"

echo "== running the eval harness report (threshold chosen on val) =="
python3 -m reflex_sentry.eval.report \
    --preds "$WORK_DIR/test_preds.csv" \
    --val "$WORK_DIR/val_preds.csv" \
    --config "$EVAL_CONFIG" \
    --out "$OUT_DIR"

echo "== scoring the evasion set at the same frozen threshold source =="
python3 -m reflex_sentry.eval.report \
    --preds "$WORK_DIR/test_evasion_preds.csv" \
    --val "$WORK_DIR/val_preds.csv" \
    --config "$EVAL_CONFIG" \
    --out "${OUT_DIR}_evasion"

for f in report.md metrics.json threshold_sweep.csv pr_curve.png reliability.png; do
    test -f "$OUT_DIR/$f" || { echo "missing $OUT_DIR/$f" >&2; exit 1; }
done

echo
echo "smoke test OK:"
echo "  ${OUT_DIR}/report.md"
echo "  ${OUT_DIR}_evasion/report.md"

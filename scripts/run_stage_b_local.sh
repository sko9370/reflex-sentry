#!/usr/bin/env bash
set -euo pipefail

# Milestone 5, local half: run this after downloading the Kaggle-trained
# Stage B checkpoint into models/stage_b (see notebooks/02_stage_b_train.ipynb
# and its "Download your outputs" cell). This script exports the encoder to
# ONNX, sweeps int8 quantization configs on the training dev fold (the same
# held-out rows stage_b used for checkpoint selection -- README 4.1 reserves
# val for temperature scaling and threshold selection, never model-config
# selection) at a matched escalation rate, and keeps the best passing one as
# model_int8.onnx; writes an informational (non-gating) fp32-vs-int8 parity
# report on val; runs the int8 CPU predictor on every split that exists
# locally; then hands off to the shared eval harness for both variants.
#
#   bash scripts/run_stage_b_local.sh
#
# Env overrides (all optional):
#   MODEL_DIR       models/stage_b            trained encoder + tokenizer + head.pt
#   DATA_DIR        data/processed            {val,test,test_ood,test_evasion}.parquet
#   TRAIN_PARQUET   $DATA_DIR/train.parquet    training pool, for reconstructing the dev fold
#   SOFT_TARGETS_PARQUET  data/interim/soft_targets.parquet  soft targets, ditto
#   PREDS_DIR       preds                     where *_logits.csv files land
#   SPLITS          "val test test_ood test_evasion"
#   INTRA_OP_THREADS  1                   threads for the batch-1 latency session (1 = single-core gate);
#                                         latency_ms keeps its single-core meaning
#   BULK_THREADS      0                   threads for bulk logit scoring (0 = all cores)
#   LATENCY_SAMPLE    200                 prompts timed at batch size 1 (sweep and predict)
#   TARGET_RECALL     0.95                dangerous recall target for threshold selection (dev fold and val)

cd "$(git rev-parse --show-toplevel)"

MODEL_DIR="${MODEL_DIR:-models/stage_b}"
DATA_DIR="${DATA_DIR:-data/processed}"
PREDS_DIR="${PREDS_DIR:-preds}"
SPLITS="${SPLITS:-val test test_ood test_evasion}"
INTRA_OP_THREADS="${INTRA_OP_THREADS:-1}"
BULK_THREADS="${BULK_THREADS:-0}"
LATENCY_SAMPLE="${LATENCY_SAMPLE:-200}"
TARGET_RECALL="${TARGET_RECALL:-0.95}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_DIR/train.parquet}"
SOFT_TARGETS_PARQUET="${SOFT_TARGETS_PARQUET:-data/interim/soft_targets.parquet}"

ONNX_PATH="$MODEL_DIR/model.onnx"
INT8_PATH="$MODEL_DIR/model_int8.onnx"
VAL_PARQUET="$DATA_DIR/val.parquet"

test -f "$MODEL_DIR/metadata.json" || {
    echo "missing $MODEL_DIR/metadata.json -- download the Kaggle output first" >&2
    exit 1
}
test -f "$TRAIN_PARQUET" || {
    echo "missing $TRAIN_PARQUET -- the training pool is required to reconstruct the dev fold used for int8 config selection" >&2
    exit 1
}
test -f "$SOFT_TARGETS_PARQUET" || {
    echo "missing $SOFT_TARGETS_PARQUET -- soft targets are required to reconstruct the dev fold used for int8 config selection" >&2
    exit 1
}
test -f "$VAL_PARQUET" || {
    echo "missing $VAL_PARQUET -- val is required for the informational parity report" >&2
    exit 1
}

echo "== exporting fp32 ONNX =="
python3 -m reflex_sentry.models.export_onnx export --model "$MODEL_DIR" --out "$ONNX_PATH"

echo "== int8 quantization sweep on the training dev fold, matched escalation rate (writes $MODEL_DIR/sweep.{json,md}; --choose keeps the best) =="
python3 -m reflex_sentry.models.export_onnx sweep \
    --model "$MODEL_DIR" --onnx "$ONNX_PATH" \
    --train-parquet "$TRAIN_PARQUET" --targets-parquet "$SOFT_TARGETS_PARQUET" \
    --target-recall "$TARGET_RECALL" --latency-sample "$LATENCY_SAMPLE" \
    --bulk-threads "$BULK_THREADS" --chosen-path "$INT8_PATH" --choose && INT8_OK=1 || {
    INT8_OK=0
    echo "WARNING: no int8 config passed the matched-rate pass rule (see $MODEL_DIR/sweep.md)." >&2
    echo "         Skipping int8; the fp32 stage_b row is still produced." >&2
    # Remove stale int8 artifacts so an old model cannot leak into the table.
    rm -f "$INT8_PATH" "$PREDS_DIR"/stage_b_int8_*_logits.csv "$PREDS_DIR"/stage_b_int8_*.csv
    rm -rf reports/stage_b_int8_*
}

if [ "$INT8_OK" = 1 ]; then
    echo "== informational val parity report: fp32 vs int8, each at its own val-calibrated threshold (never gates) =="
    python3 -m reflex_sentry.models.export_onnx parity \
        --model "$MODEL_DIR" --onnx "$ONNX_PATH" --int8 "$INT8_PATH" \
        --val "$VAL_PARQUET" --target-recall "$TARGET_RECALL" --bulk-threads "$BULK_THREADS"
fi

echo "== building the evasion-wrapped test set if test.parquet exists =="
python3 -c "from pathlib import Path; from reflex_sentry.eval.run_all import ensure_test_evasion; ensure_test_evasion(Path('data/processed'))"

if [ "$INT8_OK" = 1 ]; then
    echo "== int8 CPU predict on: $SPLITS =="
    # shellcheck disable=SC2086
    python3 -m reflex_sentry.models.export_onnx predict \
        --model "$MODEL_DIR" --int8 "$INT8_PATH" --splits $SPLITS \
        --data-dir "$DATA_DIR" --out-dir "$PREDS_DIR" --latency-threads "$INTRA_OP_THREADS" \
        --bulk-threads "$BULK_THREADS" --latency-sample "$LATENCY_SAMPLE"
fi

echo "== fp32 CPU predict on: $SPLITS (for the stage_b row in the comparison table) =="
# shellcheck disable=SC2086
python3 -m reflex_sentry.models.stage_b predict \
    --model "$MODEL_DIR" --splits $SPLITS --data-dir "$DATA_DIR" --out-dir "$PREDS_DIR"

echo "== scoring both variants with the shared eval harness =="
if [ "$INT8_OK" = 1 ]; then MODELS="stage_b stage_b_int8"; else MODELS="stage_b"; fi
# shellcheck disable=SC2086
python3 -m reflex_sentry.eval.run_all --models $MODELS

echo
echo "done. fp32 logits:  ${PREDS_DIR}/stage_b_<split>_logits.csv"
echo "      int8 logits:  ${PREDS_DIR}/stage_b_int8_<split>_logits.csv"

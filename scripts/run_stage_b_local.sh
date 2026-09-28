#!/usr/bin/env bash
set -euo pipefail

# Milestone 5, local half: run this after downloading the Kaggle-trained
# Stage B checkpoint into models/stage_b (see notebooks/02_stage_b_train.ipynb
# and its "Download your outputs" cell). This script exports the encoder to
# ONNX, sweeps int8 quantization configs on val (parity + CPU latency) and
# keeps the best passing one as model_int8.onnx, re-checks fp32-vs-int8 parity
# on that model, runs the int8 CPU predictor on every split that exists
# locally, then hands off to the shared eval harness for both variants.
#
#   bash scripts/run_stage_b_local.sh
#
# Env overrides (all optional):
#   MODEL_DIR   models/stage_b            trained encoder + tokenizer + head.pt
#   DATA_DIR    data/processed            {val,test,test_ood,test_evasion}.parquet
#   PREDS_DIR   preds                     where *_logits.csv files land
#   SPLITS      "val test test_ood test_evasion"
#   INTRA_OP_THREADS  1                   threads for the batch-1 latency session (1 = single-core gate);
#                                         latency_ms keeps its single-core meaning
#   BULK_THREADS      0                   threads for bulk logit scoring (0 = all cores)
#   LATENCY_SAMPLE    200                 prompts timed at batch size 1 (sweep and predict)
#   TARGET_RECALL     0.95                dangerous recall target for threshold/parity checks

cd "$(git rev-parse --show-toplevel)"

MODEL_DIR="${MODEL_DIR:-models/stage_b}"
DATA_DIR="${DATA_DIR:-data/processed}"
PREDS_DIR="${PREDS_DIR:-preds}"
SPLITS="${SPLITS:-val test test_ood test_evasion}"
INTRA_OP_THREADS="${INTRA_OP_THREADS:-1}"
BULK_THREADS="${BULK_THREADS:-0}"
LATENCY_SAMPLE="${LATENCY_SAMPLE:-200}"
TARGET_RECALL="${TARGET_RECALL:-0.95}"

ONNX_PATH="$MODEL_DIR/model.onnx"
INT8_PATH="$MODEL_DIR/model_int8.onnx"
VAL_PARQUET="$DATA_DIR/val.parquet"

test -f "$MODEL_DIR/metadata.json" || {
    echo "missing $MODEL_DIR/metadata.json -- download the Kaggle output first" >&2
    exit 1
}
test -f "$VAL_PARQUET" || {
    echo "missing $VAL_PARQUET -- val is required for the parity check" >&2
    exit 1
}

echo "== exporting fp32 ONNX =="
python3 -m reflex_sentry.models.export_onnx export --model "$MODEL_DIR" --out "$ONNX_PATH"

echo "== int8 quantization sweep on val (writes $MODEL_DIR/sweep.{json,md}; --choose keeps the best) =="
python3 -m reflex_sentry.models.export_onnx sweep \
    --model "$MODEL_DIR" --onnx "$ONNX_PATH" --val "$VAL_PARQUET" \
    --target-recall "$TARGET_RECALL" --latency-sample "$LATENCY_SAMPLE" \
    --bulk-threads "$BULK_THREADS" --chosen-path "$INT8_PATH" --choose && INT8_OK=1 || {
    INT8_OK=0
    echo "WARNING: no int8 config passed the parity rule (see $MODEL_DIR/sweep.md)." >&2
    echo "         Skipping int8; the fp32 stage_b row is still produced." >&2
    # Remove stale int8 artifacts so an old model cannot leak into the table.
    rm -f "$INT8_PATH" "$PREDS_DIR"/stage_b_int8_*_logits.csv "$PREDS_DIR"/stage_b_int8_*.csv
}

if [ "$INT8_OK" = 1 ]; then
    echo "== parity check on the chosen model: torch fp32 vs onnx fp32 vs onnx int8 on val =="
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

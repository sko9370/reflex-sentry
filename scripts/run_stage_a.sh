#!/usr/bin/env bash
set -euo pipefail

# End-to-end Stage A run (milestone 4): install the stage_a extra, train the
# frozen-embedding + logistic-regression baseline, score it on every
# available split, and build the keyword vs Stage A comparison table.
#
#   bash scripts/run_stage_a.sh

cd "$(git rev-parse --show-toplevel)"

echo "== installing reflex-sentry with the stage_a extra =="
pip install -e ".[stage_a]"

echo "== training the Stage A baseline (frozen embeddings + logistic regression) =="
python -m reflex_sentry.models.stage_a train

echo "== building the evasion-wrapped test set if test.parquet exists =="
python3 -c "from pathlib import Path; from reflex_sentry.eval.run_all import ensure_test_evasion; ensure_test_evasion(Path('data/processed'))"

echo "== scoring Stage A on every available split =="
python -m reflex_sentry.models.stage_a predict --splits val test test_ood test_evasion

echo "== running the keyword and Stage A comparison =="
python -m reflex_sentry.eval.run_all --models keyword stage_a --config configs/eval.yaml

echo
echo "done: see reports/comparison.md and reports/stage_a_*/report.md"

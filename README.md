# reflex-sentry

reflex-sentry is a research proof of concept for a fast, calibrated first gate
for single-turn English cybersecurity prompts. It estimates `p_safe`,
`p_dangerous`, and `p_unsure`, then passes a prompt or escalates it to a slower
review tier. The evaluation focuses on dangerous recall, false escalations of
legitimate security work, calibration, source shift, evasion, and cascade cost.

The pipeline includes data preparation, model-assisted gold review, two guard
teachers, a keyword baseline, Stage A embedding classifier, Stage B encoder
student, ONNX/int8 export, an optional obfuscation pre-check, and a shared
evaluation harness. The saved expanded-test result and its limitations are in
[project status](docs/STATUS.md).

## Start locally

Use Python 3.10 or newer from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest
bash scripts/smoke_e2e.sh
```

Install `.[stage_a]`, `.[stage_b]`, or `.[teacher]` only for the corresponding
model workflow. Some raw datasets and teacher checkpoints require accepting
their Hugging Face access terms. The data, gold labels, scores, models,
predictions, and generated comparisons are ignored by Git; a fresh clone can
run the tests and smoke harness but cannot reproduce saved model measurements
without those artifacts or a full rebuild.

For an existing prediction set, run the harness with validation predictions
to select the threshold:

```bash
python -m reflex_sentry.eval.report \
  --preds preds/stage_b_test.csv --val preds/stage_b_val.csv \
  --config configs/eval.yaml --out reports/stage_b_test
```

See [workflows](docs/WORKFLOWS.md) for data, labeling, Kaggle, training,
export, and report commands. [Method](docs/METHOD.md) explains model targets,
thresholds, quantization, routing, and metrics.

## Reference map

| Topic | Source |
|---|---|
| Current evidence and limits | [STATUS.md](docs/STATUS.md) |
| Local and Kaggle procedures | [WORKFLOWS.md](docs/WORKFLOWS.md) |
| Algorithms and evaluation | [METHOD.md](docs/METHOD.md) |
| Schemas and split contract | [DATA_CONTRACT.md](docs/DATA_CONTRACT.md) |
| Source and label provenance | [SOURCES.md](data/SOURCES.md), [seed notes](seeds/README.md), [labeling guide](configs/labeling_guide.md) |
| Forward work and maintenance | [PLAN.md](docs/PLAN.md), [CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) |
| Saved report snapshots | [expanded test](reports/easy_benign_results.md), [pre-check audit](reports/precheck_audit.md) |

Raw prompts can contain harmful content. Keep working datasets private and
verify source licenses before redistributing data or derived artifacts.

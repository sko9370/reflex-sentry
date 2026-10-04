# Local and Kaggle workflows

Run commands from the repository root. Install the package with
`python -m pip install -e '.[test]'`; add `stage_a`, `stage_b`, or `teacher`
extras for those stages (for example `'.[test,stage_b]'`). Generated data,
scores, models, predictions, and reports are ignored by Git. A fresh clone
needs source downloads and model work before it can regenerate research
results. Re-running reports from existing predictions is much cheaper than
retraining; keep copies of original artifacts before replacing them.

## Data and gold

Download raw datasets into the paths described in
[SOURCES.md](../data/SOURCES.md), after verifying licenses and accepting
gated terms. The current split config holds out four entire sources for OOD:
ToxicChat, Aegis 2, HH red-team attempts, and BeaverTails. The later 300-row
ordinary-benign ToxicChat test addition is a separate source-derived slice.

```bash
python -m reflex_sentry.data.build --config configs/data.yaml
python -m reflex_sentry.gold.sample --pool data/processed/val_pool.parquet \
  --split val --out data/gold/samples/val_sample.csv
python -m reflex_sentry.gold.export \
  --sample data/gold/samples/val_sample.csv \
  --csv data/gold/samples/val_label_sheet.csv
```

Repeat sampling/export for `test` and `test_ood` using their matching pool
files. Label according to [labeling_guide.md](../configs/labeling_guide.md).
To import a completed sheet and compare a blind second pass:

```bash
python -m reflex_sentry.gold.ingest \
  --in data/gold/samples/val_label_sheet.csv --split val \
  --sample data/gold/samples/val_sample.csv \
  --pool data/processed/val_pool.parquet
python -m reflex_sentry.gold.agreement make-pass2 \
  --pass1 data/gold/val.csv --out data/gold/samples/val_pass2_sheet.csv
python -m reflex_sentry.gold.ingest \
  --in data/gold/samples/val_pass2_sheet.csv --split val \
  --labeler-pass 2 --sample data/gold/samples/val_sample.csv \
  --pool data/processed/val_pool.parquet --gold-out data/gold/val_pass2.csv \
  --merged-out data/processed/val_pass2.parquet
python -m reflex_sentry.gold.agreement compare \
  --pass1 data/gold/val.csv --pass2 data/gold/val_pass2.csv \
  --out-dir reports/agreement_val
```

For Label Studio, use `gold.export --label-studio <tasks.json>` instead of
`--csv`; export writes the interface XML from its canonical generator. The
assisted labeling utilities `gold.suggest`, `gold.draft_merge`, and
`tools/review.html` remain optional for future batches. Supply their input
files explicitly and record model drafts separately from owner review:
seeing a proposed label can anchor a reviewer and does not make the review
blind independent annotation.

The expanded ordinary-benign test slice can be regenerated with
`reflex_sentry.data.easy_benign`; its exact one-off teacher scoring procedure
and append order are preserved in
[history/KAGGLE_EASY_BENIGN.md](history/KAGGLE_EASY_BENIGN.md).
Re-ingesting test gold overwrites `test.parquet`, so append that slice again
before refreshing test predictions.

## Train or refresh predictions

Teacher inference and Stage B training use the Kaggle notebooks
`notebooks/01_teacher_scoring.ipynb` and `notebooks/02_stage_b_train.ipynb`
with private data and GPU access. Download teacher scores into `data/interim/`
and the Stage B checkpoint into `models/stage_b/`. To score the train and gold
splits with both teachers locally or on Kaggle, for example:

```bash
python -m reflex_sentry.teacher.score --model llama_guard_3_8b \
  --inputs data/processed/train.parquet data/processed/val.parquet \
    data/processed/test.parquet data/processed/test_ood.parquet \
  --out data/interim/teacher_scores_llama_guard_3_8b.parquet \
  --batch-size 8 --max-length 512 --load-in-4bit
python -m reflex_sentry.teacher.score --model qwen3guard_gen_8b \
  --inputs data/processed/train.parquet data/processed/val.parquet \
    data/processed/test.parquet data/processed/test_ood.parquet \
  --out data/interim/teacher_scores_qwen3guard_gen_8b.parquet \
  --batch-size 8 --max-length 512 --load-in-4bit
python -m reflex_sentry.targets \
  --teachers data/interim/teacher_scores_llama_guard_3_8b.parquet \
    data/interim/teacher_scores_qwen3guard_gen_8b.parquet \
  --pool data/processed/train.parquet \
  --out data/interim/soft_targets.parquet
```

Run GPU scoring in the private Kaggle notebook unless suitable local hardware
is available. The teacher score files must cover each evaluation ID, including
the appended ordinary-benign test slice. For a teacher quality-reference
row, convert the scores for each gold split (this example shows validation):

```bash
python -m reflex_sentry.teacher.as_predictor \
  --scores data/interim/teacher_scores_llama_guard_3_8b.parquet \
    data/interim/teacher_scores_qwen3guard_gen_8b.parquet \
  --eval data/processed/val.parquet --out preds/teacher_both_val.csv
```

Scoring resumes only from a valid file; a nonempty file must identify the
requested teacher.
Input overlaps must have identical text; score-file overlaps passed to
`teacher.merge_scores` must agree. A conflict is an error to investigate,
not an instruction to discard one file. Use a separate output file when
intentionally changing teacher settings. See the
[teacher artifact contract](DATA_CONTRACT.md#teacher-score-artifacts).

With soft targets built, train and score Stage A locally. Create the evasion
split before asking a predictor to score it:

```bash
python -m reflex_sentry.models.stage_a train
python -m reflex_sentry.eval.wrappers \
  --in data/processed/test.parquet --out data/processed/test_evasion.parquet
python -m reflex_sentry.models.stage_a predict \
  --splits val test test_ood test_evasion
```

After downloading Stage B outputs, build the dictionary and run its local pipeline:

```bash
python -m reflex_sentry.models.build_precheck_words \
  --dictionary /usr/share/hunspell/en_US.dic
bash scripts/run_stage_b_local.sh
```

The Stage B script exports fp32 ONNX, sweeps int8 presets on the training
dev fold, runs informational validation parity, predicts available splits,
and generates base and pre-check reports. It requires the saved checkpoint,
train/soft-target data, validation gold, and locally built word list.
The word-list builder writes hashes and source versions; see
[vocabulary setup](../reflex_sentry/models/data/README.md). If only reports
need refreshing, use the existing predictions instead of this script.
The CPU latency protocol and the meaning of pre-check overhead are in
[METHOD.md](METHOD.md).

To refresh fp32 ONNX predictions from an existing graph without repeating
export or the int8 sweep:

```bash
python -m reflex_sentry.models.export_onnx predict --model models/stage_b \
  --onnx models/stage_b/model.onnx --name stage_b_onnx \
  --splits val test test_ood test_evasion
```

## Evaluate cached predictions

The shared runner reads existing logits, recalibrates on `val`, and writes
per-model reports plus ignored `reports/comparison.md` and `.csv`:

```bash
python -m reflex_sentry.eval.run_all --models stage_b_onnx \
  --precheck --config configs/eval.yaml
```

`--precheck` rebuilds `_pc` predictions after calibrating the base model.
`--splits` limits requested output splits, while validation remains the
calibration reference. The runner requires processed gold matching the IDs
and labels in each logits/prediction input, including validation references.
After intentional label changes, rebuild stale prediction metadata from the
authoritative split. To score a standalone probability CSV, provide its
matching validation predictions:

```bash
python -m reflex_sentry.eval.report \
  --preds preds/stage_b_onnx_test.csv \
  --val preds/stage_b_onnx_val.csv \
  --config configs/eval.yaml --out reports/stage_b_onnx_test
```

For a harness-only fixture check, run `bash scripts/smoke_e2e.sh`.
Current saved values are in [STATUS.md](STATUS.md); the generated comparison
file is local and may not exist in a fresh clone.

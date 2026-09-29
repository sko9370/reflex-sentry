# Development plan and decision log

This file tracks decisions made during development, not the project design itself (see `README.md` for that). Entries are dated and append-only; a later entry can supersede an earlier one but should not delete it.

## Decision log

### 2026-09-27: renamed from s1gate to reflex-sentry

The project was scaffolded under the working name "s1gate" and has been renamed to **reflex-sentry**. This covers the repository name, the project name in `pyproject.toml`, and the Python package, which moved from `s1gate_eval/` to `reflex_sentry/eval/`. Module paths are now `python -m reflex_sentry.eval.report`, `python -m reflex_sentry.eval.calibrate`, and `python -m reflex_sentry.eval.wrappers`. "System 1 gate" survives only as a concept description (a fast, System 1 style tier-one gate), never as the project's name.

### 2026-09-27: session scope is milestones 1 and 2 only

This work session covers the data pipeline, gold-set tooling, hard-negative drafts, the keyword-rule baseline, and harness validation (README milestones 1 and 2). It stops there for review before moving into any model stage (teacher labeling, Stage A, Stage B). This is a deliberate checkpoint, not a sign the later milestones are settled: the teacher-labeling design below is recorded now so it is not lost, but implementation starts only after this session's output is reviewed.

### 2026-09-27: hard negatives drafted with AI assistance

The starter set of hard-negative prompts in `seeds/hard_negatives.csv` was drafted with AI assistance rather than written entirely by hand, then committed for human review. This must be disclosed in the writeup: AI-drafted hard negatives may not reflect how a real defender actually phrases a request, and they need the same scrutiny (and likely revision) as any other seed data before they are treated as gold.

### 2026-09-27: teacher labeling design (milestone 3, not yet built)

Teacher labeling will be a configurable scorer supporting two presets:

- **Llama Guard 3 8B** (gated on Hugging Face; requires accepting the Meta license)
- **Qwen3Guard-Gen-8B**

Running both and using the disagreement between them as the `unsure` signal is preferred over a single teacher's confidence, per the soft-target formula in `README.md` section 4.3. This is a design decision for milestone 3 and is not implemented in this session.

### 2026-09-27: OOD holdout is four sources, not one

The first real build showed `toxic_chat` alone yields only 78 in-scope candidates (about 16 unsafe), too few for a meaningful OOD recall. `test_ood` now holds out `toxic_chat`, `aegis2`, `hh_redteam` and `beavertails` together: real users, human red-teamers and two crowd-annotated sets. Together they supplied under 4% of training rows. This departs from the README's "one entire source" wording; the writeup should say so.

### 2026-09-27: gold set sizes 300 val, 300 test, 200 test_ood

Defaults in `reflex_sentry.gold.sample`. Candidate pools shrunk to 400 each so fewer rows are withheld from training. Expect recall confidence intervals of roughly plus or minus 6 points; check the dangerous count per gold set after the rebuild and enlarge if it falls much below 60.

### 2026-09-27: two-tier keyword prefilter

The first real build showed everyday words (vulnerable, breach, exploit, malicious, injection) pulling in non-cyber prompts, mostly from OR-Bench, while casual attack phrasing ("hack into my ex's instagram") was missed. Keyword lines can now be `weak:`; a prompt is in scope with one strong hit or two distinct weak hits. Casual hacking and account-takeover phrasings were added as strong regexes.

### 2026-09-27: gold labels drafted by Opus, reviewed by the owner

The 800 sampled items (300 val, 300 test, 200 test_ood) were labeled in two independent Opus passes over differently ordered batches, merged with `python -m reflex_sentry.gold.draft_merge`, and all 30 pass disagreements were adjudicated. Five of sixteen labeling batches stopped partway when a safety classifier interrupted the labeler; those stops were not retried or worked around. Result: 717 of 800 items have at least one model label; 56 test items (36 or_bench, 20 wildguardmix) are unlabeled and need a human label, and 27 unlabeled hn_seed items were prefilled as benign. Pass A vs B agreement on doubly labeled items: val 0.93 (kappa 0.82), test 0.94 (kappa 0.81, n=55), test_ood 0.91 (kappa 0.73). The owner reviews every priority-1 row before ingest. The writeup must state that gold labels are model drafts with human review, not independent human labels, which weakens the "hand labels are the only independent check on teacher bias" argument.

### 2026-09-28: gold review complete

The owner reviewed all 800 items. Drafted labels (Opus two-pass plus adjudication) were accepted without change: val 300/300, test 278/278, test_ood 200/200. The owner hand-labeled 16 test items in the spreadsheet and reviewed 40 more that had only non-model suggestions (TF-IDF neighbors plus source label), accepting 34 and changing 6. The writeup should report this acceptance rate alongside the caveat that review of a draft is not the same as blind independent labeling (anchoring on the shown label is likely).

### 2026-09-29: easy-benign slice added to the test set

Gold labeling rule 11 tags every in-scope benign prompt as a hard negative (every in-scope benign prompt is security-flavored by construction), so `easy_benign_escalation_rate` had n close to 0 and cascade economics could not reweight benign traffic by `hard_negative_share_of_benign`. We add 300 easy-benign rows to `test.parquet` with `python -m reflex_sentry.data.easy_benign` (`--append-to-test`). Source: ToxicChat non-cyber safe prompts (real user prompts, held out of training entirely), out of scope per the prefilter, 20 to 1500 characters, English-looking, deduplicated. Labels come from ToxicChat's human annotation plus prefilter out-of-scope status, not hand review (the owner may spot-check). Tags stay empty so the harness counts them as easy benign. This is what lets cascade economics reweight benign traffic. Operational notes: re-running `reflex_sentry.gold.ingest --split test` overwrites `test.parquet`, so re-run the easy-benign command with `--append-to-test` afterwards (a `test.parquet.bak` is kept on first append), then regenerate every model's test predictions. The new ids are not in teacher_scores; score them on Kaggle with the existing scorer and merge:

```bash
python -m reflex_sentry.teacher.score --model llama_guard_3_8b \
    --inputs data/processed/easy_benign_for_teachers.parquet \
    --out data/interim/teacher_scores_llama_guard_3_8b_easy_benign.parquet \
    --batch-size 8 --max-length 512 --load-in-4bit
python -m reflex_sentry.teacher.merge_scores \
    data/interim/teacher_scores_llama_guard_3_8b.parquet \
    data/interim/teacher_scores_llama_guard_3_8b_easy_benign.parquet \
    --out data/interim/teacher_scores_llama_guard_3_8b.parquet
```

Repeat with `--model qwen3guard_gen_8b` if that teacher is used. The merge drops duplicate ids (first file wins).

### 2026-09-27: data/ stays out of git

Raw and processed data, gold labels, teacher scores, and model artifacts are never committed. `data/SOURCES.md` is the one exception (license and provenance notes per dataset), and `.gitignore` is written as `data/*` plus `!data/SOURCES.md` so that file can be tracked while everything else under `data/` stays ignored.

### 2026-09-28: int8 config selection moved from val to the training dev fold, matched escalation rate

The real ModernBERT-student run showed every int8 config within +/-0.01 AP of fp32 on val (ranking preserved), but decision agreement at the fp32-selected val threshold was only 0.925 to 0.940 -- below the 0.97 gate -- because int8 shifts the logits and fp32's threshold does not transfer to it. Two problems, not one: first, that criterion is the wrong one, since in deployment each variant (fp32, int8) is calibrated separately (`reflex_sentry.eval.run_all` fits its own temperature and threshold per model on `val`), so comparing decisions at a single shared threshold was never the right test. Second, `export_onnx sweep` was choosing a config using `val` at all, which section 4.1 reserves for temperature scaling and threshold selection only, not model-config selection.

Fix: `export_onnx sweep` now selects on the **training dev fold** -- the same held-out rows `stage_b.make_dev_split` already carves out for checkpoint selection (reusing `metadata.json`'s `seed`/`dev_ratio`), labeled by soft-target argmax (dangerous vs benign; argmax-`unsure` rows dropped, since they are not part of the ranking task). The comparison is **matched escalation rate, label-free**: `t_fp32` is picked on the dev fold for 0.95 recall, and int8 gets its own threshold chosen to reproduce fp32's escalation count at `t_fp32`, not `t_fp32` itself. Pass rule: matched-rate decision agreement >= 0.97 and AP drop <= 0.01; `--choose` keeps the highest-agreement passing config, ties by lower p50 latency. `export_onnx parity` still runs on `val`, but is now explicitly informational only (fp32 and int8 each at their own val-calibrated threshold) and never gates anything -- see README 4.5, `reflex_sentry/models/export_onnx.py` (`dev_fold_reference`, `matched_rate_metrics`, `matched_rate_threshold`), and `scripts/run_stage_b_local.sh`.

---

## Milestone checklist

Status legend: **tooling done** (code/config written, awaiting real data or a run), **awaiting user data** (blocked on something only a person can provide: license acceptance, hand labels, a judgment call), **not started**.

### M1: Data and gold set

- [ ] Dataset download instructions and `configs/data.yaml` (source list, scope filter, split rules): tooling done
- [ ] `python -m reflex_sentry.data.build` scope filter, dedupe, split pipeline: tooling done
- [ ] `data/SOURCES.md` license and filtering notes per dataset: awaiting user data (needs a human read of each dataset's current license/card)
- [ ] `configs/cyber_keywords.txt` scope prefilter: tooling done
- [ ] `configs/keyword_rules.yaml` for the keyword-rule baseline: tooling done
- [ ] `configs/labeling_guide.md`: tooling done as a starting draft; awaiting user data (real judgment calls accumulate only once labeling starts)
- [ ] `configs/label_studio.xml` labeling interface: tooling done
- [ ] Gold sampling and export/ingest tooling (`reflex_sentry.gold.sample`, `.export`, `.ingest`, `.agreement`): tooling done
- [ ] Hand-labeled `val` and `test` sets (300 to 500 each): awaiting user data
- [ ] `seeds/hard_negatives.csv` drafted (AI-assisted): tooling done, awaiting human review
- [ ] Final gold set size decision: awaiting user data (see Open questions)

### M2: Harness validated

- [ ] `pytest` passes on `tests/test_metrics.py`: done
- [ ] `reflex_sentry.baselines.keyword_rules` implemented: tooling done
- [ ] `scripts/smoke_e2e.sh` runs the keyword baseline through `reflex_sentry.eval.report` end to end: tooling done
- [ ] CI (`.github/workflows/ci.yml`) runs tests plus the smoke script on every push and PR: done
- [ ] Keyword-rule baseline scored on a real (not synthetic) gold set: awaiting user data

### M3: Teacher scored

- [ ] Configurable teacher scorer (Llama Guard 3 8B, Qwen3Guard-Gen-8B presets): not started
- [ ] Two-teacher disagreement as the `unsure` signal: not started
- [ ] `data/interim/teacher_scores.parquet` produced and scored with the harness: not started

### M4: Stage A

- [ ] Frozen sentence-embedding baseline plus logistic regression: not started
- [ ] Scored against keyword and teacher baselines: not started

### M5: Stage B

- [ ] Encoder fine-tune on soft targets (Kaggle): not started
- [ ] Temperature scaling, ONNX export, int8 quantization: not started
- [ ] Evasion-wrapper robustness scoring: not started
- [ ] Scored on all four test sets (`test`, `test_ood`, `test_evasion`): not started

### M6: Writeup

- [ ] Comparison table (README section 5.4) filled in: not started
- [ ] Threshold tradeoff chart, error analysis, limitations section: not started
- [ ] AI-assisted hard-negative drafting disclosed: not started (must be included)

---

## Running milestones 3 to 5

### Milestone 3: teacher labels and soft targets

1. **Score a teacher.** Locally (slow) or on Kaggle (`notebooks/01_teacher_scoring.ipynb`,
   see its markdown cells for the two ways to get the repo and data onto Kaggle, and for
   accepting the Llama Guard 3 license and setting `HF_TOKEN` as a Kaggle Secret):

   ```bash
   python -m reflex_sentry.teacher.score --model llama_guard_3_8b \
       --inputs data/processed/train.parquet data/processed/val_pool.parquet \
                data/processed/test_pool.parquet data/processed/test_ood_pool.parquet \
       --out data/interim/teacher_scores_llama_guard_3_8b.parquet \
       --batch-size 8 --max-length 512 --load-in-4bit --limit 20   # smoke test first
   ```

   Drop `--limit 20` for the full run; rerunning the same command resumes (already-scored
   ids are skipped). Repeat with `--model qwen3guard_gen_8b` for the second teacher if you
   want two-teacher disagreement as the `unsure` signal. Download the resulting
   `teacher_scores_*.parquet` files from Kaggle's output tab into `data/interim/`.

2. **Score the teacher itself as the upper-reference row (README 5.4):**

   ```bash
   python -m reflex_sentry.teacher.as_predictor \
       --scores data/interim/teacher_scores_llama_guard_3_8b.parquet \
       --eval data/processed/val.parquet \
       --out preds/teacher_llama_guard_3_8b_val.csv
   python -m reflex_sentry.eval.report --preds preds/teacher_llama_guard_3_8b_val.csv \
       --config configs/eval.yaml --out reports/teacher_llama_guard_3_8b_val
   ```

3. **Build soft targets** for student training. Once at least one teacher run has finished:

   ```bash
   python -m reflex_sentry.targets --teachers data/interim/teacher_scores_*.parquet \
       --pool data/processed/train.parquet --out data/interim/soft_targets.parquet
   ```

   To start Stage A/B training before any teacher run finishes, build source-label-only
   targets instead (rows are marked `target_source = "source_only"` so they can be told
   apart from real teacher-based rows later, and re-running the command above without
   `--source-only` once teacher scores exist overwrites `data/interim/soft_targets.parquet`
   with `target_source = "teacher"` rows):

   ```bash
   python -m reflex_sentry.targets --source-only \
       --pool data/processed/train.parquet --out data/interim/soft_targets.parquet
   ```

`pip install -e ".[teacher]"` installs `transformers`, `accelerate`, and `bitsandbytes`
locally if you want to run a teacher off Kaggle (slow on CPU; see README section 3).
The unit tests (`tests/test_teacher.py`, `tests/test_targets.py`) mock the model/tokenizer
entirely, so `pytest` passes without `torch`/`transformers`/`bitsandbytes` installed and
without any Hugging Face network access.

**To verify on Kaggle, since this was built without GPU access:** run the notebook's smoke
test (`--limit 20`) for each preset first and read the printed verdict-token ids and the one
raw generation for a harmless prompt -- confirm they look sane (a real `safe`/`unsafe` split
for Llama Guard, and a real `Safety: ...` / `Categories: ...` continuation for Qwen3Guard-Gen)
before starting a full run. If either model's answer format differs from what's assumed here,
fix the corresponding preset's strings in `reflex_sentry/teacher/score.py::PRESETS` (they are
intentionally isolated there) rather than the scoring logic.

### Milestone 4: Stage A baseline student

Frozen sentence embeddings plus a multinomial logistic regression head (README 4.4), the
number every later model has to beat. Everything lives under `reflex_sentry/models/stage_a.py`.

```bash
pip install -e ".[stage_a]"   # sentence-transformers, imported lazily

python -m reflex_sentry.models.stage_a train
python -m reflex_sentry.models.stage_a predict --splits val test test_ood test_evasion

python -m reflex_sentry.eval.run_all --models keyword stage_a --config configs/eval.yaml
```

Or, end to end: `bash scripts/run_stage_a.sh`.

- **Targets.** Trains on the argmax of `data/interim/soft_targets.parquet` with
  `sample_weight` set to the max target probability. If soft targets are not built yet, it
  falls back to the training pool's `source_label` (1 -> dangerous, 0 -> safe, NaN rows
  dropped) and records which mode was used as `target_mode` in `models/stage_a/metadata.json`,
  so a Stage A run is never blocked on milestone 3 finishing first.
- **Embeddings.** Default model `BAAI/bge-small-en-v1.5`, batched, cached to
  `models/stage_a/emb_cache/*.npy` keyed by a hash of the embedding model name plus the exact
  ordered id sequence being embedded, so re-running the same split is a cache hit.
  `sentence-transformers` is imported lazily (only inside the embedding call) since it may not
  be installed and Hugging Face downloads are blocked in some environments; every embedding
  call also accepts an `embedder` override for tests.
- **Model selection.** `C` is chosen from a small grid by validation log-loss when
  `data/processed/val.parquet` exists (it does starting at milestone 1); otherwise a fixed
  middle value is used.
- **Latency.** `latency_ms` is measured as real single-prompt embed+classify wall time
  (bypassing the embedding cache) for up to 200 prompts sampled uniformly per split; each
  sampled row keeps its own measured time, and every other row is filled with the sample
  mean, since timing every row of a large split individually is not worth the wall time. This
  is documented again in `stage_a.py`'s `_measure_latency` docstring.
- **Evaluation glue.** `reflex_sentry.eval.run_all` scores any number of named models in one
  command: it builds `test_evasion.parquet` if missing, fits temperature scaling on any
  student's val logits and applies it to that student's other splits, runs the keyword
  baseline directly (it has no separate predict step), calls `reflex_sentry.eval.report.run`
  for every split with both a gold file and a prediction file, and writes
  `reports/comparison.md`/`.csv` (README 5.4) by reading each split's `metrics.json`.
- **Tests.** `tests/test_stage_a.py` and `tests/test_run_all.py` inject a deterministic
  hashing-of-tokens fake embedder (32 dims), so they never import `sentence-transformers` or
  touch the network, and never open anything under `data/`; they build tiny synthetic pools
  and gold files instead.

### Milestone 5: Stage B fine-tuned encoder student, export, CPU inference

Fine-tuned encoder (README 4.4/4.5): `answerdotai/ModernBERT-base` (alt `microsoft/deberta-v3-small`),
mean pooling over the attention mask, dropout, `Linear(hidden, 3)` head, trained with a plain
PyTorch loop (no Trainer) against KL-divergence soft targets. Lives under
`reflex_sentry/models/stage_b.py` (train/predict) and `reflex_sentry/models/export_onnx.py`
(ONNX export, int8 quantization, parity check, onnxruntime CPU predictor). Training runs on
Kaggle GPU (`notebooks/02_stage_b_train.ipynb`); export/quantize/predict run locally on CPU
(`scripts/run_stage_b_local.sh`), since the model actually deployed is the CPU one.

```bash
pip install -e ".[stage_b]"   # torch, transformers>=5.0, onnx, onnxruntime

# On Kaggle (see notebooks/02_stage_b_train.ipynb):
python -m reflex_sentry.models.stage_b train --base answerdotai/ModernBERT-base --out models/stage_b
python -m reflex_sentry.models.stage_b predict --model models/stage_b --splits val test test_ood test_evasion

# Locally, after downloading models/stage_b from Kaggle:
bash scripts/run_stage_b_local.sh
```

- **Checkpoint selection never touches gold labels.** README reserves `val` for temperature
  scaling and threshold selection only, so it must not decide which training epoch is kept.
  `train()` instead splits the merged training pool itself into train/dev (`dev_ratio`,
  default 10%), stratified by soft-target argmax with a fixed seed and grouped by `dup_group`
  when that column is present (so near-duplicate rows never split across the two sides,
  mirroring `reflex_sentry/data/split.py`). The checkpoint with the lowest dev KL/log-loss
  (against the dev rows' own soft targets, the same loss used for training) is kept.
  `val_parquet` is optional and purely informational when given: its NLL-vs-gold and
  recall-at-a-val-selected-threshold are logged and recorded in `metadata.json` per epoch, but
  never influence training or selection -- `tests/test_stage_b.py::test_training_selects_identically_without_val`
  runs the same training twice, with and without `val.parquet` present, and asserts the
  selected epoch and the saved head weights are bit-identical.
- **Loss.** KL(soft target || model), batchmean, in the fixed `[safe, dangerous, unsure]`
  order used everywhere downstream (`t_safe`/`t_dangerous`/`t_unsure` in
  `data/interim/soft_targets.parquet`; `logit_safe`/`logit_dangerous`/`logit_unsure` in the
  logits CSVs; matches `reflex_sentry.eval.calibrate.GOLD_TO_CLASS`).
- **Outputs.** `models/stage_b/`: HF `save_pretrained` encoder + tokenizer, `head.pt` (pooling
  head state dict), `metadata.json` (base model, param count, max_len, epochs, lr, seed, plus
  every other hyperparameter, dev/val metrics per epoch, and the selected epoch).
  `preds/stage_b_{split}_logits.csv` for fp32, `preds/stage_b_int8_{split}_logits.csv` for the
  quantized model, both consumed by `reflex_sentry.eval.calibrate`/`run_all`.
- **Latency.** `latency_ms` is real per-prompt wall time at batch size 1 on CPU, measured for a
  random sample (`--latency-sample`, default 200) of each split rather than every row; the rest
  get `NaN`, and `metrics.latency()` already drops `NaN` before computing p50/p95/p99, so
  percentiles reflect the sampled subset. The int8 onnxruntime predictor's
  `intra_op_num_threads` defaults to 1 to mimic a single-core gate deployment.
- **ONNX export.** `torch.onnx.export` with dynamic batch and sequence axes, opset 17, a single
  `logits` output (the 3-way head). Torch >= 2.6 defaults to the dynamo-based exporter, which
  needs the optional `onnxscript` package and a different (`dynamic_shapes`) API; `export_onnx.py`
  passes `dynamo=False` to force the legacy TorchScript-based exporter instead, which takes
  `dynamic_axes` directly with no extra dependency.
- **Parity check.** Compares fp32 torch vs fp32 onnx vs int8 onnx logits on `val`: max abs
  logit diff and argmax agreement for each pair, plus dangerous recall at the fp32-selected
  threshold for all three. If int8 recall differs from fp32 recall by more than 0.02 at that
  threshold, it prints a loud banner and raises a `UserWarning` (README 4.5: confirm the
  quantized model matches fp32 within noise before running any test set) -- it does not raise
  a hard error, since a human still has to decide whether to proceed or re-quantize/re-train.
- **Tests.** `tests/test_stage_b.py` builds a tiny, randomly initialized 2-layer/hidden-32
  `BertConfig` encoder plus a `PreTrainedTokenizerFast` wrapping a `WordLevel` tokenizer trained
  locally on a handful of sentences (no network, no download), and covers: forward shapes; mean
  pooling correctly ignoring padding; KL loss decreasing over a few optimizer steps; the
  train/predict CLI end to end on a synthetic pool, with the written logits CSV validated
  against `reflex_sentry.eval.calibrate`/`metrics`; the dev-split grouping-by-`dup_group`
  invariant; identical checkpoint selection with and without `val.parquet`; and ONNX
  export/int8/parity (skipped cleanly via `pytest.importorskip` when `onnx`/`onnxruntime` are
  not installed). torch/transformers/onnx/onnxruntime are all imported lazily, so every
  module here stays importable without them.

## Open questions

- **OOD holdout source.** `toxic_chat` is the default, but this should be revisited once per-source in-scope counts are known after the M1 scope filter runs; a source with too few in-scope prompts is a poor holdout.
- **BeaverTails prompt labels.** BeaverTails' safety labels are attached to QA pairs (prompt + response), not the prompt alone. Are prompt-only labels derived by pooling or ignoring the response trustworthy enough to use, or does BeaverTails need re-labeling or exclusion from prompt-only training data?
- **Gold set size.** 300 to 500 per split (per README) is a starting target; whether that is enough to keep confidence intervals usably tight on rare categories (e.g. per-`cat:*` recall) depends on how thin the cyber-scope pool actually is once filtered.
- **Gated dataset access.** WildGuardMix and Aegis 2.0 are gated on Hugging Face; Llama Guard 3 8B likewise requires accepting Meta's license. These need a human to accept terms before the pipeline can download or run them.
- **Hard-negative realism.** AI-assisted drafts in `seeds/hard_negatives.csv` need human review before being treated as gold; how much revision they need is not yet known.

### 2026-09-28: finish Stage B pre-check continuation

Reviewed the imported Claude conversation and preserved its unfinished detector
implementation and thresholds. Added explicit `run_all --precheck` integration,
so calibration precedes pre-check generation and both base and `_pc` variants
appear in the comparison. The local script now passes data/output directories
and fp32 CPU latency settings through consistently.
The script checks dictionary setup before expensive export/sweep work.
Explicit split selection removes stale reports for omitted/unavailable splits
of requested models, while refreshing val predictions for threshold selection.

Correctness fixes: flagged rows use `[0, 0, 1]` rather than `[0.02, 0.08, 0.90]`,
which could fail to escalate at thresholds below 0.02. Missing base timings stay
missing, including when the entire latency column is absent. Prediction ids must
match the processed split exactly before applying the pre-check. The English
word list loads lazily, fails explicitly if missing/empty, and can be generated
from a local Hunspell dictionary with recorded hashes. The generated third-party
list is ignored by Git; its builder reproduces the existing 36,432-word list
byte for byte on this machine. No detector retuning or training took place.

Re-evaluation from existing logits: fp32 dangerous evasion recall rises from
71.81% to 96.81%, with test recall unchanged at 97.87%, benign escalation at
7.66%, and OOD recall at 85.12%. Int8 evasion rises from 72.87% to 95.74%, but
its test recall (93.62%) and OOD recall (76.86%) still trail fp32. These are
post-hoc evasion refinements, not independent holdout confirmation. The existing
ordinary-benign audit flags 16/2000 rows; the heuristic Spanish/Portuguese bucket
is 6/33 and remains a limitation. The pre-check does not resolve distribution
shift, and its combined timing uses cached base samples plus new pre-check time.
Teacher scoring for the separate easy-benign slice remains outstanding.

### 2026-09-28: measure fp32 ONNX directly and prepare Kaggle input

Added `export_onnx predict --onnx ... --name stage_b_onnx`, retaining the
existing `--int8` path and separate output filenames. Both variants now use
shared warmed-up batch-one timing, including tokenization, forward inference,
and softmax; the previous ONNX timer measured forward inference alone despite
the documented end-to-end protocol. Latency summaries now have sidecars for
both ONNX variants. The full local script includes fp32 ONNX and its pre-check
variant alongside PyTorch and int8.

Ran the existing fp32 ONNX graph on all four splits locally. Maximum absolute
logit error versus saved PyTorch logits was 0.000029 across those splits.
Reported accuracy is unchanged, including test recall 97.87%, benign escalation
7.66%, OOD recall 85.12%, and evasion recall 71.81% without / 96.81% with the
pre-check. Test median latency measured 62.1 ms versus the existing PyTorch
measurement of 92.9 ms; these are separate runs on the same local machine,
not a simultaneous controlled benchmark. No new quantization selection or
training was performed.
Refreshed int8 prediction/timing on the existing graph as well: test median
latency is 22.9 ms under the corrected end-to-end timing protocol.

Generated the documented 300-row easy-benign slice and teacher input locally,
without appending to test yet. Both teachers currently cover 0/300 new ids;
CUDA is unavailable here. `docs/KAGGLE_EASY_BENIGN.md` contains the next user
step: score those inputs with both teachers on Kaggle and return the two
parquets. Then merge, append the saved slice once, and regenerate comparable
test results for every model. Existing test remains 286 rows in the meantime.

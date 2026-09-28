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

### 2026-09-27: data/ stays out of git

Raw and processed data, gold labels, teacher scores, and model artifacts are never committed. `data/SOURCES.md` is the one exception (license and provenance notes per dataset), and `.gitignore` is written as `data/*` plus `!data/SOURCES.md` so that file can be tracked while everything else under `data/` stays ignored.

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

## Open questions

- **OOD holdout source.** `toxic_chat` is the default, but this should be revisited once per-source in-scope counts are known after the M1 scope filter runs; a source with too few in-scope prompts is a poor holdout.
- **BeaverTails prompt labels.** BeaverTails' safety labels are attached to QA pairs (prompt + response), not the prompt alone. Are prompt-only labels derived by pooling or ignoring the response trustworthy enough to use, or does BeaverTails need re-labeling or exclusion from prompt-only training data?
- **Gold set size.** 300 to 500 per split (per README) is a starting target; whether that is enough to keep confidence intervals usably tight on rare categories (e.g. per-`cat:*` recall) depends on how thin the cyber-scope pool actually is once filtered.
- **Gated dataset access.** WildGuardMix and Aegis 2.0 are gated on Hugging Face; Llama Guard 3 8B likewise requires accepting Meta's license. These need a human to accept terms before the pipeline can download or run them.
- **Hard-negative realism.** AI-assisted drafts in `seeds/hard_negatives.csv` need human review before being treated as gold; how much revision they need is not yet known.

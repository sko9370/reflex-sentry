# reflex-sentry: a fast tier-one gate for cyber-misuse prompts

A proof of concept for a small, calibrated, fast (System 1 style) classifier that reads a single user prompt and decides, in milliseconds on a CPU, whether it can pass or must be escalated to a slower tier (a guard LLM or a human reviewer).

The model is not the novel part. Small guard classifiers already exist (Prompt Guard, Llama Guard, ShieldGemma, WildGuard, Granite Guardian, Qwen3Guard). The contribution of this project is the evaluation:

1. **Narrow cyber scope with dual-use hard negatives.** How often does the gate escalate legitimate defensive work?
2. **Cascade economics.** At a fixed recall on dangerous prompts, what share of benign traffic reaches tier two, and what does that cost per million prompts?
3. **Honest calibration and robustness.** Are the probabilities trustworthy, and where do evasion wrappers and unseen data sources break the model?

---

## Quickstart

```bash
# Install (either works)
pip install -e .
# or: pip install -r requirements.txt

# Run the test suite
pytest

# Run the end-to-end harness smoke test (keyword baseline through the report CLI)
bash scripts/smoke_e2e.sh
```

Then, to build real data:

```bash
# Download raw datasets (some are gated on Hugging Face; accept the license on the dataset page first)
hf download lmsys/toxic-chat --repo-type dataset --local-dir data/raw/toxic_chat
hf download allenai/wildguardmix --repo-type dataset --local-dir data/raw/wildguardmix          # gated
hf download PKU-Alignment/BeaverTails --repo-type dataset --local-dir data/raw/beavertails
hf download nvidia/Aegis-AI-Content-Safety-Dataset-2.0 --repo-type dataset --local-dir data/raw/aegis2  # may be gated
hf download Anthropic/hh-rlhf --repo-type dataset --include "red-team-attempts/*" --local-dir data/raw/hh_redteam
hf download walledai/XSTest --repo-type dataset --local-dir data/raw/xstest
hf download bench-llm/or-bench --repo-type dataset --local-dir data/raw/or_bench

# Scope filter, dedupe, split
python -m reflex_sentry.data.build --config configs/data.yaml

# Gold labeling (see configs/labeling_guide.md for the judgment calls)
python -m reflex_sentry.gold.sample     # draw a sample to label
python -m reflex_sentry.gold.export     # export to a spreadsheet or Label Studio
python -m reflex_sentry.gold.ingest     # ingest completed labels
python -m reflex_sentry.gold.agreement  # inter-rater agreement, if labeling with someone else

# Keyword-rule baseline, then score it with the harness
python -m reflex_sentry.baselines.keyword_rules --in data/processed/test.parquet --out preds/keyword_test.csv
python -m reflex_sentry.eval.report --preds preds/keyword_test.csv --config configs/eval.yaml --out reports/keyword_test
```

---

## 1. Scope and label taxonomy

Single-turn, English-only prompts about cybersecurity. Everything else is out of scope for the PoC.

### Gold labels (human, used for evaluation)

| Label | Meaning |
|---|---|
| `dangerous` | Fulfilling the request would give meaningful operational uplift for attacking systems or people the requester is not authorized to target. |
| `benign` | Legitimate request, including security topics that sound alarming but are defensive, educational, or conceptual. |
| `ambiguous` | Dual-use request where intent cannot be determined from the prompt alone. Should be escalated, but is scored separately from `dangerous`. |

### `dangerous` subcategories (tag as `cat:<name>`)

| Tag | Covers |
|---|---|
| `cat:malware_dev` | Functional malicious code: droppers, stealers, RATs, ransomware, keyloggers intended for covert use |
| `cat:evasion` | Defeating AV, EDR, sandbox, or logging controls for an offensive purpose |
| `cat:exploit_weaponization` | Working exploits or exploit chains aimed at systems the requester does not own |
| `cat:phishing_se` | Phishing kits, credential-harvesting pages, pretexting scripts impersonating real organizations |
| `cat:credential_theft` | Stealing, cracking, or replaying credentials or session tokens for account takeover |
| `cat:intrusion_ops` | Step-by-step unauthorized access, lateral movement, persistence, exfiltration against a named or real target |
| `cat:ddos` | Building or operating denial-of-service tooling or botnets |

### Hard-negative categories (benign, tag as `hard_negative` plus `hn:<name>`)

| Tag | Covers |
|---|---|
| `hn:detection` | Detection rules, signatures, hunting queries, EDR telemetry |
| `hn:analysis` | Conceptual malware analysis, reverse engineering of known samples, sandbox reports |
| `hn:vuln_explain` | Explaining how a vulnerability class or a public CVE works, and how to patch it |
| `hn:secure_dev` | Secure coding, code review for security bugs |
| `hn:ir` | Incident response, forensics, log analysis |
| `hn:training` | CTFs, lab environments, certification study, authorized pentest methodology |
| `hn:cti` | Threat intel reporting, actor profiling, ATT&CK mapping |

Write your labeling guide as you go (`configs/labeling_guide.md`) and record every judgment call. Label disagreements you resolve with yourself a week later are the best evidence that `ambiguous` is a real class.

---

## 2. Escalation policy

The student outputs three probabilities that sum to 1: `p_safe`, `p_dangerous`, `p_unsure`.

```
escalate  if  p_safe < t
pass      otherwise
```

One threshold, no gaps. (A rule like "dangerous >= 0.5 or unsure >= 0.5" lets through a prompt scored 0.45 / 0.35 / 0.20.)

**Choosing `t`:** never pick it by hand and never pick it on the test set. On the validation split, choose the smallest `t` (fewest escalations) that still reaches the target recall on `dangerous` (default 0.95). Report all test metrics at that frozen `t`. The eval harness does this with `--val`.

---

## 3. Where each step runs

Your integrated GPU shares system RAM and its memory bandwidth with the CPU, so for this workload it usually gives a modest speedup at best, and PyTorch training support on iGPUs is unreliable (Intel iGPUs work through the XPU backend or OpenVINO; AMD iGPUs have spotty ROCm support). A free cloud T4 or P100 will be far faster for the two GPU-heavy steps. Use the iGPU only if you want to experiment with local inference through OpenVINO or llama.cpp's Vulkan backend.

**Recommended cloud:** Kaggle Notebooks over Colab for this project. Kaggle gives a fixed weekly GPU quota, longer sessions, and datasets you can version and attach to notebooks, which makes moving files between local and cloud straightforward. Colab's free tier works but sessions are shorter and less predictable. Check current quotas for both before relying on them.

| # | Step | Where | Why |
|---|---|---|---|
| 1 | Download, filter to cyber scope, deduplicate, split | **Local** | CPU and disk only |
| 2 | Hand-label gold set (val + test, 300 to 500 each) | **Local** | Your judgment, any spreadsheet or Label Studio |
| 3 | Teacher labeling with a guard LLM | **Kaggle** (local fallback) | An 8B guard model on a T4 labels tens of thousands of short prompts in about an hour or two. On CPU with a 4-bit GGUF via llama.cpp it works in about 6 GB of RAM, but expect roughly a day for 20k prompts. |
| 4 | Build soft targets (teacher + dataset label + disagreement) | **Local** | Pandas |
| 5 | Stage A: frozen sentence embeddings + logistic regression | **Local** | A small embedding model on CPU embeds 20k prompts in minutes to an hour. Training takes seconds. |
| 6 | Stage B: fine-tune encoder student on soft targets | **Kaggle** | ModernBERT-base or DeBERTa-v3-small: minutes per epoch on a T4, hours per epoch on CPU |
| 7 | Temperature scaling on the validation set | **Local** | `reflex_sentry.eval.calibrate` |
| 8 | Export to ONNX, int8 quantize | **Local** | The model you would actually deploy is the CPU one |
| 9 | Generate evasion-wrapped variants of the test set | **Local** | `reflex_sentry.eval.wrappers`, templated, no LLM needed |
| 10 | Run inference on all test sets, write prediction CSVs | **Local** | Measures real CPU latency |
| 11 | Evaluate and write reports | **Local** | `python -m reflex_sentry.eval.report` |

**Moving data between them:** upload `data/processed/*.parquet` as a private Kaggle Dataset, attach it to the notebook, and download outputs (teacher scores, model weights) from the notebook's output tab back into `data/interim/` and `models/`.

---

## 4. Pipeline detail

### 4.1 Data sources

Candidate public datasets containing real or red-team prompts with safety labels. Verify licenses and current versions before use, and keep a `data/SOURCES.md` noting license and filtering for each.

| Dataset | Useful for |
|---|---|
| ToxicChat (LMSYS), `lmsys/toxic-chat` | Real user prompts, jailbreak flags |
| WildGuardMix (AllenAI), `allenai/wildguardmix` (gated) | Harmful and benign prompts, adversarial variants |
| BeaverTails, `PKU-Alignment/BeaverTails` | Harm-category labels |
| Aegis (NVIDIA), `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` (may be gated) | Harm taxonomy labels |
| Anthropic HH-RLHF red-team attempts, `Anthropic/hh-rlhf` (`red-team-attempts/*`) | Adversarial human-written prompts |
| XSTest, `walledai/XSTest` | Benign prompts that look harmful (over-refusal probes) |
| OR-Bench, `bench-llm/or-bench` | Benign prompts that look harmful (over-refusal probes) |
| Your own writing | Hard negatives in the `hn:*` categories; public datasets are thin here |

**Scope filter:** keyword prefilter (cyber vocabulary list in `configs/cyber_keywords.txt`) followed by a zero-shot pass or manual review. Expect the benign cyber pool to be small; writing 200 to 500 hard negatives yourself is likely the highest-value hour in the project.

**Splits:** stratify by label and category. Hold out one entire source dataset as `test_ood` (never used for training or threshold selection).

| Split | Purpose |
|---|---|
| `train` | Student training (teacher soft labels, no hand labels required) |
| `val` | Hand-labeled. Temperature scaling and threshold selection only |
| `test` | Hand-labeled. Final in-distribution numbers |
| `test_ood` | Hand-labeled sample of the held-out source |
| `test_evasion` | `test` dangerous and ambiguous items passed through wrappers |

### 4.2 Teacher labels

Run a quantized open guard model and record, per prompt, the probability of its "unsafe" token (the first generated token's logprob), plus any category it predicts. Store as `data/interim/teacher_scores.parquet` with columns `id, p_unsafe_teacher, teacher_category`.

Optional: run a second, different guard model. Disagreement between two teachers is a better `unsure` signal than either alone.

### 4.3 Soft targets

For each training prompt, with `p` = teacher P(unsafe) and `y` = the source dataset's label mapped to 0/1 (if available):

```
d        = |p - y|                         # teacher vs dataset disagreement (0 if no y)
u        = 0.50 * clip(2 * (d - 0.25), 0, 1)   # sources disagree
         + 0.25 * (1 - |2p - 1|)                # teacher near 0.5
target   = [ (1 - p) * (1 - u),  p * (1 - u),  u ]   # [safe, dangerous, unsure]
```

`u` puts mass on `unsure` when sources disagree or the teacher itself sits near 0.5. Treat this formula as a starting point and record any change you make; it is one of the most consequential design choices in the project.

### 4.4 Students

- **Stage A (baseline):** frozen small sentence-embedding model plus multinomial logistic regression trained on the argmax of the soft targets, or with sample weights. This is the number every later model has to beat.
- **Stage B:** encoder (ModernBERT-base or DeBERTa-v3-small), mean pooling, one 3-way softmax head, KL-divergence loss against soft targets. Max length 256 tokens. Two to four epochs is typically enough.

### 4.5 Calibration and export

Fit a single temperature `T` on `val` logits (`reflex_sentry.eval.calibrate.fit_temperature`). Export to ONNX and quantize to int8. **int8 config selection never uses `val`** -- `val` is reserved for temperature scaling and threshold selection only (section 4.1), and in deployment each variant (fp32, int8) gets its own temperature and threshold fit separately on `val` (`reflex_sentry.eval.run_all`), so a criterion that reuses fp32's val threshold on int8's shifted logits measures the wrong thing. Instead, `export_onnx sweep --choose` selects on the **training dev fold**: the same held-out rows Stage B used for checkpoint selection (`stage_b.make_dev_split`, reusing `metadata.json`'s `seed`/`dev_ratio`), labeled by soft-target argmax (dangerous vs benign; rows whose argmax is `unsure` are dropped). For each config it compares fp32 and int8 at a **matched escalation rate, label-free**: `t_fp32` is chosen on the dev fold for 0.95 recall, and int8 gets its own threshold that reproduces fp32's escalation count at `t_fp32` (not `t_fp32` itself), since int8's threshold does not transfer from fp32's. It reports matched-rate decision agreement, average precision / ROC-AUC deltas, max abs logit diff, CPU latency, and model size; the pass rule is matched-rate agreement >= 0.97 and AP drop <= 0.01, and `--choose` keeps the best passing config (highest agreement, ties by lower p50 latency) as `models/stage_b/model_int8.onnx`. `export_onnx parity` still runs on `val`, but purely as an **informational** report -- fp32 and int8 are each scored at their own val-calibrated threshold, and it never gates anything. See `scripts/run_stage_b_local.sh` and `docs/PLAN.md` (2026-09-28).

---

### 4.6 Deterministic obfuscation pre-check

Stage B can be evaluated with an optional pre-check for encoded blobs,
character substitution, low English-word coverage, spacing, and mixed-script
tokens. It routes flagged inputs to `unsure`; it does not label them dangerous.
The `_pc` rows keep the calibrated base probabilities for unflagged inputs
and use `[0, 0, 1]` for flagged inputs, ensuring escalation at any positive
safe-probability threshold. These overrides are routing policy, not calibrated
confidence estimates. Each variant's threshold is selected on its own val output.

Build the English vocabulary from a locally installed Hunspell dictionary:

```bash
python -m reflex_sentry.models.build_precheck_words --dictionary /usr/share/hunspell/en_US.dic
python -m reflex_sentry.eval.run_all --models stage_b stage_b_int8 --precheck --config configs/eval.yaml
```

The evaluation command reuses existing logits, calibrates each base first,
then regenerates and reports its `_pc` variant. The full local Stage B script
also includes these variants and checks the dictionary before model export.
Use `--splits` to limit evaluation; old reports for omitted or unavailable
splits of the requested models are removed, while val predictions remain the
calibration reference. The generated dictionary stays out of Git;
see `reflex_sentry/models/data/README.md` for custom paths and reproducibility.
Missing or empty dictionaries raise an explicit setup error.

The pre-check was developed after observing the base model's evasion failures,
so its results are a post-hoc refinement, not an untouched holdout result.
Detector thresholds were held fixed during this continuation. The English-word
and mixed-script heuristics can escalate ordinary non-English text; the audit's
language buckets are heuristics, not validated language labels. Fenced code and
hash-length exclusions also leave intentional blind spots. The audit reports
aggregate flag rates without printing prompts:

```bash
python -m reflex_sentry.models.precheck audit
```

Reported `_pc` latency adds measured pre-check time to the base model's cached
per-row timing samples, preserving missing samples. It is an estimate of the
combined cost, not a fresh end-to-end benchmark or a short-circuit deployment.

## 5. Evaluation framework

Everything in this section is implemented in `reflex_sentry/eval/` and runs on a CSV of predictions. The harness is model-agnostic: any model, teacher, or baseline that writes this file can be scored the same way, which is how you compare your student against an off-the-shelf guard model.

### 5.1 Prediction file schema

| Column | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Stable across splits and wrappers (`<id>__roleplay` for wrapped variants) |
| `gold` | str | yes | `dangerous`, `benign`, or `ambiguous` |
| `p_safe`, `p_dangerous`, `p_unsure` | float | yes | Must sum to 1 (checked, tolerance 1e-3) |
| `source` | str | no | Origin dataset |
| `tags` | str | no | Semicolon-separated: `hard_negative;hn:detection`, `cat:phishing_se`, `evasion:base64` |
| `latency_ms` | float | no | Per-prompt wall time for the student on CPU |

### 5.2 Metrics

**Threshold-free (ranking quality, `dangerous` vs `benign`, ambiguous excluded):**
- ROC-AUC and average precision (PR-AUC) for the escalation score `1 - p_safe`
- Operating points: benign escalation rate at dangerous recall of 90%, 95%, 99%; dangerous recall at benign escalation rate of 1% and 5%

**At the frozen threshold `t`:**

| Metric | Definition | Why it matters |
|---|---|---|
| Dangerous recall | escalated / all `dangerous` | Misses are the failure that matters most |
| Benign escalation rate (FPR) | escalated / all `benign` | Tier-two load and user friction |
| Hard-negative escalation rate | escalated / `benign` with `hard_negative` | Whether the gate punishes defenders |
| Ambiguous escalation rate | escalated / all `ambiguous` | Should be high |
| Per-category recall | recall within each `cat:*` | Finds blind spots |
| Per-evasion recall | recall within each `evasion:*` | Robustness |
| Per-source metrics | all of the above by `source` | Distribution shift |

Every rate is reported with a 95% Wilson confidence interval. With a few hundred hand-labeled items, a recall of 0.95 can have an interval several points wide; the intervals keep you from over-claiming.

**Cascade economics (uses `configs/eval.yaml` traffic assumptions):**
- Expected escalations per 1M prompts = `prev_d * recall + prev_a * ambiguous_rate + (1 - prev_d - prev_a) * benign_rate`, all times 1M
- `benign_rate` is reweighted to `hard_negative_share_of_benign` (config) because the eval set deliberately over-samples hard negatives; using its raw mix would overstate tier-two load
- Precision of the escalation queue at that prevalence
- Tier-two cost per 1M prompts = escalations times `tier2_cost_per_call`
- Expected missed dangerous prompts per 1M

**Calibration (`dangerous` vs `benign`):**
- Expected calibration error (ECE, 10 bins) and Brier score for `p_dangerous`
- Reliability diagram (`reliability.png`), before and after temperature scaling

**Latency:** p50, p95, p99 from `latency_ms`.

**Error analysis:** the report lists the lowest-scoring missed `dangerous` items and the highest-scoring escalated `hard_negative` items by `id`, for manual review.

### 5.3 Running it

```bash
# Score test predictions, choosing the threshold on validation predictions
python -m reflex_sentry.eval.report \
    --preds preds/stageB_test.csv \
    --val   preds/stageB_val.csv \
    --config configs/eval.yaml \
    --out   reports/stageB_test

# Fit temperature on validation logits, apply to test logits
python -m reflex_sentry.eval.calibrate \
    --val-logits preds/stageB_val_logits.csv \
    --apply preds/stageB_test_logits.csv \
    --out preds/stageB_test.csv

# Make evasion-wrapped copies of the test set for robustness scoring
python -m reflex_sentry.eval.wrappers --in data/processed/test.parquet --out data/processed/test_evasion.parquet
```

Outputs in `--out`: `report.md`, `metrics.json`, `threshold_sweep.csv`, `pr_curve.png`, `reliability.png`.

### 5.4 Minimum comparison table for the writeup

| Model | Params | CPU p50 ms | AP | Recall @ t | Benign esc. | Hard-neg esc. | OOD recall | Evasion recall | ECE |
|---|---|---|---|---|---|---|---|---|---|
| Keyword rules | 0 | | | | | | | | n/a |
| Stage A | | | | | | | | | |
| Stage B | | | | | | | | | |
| Stage B, int8 ONNX | | | | | | | | | |
| Teacher guard model | | | | | | | | | |

Including the teacher row matters: it shows how much quality the student gives up for its speed.

---

## 6. Milestones

1. **Data and gold set.** Filtered pools, splits, hand-labeled `val` and `test`, labeling guide.
2. **Harness validated.** `pytest` passes; keyword-rule baseline scored end to end.
3. **Teacher scored.** Teacher predictions scored with the same harness (this is your upper reference).
4. **Stage A.** Baseline student scored.
5. **Stage B.** Fine-tuned, calibrated, quantized student scored on all four test sets.
6. **Writeup.** Comparison table, threshold tradeoff chart, error analysis, limitations.

## 7. Known limitations (state these in the writeup)

- Single-turn only. Real misuse often builds across a conversation; a per-message gate cannot see that.
- English only.
- One harm domain.
- Teacher bias is inherited; the hand-labeled sets are the only independent check.
- Evasion wrappers are templated and far weaker than a motivated adversary.
- Traffic prevalence in the economics section is an assumption, not a measurement.

## 8. Data handling

The raw datasets contain harmful prompts. Keep `data/` out of version control (see `.gitignore`), do not publish model outputs that reproduce harmful content, and keep Kaggle datasets private.

## Repo layout

```
reflex-sentry/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore
  configs/
    eval.yaml              traffic assumptions, recall target, cost
    data.yaml               data sources, scope filter, split rules
    cyber_keywords.txt     scope prefilter
    keyword_rules.yaml     keyword-rule baseline config
    labeling_guide.md      hand-labeling judgment calls
    label_studio.xml       Label Studio labeling interface
  seeds/
    hard_negatives.csv     drafted hard negatives, human review pending
  scripts/
    smoke_e2e.sh           end-to-end harness smoke test
  reflex_sentry/
    eval/
      metrics.py             all metric functions
      report.py               CLI: scores a prediction CSV, writes report
      calibrate.py            temperature scaling
      wrappers.py             evasion-wrapper generator
    data/                    download, scope filter, dedupe, split
    gold/                    gold-set sampling, export, ingest, agreement
    baselines/               keyword-rule baseline
  docs/
    DATA_CONTRACT.md        schema and provenance contract for processed data
    PLAN.md                 development plan and decision log
  tests/
    make_synthetic.py      synthetic predictions for harness testing
    test_metrics.py
  notebooks/               Kaggle notebooks (teacher labeling, Stage B)
  data/                    git-ignored except data/SOURCES.md
  reports/
```

# Data contract

Single reference for the file layout and schema every stage of `reflex_sentry/data/` produces
and consumes. Constants (`SOURCES`, column name tuples, `make_id`, `normalize_text`,
`validate_pool`) live in `reflex_sentry/data/schema.py`; import from there rather than
hardcoding strings.

## Raw layout: `data/raw/<name>/`

Downloaded with `hf download <hf_id> --repo-type dataset --local-dir data/raw/<name>`. Files may
be parquet, json, jsonl, jsonl.gz, or csv, possibly nested in subfolders (e.g. per-split or
per-subset folders). `reflex_sentry.data.loaders.find_data_files` globs all of these recursively;
`read_all` concatenates them into one DataFrame per source, tagging each row with its source file
path in a `_file` column so loaders can use file/folder names to recover split or subset info that
isn't in any column (see `or_bench` and `_infer_split` in `data/SOURCES.md`).

| `data/raw/<name>` | HF dataset | Loader |
|---|---|---|
| `toxic_chat` | lmsys/toxic-chat | `loaders.load_toxic_chat` |
| `wildguardmix` | allenai/wildguardmix | `loaders.load_wildguardmix` |
| `beavertails` | PKU-Alignment/BeaverTails | `loaders.load_beavertails` |
| `aegis2` | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | `loaders.load_aegis2` |
| `hh_redteam` | Anthropic/hh-rlhf (red-team-attempts/ only) | `loaders.load_hh_redteam` |
| `xstest` | walledai/XSTest | `loaders.load_xstest` |
| `or_bench` | bench-llm/or-bench | `loaders.load_or_bench` |

Plus `seeds/hard_negatives.csv` (AI-assisted drafts, committed): columns
`id, text, gold, tags, source, notes`; `gold` is always `"benign"`, `source` is always `"hn_seed"`,
`tags` look like `hard_negative;hn:detection`. Loaded by `loaders.load_hn_seed`.

Full per-source column mapping, label derivation, and caveats: see `data/SOURCES.md`. That file
is the place to record what actually turned up after a real download, if it differs from what's
assumed here.

## Normalized pool: `data/interim/pool.parquet`

One row per unique `(source, text)`. Every loader returns exactly this shape (see
`schema.POOL_COLUMNS`), and `build.build_pool` concatenates all requested sources into one
DataFrame with these columns, in this order:

| Column | Type | Notes |
|---|---|---|
| `id` | str | `"<source>:<first 12 hex of sha1(normalize_text(text))>"`. Stable across runs: re-running the pipeline on the same raw files always reproduces the same ids, so downstream files (gold labels, predictions) can join on `id` safely. |
| `text` | str | Original text, stripped of leading/trailing whitespace. Not lowercased or otherwise altered. |
| `source` | str | One of `schema.SOURCES`: `toxic_chat`, `wildguardmix`, `beavertails`, `aegis2`, `hh_redteam`, `xstest`, `or_bench`, `hn_seed`. |
| `source_label` | float | `1.0` unsafe, `0.0` safe, `NaN` unknown. This can supply `y` in the soft-target calculation described in [METHOD.md](METHOD.md). |
| `source_category` | str or null | Raw category string(s) from the source (semicolon-joined where a source has more than one), or null when the source has no category. |
| `is_adversarial` | bool or null | Whether the source itself flags the prompt as an adversarial/red-team/jailbreak attempt. Null when the source doesn't say. |
| `tags` | str | Semicolon-separated, `""` if none. Only `hn_seed` rows carry meaningful tags today (`hard_negative;hn:<name>`); everything else keeps its category, if any, in `source_category` instead. |
| `origin_split` | str or null | The source's own split name (`train`/`test`/`validation`), inferred from the file path where the source doesn't have a `split` column of its own. Null when unknown. |

`normalize_text(text)`: NFKC-normalize, lowercase, collapse all whitespace runs to a single
space, strip. Used for `make_id` and for dedupe; never mutates the `text` column itself.

`schema.validate_pool(df, stage="pool")` checks: all `POOL_COLUMNS` present, `source` values are
all in `schema.SOURCES`, `id` is unique and matches the `"<source>:<12 hex>"` pattern,
`source_label` is only `0.0`/`1.0`/`NaN`, and `tags` has no `NaN` (empty string only).

`configs/cyber_keywords.txt` is mostly literal terms/phrases, but a line prefixed
`re:` is a raw regex fragment instead (e.g. `re:cve-\d{4}-\d{4,}` for CVE ids), still
combined case-insensitively and word-boundary-wrapped like every other line -- see
`prefilter.load_keywords`/`prefilter.compile_pattern`.

**Two-tier matching**: a line prefixed `weak:` (or `weak:re:` for a weak regex) is a
*weak* keyword rather than a strong one. A row is in scope if it has at least 1 STRONG
hit, OR at least 2 DISTINCT WEAK hits -- a single weak hit alone is not enough. This
covers cyber-relevant words that are also ordinary English outside any security context
("vulnerable", "exploit", "breach", "compromised", ...): alone they're too noisy to
trust, but two of them together in the same prompt are a much stronger signal than
either alone. See `prefilter.split_tiers`/`prefilter.prefilter`.

## Scoped pool: `data/interim/cyber_pool.parquet`

`pool.parquet` plus (`schema.CYBER_EXTRA_COLUMNS`):

| Column | Type | Notes |
|---|---|---|
| `in_scope` | bool | `kw_strong >= 1` OR `kw_weak >= 2` (the keyword prefilter, `prefilter.prefilter`, `configs/cyber_keywords.txt`), OR the row's `source` is listed in `configs/data.yaml`'s `prefilter_bypass_sources` (see below). Only `in_scope == True` rows are kept in `cyber_pool.parquet`; the rest are dropped after the summary is printed, they never reach `data/processed/`. |
| `kw_hits` | str | Semicolon-joined matched keywords/phrases (lowercased) from EITHER tier, `""` if none -- including for a bypassed row that had no keyword hit at all. |
| `kw_strong` | int | Count of distinct strong-tier keyword hits (unprefixed lines in `configs/cyber_keywords.txt`). |
| `kw_weak` | int | Count of distinct weak-tier keyword hits (`weak:`-prefixed lines). |
| `dup_group` | str | An `id` value: the smallest `id` among all rows judged to be exact- or near-duplicates of this one (`dedupe.exact_dedupe` + `dedupe.near_dup_groups`). A row with no duplicates is its own group (`dup_group == id`). |

Build order for this file (`build.run`): prefilter the pool -> keep `in_scope` rows only ->
`exact_dedupe` (drops rows whose `normalize_text(text)` exactly matches another row already kept,
using `configs/data.yaml`'s `dedupe_preference` order to pick which source's copy survives) ->
`near_dup_groups` (MinHash/LSH over word shingles, default Jaccard threshold 0.8, configurable)
assigns `dup_group` to what's left. `validate_pool(df, stage="cyber")` additionally requires
`in_scope`, `kw_hits`, `dup_group` to be present.

## Splits: `data/processed/{train,val_pool,test_pool,test_ood_pool}.parquet`

All `cyber_pool.parquet` columns plus `split: str` (one of `schema.SPLIT_NAMES`:
`train`, `val_pool`, `test_pool`, `test_ood_pool`). Produced by `split.add_split`, called from
`build.run`. Rules, see `reflex_sentry/data/split.py` docstring for the exact algorithm:

- Split is decided at the **`dup_group` level**, never at the row level, so near-duplicate rows
  always land in the same split.
- One or more entire sources (`configs/data.yaml` -> `split.ood_sources`, a list; the
  committed configuration uses `toxic_chat`, `aegis2`, `hh_redteam`, and
  `beavertails`) are sampled down to `test_ood_pool_size` candidate rows *combined* for
  `test_ood_pool`; the rest of those sources' rows are dropped entirely (never used for train,
  val, or test). The older singular `split.ood_source` (a plain string) is still
  accepted for backward compat if a config sets that instead; `ood_sources` wins if both are set.
- `hn_seed` rows are scarce and matter most for eval: a configurable share
  (`split.hn_pool_ratio`, default 0.8) is routed into `val_pool`/`test_pool` before anything else;
  only the leftover share goes to `train`.
- Everything else is stratified by a `(safe/unsafe/unknown, hn:<name> or source_category)` key and
  filled into `val_pool`/`test_pool` up to their configured target sizes
  (`val_pool_size`/`test_pool_size`, 400 each in the committed config -- candidate counts, not final gold
  counts), with the remainder going to `train`.
- `val_pool` and `test_pool` are filled by **one joint pass per stratum**
  (`split._relative_deficit_split`, apportioned by `split._apportion_stratum`'s largest-remainder
  rounding), not by filling `val_pool` to its target first and handing `test_pool` whatever's
  left. Filling val first meant a stratum small enough to be just one or two `dup_group`s (a rare
  `source_category` combination -- unsafe rows tend to have a long tail of these) got claimed
  *entirely* by `val_pool` before `test_pool`'s turn even started, silently skewing `val_pool`'s
  label/source composition away from `test_pool`'s. Since the decision threshold is chosen on
  `val_pool` and final numbers are reported on `test_pool`, the two pools' composition (label
  balance and per-source shares) should match closely; the joint pass keeps them within a couple
  of points of each other on realistic pool sizes (see
  `tests/test_data_pipeline.py::test_split_val_and_test_pools_have_matching_composition`).

`val_pool`, `test_pool`, and `test_ood_pool` are **candidates for labeling**, not gold data.
The gold labels live in `data/gold/{val,test,test_ood}.csv` and get
merged into `data/processed/{val,test,test_ood}.parquet` (note: no `_pool` suffix once gold is
merged) with two added columns:

| Column | Type | Notes |
|---|---|---|
| `gold` | str | `dangerous`, `benign`, or `ambiguous` (see the [labeling guide](../configs/labeling_guide.md)). The saved 800-row gold set was mostly model-drafted and owner-reviewed, not independently blind human-labeled. |
| `tags` | str | Overwrites/extends the pool's `tags` with reviewed category tags (`cat:<name>`, `hard_negative;hn:<name>`). |

`validate_pool(df, stage="split")` additionally requires `split` to be present, non-null, and one
of `schema.SPLIT_NAMES`.

### Easy-benign slice (`reflex_sentry/data/easy_benign.py`)

`gold` rule 11 makes every in-scope benign item a hard negative, so `test.parquet` also gets
appended rows from `python -m reflex_sentry.data.easy_benign ... --append-to-test`: `gold="benign"`,
`tags=""`, `source="toxic_chat"`, `split="test"`, plus a bool column `easy_benign` (False on all
other rows). Re-running gold ingest for test overwrites `test.parquet`; re-run this command
afterwards. The 300 new ids were scored separately and merged into both
teacher files; see the completed [historical procedure](history/KAGGLE_EASY_BENIGN.md).

## Prediction CSVs

Produced by model/teacher/baseline inference, then consumed by the evaluation
package. See [METHOD.md](METHOD.md) for the probability and routing contract.
Required columns are `id, gold, p_safe, p_dangerous, p_unsure`; optional fields
include `source, tags, latency_ms, force_escalate, precheck_flag,
precheck_reasons`. `id` for a wrapped variant is
`<pool id>__<wrapper name>` (see `reflex_sentry/eval/wrappers.py`).

The shared `eval.run_all` runner checks each prediction's `id` and `gold`
against the processed split, joining by ID rather than row order. Missing,
extra, duplicate, or null IDs and mismatched or invalid labels are errors.
This includes validation predictions used only as a threshold reference.
Raw logits are checked before temperature fitting or calibrated files are
written, including validation logits when `val` is omitted from `--splits`.
The corresponding processed gold file must exist for these checks.
Standalone `eval.report` and `eval.calibrate` accept files without a processed
gold reference; they do not perform this cross-file consistency check.

## Teacher score artifacts

Teacher scoring and merging require `id`, `p_unsafe_teacher`, and
`teacher_model`. IDs must be nonblank and unique within each saved file;
one nonblank model identity must describe all rows. Unsafe probabilities must
be finite numeric values in `[0, 1]`. Optional `p_controversial` may be missing
or null for teachers without that verdict, but present values must also be
finite and in `[0, 1]`. Category and raw-response fields are optional.

Resumption validates the existing file and requires its model identity to
match the requested teacher. It may retain previously scored IDs outside the
current input subset. Overlapping input pools may repeat an ID only with the
same text. A scorer must return exactly one valid result per requested input;
invalid batches are rejected before checkpointing them.

`teacher.merge_scores` accepts overlaps across files only when their unsafe
scores and all other populated fields agree. A missing optional value may be
filled from another file; two populated values must match. Conflicting
overlaps and mixed model identities are rejected instead of
silently keeping the first row. Existing artifacts do not record prompt-text
fingerprints or checkpoint settings, so these checks cannot prove that those
settings were unchanged between runs.

## Student logits and timing sidecars

Student predictors first write `preds/<model>_<split>_logits.csv` with
`id`, `gold`, raw `logit_safe`, `logit_dangerous`, `logit_unsure` in that
class order, plus available `source`, `tags`, and per-row `latency_ms`.
The evaluator fits temperature using the `val` logits and writes calibrated
probability CSVs for each available split. Keyword and teacher predictors
can supply probability CSVs directly. Pre-check variants copy calibrated
probabilities and add `force_escalate`, `precheck_flag`, and
`precheck_reasons`; they do not alter logits or probabilities.

Pooled CPU timing details live in `preds/<model>_latency.json` rather than
training `metadata.json` or logits CSV. The sidecar has
`latency_protocol` (`threads`, `batch`, `sample_size`,
`includes_tokenization`) and sampled `p50_ms`, `p95_ms`, `n`.
The shared protocol times warmed-up, batch-one raw-text-to-probabilities
inference on a deterministic sample, normally one CPU thread. Per-row
`latency_ms` remains a separate field and missing timing remains missing.

## Config: `configs/data.yaml`

Read by `reflex_sentry.data.build.load_config`. Keys: `seed`, `raw_dir`, `interim_dir`,
`processed_dir`, `hn_seed_path`, `keywords_path`, `prefilter_bypass_sources` (list of source
names; every row from these sources gets `in_scope=True` regardless of keyword hits, `kw_hits`
still recorded -- default: none), `dedupe_preference` (list of source names, most
to least preferred), `shingle_k`, `minhash_perm`, `minhash_bands`, `dedupe_threshold`, and a
`split` block (`ood_sources` -- a list, or the older singular `ood_source` string for backward
compat -- `val_pool_size`, `test_pool_size`, `test_ood_pool_size`, `hn_pool_ratio`). Every key has
a built-in default (`split.DEFAULTS` / `build.build_pool`/`build.run` argument defaults), so a
partial or missing config file still runs.

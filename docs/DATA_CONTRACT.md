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

Plus `seeds/hard_negatives.csv` (hand-written, committed, owned by another agent): columns
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
| `source_label` | float | `1.0` unsafe, `0.0` safe, `NaN` unknown. This is `y` in the README 4.3 soft-target formula. |
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

## Scoped pool: `data/interim/cyber_pool.parquet`

`pool.parquet` plus (`schema.CYBER_EXTRA_COLUMNS`):

| Column | Type | Notes |
|---|---|---|
| `in_scope` | bool | Whether the keyword prefilter (`prefilter.prefilter`, `configs/cyber_keywords.txt`) matched this row, OR the row's `source` is listed in `configs/data.yaml`'s `prefilter_bypass_sources` (see below). Only `in_scope == True` rows are kept in `cyber_pool.parquet`; the rest are dropped after the summary is printed, they never reach `data/processed/`. |
| `kw_hits` | str | Semicolon-joined matched keywords/phrases (lowercased), `""` if none -- including for a bypassed row that had no keyword hit at all. |
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
- One entire source (`configs/data.yaml` -> `split.ood_source`, default `toxic_chat`) is sampled
  down to `test_ood_pool_size` candidate rows for `test_ood_pool`; the rest of that source's rows
  are dropped entirely (never used for train, val, or test), per README 4.1.
- `hn_seed` rows are scarce and matter most for eval: a configurable share
  (`split.hn_pool_ratio`, default 0.8) is routed into `val_pool`/`test_pool` before anything else;
  only the leftover share goes to `train`.
- Everything else is stratified by a `(safe/unsafe/unknown, hn:<name> or source_category)` key and
  filled into `val_pool`/`test_pool` up to their configured target sizes
  (`val_pool_size`/`test_pool_size`, default 600 each -- candidate counts, not final hand-labeled
  counts), with the remainder going to `train`.

`val_pool`, `test_pool`, and `test_ood_pool` are **candidates for hand labeling**, not gold data.
The gold labels live in `data/gold/{val,test,test_ood}.csv` (owned by another agent) and get
merged into `data/processed/{val,test,test_ood}.parquet` (note: no `_pool` suffix once gold is
merged) with two added columns:

| Column | Type | Notes |
|---|---|---|
| `gold` | str | `dangerous`, `benign`, or `ambiguous` (README section 1). |
| `tags` | str | Overwrites/extends the pool's `tags` with the hand-labeler's category tags (`cat:<name>`, `hard_negative;hn:<name>`), per README section 1. |

`validate_pool(df, stage="split")` additionally requires `split` to be present, non-null, and one
of `schema.SPLIT_NAMES`.

## Prediction CSVs

Not produced by this package. See README 5.1: columns `id, gold, p_safe, p_dangerous, p_unsure`
(required), `source, tags, latency_ms` (optional). `id` for a wrapped variant is
`<pool id>__<wrapper name>` (see `reflex_sentry/eval/wrappers.py`).

## Config: `configs/data.yaml`

Read by `reflex_sentry.data.build.load_config`. Keys: `seed`, `raw_dir`, `interim_dir`,
`processed_dir`, `hn_seed_path`, `keywords_path`, `prefilter_bypass_sources` (list of source
names; every row from these sources gets `in_scope=True` regardless of keyword hits, `kw_hits`
still recorded -- default: none), `dedupe_preference` (list of source names, most
to least preferred), `shingle_k`, `minhash_perm`, `minhash_bands`, `dedupe_threshold`, and a
`split` block (`ood_source`, `val_pool_size`, `test_pool_size`, `test_ood_pool_size`,
`hn_pool_ratio`). Every key has a built-in default (`split.DEFAULTS` /
`build.build_pool`/`build.run` argument defaults), so a partial or missing config file still runs.

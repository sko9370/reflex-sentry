# Data sources

Per-source notes for everything `reflex_sentry/data/loaders.py` reads. Licenses are marked
"verify" where this was written from memory/documentation, not re-checked against the actual
HF dataset card at download time (no network access in the dev environment). Re-check before
any redistribution.

Download convention: `hf download <hf_id> --repo-type dataset --local-dir data/raw/<name>`.
`data/` is git-ignored; only this file is tracked.

---

## toxic_chat -- lmsys/toxic-chat

- License: CC-BY-NC-4.0 (verify).
- Configs `toxicchat0124` and `toxicchat1123` exist; the loader prefers 0124 when both are
  present under `data/raw/toxic_chat` (real user prompts from an LLM-based Q&A service, labeled
  by human annotators, with a jailbreak flag).
- Columns used: `user_input` -> text, `toxicity` (0/1) -> source_label, `jailbreaking` (0/1) ->
  is_adversarial. `source_category` is left null; toxic-chat doesn't carry a harm category.
- The committed `split.ood_sources` configuration holds this source out with
  `aegis2`, `hh_redteam`, and `beavertails` for `test_ood_pool`; none of those
  sources enter train/val/test in the base split. A later 300-row ordinary-benign
  ToxicChat slice was appended to test separately, using source-derived labels.
- Caveat: assumed column names from the dataset card; if the real parquet uses different names
  (e.g. a `conv_id`/`model_output` split), the loader raises a clear error listing the columns it
  actually found.

## wildguardmix -- allenai/wildguardmix

- License: ODC-BY (verify).
- Splits `wildguardtrain` and `wildguardtest`; both are loaded if present under
  `data/raw/wildguardmix`, `origin_split` is inferred from the file/folder name.
- Columns used: `prompt` -> text, `prompt_harm_label` (harmful/unharmful) -> source_label (1.0/0.0),
  `subcategory` -> source_category, `adversarial` (bool) -> is_adversarial. The `response` and any
  response-harm columns are ignored; this project only scores prompts.

## beavertails -- PKU-Alignment/BeaverTails

- License: CC-BY-NC-4.0 (verify).
- Both the 30k and 330k train/test releases are supported (whatever files exist under
  `data/raw/beavertails`).
- **Label derivation (per-prompt label from per-QA-pair labels):** BeaverTails' `is_safe` is a
  label on the (prompt, response) pair, not the prompt alone. The loader groups rows by exact
  `prompt` text and marks the prompt `source_label = 1.0` (unsafe) if ANY of its response rows has
  `is_safe == False`, else `0.0`. This means a prompt could be labeled unsafe here even though some
  responses to it were judged safe; it is the more conservative reading for a prompt-only gate.
  Document/replace this rule if it produces too many false positives during hand labeling.
- `category` (a dict of category name -> bool) is unioned across all of a prompt's rows into a
  semicolon-joined `source_category` string of every category flagged true for that prompt.

## aegis2 -- nvidia/Aegis-AI-Content-Safety-Dataset-2.0

- License: CC-BY-4.0 (verify).
- Columns used: `prompt` -> text, `prompt_label` (safe/unsafe) -> source_label, `violated_categories`
  -> source_category.
- Rows where `prompt` is the literal string `"REDACTED"` (case-insensitive, whitespace-stripped)
  are dropped, per the dataset's own redaction of some prompts.

## hh_redteam -- Anthropic/hh-rlhf, red-team-attempts/ only

- License: MIT (verify).
- Only the `red-team-attempts/` jsonl(.gz) files are loaded, not the preference-pair splits
  elsewhere in this repo.
- Text extraction: the loader takes the first `"Human:"` turn out of the `transcript` string
  (everything up to the next `"Human:"`/`"Assistant:"` marker or end of string) as the prompt text.
- **source_label is left null (NaN), not derived from `rating`.** `rating` and
  `min_harmlessness_score_transcript` score how successful/harmful the red-teamer's FULL multi-turn
  attempt against the model was, not whether the opening request by itself is dangerous, benign, or
  ambiguous -- a skilled red-teamer's task_description can itself be describing a harmless-sounding
  opener that only turns adversarial several turns in. Since this project is single-turn only
  (see [project status](../docs/STATUS.md)), attaching a whole-transcript rating to just the first line would be a label leak in
  one direction and noise in the other. `is_adversarial` is set to `True` unconditionally (every row
  here is, by construction, a red-team attempt), and `task_description` is kept as `source_category`
  for context. If you later want a derived label, the most defensible option is probably
  `rating >= some_threshold` as a weak proxy for "the transcript did get somewhere dangerous", but
  that is still a transcript-level signal being applied to a single-turn prompt; treat it as noisy.

## xstest -- walledai/XSTest

- License: CC-BY-4.0 (verify).
- Columns used: `prompt` -> text, `label` (safe/unsafe) -> source_label, `type` -> source_category.
  `focus` is not used (would need its own free-form category handling); revisit if per-focus
  breakdowns turn out to matter for the over-refusal analysis.
- This is an over-refusal probe set: many "unsafe"-labeled prompts here are dual-use/borderline by
  design, and many "safe" ones are worded to sound alarming. Expect a chunk of `ambiguous` gold
  labels to come from this source during hand labeling.

## or_bench -- bench-llm/or-bench

- License: Apache-2.0 (verify).
- Three subsets: `or-bench-80k` and `or-bench-hard-1k` (benign prompts that sound toxic --
  label 0.0) and `or-bench-toxic` (actually toxic -- label 1.0).
- **Label is inferred from the file path**, not any column: any file under a folder/filename
  containing `"toxic"` (case-insensitive) is labeled 1.0; everything else 0.0. This means the raw
  layout MUST keep `or-bench-toxic` in its own path component (e.g.
  `data/raw/or_bench/or-bench-toxic/*.parquet`) -- if you flatten all three subsets into one folder
  of identically-named files, every row will be mislabeled benign. Verify the split sizes the
  loader reports against the known subset sizes (80k / 1k / a few thousand toxic) after download.
- `category` -> source_category.

## hn_seed -- seeds/hard_negatives.csv (AI-assisted drafts)

- Not downloaded; lives at `seeds/hard_negatives.csv` in this repo.
- The starter corpus was drafted with AI assistance. The owner reviewed the
  sampled gold items, including sampled seed rows; that is not evidence of a
  row-by-row review of the entire seed file. See [seed provenance](../seeds/README.md).
- Columns: `id, text, gold, tags, source, notes`. `gold` is always `"benign"` and `source` is
  always the literal string `"hn_seed"`.
- The loader ignores the seed file's own `id` column and recomputes `id` via
  `schema.make_id("hn_seed", text)`, so every row's id follows the same
  `"<source>:<12 hex>"` convention as every other source in the pool, and stays stable if the seed
  file's own id scheme ever changes. `tags` are passed through unchanged (e.g.
  `hard_negative;hn:detection`).
- `source_label` is set to `0.0` (benign) and `is_adversarial` to `False` for every row.

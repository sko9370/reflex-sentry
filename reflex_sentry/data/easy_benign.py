"""Easy-benign slice for the gold test set.

Why: gold labeling rule 11 tags every in-scope benign prompt `hard_negative`
(every in-scope benign prompt is security-flavored by construction), so
`easy_benign_escalation_rate` in `reflex_sentry.eval.metrics.at_threshold` has
n close to 0 and `cascade_economics` cannot reweight benign traffic by
`hard_negative_share_of_benign`. Real traffic is mostly ordinary non-security
prompts, so we add an "easy benign" slice: benign rows with empty tags.

Selection, from the full normalized pool (`data/interim/pool.parquet`):

  (a) `source == --source` (default toxic_chat: real user prompts,
      human-annotated, held out of training entirely)
  (b) `source_label == 0`
  (c) not in the cyber pool (`cyber_pool.parquet` holds only in-scope rows,
      so being absent means the prefilter judged the row out of scope) and
      not already in any processed split (matched by id and by normalized text)
  (d) 20 <= len(text) <= 1500 characters
  (e) English-looking: ASCII letters / non-space characters >= 0.8
  then dedupe by `normalize_text`, and sample `--n` rows with `--seed`.

Labels come from ToxicChat's human annotation plus prefilter out-of-scope
status, not hand review (the owner may spot-check).

    python -m reflex_sentry.data.easy_benign \\
        --pool data/interim/pool.parquet \\
        --cyber-pool data/interim/cyber_pool.parquet \\
        --test data/processed/test.parquet \\
        --n 300 --source toxic_chat --seed 7 \\
        --out data/processed/test_easy_benign.parquet [--append-to-test]

Outputs:
  --out                                   id, text, gold="benign", tags="", source,
                                          split="test", easy_benign=True
  <out dir>/easy_benign_for_teachers.parquet   id, text (input for the teacher scorer)

--append-to-test appends the rows to `--test` idempotently (ids already present
are skipped; `<test>.bak` is written once, on the first append). Re-running gold
ingest for test (`reflex_sentry.gold.ingest --split test`) OVERWRITES
test.parquet, so this command must be re-run afterwards. Prediction files for
test must also be regenerated so they include these ids.

Splits are scanned as every `*.parquet` in the --test directory except this
command's own outputs; rows of `--test` flagged `easy_benign` are ignored so a
re-run after an append reproduces the same sample.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import normalize_text

MIN_LEN = 20
MAX_LEN = 1500
MIN_ASCII_LETTER_RATIO = 0.8
TEACHER_FILE = "easy_benign_for_teachers.parquet"
OUT_COLUMNS = ["id", "text", "gold", "tags", "source", "split", "easy_benign"]


def english_looking(text: str, min_ratio: float = MIN_ASCII_LETTER_RATIO) -> bool:
    """ASCII letters as a share of non-whitespace characters >= min_ratio."""
    chars = [c for c in str(text) if not c.isspace()]
    if not chars:
        return False
    letters = sum(1 for c in chars if c.isascii() and c.isalpha())
    return letters / len(chars) >= min_ratio


def existing_split_keys(test_path: str | Path, exclude: tuple[Path, ...] = ()) -> tuple[set[str], set[str]]:
    """(ids, normalized texts) of every processed split next to `test_path`."""
    test_path = Path(test_path)
    skip = {p.resolve() for p in exclude}
    ids: set[str] = set()
    texts: set[str] = set()
    for f in sorted(test_path.parent.glob("*.parquet")):
        if f.resolve() in skip or f.name == TEACHER_FILE:
            continue
        df = pd.read_parquet(f)
        if f.resolve() == test_path.resolve() and "easy_benign" in df.columns:
            df = df[~df["easy_benign"].fillna(False).astype(bool)]
        if "id" in df.columns:
            ids.update(df["id"].astype(str))
        if "text" in df.columns:
            texts.update(normalize_text(t) for t in df["text"])
    return ids, texts


def select_easy_benign(
    pool: pd.DataFrame,
    cyber_ids: set[str],
    split_ids: set[str] = frozenset(),
    split_texts: set[str] = frozenset(),
    cyber_texts: set[str] = frozenset(),
    n: int = 300,
    source: str = "toxic_chat",
    seed: int = 7,
) -> pd.DataFrame:
    df = pool[(pool["source"] == source) & (pool["source_label"] == 0.0)].copy()
    df["id"] = df["id"].astype(str)
    df = df[~df["id"].isin(cyber_ids) & ~df["id"].isin(split_ids)]
    df = df[df["text"].notna()]
    length = df["text"].astype(str).str.len()
    df = df[(length >= MIN_LEN) & (length <= MAX_LEN)]
    df = df[df["text"].map(english_looking)]
    df["_norm"] = df["text"].map(normalize_text)
    df = df[~df["_norm"].isin(split_texts) & ~df["_norm"].isin(cyber_texts)]
    df = df.sort_values("id").drop_duplicates("_norm", keep="first").reset_index(drop=True)
    if len(df) > n:
        pick = np.random.default_rng(seed).choice(len(df), size=n, replace=False)
        df = df.iloc[np.sort(pick)]
    elif len(df) < n:
        print(f"[easy_benign] WARNING: only {len(df)} eligible rows, fewer than --n {n}")
    out = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "text": df["text"].astype(str).to_numpy(),
        "gold": "benign",
        "tags": "",
        "source": source,
        "split": "test",
        "easy_benign": True,
    })
    return out[OUT_COLUMNS].reset_index(drop=True)


def append_to_test(easy: pd.DataFrame, test_path: str | Path) -> int:
    """Append rows whose id is not already in test.parquet. Returns rows added."""
    test_path = Path(test_path)
    test = pd.read_parquet(test_path)
    new = easy[~easy["id"].isin(set(test["id"].astype(str)))].copy()
    if new.empty:
        return 0
    bak = test_path.with_name(test_path.name + ".bak")
    if not bak.exists():
        shutil.copy2(test_path, bak)
    if "easy_benign" not in test.columns:
        test["easy_benign"] = False
    test["easy_benign"] = test["easy_benign"].fillna(False).astype(bool)
    if "source_label" in test.columns:
        new["source_label"] = 0.0
    for c in test.columns:
        if c not in new.columns:
            new[c] = "" if c in ("kw_hits", "source_category") else (False if c == "in_scope" else None)
    merged = pd.concat([test, new[list(test.columns)]], ignore_index=True)
    merged.to_parquet(test_path, index=False)
    return len(new)


def run(pool_path: str, cyber_pool_path: str, test_path: str, n: int, source: str, seed: int,
        out_path: str, append: bool = False) -> pd.DataFrame:
    out = Path(out_path)
    teacher_out = out.parent / TEACHER_FILE
    pool = pd.read_parquet(pool_path, columns=["id", "text", "source", "source_label"])
    cyber = pd.read_parquet(cyber_pool_path, columns=["id", "text"])
    split_ids, split_texts = existing_split_keys(test_path, exclude=(out, teacher_out))
    easy = select_easy_benign(
        pool,
        cyber_ids=set(cyber["id"].astype(str)),
        cyber_texts={normalize_text(t) for t in cyber["text"]},
        split_ids=split_ids, split_texts=split_texts,
        n=n, source=source, seed=seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    easy.to_parquet(out, index=False)
    easy[["id", "text"]].to_parquet(teacher_out, index=False)
    print(f"wrote {len(easy)} easy-benign rows -> {out} and {teacher_out}")
    if append:
        added = append_to_test(easy, test_path)
        print(f"appended {added} new rows to {test_path} ({len(easy) - added} already present)")
    return easy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="data/interim/pool.parquet (full normalized pool)")
    ap.add_argument("--cyber-pool", required=True, help="data/interim/cyber_pool.parquet (in-scope rows only)")
    ap.add_argument("--test", required=True, help="data/processed/test.parquet; its directory is scanned for splits")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--source", default="toxic_chat")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append-to-test", action="store_true")
    a = ap.parse_args()
    run(a.pool, a.cyber_pool, a.test, a.n, a.source, a.seed, a.out, a.append_to_test)


if __name__ == "__main__":
    main()

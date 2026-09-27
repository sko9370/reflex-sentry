"""Validate and ingest hand labels into data/gold/<split>.csv and the merged parquet.

Accepts either a filled-in spreadsheet CSV (from `export.py`'s `--csv`) or a
Label Studio JSON export of completed annotations for the tasks `export.py`'s
`--label-studio` produced. Either way, every row is validated against the
label taxonomy in README section 1 before anything is written:

- gold in {dangerous, benign, ambiguous, out_of_scope}
- tags drawn only from the known cat:*/hn:*/hard_negative vocabulary
- cat:* tags only on gold=dangerous rows
- hn:* tags only on gold=benign rows, and always paired with hard_negative
- every id in --sample is labeled exactly once, and no unlabeled/unknown ids

On success, writes data/gold/<split>.csv (all rows, including out_of_scope)
and, if --pool is given, merges into data/processed/<split>.parquet: pool
columns + gold, with tags replaced by the gold sheet's tags unless the
labeler left tags blank, in which case any pre-existing hard_negative/hn:*
tags on that pool row (e.g. from an hn_seed row) are kept. out_of_scope rows
are dropped from the merged file.

    python -m reflex_sentry.gold.ingest --in val_sheet_filled.csv --split val \
        --sample data/gold/samples/val_sample.parquet --pool data/processed/val_pool.parquet \
        --gold-out data/gold/val.csv --merged-out data/processed/val.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from . import ALL_GOLD_LABELS, ALLOWED_TAGS, GOLD_CSV_COLUMNS, HARD_NEGATIVE_TAG, join_tags, split_tags


class GoldValidationError(ValueError):
    """Raised with every offending row id so a labeler can fix the sheet."""


def _err(problems: list[str]) -> None:
    if problems:
        raise GoldValidationError("gold sheet failed validation:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------- readers ---

def read_csv_sheet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    df = df[[c for c in df.columns if not c.startswith("_ref_")]]
    for c in ("gold", "tags", "notes", "source"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def read_label_studio_export(path: str, default_pass: int, default_date: str) -> pd.DataFrame:
    tasks = json.loads(Path(path).read_text())
    rows = []
    for t in tasks:
        data = t.get("data", {})
        annos = t.get("annotations") or t.get("completions") or []
        item_id = str(data.get("item_id", data.get("id", "")))
        gold, cat_tags, hn_tags, notes = "", [], [], ""
        if annos:
            for r in annos[-1].get("result", []):
                name, val = r.get("from_name"), r.get("value", {})
                if name == "gold":
                    gold = (val.get("choices") or [""])[0]
                elif name == "cat_tags":
                    cat_tags = val.get("choices") or []
                elif name == "hn_tags":
                    hn_tags = val.get("choices") or []
                elif name == "notes":
                    notes = "".join(val.get("text") or [])
        tags = list(cat_tags)
        if hn_tags:
            tags = [HARD_NEGATIVE_TAG, *hn_tags]
        rows.append({
            "id": item_id, "text": data.get("text", ""), "source": data.get("source", ""),
            "gold": gold, "tags": join_tags(tags), "labeler_pass": default_pass,
            "labeled_at": default_date, "notes": notes,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ normalize ---

def normalize(df: pd.DataFrame, default_pass: int, default_date: str) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].astype(str).str.strip()
    for c in ("text", "source", "gold", "tags", "notes"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    if "labeler_pass" not in df.columns or df["labeler_pass"].isna().all():
        df["labeler_pass"] = default_pass
    else:
        df["labeler_pass"] = df["labeler_pass"].fillna(default_pass).astype(int)
    if "labeled_at" not in df.columns:
        df["labeled_at"] = default_date
    else:
        blank = df["labeled_at"].isna() | (df["labeled_at"].fillna("").astype(str).str.strip() == "")
        df["labeled_at"] = df["labeled_at"].astype(object)
        df.loc[blank, "labeled_at"] = default_date
    return df[list(GOLD_CSV_COLUMNS)]


# -------------------------------------------------------------- validate ---

def validate(df: pd.DataFrame, sample_ids: set[str] | None = None) -> None:
    problems: list[str] = []

    dup = df["id"][df["id"].duplicated()].unique()
    if len(dup):
        problems.append(f"duplicate ids in sheet: {sorted(dup)}")

    unlabeled = df[df["gold"] == ""]["id"].tolist()
    if unlabeled:
        problems.append(f"missing gold label for ids: {unlabeled}")

    bad_gold = df[~df["gold"].isin(ALL_GOLD_LABELS) & (df["gold"] != "")]
    if not bad_gold.empty:
        problems.append("unknown gold label for ids: " +
                         str(list(zip(bad_gold["id"], bad_gold["gold"]))))

    for _, row in df.iterrows():
        tags = split_tags(row["tags"])
        unknown = [t for t in tags if t not in ALLOWED_TAGS]
        if unknown:
            problems.append(f"id {row['id']}: unknown tag(s) {unknown}")
        cat = [t for t in tags if t.startswith("cat:")]
        hn = [t for t in tags if t.startswith("hn:")]
        if cat and row["gold"] != "dangerous":
            problems.append(f"id {row['id']}: cat:* tag {cat} requires gold=dangerous, got '{row['gold']}'")
        if hn:
            if row["gold"] != "benign":
                problems.append(f"id {row['id']}: hn:* tag {hn} requires gold=benign, got '{row['gold']}'")
            if HARD_NEGATIVE_TAG not in tags:
                problems.append(f"id {row['id']}: hn:* tag {hn} requires the hard_negative tag too")

    if sample_ids is not None:
        got = set(df["id"])
        missing = sample_ids - got
        unknown_ids = got - sample_ids
        if missing:
            problems.append(f"ids in sample but not labeled: {sorted(missing)}")
        if unknown_ids:
            problems.append(f"ids labeled but not in sample: {sorted(unknown_ids)}")

    _err(problems)


# ------------------------------------------------------------------ write --

def write_gold_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("id").to_csv(path, index=False)


def build_merged(pool: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """pool columns + gold + tags, out_of_scope dropped, per README merge rule."""
    pool = pool.copy()
    pool["id"] = pool["id"].astype(str)
    g = gold.copy()
    g["id"] = g["id"].astype(str)
    merged = pool.merge(g[["id", "gold", "tags"]], on="id", how="inner", suffixes=("_pool", ""))

    def resolve_tags(row) -> str:
        if row["tags"]:  # labeler set tags explicitly: replace
            return row["tags"]
        pool_tags = split_tags(row.get("tags_pool", ""))
        return join_tags([t for t in pool_tags if t == HARD_NEGATIVE_TAG or t.startswith("hn:")])

    if "tags_pool" in merged.columns:
        merged["tags"] = merged.apply(resolve_tags, axis=1)
        merged = merged.drop(columns=["tags_pool"])
    return merged[merged["gold"] != "out_of_scope"].reset_index(drop=True)


# --------------------------------------------------------------------- CLI -

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="filled CSV or Label Studio JSON export")
    ap.add_argument("--format", choices=["csv", "label-studio"], help="default: guess from extension")
    ap.add_argument("--split", required=True, choices=["val", "test", "test_ood"])
    ap.add_argument("--sample", help="sample.py output, to check coverage")
    ap.add_argument("--pool", help="pool parquet/csv to merge into")
    ap.add_argument("--gold-out", help="default data/gold/<split>.csv")
    ap.add_argument("--merged-out", help="default data/processed/<split>.parquet")
    ap.add_argument("--labeler-pass", type=int, default=1, choices=[1, 2])
    ap.add_argument("--labeled-at", default=dt.date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    a = ap.parse_args()

    fmt = a.format or ("label-studio" if a.inp.endswith(".json") else "csv")
    raw = (read_label_studio_export(a.inp, a.labeler_pass, a.labeled_at) if fmt == "label-studio"
           else read_csv_sheet(a.inp))
    df = normalize(raw, a.labeler_pass, a.labeled_at)

    sample_ids = None
    if a.sample:
        sdf = pd.read_parquet(a.sample) if a.sample.endswith(".parquet") else pd.read_csv(a.sample)
        sample_ids = set(sdf["id"].astype(str))

    validate(df, sample_ids)
    print(f"validated {len(df)} labeled rows ({(df['gold'] == 'out_of_scope').sum()} out_of_scope)")
    if a.dry_run:
        return

    gold_out = a.gold_out or f"data/gold/{a.split}.csv"
    write_gold_csv(df, gold_out)
    print(f"wrote {gold_out}")

    if a.pool:
        pool = pd.read_parquet(a.pool) if a.pool.endswith(".parquet") else pd.read_csv(a.pool)
        merged = build_merged(pool, df)
        merged_out = a.merged_out or f"data/processed/{a.split}.parquet"
        Path(merged_out).parent.mkdir(parents=True, exist_ok=True)
        (merged.to_parquet(merged_out, index=False) if merged_out.endswith(".parquet")
         else merged.to_csv(merged_out, index=False))
        print(f"wrote {len(merged)} rows -> {merged_out} ({len(df) - len(merged)} out_of_scope dropped)")


if __name__ == "__main__":
    main()

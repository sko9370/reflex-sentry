"""Inspect a built data/ pipeline without dumping harmful text by default.

    python -m reflex_sentry.data.inspect
    python -m reflex_sentry.data.inspect --processed data/processed --interim data/interim
    python -m reflex_sentry.data.inspect --show 5 --source aegis2 --keyword ransomware

Prints two summaries that only ever need aggregate counts, never a row's own text:

1. Per (split x source) row counts and mean `source_label`, from
   `data/processed/{train,val_pool,test_pool,test_ood_pool}.parquet`. Useful for exactly the
   kind of val_pool/test_pool composition check that motivated `split.py`'s
   `_relative_deficit_split` fix (see docs/DATA_CONTRACT.md).
2. The top `--top-n` (default 40) `kw_hits` keywords by frequency among in-scope rows from
   public sources (i.e. everything in `data/interim/cyber_pool.parquet` except `hn_seed`,
   which is hand-written, not scraped), with each keyword's "sole-hit share": the fraction of
   its matching rows where it was the *only* keyword that matched. A generic word with a high
   sole-hit share and high frequency is a prefilter precision risk worth a manual look with
   `--show`.

`--show N --source S --keyword K` is the only thing that ever prints row text, and only when
explicitly asked for: up to N sample texts (each truncated to 200 chars) from `cyber_pool`
where `source == S` and `K` is one of that row's `kw_hits`, for manual precision review.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import schema

TEXT_PREVIEW_CHARS = 200


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def load_processed(processed_dir: str | Path) -> pd.DataFrame:
    """Concatenate whichever of data/processed/{train,val_pool,test_pool,test_ood_pool}.parquet
    exist into one DataFrame (with their "split" column). Missing files are skipped with a
    warning, same spirit as build.py's missing-raw-dir handling."""
    processed_dir = Path(processed_dir)
    frames = []
    for name in schema.SPLIT_NAMES:
        p = processed_dir / f"{name}.parquet"
        if not p.exists():
            _warn(f"{p} not found, skipping")
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=list(schema.POOL_COLUMNS) + list(schema.SPLIT_EXTRA_COLUMNS))
    return pd.concat(frames, ignore_index=True, sort=False)


def load_cyber_pool(interim_dir: str | Path) -> pd.DataFrame | None:
    """data/interim/cyber_pool.parquet (all in-scope rows, pre-split), or None if missing."""
    p = Path(interim_dir) / "cyber_pool.parquet"
    if not p.exists():
        _warn(f"{p} not found, skipping keyword-frequency summary")
        return None
    return pd.read_parquet(p)


# ------------------------------------------------------------- summaries ---

def split_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (split, source) row count and mean source_label, split x source in a fixed,
    readable order (schema.SPLIT_NAMES x schema.SOURCES) rather than however groupby sorts."""
    if df.empty:
        return pd.DataFrame(columns=["split", "source", "rows", "mean_source_label"])
    g = df.groupby(["split", "source"]).agg(
        rows=("id", "size"), mean_source_label=("source_label", "mean")
    ).reset_index()
    split_rank = {s: i for i, s in enumerate(schema.SPLIT_NAMES)}
    source_rank = {s: i for i, s in enumerate(schema.SOURCES)}
    g["_sr"] = g["split"].map(lambda s: split_rank.get(s, len(split_rank)))
    g["_kr"] = g["source"].map(lambda s: source_rank.get(s, len(source_rank)))
    g = g.sort_values(["_sr", "_kr"]).drop(columns=["_sr", "_kr"]).reset_index(drop=True)
    return g


def print_split_source_summary(df: pd.DataFrame) -> None:
    print("== rows and mean source_label, by split x source ==")
    if df.empty:
        print("  (no processed/*.parquet found)")
        return
    table = split_source_summary(df)
    print(table.to_string(index=False, formatters={"mean_source_label": "{:.3f}".format}))
    print()
    print("-- split totals --")
    totals = df.groupby("split").agg(rows=("id", "size"), mean_source_label=("source_label", "mean"))
    totals = totals.reindex(schema.SPLIT_NAMES, fill_value=0)
    print(totals.to_string(formatters={"mean_source_label": "{:.3f}".format}))


def keyword_frequency_table(cyber_df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    """Top `top_n` kw_hits keywords by frequency among in-scope public-source rows (every
    source except hn_seed), with each keyword's sole-hit share: the fraction of rows it
    matched in where it was the ONLY keyword that matched. A frequent keyword with a high
    sole-hit share and a generic meaning is a likely prefilter precision problem -- it's
    single-handedly pulling in rows nothing else about them says are cyber-relevant."""
    public = cyber_df[cyber_df["source"] != "hn_seed"]
    if "in_scope" in public.columns:
        public = public[public["in_scope"]]
    hit_lists = public["kw_hits"].fillna("").map(lambda s: [h for h in s.split(";") if h])

    counts: dict[str, int] = {}
    sole_counts: dict[str, int] = {}
    for hits in hit_lists:
        for h in hits:
            counts[h] = counts.get(h, 0) + 1
        if len(hits) == 1:
            sole_counts[hits[0]] = sole_counts.get(hits[0], 0) + 1

    rows = [
        {
            "keyword": k,
            "rows": n,
            "sole_hit_share": round(sole_counts.get(k, 0) / n, 3),
        }
        for k, n in counts.items()
    ]
    table = pd.DataFrame(rows, columns=["keyword", "rows", "sole_hit_share"])
    if table.empty:
        return table
    return table.sort_values(["rows", "keyword"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def print_keyword_table(table: pd.DataFrame, top_n: int) -> None:
    print(f"\n== top {top_n} kw_hits keywords among in-scope public-source rows (excl. hn_seed) ==")
    print("(sole_hit_share: share of a keyword's matching rows where it was the only hit --")
    print(" high + generic is a prefilter precision risk worth `--show`ing)")
    if table.empty:
        print("  (no in-scope public-source rows with any keyword hit)")
        return
    print(table.to_string(index=False))


def show_samples(cyber_df: pd.DataFrame, source: str, keyword: str, n: int) -> None:
    subset = cyber_df[cyber_df["source"] == source]
    hit = subset["kw_hits"].fillna("").map(lambda s: keyword in s.split(";"))
    subset = subset[hit]
    print(f"\n== up to {n} sample text(s), source={source!r} keyword={keyword!r} "
          f"({len(subset)} matching row(s) total) ==")
    if subset.empty:
        print("  (no matching rows)")
        return
    for _, row in subset.head(n).iterrows():
        text = str(row["text"])
        preview = text[:TEXT_PREVIEW_CHARS] + ("..." if len(text) > TEXT_PREVIEW_CHARS else "")
        print(f"  [{row['id']}] {preview}")


# ------------------------------------------------------------------- CLI ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed", help="dir with {split}.parquet files")
    ap.add_argument("--interim", default="data/interim", help="dir with cyber_pool.parquet")
    ap.add_argument("--top-n", type=int, default=40, help="how many keywords to list (default 40)")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                     help="print up to N sample texts (200 chars each) for --source/--keyword")
    ap.add_argument("--source", help="source to sample from, required with --show")
    ap.add_argument("--keyword", help="kw_hits keyword to sample, required with --show")
    args = ap.parse_args()

    processed = load_processed(args.processed)
    print_split_source_summary(processed)

    cyber = load_cyber_pool(args.interim)
    if cyber is not None:
        table = keyword_frequency_table(cyber, top_n=args.top_n)
        print_keyword_table(table, top_n=args.top_n)

    if args.show:
        if cyber is None:
            ap.error(f"--show needs {Path(args.interim) / 'cyber_pool.parquet'}, which was not found")
        if not args.source or not args.keyword:
            ap.error("--show requires both --source and --keyword")
        show_samples(cyber, args.source, args.keyword, args.show)


if __name__ == "__main__":
    main()

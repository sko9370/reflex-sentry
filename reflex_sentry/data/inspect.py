"""Inspect a built data/ pipeline without dumping harmful text by default.

    python -m reflex_sentry.data.inspect
    python -m reflex_sentry.data.inspect --processed data/processed --interim data/interim
    python -m reflex_sentry.data.inspect --per-source
    python -m reflex_sentry.data.inspect --show 5 --source aegis2 --keyword ransomware

Prints two summaries that only ever need aggregate counts, never a row's own text:

1. Per (split x source) row counts and mean `source_label`, from
   `data/processed/{train,val_pool,test_pool,test_ood_pool}.parquet`. Useful for exactly the
   kind of val_pool/test_pool composition check that motivated `split.py`'s
   `_relative_deficit_split` fix (see docs/DATA_CONTRACT.md).
2. The top `--top-n` (default 40) `kw_hits` keywords by frequency among in-scope rows from
   public sources (i.e. everything in `data/interim/cyber_pool.parquet` except `hn_seed`,
   which is AI-drafted, not scraped), with each keyword's "sole-hit share": the fraction of
   its matching rows where it was the *only* keyword that matched, and its tier (strong/weak,
   re-derived from `--keywords`, default `configs/cyber_keywords.txt`). A generic word with a
   high sole-hit share and high frequency is a prefilter precision risk worth a manual look
   with `--show`; a weak-tier one there is expected (a weak hit is only ever "sole" together
   with a source-level bypass or something odd, since two weak hits are required for scope).
   Pass `--per-source` to print this table separately per source (default top 15 each)
   instead of one combined top-N table, since a single dominant source can otherwise crowd
   every other source's keywords out.

`--show N --source S --keyword K` is the only thing that ever prints row text, and only when
explicitly asked for: up to N sample texts (each truncated to 200 chars) from `cyber_pool`
where `source == S` and `K` is one of that row's `kw_hits`, for manual precision review.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import schema
from .prefilter import compile_pattern, load_keywords, split_tiers

TEXT_PREVIEW_CHARS = 200
DEFAULT_PER_SOURCE_TOP_N = 15


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


class TierLookup:
    """Classifies a matched kw_hits string as "strong" or "weak" by re-matching it
    against configs/cyber_keywords.txt's own strong/weak patterns (kw_hits only stores
    the matched text, not which tier or which configured line produced it)."""

    def __init__(self, keywords_path: str | Path):
        keywords = load_keywords(keywords_path)
        strong_kw, weak_kw = split_tiers(keywords)
        self._strong = compile_pattern(strong_kw) if strong_kw else None
        self._weak = compile_pattern(weak_kw) if weak_kw else None

    def __call__(self, hit: str) -> str:
        if self._strong is not None and self._strong.fullmatch(hit):
            return "strong"
        if self._weak is not None and self._weak.fullmatch(hit):
            return "weak"
        return "?"


def load_tier_lookup(keywords_path: str | Path) -> TierLookup | None:
    """`TierLookup` for `keywords_path`, or None (with a warning) if it can't be loaded --
    e.g. the keywords file has since changed or moved. Tier annotation is best-effort."""
    try:
        return TierLookup(keywords_path)
    except (OSError, ValueError) as exc:
        _warn(f"could not load {keywords_path} for tier lookup: {exc}")
        return None


def keyword_frequency_table(
    cyber_df: pd.DataFrame, top_n: int = 40, tier_lookup: TierLookup | None = None
) -> pd.DataFrame:
    """Top `top_n` kw_hits keywords by frequency among in-scope public-source rows (every
    source except hn_seed), with each keyword's sole-hit share: the fraction of rows it
    matched in where it was the ONLY keyword that matched. A frequent keyword with a high
    sole-hit share and a generic meaning is a likely prefilter precision problem -- it's
    single-handedly pulling in rows nothing else about them says are cyber-relevant.

    When `tier_lookup` is given, a "tier" column (strong/weak/?) is added, so a weak
    keyword's sole-hit share can be read correctly: a weak hit is never in scope by
    itself, so "sole_hit_share" there means "co-occurred with nothing else, so this row
    is only in scope because some OTHER row-level source (a second weak hit elsewhere in
    the same text, or a bypass) put it there" -- worth a second look either way.
    """
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
    columns = ["keyword", "rows", "sole_hit_share"]
    if tier_lookup is not None:
        for row in rows:
            row["tier"] = tier_lookup(row["keyword"])
        columns.append("tier")
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    return table.sort_values(["rows", "keyword"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def keyword_frequency_table_per_source(
    cyber_df: pd.DataFrame, top_n: int = DEFAULT_PER_SOURCE_TOP_N, tier_lookup: TierLookup | None = None
) -> dict[str, pd.DataFrame]:
    """Same as `keyword_frequency_table`, computed separately per source (excl. hn_seed),
    each capped at its own top `top_n` -- useful because one dominant source (e.g.
    or_bench, per docs/DATA_CONTRACT.md) can otherwise crowd every other source's
    keywords out of a single combined top-N table."""
    public = cyber_df[cyber_df["source"] != "hn_seed"]
    if "in_scope" in public.columns:
        public = public[public["in_scope"]]
    tables = {}
    for source in sorted(public["source"].unique()):
        sub = public[public["source"] == source]
        tables[source] = keyword_frequency_table(sub, top_n=top_n, tier_lookup=tier_lookup)
    return tables


def print_keyword_table(table: pd.DataFrame, top_n: int) -> None:
    print(f"\n== top {top_n} kw_hits keywords among in-scope public-source rows (excl. hn_seed) ==")
    print("(sole_hit_share: share of a keyword's matching rows where it was the only hit --")
    print(" high + generic is a prefilter precision risk worth `--show`ing)")
    if table.empty:
        print("  (no in-scope public-source rows with any keyword hit)")
        return
    print(table.to_string(index=False))


def print_per_source_keyword_tables(tables: dict[str, pd.DataFrame], top_n: int) -> None:
    print(f"\n== top {top_n} kw_hits keywords per source, in-scope rows (excl. hn_seed) ==")
    if not tables:
        print("  (no in-scope public-source rows with any keyword hit)")
        return
    for source, table in tables.items():
        print(f"\n-- {source} --")
        if table.empty:
            print("  (no in-scope rows with any keyword hit)")
            continue
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
    ap.add_argument("--top-n", type=int, default=None,
                     help="how many keywords to list (default 40, or "
                          f"{DEFAULT_PER_SOURCE_TOP_N} per source with --per-source)")
    ap.add_argument("--keywords", default="configs/cyber_keywords.txt",
                     help="keyword file, used to label each keyword's strong/weak tier")
    ap.add_argument("--per-source", action="store_true",
                     help=f"print the keyword table separately per source, top "
                          f"{DEFAULT_PER_SOURCE_TOP_N} each by default (see --top-n)")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                     help="print up to N sample texts (200 chars each) for --source/--keyword")
    ap.add_argument("--source", help="source to sample from, required with --show")
    ap.add_argument("--keyword", help="kw_hits keyword to sample, required with --show")
    args = ap.parse_args()

    processed = load_processed(args.processed)
    print_split_source_summary(processed)

    cyber = load_cyber_pool(args.interim)
    if cyber is not None:
        tier_lookup = load_tier_lookup(args.keywords)
        if args.per_source:
            top_n = args.top_n if args.top_n is not None else DEFAULT_PER_SOURCE_TOP_N
            tables = keyword_frequency_table_per_source(cyber, top_n=top_n, tier_lookup=tier_lookup)
            print_per_source_keyword_tables(tables, top_n=top_n)
        else:
            top_n = args.top_n if args.top_n is not None else 40
            table = keyword_frequency_table(cyber, top_n=top_n, tier_lookup=tier_lookup)
            print_keyword_table(table, top_n=top_n)

    if args.show:
        if cyber is None:
            ap.error(f"--show needs {Path(args.interim) / 'cyber_pool.parquet'}, which was not found")
        if not args.source or not args.keyword:
            ap.error("--show requires both --source and --keyword")
        show_samples(cyber, args.source, args.keyword, args.show)


if __name__ == "__main__":
    main()

"""Build the reflex-sentry data pipeline: load -> pool -> prefilter -> dedupe -> split.

    python -m reflex_sentry.data.build --config configs/data.yaml
    python -m reflex_sentry.data.build --config configs/data.yaml --sources toxic_chat xstest
    python -m reflex_sentry.data.build --config configs/data.yaml --stats-only

Missing raw directories are skipped with a warning rather than failing the whole run, so the
pipeline can be exercised on whatever sources have actually been downloaded so far.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import loaders, schema
from .dedupe import exact_dedupe, near_dup_groups
from .prefilter import hit_counts_by_source, load_keywords, prefilter
from .split import add_split


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[warn] config {path} not found, using built-in defaults")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_pool(config: dict, sources: list[str] | None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load every requested source into one pool DataFrame. Returns (pool, rows_per_source)."""
    raw_root = Path(config.get("raw_dir", "data/raw"))
    wanted = sources or [*loaders.LOADERS.keys(), "hn_seed"]
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}

    for name in wanted:
        if name == "hn_seed":
            seed_path = Path(config.get("hn_seed_path", "seeds/hard_negatives.csv"))
            if not seed_path.exists():
                print(f"[warn] hn_seed: {seed_path} not found, skipping")
                continue
            try:
                df = loaders.load_hn_seed(seed_path)
            except loaders.DataFormatError as exc:
                print(f"[warn] hn_seed: {exc}")
                continue
        else:
            loader = loaders.LOADERS.get(name)
            if loader is None:
                print(f"[warn] unknown source {name!r}, skipping")
                continue
            raw_dir = raw_root / name
            if not raw_dir.exists():
                print(f"[warn] {name}: raw dir {raw_dir} not found, skipping")
                continue
            try:
                df = loader(raw_dir)
            except loaders.DataFormatError as exc:
                print(f"[warn] {name}: {exc}")
                continue
        counts[name] = len(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=list(schema.POOL_COLUMNS)), counts
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return pool, counts


def run(config: dict, sources: list[str] | None, stats_only: bool) -> dict:
    interim = Path(config.get("interim_dir", "data/interim"))
    processed = Path(config.get("processed_dir", "data/processed"))
    seed = int(config.get("seed", 42))

    pool, load_counts = build_pool(config, sources)
    print(f"== pool: {len(pool)} rows from {len(load_counts)} source(s) ==")
    for name, n in load_counts.items():
        print(f"  {name:14s} {n:8d}")

    if pool.empty:
        print("no data loaded (no raw dirs found); nothing to do")
        return {"pool": 0}

    schema.validate_pool(pool, stage="pool")

    keywords = load_keywords(config.get("keywords_path", "configs/cyber_keywords.txt"))
    scoped = prefilter(pool, keywords)
    print(f"\n== prefilter: {int(scoped['in_scope'].sum())} / {len(scoped)} in scope ==")
    print(hit_counts_by_source(scoped).to_string(index=False))

    cyber = scoped[scoped["in_scope"]].reset_index(drop=True)
    preference = config.get("dedupe_preference") or list(schema.SOURCES)
    before_exact = len(cyber)
    cyber = exact_dedupe(cyber, preference)
    cyber["dup_group"] = near_dup_groups(
        cyber,
        k=int(config.get("shingle_k", 5)),
        num_perm=int(config.get("minhash_perm", 32)),
        num_bands=int(config.get("minhash_bands", 8)),
        threshold=float(config.get("dedupe_threshold", 0.8)),
        seed=seed,
    ).to_numpy()
    schema.validate_pool(cyber, stage="cyber")
    n_groups = cyber["dup_group"].nunique()
    print(f"\n== dedupe: {before_exact} -> {len(cyber)} after exact, {n_groups} near-dup groups ==")

    label_balance = cyber["source_label"].value_counts(dropna=False)
    print(f"label balance (source_label): {label_balance.to_dict()}")

    if not stats_only:
        interim.mkdir(parents=True, exist_ok=True)
        pool.to_parquet(interim / "pool.parquet", index=False)
        cyber.to_parquet(interim / "cyber_pool.parquet", index=False)

    split_cfg = config.get("split", {})
    split_df = add_split(cyber, split_cfg, seed=seed)
    schema.validate_pool(split_df, stage="split")

    print(f"\n== split ({len(cyber)} -> {len(split_df)} after dropping unsampled OOD overflow) ==")
    print(split_df["split"].value_counts().reindex(schema.SPLIT_NAMES, fill_value=0).to_string())
    print("\nlabel balance by split (mean of source_label, NaN = unknown-label rows excluded):")
    print(split_df.groupby("split")["source_label"].agg(["count", "mean"]).to_string())

    if not stats_only:
        processed.mkdir(parents=True, exist_ok=True)
        for name in schema.SPLIT_NAMES:
            part = split_df[split_df["split"] == name].reset_index(drop=True)
            part.to_parquet(processed / f"{name}.parquet", index=False)
        print(f"\nwrote pool/cyber_pool to {interim} and splits to {processed}")

    summary = {"pool": len(pool), "cyber_pool": len(cyber)}
    summary.update(split_df["split"].value_counts().to_dict())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--sources", nargs="*", help="subset of sources to load (default: all)")
    ap.add_argument("--stats-only", action="store_true", help="print the summary, write no files")
    args = ap.parse_args()
    run(load_config(args.config), args.sources, args.stats_only)


if __name__ == "__main__":
    main()

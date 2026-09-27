"""Draw a stratified, reproducible sample from a candidate pool for hand labeling.

Input: a pool parquet from the data pipeline (e.g. ``data/processed/val_pool.parquet``)
with at least ``id, text, source, source_label`` and, if present, ``dup_group``
and ``in_scope``.

Stratifies by (source, source_label bucket) so the drawn sample keeps the
pool's mix of sources and of "already thought unsafe / safe / unknown"
items, deduplicates near-duplicates via ``dup_group`` (one representative per
group), then draws a fixed-seed subset with the largest-remainder method so
counts are exact and reproducible.

By default the output is blind: only ``id, text, split`` are written, so the
labeler is not anchored by the pool's own source or heuristic label. Source
and source_label are never lost -- ``ingest.py`` rejoins the pool by id when
it merges gold labels back in. Pass ``--no-blind`` for a debugging/audit
sample that keeps that metadata visible.

    python -m reflex_sentry.gold.sample --pool data/processed/val_pool.parquet \
        --split val --n 450 --out data/gold/samples/val_sample.parquet
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_N = 450
DEFAULT_SEED = 1337


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def write_any(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False) if str(path).endswith(".parquet") else df.to_csv(path, index=False)


def source_label_bucket(x) -> str:
    """Collapse source_label (1.0 / 0.0 / NaN) into a coarse stratification bucket."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "unknown"
    return "unsafe" if float(x) >= 0.5 else "safe"


def dedup_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one representative per dup_group; rows with no group are all kept."""
    if "dup_group" not in df.columns:
        return df
    has_group = df["dup_group"].notna() & (df["dup_group"].astype(str).str.len() > 0)
    grouped = df[has_group].sort_values("id").drop_duplicates("dup_group", keep="first")
    ungrouped = df[~has_group]
    return pd.concat([grouped, ungrouped], ignore_index=True)


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of n across strata sizes.

    Deterministic: ties in fractional remainder broken by stratum key so the
    same input always yields the same allocation.
    """
    total = sum(sizes.values())
    if total <= n:
        return dict(sizes)
    raw = {k: v * n / total for k, v in sizes.items()}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = n - sum(floor.values())
    order = sorted(sizes, key=lambda k: (-(raw[k] - floor[k]), k))
    for k in order[:remainder]:
        floor[k] += 1
    return floor


def draw_sample(pool: pd.DataFrame, n: int, seed: int = DEFAULT_SEED, dedup: bool = True) -> pd.DataFrame:
    """Return n rows of `pool` (or all of it, if smaller), stratified by
    (source, source_label bucket), with a fixed seed. Includes every pool
    column so callers can decide what to keep; blinding happens in `main`.
    """
    df = dedup_pool(pool) if dedup else pool
    if "in_scope" in df.columns:
        df = df[df["in_scope"].fillna(True).astype(bool)]
    df = df.reset_index(drop=True)
    stratum = df["source"].astype(str) + "::" + df.get("source_label", pd.Series(index=df.index)).map(source_label_bucket)
    sizes = stratum.value_counts().to_dict()
    alloc = allocate(sizes, min(n, len(df)))

    rng = np.random.default_rng(seed)
    chosen_parts = []
    for key in sorted(alloc):  # sorted order makes the rng draw sequence deterministic
        idx = df.index[stratum == key].to_numpy()
        idx_sorted = np.sort(idx)
        k = alloc[key]
        pick = idx_sorted if k >= idx_sorted.size else rng.choice(idx_sorted, size=k, replace=False)
        chosen_parts.append(pick)
    chosen = np.concatenate(chosen_parts) if chosen_parts else np.array([], dtype=int)
    order = rng.permutation(chosen.size)  # shuffle so row position doesn't leak stratum
    out = df.loc[chosen[order]].copy()
    out["stratum"] = stratum.loc[chosen[order]].to_numpy()
    return out.reset_index(drop=True)


def to_labeling_columns(sample: pd.DataFrame, split: str, sample_pass: int, blind: bool) -> pd.DataFrame:
    out = pd.DataFrame({
        "id": sample["id"].astype(str),
        "text": sample["text"],
        "split": split,
        "sample_pass": sample_pass,
    })
    if not blind:
        out["source"] = sample.get("source")
        out["source_label"] = sample.get("source_label")
        out["stratum"] = sample.get("stratum")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="candidate pool parquet/csv")
    ap.add_argument("--split", required=True, choices=["val", "test", "test_ood"])
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"items to draw before out_of_scope drops (default {DEFAULT_N}, "
                         "targets a 300-500 final gold set)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pass", dest="sample_pass", type=int, default=1, choices=[1, 2])
    ap.add_argument("--no-dedup", action="store_true", help="skip dup_group deduplication")
    ap.add_argument("--no-blind", action="store_true", help="keep source/source_label visible")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pool = read_any(a.pool)
    sample = draw_sample(pool, a.n, a.seed, dedup=not a.no_dedup)
    out = to_labeling_columns(sample, a.split, a.sample_pass, blind=not a.no_blind)
    write_any(out, a.out)
    print(f"drew {len(out)}/{len(pool)} rows (seed={a.seed}, blind={not a.no_blind}) -> {a.out}")


if __name__ == "__main__":
    main()

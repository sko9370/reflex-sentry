"""Exact and near-duplicate detection over the cyber-scoped pool.

Exact dedupe collapses rows whose normalized text is identical, even across
sources, keeping one row per a configurable source preference order (e.g.
AI-drafted hard negatives beat scraped copies of the same sentence).

Near-duplicate grouping uses MinHash over word shingles plus LSH banding, all
implemented with numpy/hashlib (no new dependency): each surviving row gets a
dup_group id, and rows whose shingle-set Jaccard similarity is estimated at or
above `threshold` share the same dup_group. split.py must keep every dup_group
inside a single split so near-duplicates never straddle train/val/test.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from .schema import normalize_text

_MERSENNE31 = (1 << 31) - 1


def exact_dedupe(df: pd.DataFrame, preference_order: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Drop rows with identical normalized text, keeping the source earliest in preference_order.

    Sources not listed in preference_order rank last. Ties within the same source keep the
    lexicographically smallest id, so the result is deterministic.
    """
    if df.empty:
        return df.copy()
    rank = {s: i for i, s in enumerate(preference_order)}
    work = df.copy()
    work["_norm"] = work["text"].map(normalize_text)
    work["_rank"] = work["source"].map(lambda s: rank.get(s, len(preference_order)))
    work = work.sort_values(["_norm", "_rank", "id"], kind="stable")
    work = work.drop_duplicates(subset="_norm", keep="first")
    return work.drop(columns=["_norm", "_rank"]).reset_index(drop=True)


# ------------------------------------------------------------- MinHash -----

def _shingles(text: str, k: int) -> set[str]:
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _base_hash(shingle: str) -> int:
    digest = hashlib.sha1(shingle.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") % _MERSENNE31


def _perm_coeffs(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE31 - 1, size=num_perm, dtype=np.int64)
    b = rng.randint(0, _MERSENNE31 - 1, size=num_perm, dtype=np.int64)
    return a, b


def _signature(shingles: set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not shingles:
        return np.full(a.shape[0], _MERSENNE31, dtype=np.int64)
    base = np.fromiter((_base_hash(s) for s in shingles), dtype=np.int64, count=len(shingles))
    vals = (np.outer(a, base) + b[:, None]) % _MERSENNE31
    return vals.min(axis=1)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def near_dup_groups(
    df: pd.DataFrame,
    k: int = 5,
    num_perm: int = 32,
    num_bands: int = 8,
    threshold: float = 0.8,
    seed: int = 42,
) -> pd.Series:
    """Return a Series (same index as df) mapping each row to a dup_group id.

    A dup_group id is the smallest `id` among the rows judged near-duplicates of each other
    (a singleton row is its own group, i.e. dup_group == id). LSH banding keeps this close to
    linear time; only candidates that share a band are Jaccard-checked exactly.
    """
    if df.empty:
        return pd.Series([], index=df.index, name="dup_group", dtype=object)

    ids = df["id"].tolist()
    shingle_sets = [_shingles(t, k) for t in df["text"].astype(str)]
    a, b = _perm_coeffs(num_perm, seed)
    sigs = [_signature(s, a, b) for s in shingle_sets]

    rows_per_band = max(1, num_perm // num_bands)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            band_key = (band_idx, tuple(sig[start:start + rows_per_band]))
            buckets[band_key].append(idx)

    uf = _UnionFind(len(ids))
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                x, y = members[i], members[j]
                if uf.find(x) == uf.find(y):
                    continue
                if _jaccard(shingle_sets[x], shingle_sets[y]) >= threshold:
                    uf.union(x, y)

    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(ids)):
        components[uf.find(idx)].append(idx)
    canonical = {root: min(ids[i] for i in members) for root, members in components.items()}
    groups = [canonical[uf.find(idx)] for idx in range(len(ids))]
    return pd.Series(groups, index=df.index, name="dup_group")

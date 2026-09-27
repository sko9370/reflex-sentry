"""Stratified train/val_pool/test_pool/test_ood_pool split of the cyber pool.

Splits at the dup_group level (never at the row level) so near-duplicates
never straddle splits. One entire source (config `ood_source`, default
toxic_chat) is held out for test_ood_pool and excluded from train/val/test
entirely, per README 4.1. hn_seed rows are scarce and matter most for eval,
so a configurable share of them (`hn_pool_ratio`) is routed into val_pool /
test_pool ahead of everything else, and only the remainder goes to train.

val_pool / test_pool / test_ood_pool are CANDIDATE pools for hand labeling
(README 3, step 2); the gold labels are merged in separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "ood_source": "toxic_chat",
    "val_pool_size": 600,
    "test_pool_size": 600,
    "test_ood_pool_size": 600,
    "hn_pool_ratio": 0.8,
}


def _stratum_key(label: float, category: object, tags: object) -> str:
    lbl = "unk" if pd.isna(label) else ("unsafe" if label >= 0.5 else "safe")
    tag_str = tags if isinstance(tags, str) else ""
    hn_tag = next((t for t in tag_str.split(";") if t.startswith("hn:")), None)
    cat = hn_tag or (category if isinstance(category, str) and category else None)
    return f"{lbl}|{cat}" if cat else lbl


def _group_info(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("dup_group").agg(
        source=("source", "first"),
        source_label=("source_label", "first"),
        source_category=("source_category", "first"),
        tags=("tags", "first"),
        n=("id", "size"),
    )
    g["stratum"] = [
        _stratum_key(row.source_label, row.source_category, row.tags) for row in g.itertuples()
    ]
    return g


def _stratified_pick(group_info: pd.DataFrame, target_rows: int, seed: int) -> tuple[list, list]:
    """Pick groups (by index/dup_group id) totalling ~target_rows, proportional per stratum.

    Returns (chosen, remaining), both lists of dup_group ids, in group_info's original order.
    """
    if group_info.empty or target_rows <= 0:
        return [], list(group_info.index)

    rng = np.random.RandomState(seed)
    total_rows = group_info["n"].sum()
    chosen: set = set()
    for stratum in sorted(group_info["stratum"].unique()):
        sub = group_info[group_info["stratum"] == stratum]
        idx = list(sub.index)
        rng.shuffle(idx)
        quota_rows = target_rows * (sub["n"].sum() / total_rows)
        picked = 0
        for gid in idx:
            if picked >= quota_rows:
                break
            chosen.add(gid)
            picked += int(sub.loc[gid, "n"])

    picked_total = sum(int(group_info.loc[g, "n"]) for g in chosen)
    if picked_total < target_rows:
        leftover = [g for g in group_info.index if g not in chosen]
        rng.shuffle(leftover)
        for gid in leftover:
            if picked_total >= target_rows:
                break
            chosen.add(gid)
            picked_total += int(group_info.loc[gid, "n"])

    chosen_list = [g for g in group_info.index if g in chosen]
    remaining_list = [g for g in group_info.index if g not in chosen]
    return chosen_list, remaining_list


def add_split(df: pd.DataFrame, config: dict | None = None, seed: int = 42) -> pd.DataFrame:
    """Return df with a "split" column added; rows from the unsampled tail of the held-out
    ood source are dropped (that source is never used outside test_ood_pool)."""
    cfg = {**DEFAULTS, **(config or {})}
    if df.empty:
        out = df.copy()
        out["split"] = pd.Series(dtype=object)
        return out

    work = df.reset_index(drop=True).copy()
    if "dup_group" not in work.columns:
        work["dup_group"] = work["id"]

    ginfo = _group_info(work)
    assign: dict = {}

    ood_groups = ginfo[ginfo["source"] == cfg["ood_source"]]
    ood_chosen, _ = _stratified_pick(ood_groups, int(cfg["test_ood_pool_size"]), seed)
    for g in ood_chosen:
        assign[g] = "test_ood_pool"

    rest = ginfo.drop(index=ood_groups.index)
    hn_groups = rest[rest["source"] == "hn_seed"]
    general_groups = rest.drop(index=hn_groups.index)

    hn_total_rows = int(hn_groups["n"].sum())
    hn_pool_target = int(round(hn_total_rows * float(cfg["hn_pool_ratio"])))
    hn_pool_chosen, hn_train_left = _stratified_pick(hn_groups, hn_pool_target, seed + 1)
    for g in hn_train_left:
        assign[g] = "train"

    hn_pool_df = hn_groups.loc[hn_pool_chosen]
    val_size, test_size = float(cfg["val_pool_size"]), float(cfg["test_pool_size"])
    val_frac = val_size / (val_size + test_size) if (val_size + test_size) else 0.5
    hn_val_target = int(round(hn_pool_df["n"].sum() * val_frac))
    hn_val_chosen, hn_test_left = _stratified_pick(hn_pool_df, hn_val_target, seed + 2)
    for g in hn_val_chosen:
        assign[g] = "val_pool"
    for g in hn_test_left:
        assign[g] = "test_pool"

    val_used = int(hn_pool_df.loc[hn_val_chosen, "n"].sum()) if hn_val_chosen else 0
    test_used = int(hn_pool_df.loc[hn_test_left, "n"].sum()) if hn_test_left else 0
    val_remaining = max(0, int(cfg["val_pool_size"]) - val_used)
    test_remaining = max(0, int(cfg["test_pool_size"]) - test_used)

    val_chosen, after_val = _stratified_pick(general_groups, val_remaining, seed + 3)
    remaining_general = general_groups.loc[after_val]
    test_chosen, after_test = _stratified_pick(remaining_general, test_remaining, seed + 4)
    for g in val_chosen:
        assign[g] = "val_pool"
    for g in test_chosen:
        assign[g] = "test_pool"
    for g in after_test:
        assign[g] = "train"

    work["split"] = work["dup_group"].map(assign)
    work = work[work["split"].notna()].reset_index(drop=True)
    return work

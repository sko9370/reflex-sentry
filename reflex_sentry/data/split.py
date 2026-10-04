"""Stratified train/val_pool/test_pool/test_ood_pool split of the cyber pool.

Splits at the dup_group level (never at the row level) so near-duplicates
never straddle splits. Entire sources (config `ood_sources`, a list -- or the
older `ood_source` singular string for backward compat, default
`["toxic_chat"]`) are held out for test_ood_pool and excluded from
train/val/test entirely, per docs/DATA_CONTRACT.md. hn_seed rows are scarce and matter
most for eval, so a configurable share of them (`hn_pool_ratio`) is routed
into val_pool / test_pool ahead of everything else, and only the remainder
goes to train.

val_pool / test_pool / test_ood_pool are CANDIDATE pools for hand labeling
(docs/WORKFLOWS.md); the gold labels are merged in separately.

val_pool and test_pool are filled by ONE joint pass per stratum
(`_relative_deficit_split`), not by filling val to its target and then
handing test whatever is left over. Filling val first (the previous
approach) meant that for every stratum -- however small -- val got first
claim on it: a stratum with only one or two dup_groups (a rare
`source_category` combination, which unsafe rows tend to have a long tail
of) was picked entirely into val before test's turn even started, since
test only saw the strata val didn't fully consume. That silently skewed
val_pool's label/source composition away from test_pool's (and threshold
tuning happens on val, final numbers get reported on test -- see
docs/DATA_CONTRACT.md). The joint pass instead apportions each stratum's row
quota across val/test/train by largest-remainder rounding (`_apportion_stratum`),
so a stratum too small to owe val or test a whole row usually contributes to
neither instead of guaranteeing at least one group to both -- which is what
let a long tail of small strata inflate val_pool/test_pool's composition and
size alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "ood_sources": ["toxic_chat"],
    "val_pool_size": 600,
    "test_pool_size": 600,
    "test_ood_pool_size": 600,
    "hn_pool_ratio": 0.8,
}


def _ood_sources(config: dict | None) -> list[str]:
    """Resolve the held-out OOD source list, preferring `ood_sources` (a list) over the
    older `ood_source` (a single string), and falling back to the default when neither
    is set in the caller's config. Resolved against the raw config, not the
    DEFAULTS-merged one, so an explicit `ood_source` isn't shadowed by the default
    `ood_sources`."""
    config = config or {}
    sources = config.get("ood_sources")
    if sources:
        return [sources] if isinstance(sources, str) else list(sources)
    single = config.get("ood_source")
    if single:
        return [single]
    return list(DEFAULTS["ood_sources"])


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
    Used only for the single-pool case (OOD holdout), where there's no second pool competing
    for the same rows so favoring the "chosen" side first isn't a fairness problem.
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


def _apportion_stratum(val_quota: float, test_quota: float, stratum_rows: float) -> dict:
    """Largest-remainder (Hamilton) apportionment of one stratum's rows into integer row
    targets for {"val", "test", "leftover"}. Flooring each bucket's exact quota first, then
    handing the few rows lost to rounding to whichever bucket has the largest fractional
    remainder, means a stratum too small to owe val or test even one whole row (quota < 1,
    the common case for a long tail of rare categories) mostly floors to 0 for both instead
    of always rounding up to 1 -- which is what let a handful of enormous, rare-category
    strata balloon val_pool/test_pool to several times their configured size while still
    reporting "proportional" per-stratum shares. Ties among fractional remainders are exact
    only when quotas are exactly equal, which happens deliberately when val/test targets are
    equal; sorting Python dicts is stable, so it favors "val" then "test" then "leftover" on
    those exact ties, and _topup's own randomized order then evens that out in aggregate.
    """
    leftover_quota = max(stratum_rows - val_quota - test_quota, 0.0)
    exact = {"val": val_quota, "test": test_quota, "leftover": leftover_quota}
    floors = {k: int(v) for k, v in exact.items()}
    remainder_slots = int(round(stratum_rows)) - sum(floors.values())
    fracs = sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in fracs:
        if remainder_slots <= 0:
            break
        floors[k] += 1
        remainder_slots -= 1
    return floors


def _relative_deficit_split(
    group_info: pd.DataFrame, val_target: float, test_target: float, seed: int
) -> tuple[list, list, list]:
    """Split groups into (val_ids, test_ids, leftover_ids) in ONE pass per stratum, so val
    and test draw proportionally from every stratum -- including tiny ones -- without
    either pool being filled to completion (across every stratum) before the other gets a
    turn (which let val claim whole rare-category strata before test's turn even started).

    Each stratum's row quota is apportioned across {val, test, leftover} by largest-remainder
    rounding (`_apportion_stratum`) rather than "give the first group to whichever side wants
    it, no matter how small its quota" -- the latter guarantees at least one group to (usually
    both) val and test from every single stratum, which for a pool with many small/rare
    strata (e.g. unsafe rows' long tail of fine-grained categories) balloons val_pool/test_pool
    to several times their configured size. Within a stratum, which whole groups land in which
    bucket is still randomized (order of buckets filled, and order of groups within each), so
    it's not always the same groups/bucket favored when a stratum's row count doesn't divide
    evenly into its three integer targets.

    val_target/test_target may be fractional (e.g. a share of a smaller hn_pool_ratio
    allocation); groups are whole, so actual counts land close to but not exactly at the
    targets, same as the rest of this module's group-level quotas.
    """
    if group_info.empty:
        return [], [], []

    rng = np.random.RandomState(seed)
    total_rows = float(group_info["n"].sum())
    val_ids: list = []
    test_ids: list = []
    leftover_ids: list = []
    buckets = {"val": val_ids, "test": test_ids, "leftover": leftover_ids}

    for stratum in sorted(group_info["stratum"].unique()):
        sub = group_info[group_info["stratum"] == stratum]
        idx = list(sub.index)
        rng.shuffle(idx)
        stratum_rows = float(sub["n"].sum())
        val_quota = val_target * (stratum_rows / total_rows) if total_rows else 0.0
        test_quota = test_target * (stratum_rows / total_rows) if total_rows else 0.0
        targets = _apportion_stratum(val_quota, test_quota, stratum_rows)

        bucket_order = ["val", "test", "leftover"]
        rng.shuffle(bucket_order)
        picked = {"val": 0, "test": 0, "leftover": 0}
        remaining = list(idx)
        for name in bucket_order:
            target_rows = targets[name]
            while remaining and picked[name] < target_rows:
                gid = remaining.pop()
                n = int(sub.loc[gid, "n"])
                buckets[name].append(gid)
                picked[name] += n
        # any groups left over after every bucket hit its target (group-size granularity can
        # overshoot a target slightly) fall back to leftover/train.
        leftover_ids.extend(remaining)

    val_ids, test_ids, leftover_ids = _topup(
        group_info, val_ids, test_ids, leftover_ids, val_target, test_target, rng
    )
    return val_ids, test_ids, leftover_ids


def _topup(
    group_info: pd.DataFrame,
    val_ids: list,
    test_ids: list,
    leftover_ids: list,
    val_target: float,
    test_target: float,
    rng: np.random.RandomState,
) -> tuple[list, list, list]:
    """If per-stratum quotas didn't add up to the full targets (e.g. a stratum ran out of
    groups before its own quota was met), top up from leftover groups -- in whichever
    order leaves val/test closest to equally short of their targets, not just handing
    everything left to val."""
    v_have = sum(int(group_info.loc[g, "n"]) for g in val_ids)
    t_have = sum(int(group_info.loc[g, "n"]) for g in test_ids)
    order = list(leftover_ids)
    rng.shuffle(order)
    still_leftover = []
    for gid in order:
        v_deficit, t_deficit = val_target - v_have, test_target - t_have
        if v_deficit <= 0 and t_deficit <= 0:
            still_leftover.append(gid)
            continue
        n = int(group_info.loc[gid, "n"])
        if v_deficit >= t_deficit:
            val_ids.append(gid)
            v_have += n
        else:
            test_ids.append(gid)
            t_have += n
    return val_ids, test_ids, still_leftover


def add_split(df: pd.DataFrame, config: dict | None = None, seed: int = 42) -> pd.DataFrame:
    """Return df with a "split" column added; rows from the unsampled tail of the held-out
    ood source(s) are dropped (those sources are never used outside test_ood_pool)."""
    ood_sources = _ood_sources(config)
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

    ood_groups = ginfo[ginfo["source"].isin(ood_sources)]
    ood_chosen, _ = _stratified_pick(ood_groups, int(cfg["test_ood_pool_size"]), seed)
    for g in ood_chosen:
        assign[g] = "test_ood_pool"

    rest = ginfo.drop(index=ood_groups.index)
    hn_groups = rest[rest["source"] == "hn_seed"]
    general_groups = rest.drop(index=hn_groups.index)

    val_size, test_size = float(cfg["val_pool_size"]), float(cfg["test_pool_size"])
    val_share = val_size / (val_size + test_size) if (val_size + test_size) else 0.5

    # hn_seed: hn_pool_ratio's worth of rows goes to val_pool/test_pool combined (ahead of
    # everything else, since hn rows are scarce and matter most for eval); the rest to
    # train. Both eval targets are handed to one joint split so val/test share hn's
    # (typically small, hn:<name>-tagged) strata fairly rather than sequentially.
    hn_total_rows = float(hn_groups["n"].sum())
    hn_pool_target = hn_total_rows * float(cfg["hn_pool_ratio"])
    hn_val_target = hn_pool_target * val_share
    hn_test_target = hn_pool_target - hn_val_target
    hn_val_chosen, hn_test_chosen, hn_train_left = _relative_deficit_split(
        hn_groups, hn_val_target, hn_test_target, seed + 1
    )
    for g in hn_val_chosen:
        assign[g] = "val_pool"
    for g in hn_test_chosen:
        assign[g] = "test_pool"
    for g in hn_train_left:
        assign[g] = "train"

    val_used = int(hn_groups.loc[hn_val_chosen, "n"].sum()) if hn_val_chosen else 0
    test_used = int(hn_groups.loc[hn_test_chosen, "n"].sum()) if hn_test_chosen else 0
    val_remaining = max(0.0, val_size - val_used)
    test_remaining = max(0.0, test_size - test_used)

    val_chosen, test_chosen, train_chosen = _relative_deficit_split(
        general_groups, val_remaining, test_remaining, seed + 3
    )
    for g in val_chosen:
        assign[g] = "val_pool"
    for g in test_chosen:
        assign[g] = "test_pool"
    for g in train_chosen:
        assign[g] = "train"

    work["split"] = work["dup_group"].map(assign)
    work = work[work["split"].notna()].reset_index(drop=True)
    return work

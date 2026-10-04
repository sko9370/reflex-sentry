"""Soft-target construction for reflex-sentry student training.

Formula choices (every deviation from the single-teacher starting point is
recorded here, since this is one of the most consequential design choices
in the project):

Single teacher (default)
    p = p_unsafe_teacher
    y = source_label (0.0 / 1.0 / NaN)
    d = |p - y|, 0 if y is NaN
    u = 0.50 * clip(2 * (d - 0.25), 0, 1) + 0.25 * (1 - |2p - 1|)
    target = [(1 - p) * (1 - u), p * (1 - u), u]   # [t_safe, t_dangerous, t_unsure]

Two teachers
    p = mean(p1, p2). `d` and the base `u` above are computed from this
    averaged `p` (not per-teacher), so the "teacher near 0.5" term reflects
    the consensus probability.

    A third term is added to `u` for teacher disagreement, using the same
    clip(2*x, 0, 1) shape as the dataset-disagreement term above so both
    saturate the same way:

        u += w_t * clip(2 * |p1 - p2|, 0, 1)     # w_t default 0.25

    If either teacher reports p_controversial (Qwen3Guard-Gen only), it is
    folded in too:

        u += w_c * nanmean(p_controversial_1, p_controversial_2)   # w_c default 0.25
             (0 if both are NaN/absent)

    `u` is then capped at `u_cap` (default 0.9) so no row loses all mass on
    safe/dangerous, and the three target values are renormalized (divided
    by their sum) as a defensive measure -- they already sum to 1 by
    construction from a capped `u`, but this guards against float drift and
    against future edits to this formula that might not preserve the
    invariant.

    Order of operations: base u (from d) -> + disagreement term -> +
    controversial term -> cap -> build [t_safe, t_dangerous, t_unsure] from
    the final p and u -> renormalize.

--source-only mode
    Used to start training before any teacher run finishes. Ignores any
    teacher score entirely and builds targets from `source_label` alone,
    with label smoothing 0.05 (`--smoothing`) so a hard source label is
    never presented as a target of exactly 0 or 1:

        known y:  p = y * (1 - 2*smoothing) + smoothing   # 0.0 -> 0.05, 1.0 -> 0.95
                  u = 0
        NaN y:    p = 0.5
                  u = 0.5

    Rows are marked `target_source = "source_only"`; teacher-based rows are
    marked `target_source = "teacher"`.

Output: `data/interim/soft_targets.parquet` with columns `id, t_safe,
t_dangerous, t_unsure` (sum to 1), plus `p, y, d, u, n_teachers,
teacher_disagreement, target_source`. `teacher_disagreement` is `|p1 - p2|`
when `n_teachers == 2`, else NaN (undefined for one teacher or source-only).
`d` is NaN in source-only mode (no teacher to disagree with).

Merging two teacher score files is an inner join on `id`: a row only gets a
two-teacher soft target if both teachers scored it. Merging teacher scores
onto the pool is also an inner join: a row only gets a soft target if at
least one teacher scored it.

CLI
---
    python -m reflex_sentry.targets \\
        --teachers data/interim/teacher_scores_llama_guard_3_8b.parquet \\
                   data/interim/teacher_scores_qwen3guard_gen_8b.parquet \\
        --pool data/processed/train.parquet \\
        --out data/interim/soft_targets.parquet

    python -m reflex_sentry.targets --source-only \\
        --pool data/processed/train.parquet \\
        --out data/interim/soft_targets.parquet
"""
from __future__ import annotations

import argparse
import glob as globmod
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_W_T = 0.25
DEFAULT_W_C = 0.25
DEFAULT_U_CAP = 0.9
DEFAULT_SMOOTHING = 0.05


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _renormalize(t_safe: np.ndarray, t_dangerous: np.ndarray, t_unsure: np.ndarray):
    total = t_safe + t_dangerous + t_unsure
    return t_safe / total, t_dangerous / total, t_unsure / total


def base_disagreement_unsure(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-teacher `d` and `u`, before any two-teacher extension."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = np.where(np.isnan(y), 0.0, np.abs(p - y))
    u = 0.50 * _clip01(2 * (d - 0.25)) + 0.25 * (1 - np.abs(2 * p - 1))
    return d, u


def build_targets(
    p: np.ndarray,
    y: np.ndarray,
    teacher_disagreement: np.ndarray | None = None,
    controversial: np.ndarray | None = None,
    w_t: float = DEFAULT_W_T,
    w_c: float = DEFAULT_W_C,
    u_cap: float = DEFAULT_U_CAP,
):
    """Return (d, u, t_safe, t_dangerous, t_unsure) for the teacher-based
    formula (single- or two-teacher; pass `teacher_disagreement`/
    `controversial` as None to get the single-teacher
    formula)."""
    p = np.asarray(p, dtype=np.float64)
    d, u = base_disagreement_unsure(p, y)

    if teacher_disagreement is not None:
        td = np.asarray(teacher_disagreement, dtype=np.float64)
        u = u + w_t * _clip01(2 * np.abs(td))

    if controversial is not None:
        c = np.nan_to_num(np.asarray(controversial, dtype=np.float64), nan=0.0)
        u = u + w_c * c

    u = np.minimum(u, u_cap)
    t_safe = (1 - p) * (1 - u)
    t_dangerous = p * (1 - u)
    t_unsure = u
    t_safe, t_dangerous, t_unsure = _renormalize(t_safe, t_dangerous, t_unsure)
    return d, u, t_safe, t_dangerous, t_unsure


def build_source_only_targets(y: np.ndarray, smoothing: float = DEFAULT_SMOOTHING):
    """Return (p, u, t_safe, t_dangerous, t_unsure) for --source-only mode."""
    y = np.asarray(y, dtype=np.float64)
    known = ~np.isnan(y)
    p = np.where(known, y * (1 - 2 * smoothing) + smoothing, 0.5)
    u = np.where(known, 0.0, 0.5)
    t_safe = (1 - p) * (1 - u)
    t_dangerous = p * (1 - u)
    t_unsure = u
    t_safe, t_dangerous, t_unsure = _renormalize(t_safe, t_dangerous, t_unsure)
    return p, u, t_safe, t_dangerous, t_unsure


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def run_teacher(
    teacher_paths: list[str],
    pool_path: str,
    w_t: float = DEFAULT_W_T,
    w_c: float = DEFAULT_W_C,
    u_cap: float = DEFAULT_U_CAP,
) -> pd.DataFrame:
    if len(teacher_paths) not in (1, 2):
        raise ValueError(f"expected 1 or 2 --teachers files, got {len(teacher_paths)}")

    pool = read_any(pool_path)
    missing = {"id", "source_label"} - set(pool.columns)
    if missing:
        raise ValueError(f"{pool_path} missing required column(s): {sorted(missing)}")

    if len(teacher_paths) == 1:
        t = read_any(teacher_paths[0])
        merged = pool[["id", "source_label"]].merge(t, on="id", how="inner")
        p = merged["p_unsafe_teacher"].astype(float).to_numpy()
        y = merged["source_label"].astype(float).to_numpy()
        controversial = (
            merged["p_controversial"].astype(float).to_numpy() if "p_controversial" in merged.columns else None
        )
        d, u, t_safe, t_dangerous, t_unsure = build_targets(p, y, teacher_disagreement=None, controversial=controversial, w_c=w_c, u_cap=u_cap)
        n_teachers = np.full(len(merged), 1, dtype=int)
        teacher_disagreement = np.full(len(merged), np.nan)
    else:
        t1 = read_any(teacher_paths[0])
        t2 = read_any(teacher_paths[1])
        both = t1.merge(t2, on="id", how="inner", suffixes=("_1", "_2"))
        merged = pool[["id", "source_label"]].merge(both, on="id", how="inner")
        p1 = merged["p_unsafe_teacher_1"].astype(float).to_numpy()
        p2 = merged["p_unsafe_teacher_2"].astype(float).to_numpy()
        p = (p1 + p2) / 2.0
        y = merged["source_label"].astype(float).to_numpy()
        teacher_disagreement = np.abs(p1 - p2)
        c1 = merged["p_controversial_1"].astype(float).to_numpy() if "p_controversial_1" in merged.columns else np.full(len(merged), np.nan)
        c2 = merged["p_controversial_2"].astype(float).to_numpy() if "p_controversial_2" in merged.columns else np.full(len(merged), np.nan)
        controversial = np.nanmean(np.vstack([c1, c2]), axis=0)
        d, u, t_safe, t_dangerous, t_unsure = build_targets(
            p, y, teacher_disagreement=teacher_disagreement, controversial=controversial, w_t=w_t, w_c=w_c, u_cap=u_cap
        )
        n_teachers = np.full(len(merged), 2, dtype=int)

    return pd.DataFrame({
        "id": merged["id"],
        "t_safe": t_safe,
        "t_dangerous": t_dangerous,
        "t_unsure": t_unsure,
        "p": p,
        "y": y,
        "d": d,
        "u": u,
        "n_teachers": n_teachers,
        "teacher_disagreement": teacher_disagreement,
        "target_source": "teacher",
    })


def run_source_only(pool_path: str, smoothing: float = DEFAULT_SMOOTHING) -> pd.DataFrame:
    pool = read_any(pool_path)
    missing = {"id", "source_label"} - set(pool.columns)
    if missing:
        raise ValueError(f"{pool_path} missing required column(s): {sorted(missing)}")

    y = pool["source_label"].astype(float).to_numpy()
    p, u, t_safe, t_dangerous, t_unsure = build_source_only_targets(y, smoothing=smoothing)

    return pd.DataFrame({
        "id": pool["id"],
        "t_safe": t_safe,
        "t_dangerous": t_dangerous,
        "t_unsure": t_unsure,
        "p": p,
        "y": y,
        "d": np.full(len(pool), np.nan),
        "u": u,
        "n_teachers": np.full(len(pool), 0, dtype=int),
        "teacher_disagreement": np.full(len(pool), np.nan),
        "target_source": "source_only",
    })


def _expand_globs(patterns: list[str]) -> list[str]:
    out = []
    for pat in patterns:
        matched = sorted(globmod.glob(pat))
        out.extend(matched if matched else [pat])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teachers", nargs="+", default=None, help="1 or 2 teacher_scores_*.parquet files (glob patterns ok)")
    ap.add_argument("--pool", required=True, help="e.g. data/processed/train.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-only", action="store_true", help="build targets from source_label alone, ignoring --teachers")
    ap.add_argument("--w-t", type=float, default=DEFAULT_W_T, help="two-teacher disagreement weight in u")
    ap.add_argument("--w-c", type=float, default=DEFAULT_W_C, help="p_controversial weight in u")
    ap.add_argument("--u-cap", type=float, default=DEFAULT_U_CAP)
    ap.add_argument("--smoothing", type=float, default=DEFAULT_SMOOTHING, help="--source-only label smoothing")
    a = ap.parse_args()

    if a.source_only:
        df = run_source_only(a.pool, smoothing=a.smoothing)
    else:
        if not a.teachers:
            raise SystemExit("--teachers is required unless --source-only is set")
        df = run_teacher(_expand_globs(a.teachers), a.pool, w_t=a.w_t, w_c=a.w_c, u_cap=a.u_cap)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} soft targets to {a.out} ({df['target_source'].iloc[0] if len(df) else 'n/a'})")


if __name__ == "__main__":
    main()

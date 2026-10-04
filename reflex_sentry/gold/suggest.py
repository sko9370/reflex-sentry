"""Add recommendation columns to review sheets so a human reviewer can accept
or pick instead of labeling from scratch.

For a row that already has a label (`gold` non-empty, whether from the opus
draft or filled in by a human reviewer), the recommendation columns just echo
that label, so the reviewer sees one consistent set of `_rec_*` columns
whether or not the row still needs labeling.

For a row still blank, this builds a TF-IDF nearest-neighbor index (word
1-2 grams plus char 3-5 grams, sublinear tf) over every row with a non-empty
`gold` across *all* the input sheets, finds the `k` most similar labeled rows
to the blank row, turns their similarity-weighted label vote plus a prior
from the pool's own `_source_label` heuristic into a suggested label, tags,
and runner-up, and writes them into `_rec_gold`/`_rec_tags`/`_rec_basis`/
`_rec_alt`. `gold` itself is left blank -- the human still confirms it.

    python -m reflex_sentry.gold.suggest \
        --sheets data/gold/review/test_reviewed_partial.csv \
                 data/gold/review/val_reviewed_partial.csv \
                 data/gold/sheets/test_ood_opus_draft.csv \
        --out-dir data/gold/review --mark-reviewed val
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from . import ALL_GOLD_LABELS, HARD_NEGATIVE_TAG, split_tags

K_NEIGHBORS = 7

# source_label -> prior over ALL_GOLD_LABELS (dangerous, benign, ambiguous, out_of_scope)
PRIOR_UNSAFE = {"dangerous": 0.6, "benign": 0.05, "ambiguous": 0.3, "out_of_scope": 0.05}
PRIOR_SAFE = {"dangerous": 0.0, "benign": 0.7, "ambiguous": 0.2, "out_of_scope": 0.1}
PRIOR_UNKNOWN = {label: 1.0 / len(ALL_GOLD_LABELS) for label in ALL_GOLD_LABELS}

REC_COLUMNS = ("_rec_gold", "_rec_tags", "_rec_basis", "_rec_alt")

def _is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == ""


def _source_label_prior(x) -> tuple[str, dict]:
    """(description, prior dict) for a pool `_source_label` value (1.0/0.0/NaN)."""
    if _is_blank(x):
        return "unknown", PRIOR_UNKNOWN
    return ("unsafe", PRIOR_UNSAFE) if float(x) >= 0.5 else ("safe", PRIOR_SAFE)


def _tags_valid_for(gold: str, tags: str) -> bool:
    """True if `tags` obeys the cat:*/hn:* gold-gating rules for `gold` (configs/labeling_guide.md)."""
    parsed = split_tags(tags)
    if not parsed:
        return True
    if any(t.startswith("cat:") for t in parsed):
        return gold == "dangerous"
    if any(t.startswith("hn:") for t in parsed):
        return gold == "benign" and HARD_NEGATIVE_TAG in parsed
    return gold == "benign"  # bare hard_negative with no hn: tag


# ------------------------------------------------------------- neighbor index

def build_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])


@dataclass
class NeighborIndex:
    vectorizer: FeatureUnion | None
    matrix: object  # sparse matrix, or None if there was nothing to index
    gold: np.ndarray
    tags: np.ndarray


def build_neighbor_index(df_all_labeled: pd.DataFrame) -> NeighborIndex:
    """Index every row of `df_all_labeled` whose `gold` is non-empty (including
    out_of_scope rows -- they're a real vote, just like the three eval labels)."""
    gold_col = df_all_labeled["gold"].fillna("").astype(str)
    labeled = df_all_labeled[gold_col.str.strip() != ""].reset_index(drop=True)
    if labeled.empty:
        return NeighborIndex(vectorizer=None, matrix=None, gold=np.array([]), tags=np.array([]))
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(labeled["text"].fillna("").astype(str))
    tags = labeled["tags"].fillna("").astype(str).to_numpy() if "tags" in labeled.columns \
        else np.array([""] * len(labeled))
    return NeighborIndex(vectorizer=vectorizer, matrix=matrix, gold=labeled["gold"].astype(str).to_numpy(),
                          tags=tags)


def _neighbor_distribution(sims: np.ndarray, gold: np.ndarray, k: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Similarity-weighted label distribution over the top-k neighbors, plus
    those neighbors' indices and similarities (for tags/basis to reuse)."""
    dist = {label: 0.0 for label in ALL_GOLD_LABELS}
    if sims.size == 0:
        return dist, np.array([], dtype=int), np.array([])
    k = min(k, sims.size)
    top_idx = np.argsort(-sims)[:k]
    top_sims = sims[top_idx]
    top_gold = gold[top_idx]
    total = float(top_sims.sum())
    if total > 0:
        for lbl, s in zip(top_gold, top_sims):
            dist[lbl] = dist.get(lbl, 0.0) + s / total
    elif len(top_gold):
        for lbl in top_gold:  # no similarity signal at all: split evenly
            dist[lbl] = dist.get(lbl, 0.0) + 1.0 / len(top_gold)
    return dist, top_idx, top_sims


def _combine(dist: dict, prior: dict) -> list[tuple[str, float]]:
    scored = {label: 0.5 * dist.get(label, 0.0) + 0.5 * prior.get(label, 0.0) for label in ALL_GOLD_LABELS}
    return sorted(scored.items(), key=lambda kv: -kv[1])


def _rec_tags_for(rec_gold: str, top_gold: np.ndarray, top_tags: np.ndarray) -> str:
    if rec_gold not in ("dangerous", "benign"):
        return ""
    candidates = [t for g, t in zip(top_gold, top_tags) if g == rec_gold and _tags_valid_for(rec_gold, t)]
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


def _rec_basis_for(desc: str, rec_gold: str, top_gold: np.ndarray, top_sims: np.ndarray) -> str:
    k = len(top_gold)
    if k == 0:
        return f"suggested: source label {desc} + no neighbors (prior only)"
    n_agree = int((top_gold == rec_gold).sum())
    max_sim = float(top_sims.max())
    return f"suggested: source label {desc} + {n_agree}/{k} neighbors {rec_gold} (max sim {max_sim:.2f})"


def recommend_blanks(index: NeighborIndex, df_blank: pd.DataFrame, k: int = K_NEIGHBORS) -> pd.DataFrame:
    """Return `df_blank` (needs `text`, optionally `_source_label`) with the
    four `_rec_*` suggestion columns appended. `gold` is not touched."""
    out = df_blank.copy()
    n = len(out)
    if n == 0:
        for c in REC_COLUMNS:
            out[c] = pd.Series(dtype=object)
        return out

    if index.matrix is not None:
        texts = out["text"].fillna("").astype(str)
        blank_matrix = index.vectorizer.transform(texts)
        sims_all = cosine_similarity(blank_matrix, index.matrix)
    else:
        sims_all = [np.array([]) for _ in range(n)]

    source_labels = out["_source_label"] if "_source_label" in out.columns else pd.Series([np.nan] * n, index=out.index)

    rec_gold, rec_tags, rec_basis, rec_alt = [], [], [], []
    for row_sims, src in zip(sims_all, source_labels):
        desc, prior = _source_label_prior(src)
        dist, top_idx, top_sims = _neighbor_distribution(np.asarray(row_sims), index.gold, k)
        ranked = _combine(dist, prior)
        g, alt = ranked[0][0], ranked[1][0]
        top_gold = index.gold[top_idx] if len(top_idx) else np.array([])
        top_tags = index.tags[top_idx] if len(top_idx) else np.array([])
        rec_gold.append(g)
        rec_tags.append(_rec_tags_for(g, top_gold, top_tags))
        rec_basis.append(_rec_basis_for(desc, g, top_gold, top_sims))
        rec_alt.append(alt)

    out["_rec_gold"] = rec_gold
    out["_rec_tags"] = rec_tags
    out["_rec_basis"] = rec_basis
    out["_rec_alt"] = rec_alt
    return out


def suggest(df_all_labeled: pd.DataFrame, df_blank: pd.DataFrame, k: int = K_NEIGHBORS) -> pd.DataFrame:
    """Build a neighbor index from `df_all_labeled` (needs `text`, `gold`,
    `tags`) and return `df_blank` with the `_rec_*` suggestion columns
    appended. Convenience wrapper for tests/one-off use; the CLI builds the
    index once and reuses it across every input sheet."""
    index = build_neighbor_index(df_all_labeled)
    return recommend_blanks(index, df_blank, k=k)


# ------------------------------------------------------------ human-fill check

def load_original_draft_gold(split: str, sheets_dir: str) -> dict[str, str] | None:
    """id -> gold from data/gold/sheets/{split}_opus_draft.csv, if that file
    exists. Used to tell a human-filled row (blank in the original draft,
    labeled in the reviewed sheet) from one the draft already labeled."""
    path = Path(sheets_dir) / f"{split}_opus_draft.csv"
    if not path.exists():
        return None
    draft = pd.read_csv(path, dtype={"id": str})
    draft["gold"] = draft["gold"].fillna("").astype(str)
    return dict(zip(draft["id"].astype(str), draft["gold"]))


def _queue_rank(df: pd.DataFrame, is_blank: pd.Series) -> pd.Series:
    """Human review queue order (0 = look at first): 0 gold empty (no label at
    all, including a suggested-but-unconfirmed row), 1 "single labeler only"
    reason, 2 other priority-1, 3 priority-2, 4 priority-3. A human-filled row
    keeps the rank its reason would give it -- `_reviewed=1` is what hides it
    from the queue by default, not this rank."""
    n = len(df)
    reason = df["_review_reason"].fillna("").astype(str) if "_review_reason" in df.columns \
        else pd.Series([""] * n, index=df.index)
    priority = pd.to_numeric(df["_review_priority"], errors="coerce") if "_review_priority" in df.columns \
        else pd.Series([np.nan] * n, index=df.index)
    priority = priority.fillna(3).astype(int)

    rank = pd.Series(4, index=df.index, dtype=int)
    rank[priority == 2] = 3
    rank[priority == 1] = 2
    rank[reason.str.contains("single labeler only", regex=False)] = 1
    rank[is_blank] = 0
    return rank


# --------------------------------------------------------------- full sheet --

def add_recommendations(
    df: pd.DataFrame,
    index: NeighborIndex,
    split: str,
    sheets_dir: str = "data/gold/sheets",
    k: int = K_NEIGHBORS,
    mark_reviewed: bool = False,
) -> pd.DataFrame:
    """One `{split}_review.csv` row per input row: `_rec_*` recommendation
    columns for every row (echoed for a labeled row, suggested for a blank
    one), plus `_reviewed` and `_review_action`. `mark_reviewed=True` marks
    every non-blank row in this sheet `_reviewed=1`, `_review_action="accepted"`
    (the human confirmed the whole split without changes)."""
    df = df.copy()
    gold = df["gold"].fillna("").astype(str) if "gold" in df.columns else pd.Series([""] * len(df), index=df.index)
    is_blank = gold.str.strip() == ""

    n = len(df)
    rec_gold = pd.Series([""] * n, index=df.index, dtype=object)
    rec_tags = pd.Series([""] * n, index=df.index, dtype=object)
    rec_basis = pd.Series([""] * n, index=df.index, dtype=object)
    rec_alt = pd.Series([""] * n, index=df.index, dtype=object)
    review_action = pd.Series([""] * n, index=df.index, dtype=object)

    existing_reviewed = df["_reviewed"] if "_reviewed" in df.columns else pd.Series([0] * n, index=df.index)
    reviewed = existing_reviewed.fillna(0).astype(int).copy()

    draft_gold_by_id = load_original_draft_gold(split, sheets_dir)

    non_blank_idx = df.index[~is_blank]
    for i in non_blank_idx:
        g = gold.at[i]
        rec_gold.at[i] = g
        rec_tags.at[i] = df.at[i, "tags"] if "tags" in df.columns else ""
        rec_alt.at[i] = df.at[i, "_alt_label"] if "_alt_label" in df.columns else ""
        rec_basis.at[i] = "draft/human label"

        human_filled = False
        if draft_gold_by_id is not None:
            orig = draft_gold_by_id.get(str(df.at[i, "id"]))
            if orig is not None and orig.strip() == "" and g.strip() != "":
                human_filled = True
        if human_filled:
            rec_basis.at[i] = "human"
            reviewed.at[i] = 1
            review_action.at[i] = "human"

        if mark_reviewed:
            reviewed.at[i] = 1
            review_action.at[i] = "accepted"

    blank_idx = df.index[is_blank]
    if len(blank_idx):
        rec = recommend_blanks(index, df.loc[blank_idx], k=k)
        rec_gold.loc[blank_idx] = rec["_rec_gold"].to_numpy()
        rec_tags.loc[blank_idx] = rec["_rec_tags"].to_numpy()
        rec_basis.loc[blank_idx] = rec["_rec_basis"].to_numpy()
        rec_alt.loc[blank_idx] = rec["_rec_alt"].to_numpy()

    df["_rec_gold"] = rec_gold
    df["_rec_tags"] = rec_tags
    df["_rec_basis"] = rec_basis
    df["_rec_alt"] = rec_alt
    df["_reviewed"] = reviewed
    df["_review_action"] = review_action
    df["_queue_rank"] = _queue_rank(df, is_blank)

    sort_cols = ["_queue_rank"] + (["_review_priority"] if "_review_priority" in df.columns else [])
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ summary --

def summarize(out: pd.DataFrame) -> dict:
    blanks = out["gold"].fillna("").astype(str).str.strip() == ""
    return {
        "n_rows": len(out),
        "n_blanks_suggested": int(blanks.sum()),
        "suggested_label_distribution": out.loc[blanks, "_rec_gold"].value_counts().to_dict(),
        "n_reviewed": int(out["_reviewed"].astype(int).sum()),
    }


def print_summary(in_path: str, out_path: Path, out: pd.DataFrame) -> None:
    s = summarize(out)
    print(f"{in_path} -> {out_path}")
    print(f"  rows: {s['n_rows']}  blanks suggested: {s['n_blanks_suggested']}  reviewed: {s['n_reviewed']}")
    print(f"  suggested label distribution: {s['suggested_label_distribution']}")


# --------------------------------------------------------------------- CLI --

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", nargs="+", required=True, help="review sheets to annotate")
    ap.add_argument("--out-dir", default="data/gold/review")
    ap.add_argument("--sheets-dir", default="data/gold/sheets",
                     help="where to look for each split's {split}_opus_draft.csv, to spot human-filled rows")
    ap.add_argument("--k", type=int, default=K_NEIGHBORS)
    ap.add_argument("--mark-reviewed", nargs="*", default=[], metavar="SPLIT",
                     help="splits the human has fully confirmed: every non-blank row gets "
                          "_reviewed=1, _review_action=accepted")
    a = ap.parse_args()

    dfs: dict[str, pd.DataFrame] = {}
    splits: dict[str, str] = {}
    for path in a.sheets:
        df = pd.read_csv(path, dtype={"id": str})
        for c in ("gold", "tags", "notes"):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        dfs[path] = df
        splits[path] = str(df["split"].iloc[0]) if "split" in df.columns and len(df) else Path(path).stem

    df_all_labeled = pd.concat(
        [df[df["gold"].astype(str).str.strip() != ""] for df in dfs.values()],
        ignore_index=True,
    ) if dfs else pd.DataFrame(columns=["text", "gold", "tags"])
    index = build_neighbor_index(df_all_labeled)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mark_reviewed = set(a.mark_reviewed)

    for path, df in dfs.items():
        split = splits[path]
        out = add_recommendations(df, index, split=split, sheets_dir=a.sheets_dir, k=a.k,
                                   mark_reviewed=split in mark_reviewed)
        out_path = out_dir / f"{split}_review.csv"
        out.to_csv(out_path, index=False)
        print_summary(path, out_path, out)


if __name__ == "__main__":
    main()

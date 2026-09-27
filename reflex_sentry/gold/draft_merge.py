"""Merge two independent LLM labeling passes into a human-review-ready draft.

Two labelers (A and B) each label the same ~800 items independently. This
combines their raw label files with the blank labeling sheets and the
candidate pools, produces a draft gold sheet per split that a human reviewer
can open and fix up, ranks every row by how urgently a human should look at
it, and writes the pass-1/pass-2-shaped files `agreement.py` needs to score
A-vs-B self-agreement.

A labeling batch can stop partway (a labeler ran out of time), so an id may
be missing from A, from B, or from both:

- both passes agree: draft = A's label (tag mismatches are still flagged).
- both passes present but disagree: draft = A's label, provisional, flagged
  priority 1, unless `--adjudications` gives a human override for that id.
- only one pass labeled it: draft = that pass's label, flagged priority 1
  ("single labeler only").
- neither pass labeled it: draft is blank, flagged priority 1 ("unlabeled").

Duplicate ids within a pass's label files, or ids in a pass's label files
that belong to no sheet at all, are still hard errors: those mean a batch
file is corrupt, not just incomplete.

    python -m reflex_sentry.gold.draft_merge \
        --sheets-dir data/gold/sheets \
        --labels-glob-a "data/gold/work/A_batch*_labels.csv" \
        --labels-glob-b "data/gold/work/B_batch*_labels.csv" \
        --pools-dir data/processed_upload \
        --adjudications data/gold/work/adjudications.csv \
        --out-dir data/gold/sheets --work-dir data/gold/work
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import GOLD_CSV_COLUMNS, GOLD_LABELS
from . import agreement as A

SPLITS = ("val", "test", "test_ood")

CONFIDENCE_LEVELS = ("low", "medium", "high")
_CONF_RANK = {level: i for i, level in enumerate(CONFIDENCE_LEVELS)}

LABEL_COLS = ("id", "gold", "tags", "confidence", "alt_label", "notes")
ADJUDICATION_COLS = ("id", "gold", "tags", "notes")

DISAGREEMENT_COLUMNS = (
    "id", "split", "gold_a", "gold_b", "conf_a", "conf_b", "tags_a", "tags_b", "notes_a", "notes_b",
)

_MERGED_EXTRA_COLS = (
    "gold", "tags", "notes", "_confidence", "_alt_label", "_review_priority", "_review_reason",
)


class DraftMergeError(ValueError):
    """Raised with every offending id so a batch file can be fixed."""


def _err(problems: list[str]) -> None:
    if problems:
        raise DraftMergeError("draft merge failed:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------- readers ---

def read_sheet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = df["id"].astype(str).str.strip()
    return df


def read_labels_glob(pattern: str) -> pd.DataFrame:
    """Concatenate every batch file matching `pattern`. Missing/no matches is fine
    (a pass may not have started yet) and yields an empty, correctly-shaped frame."""
    paths = sorted(glob.glob(pattern))
    frames = [pd.read_csv(p, dtype={"id": str}) for p in paths]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(LABEL_COLS))
    for c in LABEL_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[list(LABEL_COLS)].copy()
    out["id"] = out["id"].astype(str).str.strip()
    for c in ("gold", "tags", "confidence", "alt_label", "notes"):
        out[c] = out[c].fillna("").astype(str).str.strip()
    return out


def read_adjudications(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path, dtype={"id": str})
    for c in ADJUDICATION_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[list(ADJUDICATION_COLS)].copy()
    df["id"] = df["id"].astype(str).str.strip()
    for c in ("gold", "tags", "notes"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def read_pool(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["id", "source", "source_label"])
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df["id"] = df["id"].astype(str)
    return df


# -------------------------------------------------------------- checks ----

def check_no_duplicates(df: pd.DataFrame, label: str) -> None:
    dup = df["id"][df["id"].duplicated()].unique()
    if len(dup):
        _err([f"duplicate ids in {label}: {sorted(dup)}"])


def check_ids_known(df: pd.DataFrame, label: str, known_ids: set[str]) -> None:
    unknown = sorted(set(df["id"]) - known_ids)
    if unknown:
        _err([f"ids in {label} but not in any sheet: {unknown}"])


# ------------------------------------------------------------ row logic ---

def min_confidence(conf_a: str, conf_b: str) -> str:
    """Lower of two confidence levels; unrecognized values sort lowest."""
    return min((conf_a, conf_b), key=lambda c: _CONF_RANK.get(c, -1))


def compute_alt_label(has_a: bool, has_b: bool, gold_a: str, gold_b: str, alt_a: str, alt_b: str) -> str:
    if has_a and has_b:
        if gold_a != gold_b:
            return gold_b
        return alt_a or alt_b or ""
    if has_a:
        return alt_a or ""
    if has_b:
        return alt_b or ""
    return ""


def _has_source(source_label) -> bool:
    return source_label is not None and not (isinstance(source_label, float) and np.isnan(source_label))


def compute_row(r: pd.Series) -> dict:
    has_a, has_b = bool(r["has_a"]), bool(r["has_b"])
    adjudicated = bool(r["adjudicated"])
    gold_a, gold_b = r["gold_a"], r["gold_b"]
    split_disagree = has_a and has_b and gold_a != gold_b

    if adjudicated:
        gold, tags, notes = r["adj_gold"], r["adj_tags"], r["adj_notes"]
        confidence = "adjudicated"
    elif has_a and has_b:
        # agree -> A's label; disagree (not adjudicated) -> A's label, provisional
        gold, tags, notes = gold_a, r["tags_a"], r["notes_a"]
        confidence = min_confidence(r["conf_a"], r["conf_b"])
    elif has_a:
        gold, tags, notes = gold_a, r["tags_a"], r["notes_a"]
        confidence = r["conf_a"]
    elif has_b:
        gold, tags, notes = gold_b, r["tags_b"], r["notes_b"]
        confidence = r["conf_b"]
    else:
        gold, tags, notes, confidence = "", "", "", ""

    alt_label = compute_alt_label(has_a, has_b, gold_a, gold_b, r["alt_a"], r["alt_b"])

    p1: list[str] = []
    if split_disagree:
        p1.append(f"adjudicated split {gold_a}/{gold_b}" if adjudicated else f"labelers split {gold_a}/{gold_b}")
    if has_a and not has_b:
        p1.append("single labeler only (pass B missing)")
    elif has_b and not has_a:
        p1.append("single labeler only (pass A missing)")
    elif not has_a and not has_b:
        p1.append("unlabeled, needs human label")

    has_src = _has_source(r["_source_label"])
    src_unsafe = has_src and float(r["_source_label"]) == 1.0
    src_safe = has_src and float(r["_source_label"]) == 0.0
    if src_unsafe and gold == "benign":
        p1.append("source says unsafe, draft benign")
    if src_unsafe and gold == "out_of_scope":
        p1.append("source says unsafe, draft out_of_scope")
    if src_safe and gold == "dangerous":
        p1.append("source says safe, draft dangerous")
    if gold == "ambiguous":
        p1.append("ambiguous")
    if confidence == "low":
        p1.append("low confidence")

    if p1:
        priority, reason = 1, ";".join(p1)
    else:
        p2: list[str] = []
        if confidence == "medium":
            p2.append("medium confidence")
        if has_a and has_b and r["tags_a"] != r["tags_b"]:
            p2.append(f"tag mismatch {r['tags_a']}/{r['tags_b']}")
        if gold == "out_of_scope":
            p2.append("out_of_scope check")
        if not has_src and gold == "dangerous":
            p2.append("no source label, draft dangerous")
        if p2:
            priority, reason = 2, ";".join(p2)
        else:
            priority, reason = 3, "agree, high confidence, consistent with source"

    return {
        "gold": gold, "tags": tags, "notes": notes,
        "_confidence": confidence, "_alt_label": alt_label,
        "_review_priority": priority, "_review_reason": reason,
    }


# ------------------------------------------------------------ merge core --

def merge_all(
    sheet_df: pd.DataFrame,
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    adjudications_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per sheet id, carrying both passes' raw labels, the pool's source
    label, any adjudication, and the computed draft gold/tags/notes/priority/reason."""
    base = sheet_df[["id", "text", "split"]].copy()
    base["id"] = base["id"].astype(str)

    a_ids, b_ids = set(a_df["id"]), set(b_df["id"])
    base["has_a"] = base["id"].isin(a_ids)
    base["has_b"] = base["id"].isin(b_ids)

    a_small = a_df.rename(columns={
        "gold": "gold_a", "tags": "tags_a", "confidence": "conf_a", "alt_label": "alt_a", "notes": "notes_a",
    })
    b_small = b_df.rename(columns={
        "gold": "gold_b", "tags": "tags_b", "confidence": "conf_b", "alt_label": "alt_b", "notes": "notes_b",
    })

    merged = base.merge(a_small, on="id", how="left").merge(b_small, on="id", how="left")
    for c in ("gold_a", "tags_a", "conf_a", "alt_a", "notes_a", "gold_b", "tags_b", "conf_b", "alt_b", "notes_b"):
        merged[c] = merged[c].fillna("")

    if not pool_df.empty:
        pool_small = pool_df[["id", "source", "source_label"]].copy()
        pool_small["id"] = pool_small["id"].astype(str)
        pool_small = pool_small.rename(columns={"source": "_source", "source_label": "_source_label"})
        merged = merged.merge(pool_small, on="id", how="left")
    else:
        merged["_source"] = ""
        merged["_source_label"] = np.nan
    merged["_source"] = merged["_source"].fillna("")

    merged["adjudicated"] = False
    merged["adj_gold"] = ""
    merged["adj_tags"] = ""
    merged["adj_notes"] = ""
    if adjudications_df is not None and not adjudications_df.empty:
        adj = adjudications_df.rename(columns={"gold": "_adj_g", "tags": "_adj_t", "notes": "_adj_n"})
        merged = merged.merge(adj[["id", "_adj_g", "_adj_t", "_adj_n"]], on="id", how="left")
        has_adj = merged["_adj_g"].notna()
        merged.loc[has_adj, "adjudicated"] = True
        merged.loc[has_adj, "adj_gold"] = merged.loc[has_adj, "_adj_g"]
        merged.loc[has_adj, "adj_tags"] = merged.loc[has_adj, "_adj_t"].fillna("")
        merged.loc[has_adj, "adj_notes"] = merged.loc[has_adj, "_adj_n"].fillna("")
        merged = merged.drop(columns=["_adj_g", "_adj_t", "_adj_n"])

    computed = merged.apply(lambda r: pd.Series(compute_row(r)), axis=1)
    return pd.concat([merged.reset_index(drop=True), computed.reset_index(drop=True)], axis=1)


# ------------------------------------------------------------- outputs ----

def to_draft_sheet(merged: pd.DataFrame, sheet_df: pd.DataFrame) -> pd.DataFrame:
    sheet_cols = list(sheet_df.columns)
    keep = [c for c in sheet_cols if c not in ("gold", "tags", "notes", "labeler_pass", "labeled_at")]
    out = sheet_df[keep].copy()
    out["id"] = out["id"].astype(str)
    extra = ["id", "gold", "tags", "notes", "_review_priority", "_review_reason", "_confidence",
             "_alt_label", "_source", "_source_label"]
    out = out.merge(merged[extra], on="id", how="left")
    out["labeler_pass"] = 1
    out["labeled_at"] = ""
    ordered = sheet_cols + ["_review_priority", "_review_reason", "_confidence", "_alt_label",
                            "_source", "_source_label"]
    out = out[ordered]
    return out.sort_values(["_review_priority", "_source", "id"]).reset_index(drop=True)


def to_pass_gold(merged: pd.DataFrame, pass_num: int, labeled_at: str) -> pd.DataFrame:
    if pass_num == 1:
        sub = merged[merged["has_a"]].copy()
        sub["gold"], sub["tags"], sub["notes"] = sub["gold_a"], sub["tags_a"], sub["notes_a"]
    else:
        sub = merged[merged["has_b"]].copy()
        sub["gold"], sub["tags"], sub["notes"] = sub["gold_b"], sub["tags_b"], sub["notes_b"]
    sub["source"] = sub["_source"]
    sub["labeler_pass"] = pass_num
    sub["labeled_at"] = labeled_at
    return sub[list(GOLD_CSV_COLUMNS)].sort_values("id").reset_index(drop=True)


def to_disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    m = merged[merged["has_a"] & merged["has_b"] & (merged["gold_a"] != merged["gold_b"]) & (~merged["adjudicated"])]
    return m[list(DISAGREEMENT_COLUMNS)].sort_values(["split", "id"]).reset_index(drop=True)


# ------------------------------------------------------------- summary ----

def summarize(merged: pd.DataFrame) -> dict:
    n = len(merged)
    n_single = int((merged["has_a"] ^ merged["has_b"]).sum())
    n_unlabeled = int((~merged["has_a"] & ~merged["has_b"]).sum())
    n_both = int((merged["has_a"] & merged["has_b"]).sum())

    ab = merged.loc[merged["has_a"] & merged["has_b"], ["gold_a", "gold_b"]].rename(
        columns={"gold_a": "gold_1", "gold_b": "gold_2"})
    in_scope = ab[ab["gold_1"].isin(GOLD_LABELS) & ab["gold_2"].isin(GOLD_LABELS)]
    raw_agreement = float((in_scope["gold_1"] == in_scope["gold_2"]).mean()) if len(in_scope) else float("nan")
    kappa = A.kappa_overall(ab) if len(ab) else float("nan")

    reasons = merged["_review_reason"].str.split(";").explode()
    reasons = reasons[reasons != ""]
    top_reasons = reasons.value_counts().head(5)

    return {
        "n_rows": n,
        "n_both_passes": n_both,
        "n_single_labeled": n_single,
        "n_unlabeled": n_unlabeled,
        "label_counts": merged["gold"].value_counts(dropna=False).to_dict(),
        "raw_agreement": raw_agreement,
        "kappa": kappa,
        "priority_counts": merged["_review_priority"].value_counts().sort_index().to_dict(),
        "top_reasons": top_reasons,
    }


def print_summary(split: str, merged: pd.DataFrame) -> None:
    s = summarize(merged)
    print(f"\n== {split} ({s['n_rows']} rows) ==")
    print(f"  both passes: {s['n_both_passes']}  single-labeled: {s['n_single_labeled']}  "
          f"unlabeled: {s['n_unlabeled']}")
    print(f"  draft label counts: {s['label_counts']}")
    print(f"  A/B raw agreement: {s['raw_agreement']:.3f}  Cohen's kappa: {s['kappa']:.3f}")
    print(f"  review priority counts: {s['priority_counts']}")
    print("  top reasons:")
    for reason, count in s["top_reasons"].items():
        print(f"    {count:4d}  {reason}")


# --------------------------------------------------------------------- CLI -

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets-dir", default="data/gold/sheets")
    ap.add_argument("--labels-glob-a", default="data/gold/work/A_batch*_labels.csv")
    ap.add_argument("--labels-glob-b", default="data/gold/work/B_batch*_labels.csv")
    ap.add_argument("--pools-dir", default="data/processed_upload")
    ap.add_argument("--adjudications", default=None)
    ap.add_argument("--out-dir", default="data/gold/sheets")
    ap.add_argument("--work-dir", default="data/gold/work")
    a = ap.parse_args()

    sheets = {s: read_sheet(f"{a.sheets_dir}/{s}_pass1.csv") for s in SPLITS}
    for s, df in sheets.items():
        check_no_duplicates(df, f"sheet[{s}]")
    all_ids: set[str] = set()
    for df in sheets.values():
        all_ids |= set(df["id"])

    a_df, b_df = read_labels_glob(a.labels_glob_a), read_labels_glob(a.labels_glob_b)
    check_no_duplicates(a_df, "pass A labels")
    check_no_duplicates(b_df, "pass B labels")
    check_ids_known(a_df, "pass A labels", all_ids)
    check_ids_known(b_df, "pass B labels", all_ids)

    adjudications_df = read_adjudications(a.adjudications)
    if adjudications_df is not None:
        check_no_duplicates(adjudications_df, "adjudications")
        check_ids_known(adjudications_df, "adjudications", all_ids)

    labeled_at = dt.date.today().isoformat()
    out_dir, work_dir = Path(a.out_dir), Path(a.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    all_disagreements = []
    for s in SPLITS:
        sheet_df = sheets[s]
        ids_s = set(sheet_df["id"])
        a_s = a_df[a_df["id"].isin(ids_s)].reset_index(drop=True)
        b_s = b_df[b_df["id"].isin(ids_s)].reset_index(drop=True)
        adj_s = (adjudications_df[adjudications_df["id"].isin(ids_s)].reset_index(drop=True)
                 if adjudications_df is not None else None)
        pool_df = read_pool(str(Path(a.pools_dir) / f"{s}_pool.parquet"))

        merged = merge_all(sheet_df, a_s, b_s, pool_df, adj_s)

        to_draft_sheet(merged, sheet_df).to_csv(out_dir / f"{s}_opus_draft.csv", index=False)
        to_pass_gold(merged, 1, labeled_at).to_csv(work_dir / f"{s}_passA_gold.csv", index=False)
        to_pass_gold(merged, 2, labeled_at).to_csv(work_dir / f"{s}_passB_gold.csv", index=False)
        all_disagreements.append(to_disagreements(merged))

        print_summary(s, merged)

    disagreements = (pd.concat(all_disagreements, ignore_index=True) if all_disagreements
                      else pd.DataFrame(columns=list(DISAGREEMENT_COLUMNS)))
    disagreements.to_csv(work_dir / "disagreements_for_adjudication.csv", index=False)
    print(f"\nwrote {len(disagreements)} rows needing adjudication -> "
          f"{work_dir / 'disagreements_for_adjudication.csv'}")


if __name__ == "__main__":
    main()

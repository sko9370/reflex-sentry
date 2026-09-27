import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflex_sentry.gold import GOLD_CSV_COLUMNS, draft_merge as D  # noqa: E402
from reflex_sentry.gold import ingest as I  # noqa: E402

SHEET_COLUMNS = ["id", "text", "split", "gold", "tags", "notes", "labeler_pass", "labeled_at",
                 "_ref_gold", "_ref_tags"]


def _sheet_row(i, text="placeholder request text"):
    return {"id": f"s{i:03d}", "text": text, "split": "val", "gold": "", "tags": "", "notes": "",
            "labeler_pass": "", "labeled_at": "", "_ref_gold": "", "_ref_tags": ""}


def _sheet(n):
    return pd.DataFrame([_sheet_row(i) for i in range(n)])[SHEET_COLUMNS]


def _label_row(i, gold, tags="", confidence="high", alt_label="", notes=""):
    return {"id": f"s{i:03d}", "gold": gold, "tags": tags, "confidence": confidence,
            "alt_label": alt_label, "notes": notes}


def _pool_row(i, source="src_a", source_label=0.0):
    return {"id": f"s{i:03d}", "source": source, "source_label": source_label}


# ------------------------------------------------------------------ merge -

def test_agreement_uses_a_label_and_a_tags():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "benign", tags="")])
    b = pd.DataFrame([_label_row(0, "benign", tags="")])
    pool = pd.DataFrame([_pool_row(0, source_label=0.0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["gold"] == "benign"
    assert row["_review_priority"] == 3
    assert row["_confidence"] == "high"


def test_split_flags_priority_one_and_uses_a_as_provisional():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    b = pd.DataFrame([_label_row(0, "benign", tags="")])
    pool = pd.DataFrame([_pool_row(0, source_label=np.nan)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["gold"] == "dangerous"  # provisional = A
    assert row["_review_priority"] == 1
    assert "labelers split dangerous/benign" in row["_review_reason"]
    assert row["_alt_label"] == "benign"  # B's differing gold


def test_source_says_unsafe_draft_benign_flags_priority_one():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "benign")])
    b = pd.DataFrame([_label_row(0, "benign")])
    pool = pd.DataFrame([_pool_row(0, source_label=1.0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["_review_priority"] == 1
    assert "source says unsafe, draft benign" in row["_review_reason"]


def test_source_says_safe_draft_dangerous_flags_priority_one():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    b = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    pool = pd.DataFrame([_pool_row(0, source_label=0.0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["_review_priority"] == 1
    assert "source says safe, draft dangerous" in row["_review_reason"]


def test_nan_source_dangerous_is_priority_two_no_source_check():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    b = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    pool = pd.DataFrame([_pool_row(0, source_label=np.nan)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["_review_priority"] == 2
    assert "no source label, draft dangerous" in row["_review_reason"]


def test_adjudication_overrides_and_keeps_priority_one_with_adjudicated_reason():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    b = pd.DataFrame([_label_row(0, "benign")])
    pool = pd.DataFrame([_pool_row(0, source_label=np.nan)])
    adj = pd.DataFrame([{"id": "s000", "gold": "benign", "tags": "", "notes": "reviewer: clearly benign"}])
    merged = D.merge_all(sheet, a, b, pool, adj)
    row = merged.iloc[0]
    assert row["gold"] == "benign"
    assert row["notes"] == "reviewer: clearly benign"
    assert row["_confidence"] == "adjudicated"
    assert row["_review_priority"] == 1
    assert "adjudicated split dangerous/benign" in row["_review_reason"]


def test_tag_mismatch_on_agreement_is_priority_two():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    b = pd.DataFrame([_label_row(0, "dangerous", tags="cat:evasion")])
    pool = pd.DataFrame([_pool_row(0, source_label=1.0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["gold"] == "dangerous"
    assert row["_review_priority"] == 2
    assert "tag mismatch cat:ddos/cat:evasion" in row["_review_reason"]


def test_single_labeler_only_pass_a():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "benign", confidence="medium")])
    b = pd.DataFrame(columns=D.LABEL_COLS)
    pool = pd.DataFrame([_pool_row(0, source_label=0.0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["gold"] == "benign"
    assert row["_confidence"] == "medium"
    assert row["_review_priority"] == 1
    assert "single labeler only (pass B missing)" in row["_review_reason"]


def test_single_labeler_only_pass_b():
    sheet = _sheet(1)
    a = pd.DataFrame(columns=D.LABEL_COLS)
    b = pd.DataFrame([_label_row(0, "dangerous", tags="cat:ddos")])
    pool = pd.DataFrame([_pool_row(0, source_label=1.0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["gold"] == "dangerous"
    assert row["_review_priority"] == 1
    assert "single labeler only (pass A missing)" in row["_review_reason"]


def test_unlabeled_both_passes():
    sheet = _sheet(1)
    a = pd.DataFrame(columns=D.LABEL_COLS)
    b = pd.DataFrame(columns=D.LABEL_COLS)
    pool = pd.DataFrame([_pool_row(0)])
    merged = D.merge_all(sheet, a, b, pool)
    row = merged.iloc[0]
    assert row["gold"] == "" and row["tags"] == "" and row["notes"] == ""
    assert row["_review_priority"] == 1
    assert row["_review_reason"] == "unlabeled, needs human label"


def test_priority_ordering_in_draft_sheet():
    sheet = _sheet(3)
    a = pd.DataFrame([
        _label_row(0, "benign"),                       # agree, high conf -> priority 3
        _label_row(1, "dangerous", tags="cat:ddos"),    # split -> priority 1
        _label_row(2, "benign", confidence="medium"),   # medium conf -> priority 2
    ])
    b = pd.DataFrame([
        _label_row(0, "benign"),
        _label_row(1, "benign"),
        _label_row(2, "benign", confidence="medium"),
    ])
    pool = pd.DataFrame([_pool_row(i, source_label=0.0) for i in range(3)])
    merged = D.merge_all(sheet, a, b, pool)
    draft = D.to_draft_sheet(merged, sheet)
    assert list(draft["_review_priority"]) == sorted(draft["_review_priority"])
    assert draft.iloc[0]["id"] == "s001"  # the priority-1 split row sorts first


# --------------------------------------------------------------- outputs --

def test_to_pass_gold_only_contains_ids_that_pass_labeled():
    sheet = _sheet(3)
    a = pd.DataFrame([_label_row(0, "benign"), _label_row(1, "benign")])
    b = pd.DataFrame([_label_row(1, "benign"), _label_row(2, "benign")])
    pool = pd.DataFrame([_pool_row(i, source_label=0.0) for i in range(3)])
    merged = D.merge_all(sheet, a, b, pool)
    pass_a = D.to_pass_gold(merged, 1, "2026-09-27")
    pass_b = D.to_pass_gold(merged, 2, "2026-09-27")
    assert set(pass_a["id"]) == {"s000", "s001"}
    assert set(pass_b["id"]) == {"s001", "s002"}
    assert list(pass_a.columns) == list(GOLD_CSV_COLUMNS)


def test_agreement_compare_handles_non_identical_id_sets():
    sheet = _sheet(4)
    a = pd.DataFrame([
        _label_row(0, "benign"),                        # A only
        _label_row(1, "dangerous", tags="cat:ddos"),     # in both, agree
        _label_row(2, "benign"),                         # in both, agree
    ])
    b = pd.DataFrame([
        _label_row(1, "dangerous", tags="cat:ddos"),     # in both, agree
        _label_row(2, "benign"),                         # in both, agree
        _label_row(3, "benign"),                         # B only
    ])
    pool = pd.DataFrame([_pool_row(i, source_label=0.0) for i in range(4)])
    merged = D.merge_all(sheet, a, b, pool)
    pass_a = D.to_pass_gold(merged, 1, "2026-09-27")
    pass_b = D.to_pass_gold(merged, 2, "2026-09-27")

    from reflex_sentry.gold import agreement as A
    m = A.compare(pass_a, pass_b)
    assert set(m["id"]) == {"s001", "s002"}  # inner join keeps only the intersection
    assert A.kappa_overall(m) == pytest.approx(1.0)


def test_disagreements_file_excludes_adjudicated_and_single_labeled():
    sheet = _sheet(3)
    a = pd.DataFrame([
        _label_row(0, "dangerous", tags="cat:ddos"),
        _label_row(1, "dangerous", tags="cat:ddos"),
        _label_row(2, "benign"),
    ])
    b = pd.DataFrame([
        _label_row(0, "benign"),
        _label_row(1, "benign"),
    ])
    pool = pd.DataFrame([_pool_row(i, source_label=np.nan) for i in range(3)])
    adj = pd.DataFrame([{"id": "s000", "gold": "benign", "tags": "", "notes": ""}])
    merged = D.merge_all(sheet, a, b, pool, adj)
    dis = D.to_disagreements(merged)
    assert list(dis["id"]) == ["s001"]  # s000 adjudicated away, s002 single-labeled


# --------------------------------------------------------- error handling -

def test_duplicate_id_within_pass_a_errors():
    sheet = _sheet(1)
    a = pd.DataFrame([_label_row(0, "benign"), _label_row(0, "dangerous", tags="cat:ddos")])
    with pytest.raises(D.DraftMergeError, match="duplicate ids"):
        D.check_no_duplicates(a, "pass A labels")


def test_id_not_in_any_sheet_errors():
    sheet = _sheet(1)
    known_ids = {"s000"}
    b = pd.DataFrame([_label_row(0, "benign"), _label_row(99, "benign")])
    with pytest.raises(D.DraftMergeError, match="not in any sheet"):
        D.check_ids_known(b, "pass B labels", known_ids)


# ------------------------------------------------------------ ingest e2e --

def test_draft_sheet_is_ingestible(tmp_path):
    sheet = _sheet(3)
    a = pd.DataFrame([
        _label_row(0, "dangerous", tags="cat:ddos", notes="clear malware ask"),
        _label_row(1, "benign", tags="hard_negative;hn:detection"),
        _label_row(2, "benign"),
    ])
    b = pd.DataFrame([
        _label_row(0, "dangerous", tags="cat:ddos"),
        _label_row(1, "benign", tags="hard_negative;hn:detection"),
        _label_row(2, "benign"),
    ])
    pool = pd.DataFrame([_pool_row(i, source_label=0.0) for i in range(3)])
    merged = D.merge_all(sheet, a, b, pool)
    draft = D.to_draft_sheet(merged, sheet)

    out = tmp_path / "val_opus_draft.csv"
    draft.to_csv(out, index=False)

    read_back = I.read_csv_sheet(str(out))
    normalized = I.normalize(read_back, default_pass=1, default_date="2026-09-27")
    I.validate(normalized, sample_ids=set(sheet["id"]))  # must not raise
    assert (normalized["labeled_at"] == "2026-09-27").all()
    assert set(normalized["gold"]) == {"dangerous", "benign"}

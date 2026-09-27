import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflex_sentry.gold import GOLD_LABELS, agreement as A  # noqa: E402
from reflex_sentry.gold import export as E  # noqa: E402
from reflex_sentry.gold import ingest as I  # noqa: E402
from reflex_sentry.gold import sample as S  # noqa: E402


# --------------------------------------------------------------- fixtures --

def _pool_row(i, source, source_label, tags="", dup_group=None):
    return {
        "id": f"p{i:03d}", "text": f"placeholder request text {i}", "source": source,
        "source_label": source_label, "source_category": "", "is_adversarial": False,
        "tags": tags, "origin_split": "train", "in_scope": True, "kw_hits": 1,
        "dup_group": dup_group or "", "split": "val",
    }


@pytest.fixture
def pool():
    rows = []
    i = 0
    for _ in range(6):
        rows.append(_pool_row(i, "src_a", 1.0)); i += 1
    for _ in range(4):
        rows.append(_pool_row(i, "src_a", np.nan)); i += 1
    for _ in range(10):
        rows.append(_pool_row(i, "src_b", 0.0)); i += 1
    for _ in range(10):
        rows.append(_pool_row(i, "hn_seed", 0.0, tags="hard_negative;hn:detection")); i += 1
    df = pd.DataFrame(rows)
    # two near-duplicates that should collapse to one representative
    df.loc[df["id"] == "p001", "dup_group"] = "dg1"
    df.loc[df["id"] == "p002", "dup_group"] = "dg1"
    return df


def _gold_row(i, text, source, gold, tags="", labeler_pass=1, labeled_at="2026-09-27", notes=""):
    return {"id": f"g{i:03d}", "text": text, "source": source, "gold": gold, "tags": tags,
            "labeler_pass": labeler_pass, "labeled_at": labeled_at, "notes": notes}


# ------------------------------------------------------------------ sample -

def test_sample_is_deterministic(pool):
    a = S.draw_sample(pool, n=15, seed=42)
    b = S.draw_sample(pool, n=15, seed=42)
    assert list(a["id"]) == list(b["id"])


def test_sample_different_seed_can_differ(pool):
    a = S.draw_sample(pool, n=15, seed=1)
    b = S.draw_sample(pool, n=15, seed=2)
    assert list(a["id"]) != list(b["id"])


def test_sample_dedups_dup_group(pool):
    out = S.draw_sample(pool, n=30, seed=1)  # ask for (almost) everything
    assert not {"p001", "p002"}.issubset(set(out["id"]))  # only one of the pair survives


def test_sample_stratification_matches_pool_share(pool):
    out = S.draw_sample(pool, n=20, seed=7)
    src_b_share_pool = (pool["source"] == "src_b").mean()
    src_b_share_sample = (out["source"] == "src_b").mean()
    assert abs(src_b_share_pool - src_b_share_sample) < 0.15


def test_allocate_sums_to_n_and_respects_capacity():
    sizes = {"a": 3, "b": 7, "c": 1}
    alloc = S.allocate(sizes, 6)
    assert sum(alloc.values()) == 6
    assert all(alloc[k] <= sizes[k] for k in sizes)


def test_blind_hides_source_columns(pool):
    drawn = S.draw_sample(pool, n=10, seed=3)
    blind = S.to_labeling_columns(drawn, "val", 1, blind=True)
    assert "source" not in blind.columns and "source_label" not in blind.columns
    visible = S.to_labeling_columns(drawn, "val", 1, blind=False)
    assert {"source", "source_label"} <= set(visible.columns)


# ------------------------------------------------------------------ export -

def test_export_spreadsheet_has_empty_label_columns_and_refs():
    sample = pd.DataFrame({"id": ["a1", "a2"], "text": ["t1", "t2"], "split": "val", "sample_pass": 1})
    sheet = E.to_spreadsheet(sample)
    assert (sheet["gold"] == "").all() and (sheet["tags"] == "").all() and (sheet["notes"] == "").all()
    assert "_ref_gold" in sheet.columns and "_ref_tags" in sheet.columns
    assert "dangerous" in sheet["_ref_gold"].tolist()
    assert "cat:malware_dev" in sheet["_ref_tags"].tolist()


def test_export_label_studio_tasks_respect_blindness():
    sample = pd.DataFrame({"id": ["a1"], "text": ["t1"], "split": "val", "sample_pass": 1})
    tasks = E.to_label_studio_tasks(sample)
    assert tasks[0]["data"]["item_id"] == "a1"
    assert "source" not in tasks[0]["data"]

    sample2 = sample.assign(source="src_a", source_label=1.0)
    tasks2 = E.to_label_studio_tasks(sample2)
    assert tasks2[0]["data"]["source"] == "src_a"


def test_label_studio_xml_has_taxonomy():
    xml = E.label_studio_xml()
    assert 'name="gold"' in xml and 'name="cat_tags"' in xml and 'name="hn_tags"' in xml
    assert "cat:ddos" in xml and "hn:cti" in xml


# ------------------------------------------------------------------ ingest -

def test_ingest_rejects_unknown_gold():
    df = pd.DataFrame([_gold_row(1, "t", "s", "maybe")])
    with pytest.raises(I.GoldValidationError, match="unknown gold"):
        I.validate(I.normalize(df, 1, "2026-09-27"))


def test_ingest_rejects_cat_tag_on_non_dangerous():
    df = pd.DataFrame([_gold_row(1, "t", "s", "benign", tags="cat:ddos")])
    with pytest.raises(I.GoldValidationError, match="requires gold=dangerous"):
        I.validate(I.normalize(df, 1, "2026-09-27"))


def test_ingest_rejects_hn_tag_without_hard_negative():
    df = pd.DataFrame([_gold_row(1, "t", "s", "benign", tags="hn:detection")])
    with pytest.raises(I.GoldValidationError, match="requires the hard_negative tag"):
        I.validate(I.normalize(df, 1, "2026-09-27"))


def test_ingest_rejects_missing_and_unknown_ids():
    df = pd.DataFrame([_gold_row(1, "t", "s", "benign")])
    with pytest.raises(I.GoldValidationError, match="not labeled"):
        I.validate(I.normalize(df, 1, "2026-09-27"), sample_ids={"g001", "g002"})
    with pytest.raises(I.GoldValidationError, match="not in sample"):
        I.validate(I.normalize(df, 1, "2026-09-27"), sample_ids={"g999"})


def test_ingest_accepts_valid_sheet_and_writes_gold_csv(tmp_path):
    df = pd.DataFrame([
        _gold_row(1, "t1", "s", "dangerous", tags="cat:ddos"),
        _gold_row(2, "t2", "s", "benign", tags="hard_negative;hn:detection"),
        _gold_row(3, "t3", "s", "out_of_scope"),
    ])
    norm = I.normalize(df, 1, "2026-09-27")
    I.validate(norm, sample_ids={"g001", "g002", "g003"})
    out = tmp_path / "val.csv"
    I.write_gold_csv(norm, out)
    back = pd.read_csv(out, dtype={"id": str})
    assert list(back["id"]) == ["g001", "g002", "g003"]  # sorted, out_of_scope kept


def test_build_merged_keeps_pool_hn_tags_when_gold_blank_and_drops_out_of_scope(pool):
    hn_row = pool.iloc[20]  # hn_seed rows start at index 20 in the fixture
    gold = pd.DataFrame([
        _gold_row(0, pool.iloc[0]["text"], "src_a", "dangerous", tags="cat:malware_dev"),
        _gold_row(1, hn_row["text"], "hn_seed", "benign", tags=""),  # blank tags
        _gold_row(2, pool.iloc[2]["text"], "src_a", "out_of_scope", tags=""),
    ])
    gold["id"] = [pool.iloc[0]["id"], hn_row["id"], pool.iloc[2]["id"]]
    merged = I.build_merged(pool, gold)
    assert set(merged["id"]) == {pool.iloc[0]["id"], hn_row["id"]}  # out_of_scope dropped
    row0 = merged[merged["id"] == pool.iloc[0]["id"]].iloc[0]
    assert row0["tags"] == "cat:malware_dev"
    row_hn = merged[merged["id"] == hn_row["id"]].iloc[0]
    assert row_hn["tags"] == "hard_negative;hn:detection"  # kept from the hn_seed pool row


def test_build_merged_gold_tags_override_pool_tags(pool):
    hn_row = pool.iloc[20]
    gold = pd.DataFrame([_gold_row(0, hn_row["text"], "hn_seed", "benign", tags="hard_negative;hn:ir")])
    gold["id"] = [hn_row["id"]]
    merged = I.build_merged(pool, gold)
    assert merged.iloc[0]["tags"] == "hard_negative;hn:ir"  # labeler's tags win, not the pool's hn:detection


# --------------------------------------------------------------- agreement -

def _kappa_ids():
    g1 = ["dangerous"] * 3 + ["benign"] * 3 + ["ambiguous"] * 3
    g2 = ["dangerous", "dangerous", "benign", "benign", "benign", "ambiguous", "ambiguous", "ambiguous", "dangerous"]
    p1 = pd.DataFrame([_gold_row(i, f"text {i}", "s", g, labeler_pass=1) for i, g in enumerate(g1)])
    p2 = pd.DataFrame([_gold_row(i, f"text {i}", "s", g, labeler_pass=2) for i, g in enumerate(g2)])
    return p1, p2


def test_kappa_matches_hand_computed_value():
    p1, p2 = _kappa_ids()
    merged = A.compare(p1, p2)
    # hand computation: 3x3 confusion is diag [2,2,2], all marginals 3/9;
    # po = 6/9, pe = 3*(1/3*1/3) = 1/3, kappa = (po-pe)/(1-pe) = 0.5
    assert A.kappa_overall(merged) == pytest.approx(0.5, abs=1e-9)


def test_confusion_matrix_matches_hand_computed_matrix():
    p1, p2 = _kappa_ids()
    cm = A.confusion(A.compare(p1, p2))
    expected = pd.DataFrame(
        [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
        index=[f"pass1_{g}" for g in GOLD_LABELS], columns=[f"pass2_{g}" for g in GOLD_LABELS],
    )
    pd.testing.assert_frame_equal(cm, expected, check_dtype=False)


def test_ambiguous_flip_rate_matches_hand_count():
    p1, p2 = _kappa_ids()
    r = A.ambiguous_flip_rate(A.compare(p1, p2))
    assert r["n_disagreements"] == 3
    assert r["n_ambiguous_flips"] == 2
    assert r["rate_of_disagreements"] == pytest.approx(2 / 3)
    assert r["rate_of_total"] == pytest.approx(2 / 9)


def test_make_pass2_sheet_hides_labels_and_shuffles(pool):
    pass1 = pd.DataFrame([_gold_row(i, f"placeholder text {i}", "s", "benign") for i in range(20)])
    sheet = A.make_pass2_sheet(pass1, seed=99)
    assert (sheet["gold"] == "").all() and (sheet["tags"] == "").all()
    assert set(sheet["id"]) == set(pass1["id"])
    assert list(sheet["id"]) != list(pass1["id"])  # order shuffled
    assert "source" not in sheet.columns  # blind by default

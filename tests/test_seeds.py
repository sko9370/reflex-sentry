"""Validation tests for seeds/hard_negatives.csv (the benign hard-negative seed set).

Run with: python -m pytest -q tests/test_seeds.py
"""
import csv
import re
from collections import Counter
from pathlib import Path

import pytest

SEEDS_PATH = Path(__file__).resolve().parents[1] / "seeds" / "hard_negatives.csv"

REQUIRED_COLUMNS = ["id", "text", "gold", "tags", "source", "notes"]
CATEGORIES = {"detection", "analysis", "vuln_explain", "secure_dev", "ir", "training", "cti"}
ID_RE = re.compile(r"^hn_(" + "|".join(sorted(CATEGORIES)) + r")_(\d{3})$")
MIN_PER_CATEGORY = 30
EM_DASH = "\u2014"


def _read_rows():
    with open(SEEDS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


@pytest.fixture(scope="module")
def rows():
    data, _ = _read_rows()
    return data


@pytest.fixture(scope="module")
def fieldnames():
    _, names = _read_rows()
    return names


def test_file_exists():
    assert SEEDS_PATH.exists(), f"missing {SEEDS_PATH}"


def test_required_columns(fieldnames):
    for col in REQUIRED_COLUMNS:
        assert col in fieldnames, f"missing required column {col}"


def test_ids_unique_and_match_pattern(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids found"
    for r in rows:
        m = ID_RE.match(r["id"])
        assert m, f"id {r['id']!r} does not match hn_<category>_<3-digit n>"


def test_gold_is_benign(rows):
    for r in rows:
        assert r["gold"] == "benign", f"{r['id']} has gold={r['gold']!r}, expected 'benign'"


def test_source_is_hn_seed(rows):
    for r in rows:
        assert r["source"] == "hn_seed", f"{r['id']} has source={r['source']!r}, expected 'hn_seed'"


def test_tags_match_category_and_id(rows):
    for r in rows:
        m = ID_RE.match(r["id"])
        assert m, f"id {r['id']!r} malformed"
        id_category = m.group(1)
        expected_tags = f"hard_negative;hn:{id_category}"
        assert r["tags"] == expected_tags, (
            f"{r['id']} has tags={r['tags']!r}, expected {expected_tags!r}"
        )


def test_non_empty_text(rows):
    for r in rows:
        assert r["text"] is not None and r["text"].strip() != "", f"{r['id']} has empty text"


def test_at_least_min_per_category(rows):
    counts = Counter()
    for r in rows:
        m = ID_RE.match(r["id"])
        counts[m.group(1)] += 1
    for cat in CATEGORIES:
        assert counts.get(cat, 0) >= MIN_PER_CATEGORY, (
            f"category {cat} has only {counts.get(cat, 0)} rows, need >= {MIN_PER_CATEGORY}"
        )


def test_no_exact_duplicates_after_normalization(rows):
    seen = {}
    dupes = []
    for r in rows:
        norm = re.sub(r"\s+", " ", r["text"].strip().lower())
        if norm in seen:
            dupes.append((r["id"], seen[norm]))
        else:
            seen[norm] = r["id"]
    assert not dupes, f"duplicate texts (after lowercasing/whitespace collapse) found: {dupes}"


def test_no_em_dash_in_any_field(rows, fieldnames):
    offenders = []
    for r in rows:
        for col in fieldnames:
            value = r.get(col) or ""
            if EM_DASH in value:
                offenders.append((r["id"], col))
    assert not offenders, f"em-dash (U+2014) found in fields: {offenders}"

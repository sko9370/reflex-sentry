"""Shared constants and validation for the reflex-sentry data pipeline.

Kept dependency-light (pandas only) since other agents import from here
(loaders.py, prefilter.py, dedupe.py, split.py, build.py, and the eval/model
code that reads data/processed/*.parquet).

See docs/DATA_CONTRACT.md for the full contract these constants encode.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

# Canonical source names used in the "source" column of every pool stage.
SOURCES = (
    "toxic_chat",
    "wildguardmix",
    "beavertails",
    "aegis2",
    "hh_redteam",
    "xstest",
    "or_bench",
    "hn_seed",
)

# data/interim/pool.parquet columns (one row per unique (source, text)).
POOL_COLUMNS = (
    "id",
    "text",
    "source",
    "source_label",
    "source_category",
    "is_adversarial",
    "tags",
    "origin_split",
)

# Extra columns added by prefilter.py + dedupe.py -> data/interim/cyber_pool.parquet
CYBER_EXTRA_COLUMNS = ("in_scope", "kw_hits", "dup_group")

# Extra column added by split.py -> data/processed/{train,val_pool,test_pool,test_ood_pool}.parquet
SPLIT_EXTRA_COLUMNS = ("split",)

SPLIT_NAMES = ("train", "val_pool", "test_pool", "test_ood_pool")

_ID_RE = re.compile(r"^[a-z0-9_]+:[0-9a-f]{12}$")


class SchemaError(ValueError):
    """Raised when a dataframe violates the pool/cyber_pool/split contract."""


def normalize_text(text: object) -> str:
    """NFKC-normalize, lowercase, and collapse whitespace. Used to build ids and to dedupe."""
    if text is None:
        return ""
    norm = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", norm).strip()


def make_id(source: str, text: str) -> str:
    """Stable id: '<source>:<first 12 hex of sha1(normalize_text(text))>'."""
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:12]
    return f"{source}:{digest}"


def required_columns(stage: str) -> list[str]:
    cols = list(POOL_COLUMNS)
    if stage in ("cyber", "split"):
        cols += list(CYBER_EXTRA_COLUMNS)
    if stage == "split":
        cols += list(SPLIT_EXTRA_COLUMNS)
    return cols


def validate_pool(df: pd.DataFrame, stage: str = "pool") -> None:
    """Raise SchemaError if df violates the contract for the given stage.

    stage is one of "pool", "cyber", "split", matching the three parquet
    families in docs/DATA_CONTRACT.md.
    """
    if stage not in ("pool", "cyber", "split"):
        raise ValueError(f"unknown stage {stage!r}")

    required = required_columns(stage)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{stage} dataframe missing required columns: {missing}")

    if df.empty:
        return

    bad_source = sorted(set(df["source"].unique()) - set(SOURCES))
    if bad_source:
        raise SchemaError(f"unknown source value(s): {bad_source}")

    if not df["id"].is_unique:
        dupes = df.loc[df["id"].duplicated(), "id"].unique()[:5].tolist()
        raise SchemaError(f"duplicate ids in {stage} dataframe, e.g. {dupes}")

    bad_ids = [i for i in df["id"].sample(min(50, len(df)), random_state=0) if not _ID_RE.match(str(i))]
    if bad_ids:
        raise SchemaError(f"id(s) not matching '<source>:<12 hex>' pattern, e.g. {bad_ids[:5]}")

    labels = df["source_label"].dropna()
    if not labels.empty and not labels.isin([0.0, 1.0]).all():
        bad = sorted(labels[~labels.isin([0.0, 1.0])].unique())[:5]
        raise SchemaError(f"source_label values must be 0.0, 1.0, or NaN, found: {bad}")

    if df["tags"].isna().any():
        raise SchemaError("tags column must not contain NaN; use '' for no tags")

    if stage == "split":
        bad_split = sorted(set(df["split"].dropna().unique()) - set(SPLIT_NAMES))
        if bad_split:
            raise SchemaError(f"unknown split value(s): {bad_split}")
        if df["split"].isna().any():
            raise SchemaError("split column must not contain NaN in a split-stage dataframe")

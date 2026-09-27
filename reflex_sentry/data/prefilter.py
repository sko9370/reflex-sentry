"""Keyword prefilter that narrows the normalized pool to cyber-relevant prompts.

Reads configs/cyber_keywords.txt: one term or phrase per line, '#' starts a
comment (inline or full-line), blank lines ignored. Matching is case
insensitive and word-boundary aware, so multiword phrases like "cobalt
strike" match as a unit and single words like "shell" do not match inside
"shellfish". This is a recall-oriented filter (README 4.1): it is meant to
catch defensive/IR vocabulary too, not just offensive terms, so downstream
labeling has both dangerous and hard-negative candidates to work with.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_keywords(path: str | Path) -> list[str]:
    words: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            words.append(line)
    if not words:
        raise ValueError(f"no keywords found in {path}")
    # longest first so multiword phrases take precedence over their substrings
    return sorted(set(words), key=len, reverse=True)


def compile_pattern(keywords: list[str]) -> re.Pattern:
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def prefilter(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Add in_scope (bool) and kw_hits (semicolon-joined matched keywords) columns."""
    pattern = compile_pattern(keywords)
    hits, in_scope = [], []
    for text in df["text"].astype(str):
        found = sorted({m.group(0).lower() for m in pattern.finditer(text)})
        hits.append(";".join(found))
        in_scope.append(bool(found))
    out = df.copy()
    out["in_scope"] = in_scope
    out["kw_hits"] = hits
    return out


def hit_counts_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source rows / in-scope hit counts, for build.py's summary."""
    g = df.groupby("source").agg(rows=("id", "size"), in_scope=("in_scope", "sum")).reset_index()
    g["in_scope_rate"] = (g["in_scope"] / g["rows"]).round(4)
    return g

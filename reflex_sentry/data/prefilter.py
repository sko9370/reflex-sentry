"""Keyword prefilter that narrows the normalized pool to cyber-relevant prompts.

Reads configs/cyber_keywords.txt: one term or phrase per line, '#' starts a
comment (inline or full-line), blank lines ignored. Matching is case
insensitive and word-boundary aware, so multiword phrases like "cobalt
strike" match as a unit and single words like "shell" do not match inside
"shellfish". This is a recall-oriented filter (README 4.1): it is meant to
catch defensive/IR vocabulary too, not just offensive terms, so downstream
labeling has both dangerous and hard-negative candidates to work with.

A line prefixed with "re:" is a raw regex fragment instead of a literal
phrase (e.g. `re:cve-\\d{4}-\\d{4,}`), for identifier formats that a plain
word list can't express. It is combined into the same case-insensitive,
word-boundary-wrapped alternation as every literal keyword, so it still
only matches on a word boundary and still shows up in kw_hits.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REGEX_PREFIX = "re:"


def load_keywords(path: str | Path) -> list[str]:
    """Return keyword/phrase lines lowercased, and `re:`-prefixed regex lines as-is.

    A regex line keeps its "re:" prefix in the returned list so compile_pattern can
    tell it apart from a literal phrase; its own case is preserved (not lowercased)
    since regex escapes like `\\d`/`\\b` are meaningful, but matching stays
    case-insensitive regardless because compile_pattern compiles with re.IGNORECASE.
    """
    words: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith(REGEX_PREFIX):
            words.append(REGEX_PREFIX + line[len(REGEX_PREFIX):].strip())
        else:
            words.append(line.lower())
    if not words:
        raise ValueError(f"no keywords found in {path}")
    # longest first so multiword phrases take precedence over their substrings
    return sorted(set(words), key=len, reverse=True)


def compile_pattern(keywords: list[str]) -> re.Pattern:
    fragments = []
    for k in keywords:
        if k.startswith(REGEX_PREFIX):
            fragments.append(k[len(REGEX_PREFIX):])
        else:
            fragments.append(re.escape(k))
    return re.compile(r"\b(?:" + "|".join(fragments) + r")\b", re.IGNORECASE)


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

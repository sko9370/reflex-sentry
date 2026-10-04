"""Keyword prefilter that narrows the normalized pool to cyber-relevant prompts.

Reads configs/cyber_keywords.txt: one term or phrase per line, '#' starts a
comment (inline or full-line), blank lines ignored. Matching is case
insensitive and word-boundary aware, so multiword phrases like "cobalt
strike" match as a unit and single words like "shell" do not match inside
"shellfish". This is a recall-oriented filter (docs/DATA_CONTRACT.md): it is meant to
catch defensive/IR vocabulary too, not just offensive terms, so downstream
labeling has both dangerous and hard-negative candidates to work with.

A line prefixed with "re:" is a raw regex fragment instead of a literal
phrase (e.g. `re:cve-\\d{4}-\\d{4,}`), for identifier formats that a plain
word list can't express. It is combined into the same case-insensitive,
word-boundary-wrapped alternation as every literal keyword, so it still
only matches on a word boundary and still shows up in kw_hits.

TWO-TIER MATCHING: a line prefixed "weak:" (optionally combined as
"weak:re:" for a weak regex) is a *weak* keyword rather than a strong one.
A row is in scope if it has at least 1 STRONG hit, OR at least 2 DISTINCT
WEAK hits -- a single weak hit alone is not enough. This lets ambiguous,
everyday-English cyber words ("vulnerable", "exploit", "breach", ...) stay
in the list for recall without each one alone dragging in unrelated rows:
two of them together in the same prompt is a much stronger signal than
either alone. `prefilter()` records this as `kw_strong`/`kw_weak` (distinct
hit counts per tier) alongside the existing `kw_hits` (the union of every
matched string, from either tier, same as before).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REGEX_PREFIX = "re:"
WEAK_PREFIX = "weak:"


def load_keywords(path: str | Path) -> list[str]:
    """Return keyword/phrase lines lowercased, and `re:`-prefixed regex lines as-is.

    A regex line keeps its "re:" prefix in the returned list so compile_pattern can
    tell it apart from a literal phrase; its own case is preserved (not lowercased)
    since regex escapes like `\\d`/`\\b` are meaningful, but matching stays
    case-insensitive regardless because compile_pattern compiles with re.IGNORECASE.

    A line prefixed "weak:" (before an optional "re:") keeps that "weak:" prefix too,
    so callers can split the list into tiers with `split_tiers` before compiling.
    """
    words: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tier_prefix = ""
        rest = line
        if rest.lower().startswith(WEAK_PREFIX):
            tier_prefix = WEAK_PREFIX
            rest = rest[len(WEAK_PREFIX):].strip()
        if rest.lower().startswith(REGEX_PREFIX):
            words.append(tier_prefix + REGEX_PREFIX + rest[len(REGEX_PREFIX):].strip())
        else:
            words.append(tier_prefix + rest.lower())
    if not words:
        raise ValueError(f"no keywords found in {path}")
    # longest first so multiword phrases take precedence over their substrings
    return sorted(set(words), key=len, reverse=True)


def split_tiers(keywords: list[str]) -> tuple[list[str], list[str]]:
    """Split a `load_keywords` list into (strong, weak), stripping the "weak:" prefix.

    Each returned list keeps whatever relative order it had in `keywords` (still
    longest-first within each tier, since that's a stable filter of an already
    length-sorted list), and each entry is either a plain lowercased phrase or a
    "re:"-prefixed regex fragment, exactly what `compile_pattern` expects.
    """
    strong, weak = [], []
    for k in keywords:
        if k.startswith(WEAK_PREFIX):
            weak.append(k[len(WEAK_PREFIX):])
        else:
            strong.append(k)
    return strong, weak


def compile_pattern(keywords: list[str]) -> re.Pattern:
    fragments = []
    for k in keywords:
        if k.startswith(REGEX_PREFIX):
            fragments.append(k[len(REGEX_PREFIX):])
        else:
            fragments.append(re.escape(k))
    return re.compile(r"\b(?:" + "|".join(fragments) + r")\b", re.IGNORECASE)


def _find_hits(pattern: re.Pattern | None, text: str) -> list[str]:
    if pattern is None:
        return []
    return sorted({m.group(0).lower() for m in pattern.finditer(text)})


def prefilter(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Add in_scope/kw_hits/kw_strong/kw_weak columns.

    in_scope: True if the row has >=1 strong hit, OR >=2 distinct weak hits.
    kw_hits: semicolon-joined union of every matched string (strong and weak both,
        same shape as before two-tier matching was added).
    kw_strong / kw_weak: count of distinct strong / weak hit strings for the row.
    """
    strong_kw, weak_kw = split_tiers(keywords)
    strong_pattern = compile_pattern(strong_kw) if strong_kw else None
    weak_pattern = compile_pattern(weak_kw) if weak_kw else None

    hits, in_scope, kw_strong, kw_weak = [], [], [], []
    for text in df["text"].astype(str):
        s_found = _find_hits(strong_pattern, text)
        w_found = _find_hits(weak_pattern, text)
        found = sorted(set(s_found) | set(w_found))
        hits.append(";".join(found))
        kw_strong.append(len(s_found))
        kw_weak.append(len(w_found))
        in_scope.append(len(s_found) >= 1 or len(w_found) >= 2)
    out = df.copy()
    out["in_scope"] = in_scope
    out["kw_hits"] = hits
    out["kw_strong"] = kw_strong
    out["kw_weak"] = kw_weak
    return out


def hit_counts_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source rows / in-scope hit counts, for build.py's summary."""
    g = df.groupby("source").agg(rows=("id", "size"), in_scope=("in_scope", "sum")).reset_index()
    g["in_scope_rate"] = (g["in_scope"] / g["rows"]).round(4)
    return g


def scope_reason(row_kw_strong: int, row_kw_weak: int, bypassed: bool) -> str:
    """Classify why a single row is in scope: "strong", "weak_pair", "bypass", or
    "none" (not in scope). Used by build.py's per-source scope-reason summary."""
    if row_kw_strong >= 1:
        return "strong"
    if row_kw_weak >= 2:
        return "weak_pair"
    if bypassed:
        return "bypass"
    return "none"


def scope_reason_counts_by_source(
    df: pd.DataFrame, bypass_sources: set[str] | None = None
) -> pd.DataFrame:
    """Per-source in-scope counts broken down by reason (strong / weak_pair / bypass).

    `df` must already have kw_strong/kw_weak (i.e. it's the output of `prefilter`),
    before any bypass override has been applied to `in_scope`. Rows from a
    `bypass_sources` source with no keyword hit at all are counted as "bypass";
    a bypass-source row that already had a strong/weak hit is counted under that
    hit's own reason, since it would have been in scope anyway.
    """
    bypass_sources = bypass_sources or set()
    reasons = [
        scope_reason(s, w, src in bypass_sources)
        for s, w, src in zip(df["kw_strong"], df["kw_weak"], df["source"])
    ]
    tmp = df[["source"]].copy()
    tmp["reason"] = reasons
    table = (
        tmp[tmp["reason"] != "none"]
        .groupby(["source", "reason"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ("strong", "weak_pair", "bypass"):
        if col not in table.columns:
            table[col] = 0
    table = table[["strong", "weak_pair", "bypass"]]
    table["total_in_scope"] = table.sum(axis=1)
    return table.reset_index()

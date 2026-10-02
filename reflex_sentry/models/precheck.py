"""A cheap, deterministic pre-check gate in front of the Stage B student.

Context: the fine-tuned Stage B student catches 94-98% of dangerous prompts
under semantic wrappers (roleplay, fiction, authority framing, ...) but 0%
under base64 and leetspeak encodings -- unreadable input maps to "safe". This
module was built after seeing that failure, so it is deliberately generic
(it looks for the *shape* of obfuscated or unreadable text, not for
base64/leetspeak by name) rather than two hand-written regexes for the two
known failures; `audit` below checks it against encodings the eval wrappers
never generate, to keep it honest about that distinction.

The pre-check never tries to *classify* a prompt as dangerous. It only
answers "can a downstream reader make sense of this text at all?" and, when
the answer is no, forces escalation through a separate routing flag. The
student's calibrated probabilities stay intact for ranking and calibration.

    python -m reflex_sentry.models.precheck apply --base stage_b \
        --splits val test test_ood test_evasion
    python -m reflex_sentry.models.precheck audit
"""
from __future__ import annotations

import argparse
import base64
import binascii
import codecs
import os
import re
import string
import time
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
COMMON_WORDS_PATH = Path(os.environ.get("REFLEX_SENTRY_COMMON_WORDS") or DATA_DIR / "common_words.txt")

DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_PREDS_DIR = "preds"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_INTERIM_DIR = "data/interim"
APPLY_SPLITS = ("val", "test", "test_ood", "test_evasion")


def _load_common_words() -> frozenset:
    if not COMMON_WORDS_PATH.exists():
        raise FileNotFoundError(
            f"Pre-check dictionary missing: {COMMON_WORDS_PATH}. "
            "Run python -m reflex_sentry.models.build_precheck_words "
            "--dictionary /path/to/en_US.dic, or set REFLEX_SENTRY_COMMON_WORDS."
        )
    words = frozenset(w.strip().lower() for w in COMMON_WORDS_PATH.read_text(encoding="utf-8").splitlines() if w.strip())
    if not words:
        raise ValueError(f"Pre-check dictionary is empty: {COMMON_WORDS_PATH}")
    return words


# A standard American English dictionary word list (~36k words), used only to
# measure how much of a prompt reads as ordinary English -- see
# detect_low_nl_ratio. Provenance, licenses, and construction method are
# documented in reflex_sentry/models/data/README.md; not downloaded, built
# entirely from files already present on the build machine.
COMMON_WORDS: frozenset | None = None


def _common_words() -> frozenset:
    global COMMON_WORDS
    if COMMON_WORDS is None:
        COMMON_WORDS = _load_common_words()
    return COMMON_WORDS

# =============================================================================
# Thresholds. Every number a detector compares against lives here, as a
# module constant with a one-line rationale, rather than inline in the
# detector -- so the whole "what counts as obfuscated" policy is readable in
# one place and tunable without touching detector logic.
# =============================================================================

# --- (a) encoded blobs -------------------------------------------------------
# A whitespace-free token this long, drawn only from an encoding alphabet, is
# not something that occurs by accident in English prose, code identifiers,
# or file paths (a 24+ character run mixing case or digits basically never
# is). Long enough to leave short tokens (UUID segments, hex colors, acronyms)
# alone; short enough to catch "ignore all safety rules" (27 base64 chars).
MIN_B64_RUN = 24
# 16 byte-pairs = 32 hex characters. This is also an MD5 digest's length,
# which is extremely common in benign security text (IOC lists, hn:analysis,
# hn:cti); see HASH_DIGEST_LENS -- tokens at those exact lengths are treated
# as hashes, not an encoded payload, however many appear in the prompt.
MIN_HEX_RUN = 32
HASH_DIGEST_LENS = frozenset({32, 40, 64})  # md5, sha1, sha256 hex digest lengths
# Base32's alphabet (A-Z2-7) is conventionally all upper-case; a run this
# long of nothing else is not something normal writing produces (English has
# no 24-letter all-caps words), but short enough to catch a brief command.
MIN_B32_RUN = 24
# A couple of %XX escapes shows up in an ordinary pasted URL; a prompt built
# mostly out of them is percent-encoding an instruction, not linking to one.
MIN_PERCENT_ESCAPES = 6
# Decode confirmation is optional (README-adjacent spec: "confirmed by
# successful decode to mostly printable text" is a *strengthening* signal,
# not a gate) -- an encoded blob that decodes to binary garbage is still
# unreadable input and still gets escalated, it just doesn't get the
# "->readable" annotation in the reason string.
PRINTABLE_CHARS = frozenset(string.printable)
PRINTABLE_RATIO_FOR_READABLE = 0.85

# --- (b) leetspeak / character substitution ---------------------------------
LEET_CHARS = frozenset("0134578@$")
# Below this many word-like tokens, a ratio is too noisy to trust (a 3-word
# prompt with one leet-looking token is not a pattern).
LEET_MIN_TOKENS = 6
# A genuinely leet-substituted sentence touches most of its words; normal
# text has isolated alphanumeric identifiers (sha256, log4j, win32, b2b, ...)
# at a much lower rate.
LEET_RATIO_THRESHOLD = 0.30

# --- (c) low natural-language ratio -----------------------------------------
# Only judged once there is enough alphabetic material to be meaningful: a
# short prompt or a code/log snippet (few or no pure-alphabetic tokens once
# identifiers, numbers, hashes, and paths are excluded by construction, see
# detect_low_nl_ratio) should not be flagged on this signal alone.
#
# These four were grid-searched (2026-09-28) against 2,000 out-of-scope
# benign prompts sampled from wildguardmix + or_bench (data/interim/pool.
# parquet, source_label 0, id not in data/interim/cyber_pool.parquet, fixed
# seed 0 -- see reflex_sentry.models.precheck.audit's "tuning_ood_benign"
# row) with the goal of a <=0.5% flag rate on that set while keeping the
# generalization encodings that lean on this detector (rot13, reversed,
# char_spacing) at >=95% catch and base64/leetspeak at 100%. The chosen
# values clear all of that with margin: 0.15% flag rate on the tuning set
# (3/2000) and 100% catch on every generalization encoding. Never tuned
# against ToxicChat (the easy-benign test slice is drawn from it; audit's
# separate "toxic_chat_ordinary_benign" row, same selection rule, is
# report-only -- it came in at 0.8%, confirming these thresholds generalize
# rather than having been fit to this one tuning set).
NL_MIN_ALPHA_TOKENS = 8
NL_WORD_RATIO_THRESHOLD = 0.20
# Two- and single-letter tokens ("ok", "hi", "vs", "id", "ip", "re", ...) are
# disproportionately either abbreviations absent from the dictionary or too
# short to carry a real "is this English" signal; requiring 3+ characters
# removes them from both the numerator and denominator.
NL_MIN_TOKEN_LEN = 3
# A token that starts a sentence is capitalized regardless of whether it is a
# common word (already handled: membership is checked on the lowercased
# form), but a capitalized token *mid*-sentence is usually a proper noun --
# a person, product, or place name a general-purpose dictionary will never
# contain. Counting those against the ratio penalizes ordinary prompts that
# simply mention a name (Kaggle, ModernBERT, OpenAI) or a country/person.
NL_IGNORE_MIDSENTENCE_CAPITALIZED = True

# --- (d) spacing and homoglyphs ----------------------------------------------
# "s p a c e d  l e t t e r s" needs a run of single-letter tokens; 6 in a
# row spells at least a short word and will not fire on "a b" style short
# enumerations or lettered list items.
SPACING_MIN_RUN = 6
# One mixed-script token (a smart quote, an accented loanword copy-pasted
# from somewhere) is normal; two or more Latin/non-Latin mixed tokens in one
# prompt is the homoglyph-substitution pattern (Cyrillic "a" for Latin "a").
HOMOGLYPH_MIN_MIXED_TOKENS = 2

# =============================================================================
# Shared text helpers
# =============================================================================

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _strip_code(text: str) -> str:
    """Drop fenced and inline code spans before scanning for obfuscation.
    Benign security prompts routinely paste code, stack traces, and log
    lines; those are not obfuscated instructions and should never feed any
    detector below."""
    text = _FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return text


def _tokens(text: str) -> list:
    return text.split()


def _strip_punct(tok: str) -> str:
    return tok.strip(string.punctuation)


# =============================================================================
# (a) encoded blobs
# =============================================================================

def _try_decode_readable(kind: str, tok: str) -> bool:
    try:
        if kind == "base64":
            pad = "=" * (-len(tok) % 4)
            raw = base64.b64decode(tok + pad, validate=False)
        elif kind == "base32":
            pad = "=" * (-len(tok) % 8)
            raw = base64.b32decode(tok.upper() + pad, casefold=True)
        elif kind == "hex":
            raw = bytes.fromhex(tok)
        else:
            return False
    except (binascii.Error, ValueError):
        return False
    if not raw:
        return False
    text = raw.decode("utf-8", errors="ignore")
    if not text:
        return False
    printable = sum(1 for c in text if c in PRINTABLE_CHARS)
    return printable / len(text) >= PRINTABLE_RATIO_FOR_READABLE


def detect_encoded_blob(text: str) -> tuple:
    """Long whitespace-free runs from the base64/base32/hex alphabets, or
    several %XX percent-encoding escapes. A single hash-length hex token (or
    any number of them -- an IOC list) is treated as an identifier, not an
    encoded instruction (HASH_DIGEST_LENS)."""
    clean = _strip_code(text)
    hits = []

    for raw_tok in _tokens(clean):
        tok = _strip_punct(raw_tok)
        if len(tok) < min(MIN_B64_RUN, MIN_HEX_RUN, MIN_B32_RUN):
            continue

        if len(tok) >= MIN_HEX_RUN and re.fullmatch(r"[0-9A-Fa-f]+", tok):
            if len(tok) in HASH_DIGEST_LENS:
                continue  # a hash (or a list of them) is not an encoded instruction
            readable = _try_decode_readable("hex", tok)
            hits.append(f"hex[{len(tok)}]" + (" (decodes to text)" if readable else ""))
            continue

        core = tok.rstrip("=")
        if len(core) >= MIN_B32_RUN and re.fullmatch(r"[A-Z2-7]+", core):
            readable = _try_decode_readable("base32", core)
            hits.append(f"base32[{len(core)}]" + (" (decodes to text)" if readable else ""))
            continue

        if len(tok) >= MIN_B64_RUN and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", tok):
            has_digit = any(c.isdigit() for c in tok)
            has_mixed_case = any(c.islower() for c in tok) and any(c.isupper() for c in tok)
            if has_digit or has_mixed_case:
                readable = _try_decode_readable("base64", tok.rstrip("="))
                hits.append(f"base64[{len(tok)}]" + (" (decodes to text)" if readable else ""))

    n_percent = len(re.findall(r"%[0-9A-Fa-f]{2}", clean))
    if n_percent >= MIN_PERCENT_ESCAPES:
        hits.append(f"percent_encoding[{n_percent} escapes]")

    if hits:
        return True, "encoded blob: " + ", ".join(hits[:3])
    return False, None


# =============================================================================
# (b) leetspeak / character substitution
# =============================================================================

_LEET_TOKEN_RE = re.compile(r"^[A-Za-z0-9@$]+$")
_PURE_HEX_RE = re.compile(r"^[0-9A-Fa-f]+$")
# A token drawn only from the hex alphabet (0-9a-f) and this long is almost
# always a hash, id, or color, not a word -- and it trivially "mixes letters
# with digits" the same way every hex string does, which would otherwise
# make hash lists (a common hn:analysis/hn:cti hard negative) look like
# heavy leetspeak. Below this length, ordinary short hex-alphabet words
# ("cab", "dead", "beef", "cafe", ...) stay eligible.
HEX_LOOKING_MIN_LEN = 8


def _is_word_like(tok: str) -> bool:
    if not (_LEET_TOKEN_RE.fullmatch(tok) and len(tok) >= 3 and any(c.isalpha() for c in tok)):
        return False
    if len(tok) >= HEX_LOOKING_MIN_LEN and _PURE_HEX_RE.fullmatch(tok):
        return False
    return True


def _is_leet_mixed(tok: str) -> bool:
    return any(c.isalpha() for c in tok) and any(c in LEET_CHARS for c in tok)


def detect_leetspeak(text: str) -> tuple:
    """Share of word-like tokens that mix letters with digit/symbol
    stand-ins (0,1,3,4,5,7,@,$), above LEET_RATIO_THRESHOLD, with a minimum
    token count so a single "l33t" in an otherwise normal sentence doesn't
    trip it."""
    clean = _strip_code(text)
    toks = [_strip_punct(t) for t in _tokens(clean)]
    word_like = [t for t in toks if _is_word_like(t)]
    if len(word_like) < LEET_MIN_TOKENS:
        return False, None
    mixed = [t for t in word_like if _is_leet_mixed(t)]
    ratio = len(mixed) / len(word_like)
    if ratio >= LEET_RATIO_THRESHOLD:
        return True, f"leetspeak: {len(mixed)}/{len(word_like)} word-like tokens ({ratio:.0%}) mix letters with leet substitutions"
    return False, None


# =============================================================================
# (c) low natural-language ratio
# =============================================================================

_ALPHA_ONLY_RE = re.compile(r"^[A-Za-z]+$")


def _looks_like_midsentence_name(tok: str) -> bool:
    """Title-case ("Kaggle", "Microsoft"), not all-caps (an acronym stays in
    the ratio, e.g. "NIST", "EDR" -- ALLCAPS tokens are rare enough in
    common-word text that they are not worth a special case either way)."""
    return len(tok) >= 2 and tok[0].isupper() and tok[1:].islower()


def detect_low_nl_ratio(text: str) -> tuple:
    """Share of purely-alphabetic tokens that are common English words,
    below NL_WORD_RATIO_THRESHOLD, only judged once there are at least
    NL_MIN_ALPHA_TOKENS such tokens. Identifiers, hashes, IPs, CVE ids, and
    file paths are excluded from the denominator by construction: none of
    them are purely alphabetic once their digits/punctuation are counted.
    Tokens shorter than NL_MIN_TOKEN_LEN, and (if
    NL_IGNORE_MIDSENTENCE_CAPITALIZED) capitalized tokens that do not start a
    sentence, are excluded too -- see the constants' rationale above."""
    words = _common_words()
    clean = _strip_code(text)
    raw_toks = _tokens(clean)
    alpha_toks = []
    sentence_start = True
    for raw in raw_toks:
        core = _strip_punct(raw)
        if _ALPHA_ONLY_RE.fullmatch(core) and len(core) >= NL_MIN_TOKEN_LEN:
            skip_name = NL_IGNORE_MIDSENTENCE_CAPITALIZED and not sentence_start and _looks_like_midsentence_name(core)
            if not skip_name:
                alpha_toks.append(core)
        sentence_start = bool(re.search(r"[.!?]$", raw))
    if len(alpha_toks) < NL_MIN_ALPHA_TOKENS:
        return False, None
    common = sum(1 for t in alpha_toks if t.lower() in words)
    ratio = common / len(alpha_toks)
    if ratio < NL_WORD_RATIO_THRESHOLD:
        return True, f"low natural-language ratio: {common}/{len(alpha_toks)} alphabetic tokens ({ratio:.0%}) are common English words"
    return False, None


# =============================================================================
# (d) spacing and homoglyphs
# =============================================================================

def detect_spacing(text: str) -> tuple:
    """A run of SPACING_MIN_RUN or more consecutive single-letter
    whitespace-delimited tokens ("s p a c e d  l e t t e r s")."""
    clean = _strip_code(text)
    run = best = 0
    for t in _tokens(clean):
        core = _strip_punct(t)
        if len(core) == 1 and core.isalpha():
            run += 1
            best = max(best, run)
        else:
            run = 0
    if best >= SPACING_MIN_RUN:
        return True, f"character spacing: {best} consecutive single-letter tokens"
    return False, None


def _is_mixed_script_token(tok: str) -> bool:
    has_ascii_letter = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in tok)
    has_other_letter = any(ord(c) > 127 and unicodedata.category(c).startswith("L") for c in tok)
    return has_ascii_letter and has_other_letter


def detect_homoglyph(text: str) -> tuple:
    """HOMOGLYPH_MIN_MIXED_TOKENS or more tokens that mix ASCII Latin
    letters with non-ASCII letters in the same token (e.g. Cyrillic "a"
    standing in for Latin "a") -- one is normal (a stray smart character),
    several in one prompt is the homoglyph-substitution pattern."""
    clean = _strip_code(text)
    mixed = [t for t in _tokens(clean) if _is_mixed_script_token(t)]
    if len(mixed) >= HOMOGLYPH_MIN_MIXED_TOKENS:
        return True, f"homoglyph/mixed-script: {len(mixed)} tokens mix Latin and non-Latin letters"
    return False, None


# =============================================================================
# Aggregation
# =============================================================================

DETECTORS: dict = {
    "encoded_blob": detect_encoded_blob,
    "leetspeak": detect_leetspeak,
    "low_nl_ratio": detect_low_nl_ratio,
    "spacing": detect_spacing,
    "homoglyph": detect_homoglyph,
}


def obfuscation_signals(text: str) -> dict:
    """Run every detector independently (each one only looks at the raw
    text; none depends on another's outcome). Returns
    {detector_name: {"flag": bool, "reason": str | None}}."""
    text = "" if text is None else str(text)
    out = {}
    for name, fn in DETECTORS.items():
        flag, reason = fn(text)
        out[name] = {"flag": bool(flag), "reason": reason}
    return out


def precheck(text: str) -> tuple:
    """(flagged, reasons): flagged is True if any detector fired; reasons is
    the list of human-readable explanations from the detectors that fired,
    in a stable (dict-insertion) order."""
    sig = obfuscation_signals(text)
    reasons = [v["reason"] for v in sig.values() if v["flag"] and v["reason"]]
    return (len(reasons) > 0, reasons)


# =============================================================================
# CLI: apply
# =============================================================================

def apply_precheck(base: str, splits: Sequence = APPLY_SPLITS, preds_dir="preds",
                    processed_dir="data/processed") -> dict:
    """For every split with both preds/<base>_<split>.csv and
    data/processed/<split>.parquet, join predictions to text on `id`, run
    the pre-check per row, retain the base probabilities, and write a
    force_escalate policy column to preds/<base>_pc_<split>.csv. No logits
    file is written for the "_pc" model, so run_all
    (reflex_sentry.eval.run_all) treats it as already-calibrated predictions.
    Returns {split: {"path": ..., "n": ..., "n_flagged": ...}} for splits
    actually written."""
    preds_dir, processed_dir = Path(preds_dir), Path(processed_dir)
    # Load once before timing; dictionary I/O is startup cost, not per-prompt
    # overhead. Also fail before writing any split if setup is incomplete.
    _common_words()
    written = {}
    for split in splits:
        preds_path = preds_dir / f"{base}_{split}.csv"
        proc_path = processed_dir / f"{split}.parquet"
        if not preds_path.exists() or not proc_path.exists():
            continue

        preds = pd.read_csv(preds_path)
        proc = pd.read_parquet(proc_path, columns=["id", "text"])
        if preds["id"].isna().any() or proc["id"].isna().any():
            raise ValueError("Pre-check inputs contain null ids")
        if set(preds["id"]) != set(proc["id"]):
            raise ValueError(f"Prediction ids do not match {proc_path}; regenerate {preds_path}")
        merged = preds.merge(proc, on="id", how="left", validate="one_to_one")
        n_missing = int(merged["text"].isna().sum())
        if n_missing:
            raise ValueError(f"{n_missing} row(s) in {preds_path} have no matching id in {proc_path}")

        flags, reasons_col, pc_ms = [], [], []
        for text in merged["text"]:
            t0 = time.perf_counter()
            flagged, reasons = precheck(text)
            pc_ms.append((time.perf_counter() - t0) * 1000.0)
            flags.append(flagged)
            reasons_col.append(";".join(reasons))

        out = merged.drop(columns=["text"]).copy()
        out["precheck_flag"] = flags
        out["precheck_reasons"] = reasons_col
        out["force_escalate"] = flags

        base_latency = (out["latency_ms"].to_numpy(dtype=float) if "latency_ms" in out.columns
                         else np.full(len(out), np.nan))
        # Rows the base model did not time stay NaN (metrics.latency drops
        # NaNs); filling them with 0 would drag the reported p50 down.
        out["latency_ms"] = base_latency + np.asarray(pc_ms, dtype=float)

        out_path = preds_dir / f"{base}_pc_{split}.csv"
        out.to_csv(out_path, index=False)
        written[split] = {"path": str(out_path), "n": int(len(out)), "n_flagged": int(sum(flags))}
    return written


# =============================================================================
# CLI: audit  (generalization encodings -- deliberately NOT the eval wrappers)
# =============================================================================
# reflex_sentry.eval.wrappers ships roleplay/fiction/authority/research
# framing plus base64 and leetspeak transforms. To check the pre-check is a
# genuine "unreadable input" detector rather than something that memorized
# those two specific transforms, this audit encodes val's dangerous+benign
# text with a *different* set of encodings the wrappers never produce: hex,
# ROT13, URL percent-encoding, base32, character spacing, reversed text, and
# a homoglyph substitution. None of this code is shared with wrappers.py.

def _enc_hex(t: str) -> str:
    return t.encode("utf-8").hex()


def _enc_rot13(t: str) -> str:
    return codecs.encode(t, "rot13")


def _enc_url_percent(t: str) -> str:
    return urllib.parse.quote(t)


def _enc_base32(t: str) -> str:
    return base64.b32encode(t.encode("utf-8")).decode("ascii")


def _enc_char_spacing(t: str) -> str:
    return " ".join(t)


def _enc_reversed(t: str) -> str:
    return t[::-1]


_HOMOGLYPH_MAP = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р",
    "c": "с", "x": "х", "y": "у", "i": "і",
})


def _enc_homoglyph(t: str) -> str:
    return t.translate(_HOMOGLYPH_MAP)


GENERALIZATION_ENCODINGS: dict = {
    "hex": _enc_hex,
    "rot13": _enc_rot13,
    "url_percent": _enc_url_percent,
    "base32": _enc_base32,
    "char_spacing": _enc_char_spacing,
    "reversed": _enc_reversed,
    "homoglyph": _enc_homoglyph,
}


def _rate(texts) -> dict:
    texts = [t for t in texts]
    n = len(texts)
    if n == 0:
        return {"k": 0, "n": 0, "rate": float("nan")}
    k = sum(1 for t in texts if precheck(t)[0])
    return {"k": k, "n": n, "rate": k / n}


# =============================================================================
# Ordinary out-of-scope benign traffic (audit only -- never used by precheck
# or apply_precheck). Two disjoint samples of "clearly not a cyber prompt at
# all" benign text, both drawn the same way (same selection rule, same fixed
# seed, same size) so they are comparable:
#   - tuning_ood_benign: wildguardmix + or_bench. This is the set the low
#     natural-language-ratio thresholds above were grid-searched against.
#   - toxic_chat_ordinary_benign: ToxicChat. Report-only, NEVER used to pick
#     a threshold -- ToxicChat also backs the easy-benign slice of the real
#     test set, so tuning against it would leak into the eval.
# =============================================================================

OOS_TUNING_SOURCES = ("wildguardmix", "or_bench")
OOS_REPORT_SOURCES = ("toxic_chat",)
OOS_SAMPLE_SIZE = 2000
OOS_SAMPLE_SEED = 0


def _sample_oos_benign(interim_dir: Path, sources: Sequence, n: int, seed: int):
    """Rows from data/interim/pool.parquet whose source is in `sources`,
    source_label == 0 (benign), and whose id was dropped by the cyber scope
    filter (i.e. NOT in data/interim/cyber_pool.parquet) -- ordinary traffic
    that never made it into this project's cyber-scope pipeline at all.
    Returns None if the interim files are not present (e.g. a synthetic test
    tree), a fixed-seed sample of up to `n` texts otherwise."""
    pool_path, cyber_path = interim_dir / "pool.parquet", interim_dir / "cyber_pool.parquet"
    if not pool_path.exists() or not cyber_path.exists():
        return None
    pool = pd.read_parquet(pool_path, columns=["id", "text", "source", "source_label"])
    cyber_ids = set(pd.read_parquet(cyber_path, columns=["id"])["id"])
    cand = pool[pool["source"].isin(sources) & (pool["source_label"] == 0.0) & ~pool["id"].isin(cyber_ids)]
    if cand.empty:
        return cand["text"].astype(str)
    return cand.sample(n=min(n, len(cand)), random_state=seed)["text"].astype(str)


# A simple, cheap stopword/script heuristic for the audit's diagnostic
# language breakdown ONLY -- it is never used by any detector or by
# precheck/apply_precheck, and its job is only to explain where
# toxic_chat_ordinary_benign's flags come from (README section 1: this
# project is English-only by scope, so non-English text is expected to look
# "not natural-language" to detect_low_nl_ratio's English word list).
_LANG_STOPWORDS = {
    "es_pt": frozenset({"el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es",
                         "por", "para", "no", "se", "con", "como", "pero", "sus", "le", "ya",
                         "esta", "muy", "sin", "sobre", "tambien", "nao", "voce", "isso",
                         "mais", "muito", "foi", "pelo", "pela", "esse", "essa", "sao", "uma"}),
    "fr": frozenset({"le", "la", "les", "de", "des", "et", "un", "une", "est", "que", "pour",
                      "dans", "ne", "pas", "se", "ce", "qui", "sur", "avec", "comme", "mais",
                      "plus", "tout", "nous", "vous", "ils", "elles", "tres", "cette", "leur"}),
    "de": frozenset({"der", "die", "das", "und", "ist", "nicht", "ich", "du", "er", "sie",
                      "es", "ein", "eine", "zu", "mit", "auf", "fur", "von", "dem", "den",
                      "im", "sich", "auch", "aber", "wie", "was", "sind", "werden", "nach"}),
}
_LANG_MIN_HITS = 2  # a couple of stopword hits distinguishes a language from a stray loanword


def _is_latin_letter(c: str) -> bool:
    return ord(c) < 0x250  # Basic Latin + Latin-1 Supplement + Latin Extended-A/B


def classify_language(text: str) -> str:
    """"non_latin_script" if more than 30% of the text's letters fall
    outside the Latin ranges; otherwise the language (of es_pt/fr/de) whose
    stopword set the text overlaps most, if that overlap is at least
    _LANG_MIN_HITS; otherwise "en_or_other"."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    non_latin = sum(1 for c in letters if not _is_latin_letter(c))
    if non_latin / len(letters) > 0.3:
        return "non_latin_script"
    toks = set(re.findall(r"[a-zA-ZÀ-ɏ]+", text.lower()))
    lang, hits = max(((lang, len(toks & words)) for lang, words in _LANG_STOPWORDS.items()),
                      key=lambda kv: kv[1])
    return lang if hits >= _LANG_MIN_HITS else "en_or_other"


def audit(processed_dir="data/processed", reports_dir="reports", interim_dir=DEFAULT_INTERIM_DIR) -> dict:
    """Counts-only flag-rate audit (never prints or stores prompt text):
    train/val/test/test_ood benign and dangerous rows, val/test split by
    hard_negative tag, per-wrapper rates on test_evasion, a generalization
    check on encodings the eval wrappers never produce, and two ordinary
    out-of-scope benign samples (tuning_ood_benign, the set the low
    natural-language-ratio thresholds were tuned against, and
    toxic_chat_ordinary_benign, report-only with a language breakdown).
    Writes reports_dir/precheck_audit.md (aggregate numbers only) and
    returns the same numbers as a dict."""
    processed_dir, reports_dir, interim_dir = Path(processed_dir), Path(reports_dir), Path(interim_dir)
    results: dict = {}

    def _read(split, columns=None):
        path = processed_dir / f"{split}.parquet"
        return pd.read_parquet(path, columns=columns) if path.exists() else None

    train = _read("train", columns=["text", "source_label"])
    if train is not None and "source_label" in train.columns:
        results["train_benign"] = _rate(train.loc[train["source_label"] == 0.0, "text"])

    tuning_texts = _sample_oos_benign(interim_dir, OOS_TUNING_SOURCES, OOS_SAMPLE_SIZE, OOS_SAMPLE_SEED)
    if tuning_texts is not None:
        results["tuning_ood_benign"] = _rate(tuning_texts)

    toxic_texts = _sample_oos_benign(interim_dir, OOS_REPORT_SOURCES, OOS_SAMPLE_SIZE, OOS_SAMPLE_SEED)
    if toxic_texts is not None:
        results["toxic_chat_ordinary_benign"] = _rate(toxic_texts)
        langs = toxic_texts.map(classify_language)
        results["toxic_chat_ordinary_benign_by_language"] = {
            lang: _rate(toxic_texts[langs == lang]) for lang in sorted(set(langs))
        }

    for split in ("val", "test", "test_ood"):
        df = _read(split)
        if df is None:
            continue
        hn = df["tags"].fillna("").str.contains("hard_negative") if "tags" in df.columns \
            else pd.Series(False, index=df.index)
        benign = df["gold"] == "benign"
        results[f"{split}_benign_hard_negative"] = _rate(df.loc[benign & hn, "text"])
        results[f"{split}_benign_other"] = _rate(df.loc[benign & ~hn, "text"])
        results[f"{split}_dangerous"] = _rate(df.loc[df["gold"] == "dangerous", "text"])

    evasion = _read("test_evasion")
    if evasion is not None:
        per_wrapper = {}
        tags = evasion["tags"].fillna("") if "tags" in evasion.columns else pd.Series("", index=evasion.index)
        wrappers = sorted({t[len("evasion:"):] for s in tags for t in s.split(";") if t.startswith("evasion:")})
        for w in wrappers:
            mask = tags.map(lambda s, w=w: f"evasion:{w}" in [x.strip() for x in s.split(";")])
            per_wrapper[w] = _rate(evasion.loc[mask, "text"])
        results["test_evasion_by_wrapper"] = per_wrapper

    val = _read("val")
    if val is not None:
        base_texts = val.loc[val["gold"].isin(["dangerous", "benign"]), "text"].astype(str).tolist()
        generalization = {}
        for name, enc in GENERALIZATION_ENCODINGS.items():
            encoded = [enc(t) for t in base_texts]
            generalization[name] = _rate(encoded)
        results["generalization"] = generalization

    _write_audit_markdown(results, reports_dir / "precheck_audit.md")
    return results


def _fmt_rate(r) -> str:
    if r is None or r.get("n", 0) == 0:
        return "n/a (n=0)"
    return f"{r['rate']:.3f} ({r['k']}/{r['n']})"


def _audit_rows(results: dict) -> list:
    rows = []
    for key in ("train_benign",
                "tuning_ood_benign (wildguardmix+or_bench, used for tuning)",
                "toxic_chat_ordinary_benign (report only, never tuned)"):
        lookup = key.split(" ", 1)[0]
        if lookup in results:
            rows.append((key, results[lookup]))
    for lang, r in results.get("toxic_chat_ordinary_benign_by_language", {}).items():
        rows.append((f"toxic_chat_ordinary_benign:{lang}", r))
    for key in ("val_benign_hard_negative", "val_benign_other", "val_dangerous",
                "test_benign_hard_negative", "test_benign_other", "test_dangerous",
                "test_ood_benign_hard_negative", "test_ood_benign_other", "test_ood_dangerous"):
        if key in results:
            rows.append((key, results[key]))
    for w, r in results.get("test_evasion_by_wrapper", {}).items():
        rows.append((f"test_evasion:{w}", r))
    for name, r in results.get("generalization", {}).items():
        rows.append((f"generalization:{name}", r))
    return rows


def _write_audit_markdown(results: dict, path: Path) -> None:
    lines = ["# Pre-check flag-rate audit\n",
             "Counts only -- see reflex_sentry.models.precheck.audit. Higher is better on "
             "dangerous/generalization rows, lower is better on benign rows.\n",
             "| Slice | Flag rate (k/n) |", "|---|---|"]
    for name, r in _audit_rows(results):
        lines.append(f"| {name} | {_fmt_rate(r)} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def format_audit_table(results: dict) -> str:
    lines = [f"{'slice':40s} {'flag_rate':>20s}"]
    for name, r in _audit_rows(results):
        lines.append(f"{name:40s} {_fmt_rate(r):>20s}")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("apply", help="apply the pre-check on top of a base model's predictions")
    pa.add_argument("--base", required=True, help="base model name, e.g. stage_b")
    pa.add_argument("--splits", nargs="+", default=list(APPLY_SPLITS))
    pa.add_argument("--preds-dir", default=DEFAULT_PREDS_DIR)
    pa.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)

    pu = sub.add_parser("audit", help="counts-only flag-rate audit against benign/dangerous/evasion/generalization sets")
    pu.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    pu.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    pu.add_argument("--interim-dir", default=DEFAULT_INTERIM_DIR)

    a = ap.parse_args()
    if a.cmd == "apply":
        written = apply_precheck(a.base, a.splits, preds_dir=a.preds_dir, processed_dir=a.processed_dir)
        for split, info in written.items():
            print(f"{a.base}_pc_{split}: {info['n_flagged']}/{info['n']} flagged -> {info['path']}")
        if not written:
            print(f"no splits scored: no matching preds/{a.base}_<split>.csv + "
                  f"{a.processed_dir}/<split>.parquet pairs found")
    elif a.cmd == "audit":
        results = audit(processed_dir=a.processed_dir, reports_dir=a.reports_dir, interim_dir=a.interim_dir)
        print(format_audit_table(results))
        print(f"\nwrote {Path(a.reports_dir) / 'precheck_audit.md'}")


if __name__ == "__main__":
    main()

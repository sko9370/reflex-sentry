"""One loader per raw data source, normalizing each to the pool schema.

Every loader takes a raw directory (or, for hn_seed, a csv path) and returns
a DataFrame with the columns in schema.POOL_COLUMNS (no "id" duplication
across sources is attempted here; that happens once when pools are merged in
build.py, keeping "one row per unique (source, text)" true per loader too).

Schemas are assumed from public documentation, not verified against the real
files (no network access in this environment). Each loader is defensive: it
checks for the columns it needs and raises DataFormatError with the columns
it actually found, so a schema drift after `hf download` fails loudly instead
of silently mislabeling data. Verify against data/SOURCES.md after download.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import schema

_EXTS = (".parquet", ".json", ".jsonl", ".jsonl.gz", ".csv")


class DataFormatError(RuntimeError):
    """A raw file did not have the columns a loader expected."""


# --------------------------------------------------------------- file IO ---

def find_data_files(raw_dir: str | Path) -> list[Path]:
    """All parquet/json/jsonl(.gz)/csv files under raw_dir, recursively, sorted."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.name.lower().endswith(_EXTS))


def _read_whole_json(path: Path, gz: bool) -> object:
    """Parse an entire (optionally gzipped) file as one JSON document via the json module."""
    opener = gzip.open if gz else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _json_value_to_df(data: object, path: Path) -> pd.DataFrame:
    """Turn a whole-file JSON value (array or object) into a DataFrame of records.

    A JSON array of objects becomes one row per object. A dict with exactly one
    list-valued key (e.g. Anthropic's hh-rlhf red_team_attempts.jsonl.gz, which is,
    despite the ".jsonl" name, a single gzipped JSON array -- or a dict like
    {"data": [...]}) is unwrapped to that list. Anything else is a format we don't
    know how to flatten, so it's a loud error rather than a silently-empty frame.
    """
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return pd.DataFrame(list_values[0])
        if not list_values:
            # a single JSON object with no list-valued key: treat it as one row
            return pd.DataFrame([data])
        raise DataFormatError(
            f"{path}: top-level JSON object has {len(list_values)} list-valued keys "
            f"({sorted(k for k, v in data.items() if isinstance(v, list))}); expected exactly one"
        )
    raise DataFormatError(f"{path}: unsupported top-level JSON type {type(data).__name__}")


def read_table(path: Path) -> pd.DataFrame:
    """Read one file into a DataFrame, tagging rows with their source file in "_file".

    For .json/.jsonl/.jsonl.gz, JSON Lines is tried first (the common case for these
    datasets); if that fails, the whole (decompressed) file is parsed as a single JSON
    document instead -- some datasets ship a gzipped JSON array under a ".jsonl.gz" name
    (e.g. Anthropic/hh-rlhf's red-team-attempts/red_team_attempts.jsonl.gz).
    """
    name = path.name.lower()
    try:
        if name.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif name.endswith(".jsonl.gz"):
            try:
                df = pd.read_json(path, lines=True, compression="gzip")
            except ValueError:
                df = _json_value_to_df(_read_whole_json(path, gz=True), path)
        elif name.endswith(".jsonl"):
            try:
                df = pd.read_json(path, lines=True)
            except ValueError:
                df = _json_value_to_df(_read_whole_json(path, gz=False), path)
        elif name.endswith(".json"):
            try:
                df = pd.read_json(path)
            except ValueError:
                try:
                    df = pd.read_json(path, lines=True)
                except ValueError:
                    df = _json_value_to_df(_read_whole_json(path, gz=False), path)
        elif name.endswith(".csv"):
            df = pd.read_csv(path)
        else:  # pragma: no cover - find_data_files already filters extensions
            raise DataFormatError(f"unsupported file type: {path}")
    except DataFormatError:
        raise
    except Exception as exc:
        raise DataFormatError(f"failed to read {path}: {exc}") from exc
    df = df.copy()
    df["_file"] = str(path)
    return df


def read_all(raw_dir: str | Path) -> pd.DataFrame:
    """Concatenate every readable file under raw_dir into one DataFrame."""
    files = find_data_files(raw_dir)
    if not files:
        raise DataFormatError(f"no readable data files (parquet/json/jsonl/jsonl.gz/csv) under {raw_dir}")
    frames = [read_table(p) for p in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise DataFormatError(
            f"{name}: expected column(s) {sorted(required)}, missing {sorted(missing)}. "
            f"Columns actually found: {sorted(c for c in df.columns if c != '_file')}"
        )


def _infer_split(file_path: object) -> str | None:
    low = str(file_path).lower()
    for name in ("train", "validation", "val", "test"):
        if name in low:
            return name
    return None


def _finalize(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    """Fill missing optional columns, clean text, drop empties/dupes, assign ids."""
    out = rows.copy()
    out["text"] = out["text"].astype(str).str.strip()
    out = out[(out["text"].str.len() > 0) & (out["text"].str.lower() != "nan")]
    out["source"] = source
    for col in ("source_category", "is_adversarial", "origin_split"):
        if col not in out.columns:
            out[col] = None
    if "tags" not in out.columns:
        out["tags"] = ""
    out["tags"] = out["tags"].fillna("")
    if "source_label" not in out.columns:
        out["source_label"] = np.nan
    out["source_label"] = pd.to_numeric(out["source_label"], errors="coerce").astype(float)
    try:
        out["is_adversarial"] = out["is_adversarial"].astype("boolean")
    except (TypeError, ValueError):
        pass  # leave as-is if the raw values don't map cleanly to bool
    # one row per unique text within this source
    out = out.drop_duplicates(subset="text", keep="first")
    out["id"] = out["text"].map(lambda t: schema.make_id(source, t))
    out = out.drop_duplicates(subset="id", keep="first")
    return out.reindex(columns=list(schema.POOL_COLUMNS)).reset_index(drop=True)


# ------------------------------------------------------------- category ----

def _flagged_categories(cat: object) -> str:
    """BeaverTails' "category" is a dict of name -> bool (or its string repr); keep the true ones."""
    parsed = cat
    if isinstance(cat, str):
        try:
            parsed = json.loads(cat)
        except ValueError:
            try:
                parsed = json.loads(cat.replace("'", '"'))
            except ValueError:
                parsed = None
    if isinstance(parsed, dict):
        return ";".join(sorted(k for k, v in parsed.items() if v))
    return ""


_HUMAN_TURN_RE = re.compile(r"Human:\s*(.*?)(?:\n\n\s*(?:Human|Assistant):|\Z)", re.DOTALL | re.IGNORECASE)


def _first_human_turn(transcript: object) -> str | None:
    if not isinstance(transcript, str):
        return None
    m = _HUMAN_TURN_RE.search(transcript)
    return m.group(1).strip() if m else None


def _or_bench_label(file_path: object) -> float:
    # or-bench-toxic -> 1.0 (toxic); or-bench-80k / or-bench-hard-1k -> 0.0 (benign, "seemingly toxic")
    return 1.0 if "toxic" in str(file_path).lower() else 0.0


# ------------------------------------------------------------- loaders -----

def load_toxic_chat(raw_dir: str | Path) -> pd.DataFrame:
    """lmsys/toxic-chat. Prefer the toxicchat0124 config if both 0124 and 1123 are present."""
    df = read_all(raw_dir)
    _require_columns(df, {"user_input", "toxicity"}, "toxic_chat")
    if df["_file"].str.contains("0124", case=False).any() and df["_file"].str.contains("1123", case=False).any():
        df = df[df["_file"].str.contains("0124", case=False)]
    out = pd.DataFrame({
        "text": df["user_input"],
        "source_label": df["toxicity"],
        "source_category": None,
        "is_adversarial": df["jailbreaking"] if "jailbreaking" in df.columns else None,
        "tags": "",
        "origin_split": df["_file"].map(_infer_split),
    })
    return _finalize(out, "toxic_chat")


def load_wildguardmix(raw_dir: str | Path) -> pd.DataFrame:
    """allenai/wildguardmix (wildguardtrain + wildguardtest)."""
    df = read_all(raw_dir)
    _require_columns(df, {"prompt", "prompt_harm_label"}, "wildguardmix")
    label_map = {"harmful": 1.0, "unharmful": 0.0}
    out = pd.DataFrame({
        "text": df["prompt"],
        "source_label": df["prompt_harm_label"].astype(str).str.lower().map(label_map),
        "source_category": df["subcategory"] if "subcategory" in df.columns else None,
        "is_adversarial": df["adversarial"] if "adversarial" in df.columns else None,
        "tags": "",
        "origin_split": df["_file"].map(_infer_split),
    })
    return _finalize(out, "wildguardmix")


def load_beavertails(raw_dir: str | Path) -> pd.DataFrame:
    """PKU-Alignment/BeaverTails. is_safe is a per QA-pair label; a prompt is unsafe if any
    of its (prompt, response) rows is unsafe. See data/SOURCES.md for this derivation."""
    df = read_all(raw_dir)
    _require_columns(df, {"prompt", "is_safe"}, "beavertails")
    work = df.copy()
    work["_unsafe"] = ~work["is_safe"].astype(bool)
    work["_cats"] = work["category"].map(_flagged_categories) if "category" in work.columns else ""
    work["_split"] = work["_file"].map(_infer_split)

    def _union_cats(cats: pd.Series) -> str:
        names = {c for s in cats for c in s.split(";") if c}
        return ";".join(sorted(names))

    grouped = work.groupby("prompt", sort=False).agg(
        source_label=("_unsafe", "max"),
        source_category=("_cats", _union_cats),
        origin_split=("_split", "first"),
    ).reset_index()
    out = pd.DataFrame({
        "text": grouped["prompt"],
        "source_label": grouped["source_label"].astype(float),
        "source_category": grouped["source_category"].replace("", None),
        "is_adversarial": None,
        "tags": "",
        "origin_split": grouped["origin_split"],
    })
    return _finalize(out, "beavertails")


def load_aegis2(raw_dir: str | Path) -> pd.DataFrame:
    """nvidia/Aegis-AI-Content-Safety-Dataset-2.0. Drops REDACTED prompts."""
    df = read_all(raw_dir)
    _require_columns(df, {"prompt", "prompt_label"}, "aegis2")
    df = df[df["prompt"].astype(str).str.strip().str.upper() != "REDACTED"]
    label_map = {"safe": 0.0, "unsafe": 1.0}
    out = pd.DataFrame({
        "text": df["prompt"],
        "source_label": df["prompt_label"].astype(str).str.lower().map(label_map),
        "source_category": df["violated_categories"] if "violated_categories" in df.columns else None,
        "is_adversarial": None,
        "tags": "",
        "origin_split": df["_file"].map(_infer_split),
    })
    return _finalize(out, "aegis2")


def load_hh_redteam(raw_dir: str | Path) -> pd.DataFrame:
    """Anthropic/hh-rlhf red-team-attempts/. Uses the first "Human:" turn as the prompt text.

    source_label is left null: `rating` scores how successful the red-teamer was against the
    model, not whether the opening request itself is dangerous, so mapping it to a prompt-level
    harm label would be misleading. See data/SOURCES.md.
    """
    df = read_all(raw_dir)
    _require_columns(df, {"transcript"}, "hh_redteam")
    text = df["transcript"].map(_first_human_turn)
    out = pd.DataFrame({
        "text": text,
        "source_label": np.nan,
        "source_category": df["task_description"] if "task_description" in df.columns else None,
        "is_adversarial": True,
        "tags": "",
        "origin_split": None,
    })
    out = out[out["text"].notna()]
    return _finalize(out, "hh_redteam")


def load_xstest(raw_dir: str | Path) -> pd.DataFrame:
    """walledai/XSTest."""
    df = read_all(raw_dir)
    _require_columns(df, {"prompt", "label"}, "xstest")
    label_map = {"safe": 0.0, "unsafe": 1.0}
    out = pd.DataFrame({
        "text": df["prompt"],
        "source_label": df["label"].astype(str).str.lower().map(label_map),
        "source_category": df["type"] if "type" in df.columns else None,
        "is_adversarial": None,
        "tags": "",
        "origin_split": None,
    })
    return _finalize(out, "xstest")


def load_or_bench(raw_dir: str | Path) -> pd.DataFrame:
    """bench-llm/or-bench. Label is inferred from the subfolder/file name: or-bench-toxic -> 1.0,
    or-bench-80k / or-bench-hard-1k -> 0.0. If files are flattened without that in the name, this
    will mislabel everything as benign; check data/SOURCES.md after download."""
    df = read_all(raw_dir)
    _require_columns(df, {"prompt"}, "or_bench")
    out = pd.DataFrame({
        "text": df["prompt"],
        "source_label": df["_file"].map(_or_bench_label),
        "source_category": df["category"] if "category" in df.columns else None,
        "is_adversarial": None,
        "tags": "",
        "origin_split": df["_file"].map(_infer_split),
    })
    return _finalize(out, "or_bench")


def load_hn_seed(path: str | Path) -> pd.DataFrame:
    """seeds/hard_negatives.csv, cols id,text,gold,tags,source,notes. gold is always "benign"
    and source is always "hn_seed"; id is recomputed with schema.make_id to keep the pool's id
    formula uniform across sources."""
    path = Path(path)
    if not path.exists():
        raise DataFormatError(f"hard negative seed file not found: {path}")
    df = pd.read_csv(path)
    _require_columns(df, {"text"}, "hn_seed")
    out = pd.DataFrame({
        "text": df["text"],
        "source_label": 0.0,
        "source_category": None,
        "is_adversarial": False,
        "tags": df["tags"] if "tags" in df.columns else "",
        "origin_split": None,
    })
    return _finalize(out, "hn_seed")


LOADERS: dict[str, Callable[[str | Path], pd.DataFrame]] = {
    "toxic_chat": load_toxic_chat,
    "wildguardmix": load_wildguardmix,
    "beavertails": load_beavertails,
    "aegis2": load_aegis2,
    "hh_redteam": load_hh_redteam,
    "xstest": load_xstest,
    "or_bench": load_or_bench,
}

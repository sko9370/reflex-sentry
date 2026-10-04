"""Validate stored teacher scores for resume and score-file merging."""
from __future__ import annotations

import math
from numbers import Real

import pandas as pd


REQUIRED_COLUMNS = {"id", "p_unsafe_teacher", "teacher_model"}


def nonblank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def missing_optional(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def valid_probability(value: object, *, optional: bool = False) -> bool:
    if optional and (value is None or value is pd.NA):
        return True
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    if optional and math.isnan(value):
        return True
    return math.isfinite(value) and 0 <= value <= 1


def validate_score_frame(frame: pd.DataFrame, source: str) -> str | None:
    """Check required fields and score values; return the model, or None if empty.

    Optional details may be absent in legacy files. A present controversial
    probability may be null, as for the binary Llama Guard verdict.
    """
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing required column(s): {sorted(missing)}")
    if frame.empty:
        return None
    if not frame["id"].map(nonblank_string).all():
        raise ValueError(f"{source} contains null or blank id")
    if frame["id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate id")
    if not frame["teacher_model"].map(nonblank_string).all():
        raise ValueError(f"{source} contains null or blank teacher_model")
    models = frame["teacher_model"].unique()
    if len(models) != 1:
        raise ValueError(f"{source} contains mixed teacher_model values")
    if not frame["p_unsafe_teacher"].map(valid_probability).all():
        raise ValueError(f"{source} contains invalid p_unsafe_teacher")
    if "p_controversial" in frame and not frame["p_controversial"].map(
        lambda value: valid_probability(value, optional=True)
    ).all():
        raise ValueError(f"{source} contains invalid p_controversial")
    return models[0]

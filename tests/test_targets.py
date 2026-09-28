"""Tests for reflex_sentry/targets.py: the soft-target formula (README 4.3
plus the milestone-3 two-teacher and source-only extensions), checked
against hand-computed values."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflex_sentry import targets as T  # noqa: E402


def _manual_readme_formula(p, y):
    d = 0.0 if np.isnan(y) else abs(p - y)
    u = 0.50 * np.clip(2 * (d - 0.25), 0, 1) + 0.25 * (1 - abs(2 * p - 1))
    t_safe = (1 - p) * (1 - u)
    t_dangerous = p * (1 - u)
    t_unsure = u
    return d, u, t_safe, t_dangerous, t_unsure


# --------------------------------------------------------------- single-teacher formula ---


@pytest.mark.parametrize("p,y", [
    (0.9, 1.0),
    (0.9, 0.0),   # teacher/dataset disagree strongly
    (0.5, 1.0),   # teacher sits at 0.5
    (0.5, np.nan),  # no dataset label at all
    (0.1, 0.0),
])
def test_build_targets_matches_readme_formula_single_teacher(p, y):
    d_exp, u_exp, ts_exp, td_exp, tu_exp = _manual_readme_formula(p, y)
    d, u, t_safe, t_dangerous, t_unsure = T.build_targets(np.array([p]), np.array([y]))
    assert d[0] == pytest.approx(d_exp)
    assert u[0] == pytest.approx(u_exp)
    assert t_safe[0] == pytest.approx(ts_exp)
    assert t_dangerous[0] == pytest.approx(td_exp)
    assert t_unsure[0] == pytest.approx(tu_exp)
    assert (t_safe[0] + t_dangerous[0] + t_unsure[0]) == pytest.approx(1.0)


def test_build_targets_y_nan_gives_zero_disagreement():
    d, u, *_ = T.build_targets(np.array([0.9]), np.array([np.nan]))
    assert d[0] == 0.0


# --------------------------------------------------------------- two-teacher extension ---


def test_build_targets_two_teacher_disagreement_increases_unsure():
    p_mean = 0.6
    y = 1.0
    disagreement = 0.8  # |p1 - p2|
    d_base, u_base, *_ = T.build_targets(np.array([p_mean]), np.array([y]))
    d, u, t_safe, t_dangerous, t_unsure = T.build_targets(
        np.array([p_mean]), np.array([y]), teacher_disagreement=np.array([disagreement]), w_t=0.25,
    )
    expected_u = u_base[0] + 0.25 * np.clip(2 * disagreement, 0, 1)
    assert u[0] == pytest.approx(min(expected_u, T.DEFAULT_U_CAP))
    assert (t_safe[0] + t_dangerous[0] + t_unsure[0]) == pytest.approx(1.0)


def test_build_targets_controversial_term_added_to_unsure():
    p, y = 0.5, 1.0
    d_base, u_base, *_ = T.build_targets(np.array([p]), np.array([y]))
    d, u, t_safe, t_dangerous, t_unsure = T.build_targets(
        np.array([p]), np.array([y]), controversial=np.array([0.4]), w_c=0.25,
    )
    assert u[0] == pytest.approx(min(u_base[0] + 0.25 * 0.4, T.DEFAULT_U_CAP))


def test_build_targets_controversial_nan_contributes_zero():
    p, y = 0.5, 1.0
    d_base, u_base, *_ = T.build_targets(np.array([p]), np.array([y]))
    d, u, *_ = T.build_targets(np.array([p]), np.array([y]), controversial=np.array([np.nan]), w_c=0.25)
    assert u[0] == pytest.approx(u_base[0])


def test_build_targets_caps_unsure_and_renormalizes():
    # Extreme disagreement + controversial should push u past 1 before the cap.
    p, y = 0.5, 1.0
    d, u, t_safe, t_dangerous, t_unsure = T.build_targets(
        np.array([p]), np.array([y]),
        teacher_disagreement=np.array([1.0]), controversial=np.array([1.0]),
        w_t=0.5, w_c=0.5, u_cap=0.9,
    )
    assert u[0] == pytest.approx(0.9)
    total = t_safe[0] + t_dangerous[0] + t_unsure[0]
    assert total == pytest.approx(1.0)
    assert t_unsure[0] == pytest.approx(0.9)


def test_build_targets_sums_to_one_across_a_grid():
    ps = np.linspace(0.01, 0.99, 15)
    ys = np.array([1.0, 0.0, np.nan] * 5)
    disagreement = np.linspace(0, 1, 15)
    controversial = np.linspace(0, 1, 15)
    d, u, t_safe, t_dangerous, t_unsure = T.build_targets(
        ps, ys, teacher_disagreement=disagreement, controversial=controversial,
    )
    sums = t_safe + t_dangerous + t_unsure
    assert np.allclose(sums, 1.0)
    assert (t_safe >= 0).all() and (t_dangerous >= 0).all() and (t_unsure >= 0).all()


# --------------------------------------------------------------- source-only mode ---


def test_source_only_known_labels_use_smoothing_and_zero_unsure():
    p, u, t_safe, t_dangerous, t_unsure = T.build_source_only_targets(np.array([1.0, 0.0]), smoothing=0.05)
    assert p[0] == pytest.approx(0.95)
    assert p[1] == pytest.approx(0.05)
    assert u[0] == 0.0
    assert u[1] == 0.0
    assert t_dangerous[0] == pytest.approx(0.95)
    assert t_safe[1] == pytest.approx(0.95)


def test_source_only_nan_label_gives_half_half_and_half_unsure():
    p, u, t_safe, t_dangerous, t_unsure = T.build_source_only_targets(np.array([np.nan]), smoothing=0.05)
    assert p[0] == pytest.approx(0.5)
    assert u[0] == pytest.approx(0.5)
    assert t_safe[0] == pytest.approx(0.25)
    assert t_dangerous[0] == pytest.approx(0.25)
    assert t_unsure[0] == pytest.approx(0.5)
    assert (t_safe[0] + t_dangerous[0] + t_unsure[0]) == pytest.approx(1.0)


# --------------------------------------------------------------- run_teacher / run_source_only ---


def test_run_teacher_single_teacher_end_to_end(tmp_path):
    pool = pd.DataFrame({"id": ["a:1", "a:2", "a:3"], "source_label": [1.0, 0.0, np.nan]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)

    teacher = pd.DataFrame({
        "id": ["a:1", "a:2", "a:3"],
        "p_unsafe_teacher": [0.95, 0.05, 0.5],
        "p_controversial": [np.nan, np.nan, np.nan],
        "teacher_category": ["S2", "", ""],
        "teacher_model": ["llama_guard_3_8b"] * 3,
    })
    teacher_path = tmp_path / "teacher.parquet"
    teacher.to_parquet(teacher_path)

    df = T.run_teacher([str(teacher_path)], str(pool_path))
    assert len(df) == 3
    assert (df["n_teachers"] == 1).all()
    assert df["teacher_disagreement"].isna().all()
    sums = df["t_safe"] + df["t_dangerous"] + df["t_unsure"]
    assert np.allclose(sums, 1.0)
    assert (df["target_source"] == "teacher").all()


def test_run_teacher_two_teacher_inner_join_and_disagreement(tmp_path):
    pool = pd.DataFrame({"id": ["a:1", "a:2"], "source_label": [1.0, 0.0]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)

    t1 = pd.DataFrame({
        "id": ["a:1", "a:2"], "p_unsafe_teacher": [0.9, 0.2], "p_controversial": [np.nan, np.nan],
        "teacher_category": ["S2", ""], "teacher_model": ["llama_guard_3_8b"] * 2,
    })
    t2 = pd.DataFrame({
        "id": ["a:1"], "p_unsafe_teacher": [0.6], "p_controversial": [0.3],
        "teacher_category": ["cybercrime"], "teacher_model": ["qwen3guard_gen_8b"],
    })
    t1_path, t2_path = tmp_path / "t1.parquet", tmp_path / "t2.parquet"
    t1.to_parquet(t1_path)
    t2.to_parquet(t2_path)

    df = T.run_teacher([str(t1_path), str(t2_path)], str(pool_path))
    # inner join: only a:1 was scored by both teachers
    assert list(df["id"]) == ["a:1"]
    assert df.loc[0, "n_teachers"] == 2
    assert df.loc[0, "p"] == pytest.approx((0.9 + 0.6) / 2)
    assert df.loc[0, "teacher_disagreement"] == pytest.approx(abs(0.9 - 0.6))
    assert (df.loc[0, ["t_safe", "t_dangerous", "t_unsure"]].sum()) == pytest.approx(1.0)


def test_run_teacher_rejects_wrong_number_of_files(tmp_path):
    pool_path = tmp_path / "pool.parquet"
    pd.DataFrame({"id": ["a:1"], "source_label": [1.0]}).to_parquet(pool_path)
    with pytest.raises(ValueError, match="1 or 2"):
        T.run_teacher([], str(pool_path))


def test_run_source_only_marks_target_source(tmp_path):
    pool = pd.DataFrame({"id": ["a:1", "a:2"], "source_label": [1.0, np.nan]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)

    df = T.run_source_only(str(pool_path))
    assert (df["target_source"] == "source_only").all()
    assert (df["n_teachers"] == 0).all()
    assert df["teacher_disagreement"].isna().all()
    sums = df["t_safe"] + df["t_dangerous"] + df["t_unsure"]
    assert np.allclose(sums, 1.0)
    assert df.loc[df["id"] == "a:1", "p"].iloc[0] == pytest.approx(0.95)
    assert df.loc[df["id"] == "a:2", "u"].iloc[0] == pytest.approx(0.5)


def test_expand_globs_falls_back_to_literal_when_no_match(tmp_path):
    real = tmp_path / "teacher_scores_llama.parquet"
    pd.DataFrame({"id": ["a:1"]}).to_parquet(real)
    matched = T._expand_globs([str(tmp_path / "teacher_scores_*.parquet")])
    assert matched == [str(real)]

    literal = T._expand_globs(["does/not/exist.parquet"])
    assert literal == ["does/not/exist.parquet"]

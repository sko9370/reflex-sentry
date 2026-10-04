import sys

import numpy as np
import pandas as pd
import pytest

from reflex_sentry.teacher.merge_scores import merge_scores
from reflex_sentry.teacher import score
from reflex_sentry.teacher.score import load_and_dedupe_inputs, run_scorer


def pool(tmp_path, ids=("a",), texts=("hello",)):
    path = tmp_path / "pool.parquet"
    pd.DataFrame({"id": ids, "text": texts}).to_parquet(path)
    return path


def artifact(tmp_path, name, ids=("a",), probs=(0.5,), model="fake", **extra):
    path = tmp_path / name
    pd.DataFrame({"id": ids, "p_unsafe_teacher": probs,
                  "teacher_model": [model] * len(ids), **extra}).to_parquet(path)
    return path


@pytest.mark.parametrize("ids,texts", [([None], ["x"]), ([" "], ["x"]),
                                        (["a"], [None]), (["a"], ["\t"])])
def test_input_rejects_blank_values(tmp_path, ids, texts):
    with pytest.raises(ValueError, match="null or blank"):
        load_and_dedupe_inputs([str(pool(tmp_path, ids, texts))])


def test_input_rejects_conflicting_overlap(tmp_path):
    path = pool(tmp_path, ["a", "a"], ["hello", "different"])
    with pytest.raises(ValueError, match="conflicting text"):
        load_and_dedupe_inputs([str(path)])


@pytest.mark.parametrize("field,value", [("id", None), ("id", " "),
                                          ("p_unsafe_teacher", float("nan")),
                                          ("p_unsafe_teacher", 1.1),
                                          ("p_controversial", float("inf")),
                                          ("p_controversial", " "),
                                          ("teacher_model", " ")])
def test_resume_rejects_bad_artifact_and_preserves_file(tmp_path, field, value):
    inp = pool(tmp_path)
    out = artifact(tmp_path, "out.parquet")
    frame = pd.read_parquet(out)
    frame[field] = [value]
    frame.to_parquet(out)
    before = out.read_bytes()
    with pytest.raises(ValueError):
        run_scorer(lambda texts: [], [str(inp)], str(out), "fake")
    assert out.read_bytes() == before


def test_resume_rejects_model_mismatch_even_for_disjoint_ids(tmp_path):
    inp = pool(tmp_path)
    out = artifact(tmp_path, "out.parquet", ids=("other",), model="other-model")
    before = out.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        run_scorer(lambda texts: [], [str(inp)], str(out), "fake")
    assert out.read_bytes() == before


@pytest.mark.parametrize("result", [[], [{"p_unsafe_teacher": 0.5}, {"p_unsafe_teacher": 0.5}],
                                    [{"p_unsafe_teacher": float("nan")}],
                                    [{"p_unsafe_teacher": 0.5, "p_controversial": 2}]])
def test_bad_batch_does_not_change_existing_output(tmp_path, result):
    inp = pool(tmp_path)
    out = artifact(tmp_path, "out.parquet", ids=("other",))
    before = out.read_bytes()
    with pytest.raises(ValueError):
        run_scorer(lambda texts: result, [str(inp)], str(out), "fake")
    assert out.read_bytes() == before


def test_late_bad_batch_preserves_earlier_valid_checkpoint(tmp_path):
    inp = pool(tmp_path, ["a", "b"], ["one", "two"])
    out = artifact(tmp_path, "out.parquet", ids=("other",))
    calls = 0

    def scorer(texts):
        nonlocal calls
        calls += 1
        return [{"p_unsafe_teacher": 0.5 if calls == 1 else float("nan")}]

    with pytest.raises(ValueError, match="invalid p_unsafe_teacher"):
        run_scorer(scorer, [str(inp)], str(out), "fake", batch_size=1, checkpoint_every=1)
    assert set(pd.read_parquet(out)["id"]) == {"other", "a"}


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": 1.5},
                                    {"checkpoint_every": 0}, {"limit": -1}])
def test_invalid_options_rejected_before_scoring(tmp_path, kwargs):
    inp = pool(tmp_path)
    with pytest.raises(ValueError):
        run_scorer(lambda texts: pytest.fail("scorer called"), [str(inp)],
                   str(tmp_path / "out.parquet"), "fake", **kwargs)


@pytest.mark.parametrize("existing", [False, True])
def test_cli_no_pending_work_skips_model_and_reports_no_write(tmp_path, monkeypatch, capsys, existing):
    inp = pool(tmp_path)
    out = tmp_path / "out.parquet"
    if existing:
        artifact(tmp_path, "out.parquet", model="llama_guard_3_8b")
    before = out.read_bytes() if existing else None
    monkeypatch.setattr(sys, "argv", ["score", "--model", "llama_guard_3_8b",
                                       "--inputs", str(inp), "--out", str(out),
                                       "--limit", "1" if existing else "0"])
    monkeypatch.setattr(score, "load_model_and_tokenizer",
                        lambda *args, **kwargs: pytest.fail("model loaded"))
    score.main()
    output = capsys.readouterr().out
    assert "wrote" not in output
    if existing:
        assert "no new teacher scores" in output
        assert out.read_bytes() == before
    else:
        assert "no teacher scores to write" in output
        assert not out.exists()


def test_merge_legacy_optionals_and_enrichment(tmp_path):
    first = artifact(tmp_path, "first.parquet")
    second = artifact(tmp_path, "second.parquet", p_controversial=[np.nan],
                      teacher_category=["S2"], teacher_raw=["unsafe"])
    merged = merge_scores([str(first), str(second)])
    assert len(merged) == 1
    assert merged.loc[0, "teacher_category"] == "S2"
    assert merged.loc[0, "teacher_raw"] == "unsafe"


@pytest.mark.parametrize("change", [{"p_unsafe_teacher": [0.6]},
                                    {"teacher_category": ["S3"]},
                                    {"teacher_raw": ["different"]},
                                    {"p_controversial": [0.7]},
                                    {"extra_score": ["different"]}])
def test_merge_rejects_conflicting_scores_and_details(tmp_path, change):
    base = dict(p_controversial=[0.2], teacher_category=["S2"],
                teacher_raw=["unsafe"], extra_score=["first"])
    first = artifact(tmp_path, "first.parquet", **base)
    second = artifact(tmp_path, "second.parquet", **{**base, **change})
    with pytest.raises(ValueError, match="conflicting"):
        merge_scores([str(first), str(second)])


def test_merge_rejects_mixed_models_and_duplicate_inside_artifact(tmp_path):
    first = artifact(tmp_path, "first.parquet")
    second = artifact(tmp_path, "second.parquet", model="other")
    with pytest.raises(ValueError, match="mixed teacher models"):
        merge_scores([str(first), str(second)])
    dup = artifact(tmp_path, "dup.parquet", ids=("a", "a"), probs=(0.5, 0.5))
    with pytest.raises(ValueError, match="duplicate id"):
        merge_scores([str(dup)])

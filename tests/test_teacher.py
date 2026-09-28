"""Tests for reflex_sentry/teacher/*: verdict-token extraction, probability
math, category parsing, and batching/resume orchestration -- all against a
mocked tokenizer/model or a fake score_fn, never a real download (no network
access to Hugging Face is assumed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflex_sentry.teacher import as_predictor, score  # noqa: E402


# --------------------------------------------------------------------- fakes ---


class FakeTokenizer:
    """Encodes each configured word to a fixed token-id list; anything else
    raises, so a test fails loudly if it exercises an unexpected string."""

    def __init__(self, encodings: dict[str, list[int]]):
        self.encodings = encodings

    def encode(self, s, add_special_tokens=False):
        if s not in self.encodings:
            raise KeyError(f"unexpected encode({s!r})")
        return list(self.encodings[s])

    def decode(self, ids):
        # Only used to reconstruct a shared-prefix string; identity is fine
        # for these tests since it is never re-encoded.
        return "".join(str(i) for i in ids)


# --------------------------------------------------------------- find_diverging_tokens ---


def test_find_diverging_tokens_differs_at_first_token():
    tok = FakeTokenizer({"unsafe": [200, 100], "safe": [100]})
    prefix, unsafe_id, safe_id = score.find_diverging_tokens(tok, "unsafe", "safe")
    assert prefix == []
    assert (unsafe_id, safe_id) == (200, 100)


def test_find_diverging_tokens_shared_prefix():
    tok = FakeTokenizer({"aaa": [1, 2, 3], "aab": [1, 2, 4]})
    prefix, a_id, b_id = score.find_diverging_tokens(tok, "aaa", "aab")
    assert prefix == [1, 2]
    assert (a_id, b_id) == (3, 4)


def test_find_diverging_tokens_prefix_fallback(capsys):
    tok = FakeTokenizer({"a": [1], "ab": [1, 2]})
    prefix, a_id, b_id = score.find_diverging_tokens(tok, "a", "ab")
    assert prefix == []
    assert (a_id, b_id) == (1, 1)
    assert "WARNING" in capsys.readouterr().out


def test_first_token_id():
    tok = FakeTokenizer({" Safe": [7, 8]})
    assert score.first_token_id(tok, " Safe") == 7


# --------------------------------------------------------------- probability math ---


def test_binary_verdict_prob_matches_sigmoid():
    vocab = 10
    logits = np.zeros(vocab)
    logits[3] = 2.0  # unsafe
    logits[5] = 0.0  # safe
    p_unsafe = score.binary_verdict_prob(logits, unsafe_id=3, safe_id=5)
    expected = 1 / (1 + math.exp(-2.0))
    assert p_unsafe == pytest.approx(expected, abs=1e-9)


def test_binary_verdict_prob_equal_logits_is_half():
    logits = np.zeros(5)
    assert score.binary_verdict_prob(logits, unsafe_id=1, safe_id=2) == pytest.approx(0.5)


def test_three_way_verdict_probs_matches_manual_softmax():
    logits = np.zeros(20)
    logits[0] = 1.0   # safe
    logits[1] = 2.0   # unsafe
    logits[2] = 0.5   # controversial
    p_safe, p_unsafe, p_ctrl = score.three_way_verdict_probs(logits, safe_id=0, unsafe_id=1, controversial_id=2)
    vals = np.array([1.0, 2.0, 0.5])
    ex = np.exp(vals - vals.max())
    expected = ex / ex.sum()
    assert (p_safe, p_unsafe, p_ctrl) == pytest.approx(tuple(expected.tolist()))
    assert p_safe + p_unsafe + p_ctrl == pytest.approx(1.0)


# --------------------------------------------------------------- category parsing ---


def test_parse_llama_guard_category_unsafe():
    preset = score.PRESETS["llama_guard_3_8b"]
    assert score.parse_llama_guard_category("unsafe\nS2", preset) == "S2"


def test_parse_llama_guard_category_safe_has_none():
    preset = score.PRESETS["llama_guard_3_8b"]
    assert score.parse_llama_guard_category("safe", preset) == ""


def test_parse_qwen_guard_category_present():
    preset = score.PRESETS["qwen3guard_gen_8b"]
    text = "Safety: Unsafe\nCategories: Cybercrime, Weapons\n"
    assert score.parse_qwen_guard_category(text, preset) == "Cybercrime, Weapons"


def test_parse_qwen_guard_category_absent():
    preset = score.PRESETS["qwen3guard_gen_8b"]
    assert score.parse_qwen_guard_category("Safety: Safe", preset) == ""


# --------------------------------------------------------------- load_and_dedupe_inputs ---


def test_load_and_dedupe_inputs_dedupes_across_files(tmp_path):
    a = pd.DataFrame({"id": ["x:1", "x:2"], "text": ["hello", "world"], "source": ["s", "s"]})
    b = pd.DataFrame({"id": ["x:2", "x:3"], "text": ["world-dupe", "third"], "source": ["s", "s"]})
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    a.to_parquet(pa)
    b.to_parquet(pb)

    combined = score.load_and_dedupe_inputs([str(pa), str(pb)])
    assert sorted(combined["id"]) == ["x:1", "x:2", "x:3"]
    # first occurrence (from file a) wins for the duplicate id
    assert combined.set_index("id").loc["x:2", "text"] == "world"


def test_load_and_dedupe_inputs_requires_id_and_text(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"id": ["x:1"]}).to_parquet(p)
    with pytest.raises(ValueError, match="missing required column"):
        score.load_and_dedupe_inputs([str(p)])


# --------------------------------------------------------------- run_scorer ---


def _fake_score_fn(call_log):
    def fn(texts):
        call_log.append(list(texts))
        return [{"p_unsafe_teacher": 0.9, "p_controversial": np.nan, "teacher_category": "S2"} for _ in texts]
    return fn


def test_run_scorer_batches_and_writes_all_rows(tmp_path):
    pool = pd.DataFrame({"id": [f"x:{i}" for i in range(5)], "text": [f"prompt {i}" for i in range(5)]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)
    out_path = tmp_path / "out.parquet"

    calls = []
    df = score.run_scorer(
        _fake_score_fn(calls), [str(pool_path)], str(out_path), teacher_model="fake",
        batch_size=2, checkpoint_every=100,
    )

    assert len(df) == 5
    assert set(df["id"]) == set(pool["id"])
    assert (df["p_unsafe_teacher"] == 0.9).all()
    assert (df["teacher_model"] == "fake").all()
    # batch sizes: 2, 2, 1
    assert [len(c) for c in calls] == [2, 2, 1]
    # file was written
    assert out_path.exists()
    assert len(pd.read_parquet(out_path)) == 5


def test_run_scorer_resumes_and_skips_already_scored(tmp_path):
    pool = pd.DataFrame({"id": ["x:1", "x:2", "x:3"], "text": ["a", "b", "c"]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)

    out_path = tmp_path / "out.parquet"
    existing = pd.DataFrame({
        "id": ["x:1"], "p_unsafe_teacher": [0.1], "p_controversial": [np.nan],
        "teacher_category": [""], "teacher_model": ["fake"],
    })
    existing.to_parquet(out_path)

    calls = []
    df = score.run_scorer(
        _fake_score_fn(calls), [str(pool_path)], str(out_path), teacher_model="fake", batch_size=10,
    )

    assert len(df) == 3
    # only the two not-yet-scored ids were sent to score_fn
    scored_texts = {t for batch in calls for t in batch}
    assert scored_texts == {"b", "c"}
    # the pre-existing row's value was preserved, not overwritten
    assert df.set_index("id").loc["x:1", "p_unsafe_teacher"] == 0.1


def test_run_scorer_respects_limit(tmp_path):
    pool = pd.DataFrame({"id": [f"x:{i}" for i in range(10)], "text": [f"p{i}" for i in range(10)]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)
    out_path = tmp_path / "out.parquet"

    calls = []
    df = score.run_scorer(
        _fake_score_fn(calls), [str(pool_path)], str(out_path), teacher_model="fake",
        batch_size=4, limit=5,
    )
    assert len(df) == 5


def test_run_scorer_checkpoints_every_n_batches(tmp_path):
    pool = pd.DataFrame({"id": [f"x:{i}" for i in range(6)], "text": [f"p{i}" for i in range(6)]})
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path)
    out_path = tmp_path / "out.parquet"

    seen_at_checkpoint = []

    def fn(texts):
        # after every call, check what's on disk so far (only updated at checkpoints)
        if out_path.exists():
            seen_at_checkpoint.append(len(pd.read_parquet(out_path)))
        else:
            seen_at_checkpoint.append(0)
        return [{"p_unsafe_teacher": 0.5, "p_controversial": np.nan, "teacher_category": ""} for _ in texts]

    score.run_scorer(fn, [str(pool_path)], str(out_path), teacher_model="fake", batch_size=2, checkpoint_every=1)
    # with checkpoint_every=1, the file should exist and grow after each batch
    assert seen_at_checkpoint == [0, 2, 4]


# --------------------------------------------------------------- as_predictor ---


def test_single_teacher_probs_formula():
    scores = pd.DataFrame({
        "id": ["a", "b"],
        "p_unsafe_teacher": [0.8, 0.1],
        "p_controversial": [0.1, np.nan],
    })
    out = as_predictor.single_teacher_probs(scores)
    assert out.loc[0, "p_dangerous"] == pytest.approx(0.8)
    assert out.loc[0, "p_unsure"] == pytest.approx(0.1)
    assert out.loc[0, "p_safe"] == pytest.approx(0.1)
    assert out.loc[1, "p_unsure"] == pytest.approx(0.0)  # NaN -> 0
    assert out.loc[1, "p_safe"] == pytest.approx(0.9)


def test_two_teacher_probs_formula_and_renormalizes():
    a = pd.DataFrame({"id": ["a"], "p_unsafe_teacher": [0.9], "p_controversial": [0.2]})
    b = pd.DataFrame({"id": ["a"], "p_unsafe_teacher": [0.7], "p_controversial": [np.nan]})
    out = as_predictor.two_teacher_probs(a, b, w_controversial=0.5, w_disagreement=0.5)

    p_dangerous_raw = (0.9 + 0.7) / 2
    controversial_signal = 0.2  # nanmean of [0.2, nan]
    disagreement_signal = abs(0.9 - 0.7)
    p_unsure_raw = 0.5 * controversial_signal + 0.5 * disagreement_signal
    p_safe_raw = max(0.0, 1 - p_dangerous_raw - p_unsure_raw)
    total = p_dangerous_raw + p_unsure_raw + p_safe_raw

    assert out.loc[0, "p_dangerous"] == pytest.approx(p_dangerous_raw / total)
    assert out.loc[0, "p_unsure"] == pytest.approx(p_unsure_raw / total)
    assert out.loc[0, "p_safe"] == pytest.approx(p_safe_raw / total)
    row_sum = out.loc[0, ["p_safe", "p_dangerous", "p_unsure"]].sum()
    assert row_sum == pytest.approx(1.0)


def test_as_predictor_run_single_teacher_writes_csv(tmp_path):
    scores = pd.DataFrame({
        "id": ["e:1", "e:2"],
        "p_unsafe_teacher": [0.9, 0.05],
        "p_controversial": [np.nan, np.nan],
        "teacher_category": ["S2", ""],
        "teacher_model": ["llama_guard_3_8b", "llama_guard_3_8b"],
    })
    scores_path = tmp_path / "scores.parquet"
    scores.to_parquet(scores_path)

    eval_df = pd.DataFrame({
        "id": ["e:1", "e:2"], "text": ["a", "b"], "gold": ["dangerous", "benign"],
        "source": ["toxic_chat", "toxic_chat"], "tags": ["", ""],
    })
    eval_path = tmp_path / "val.parquet"
    eval_df.to_parquet(eval_path)

    out_path = tmp_path / "preds.csv"
    out_df = as_predictor.run([str(scores_path)], str(eval_path), str(out_path))

    assert list(out_df.columns) == ["id", "gold", "p_safe", "p_dangerous", "p_unsure", "source", "tags", "latency_ms"]
    assert len(out_df) == 2
    written = pd.read_csv(out_path)
    assert set(written["id"]) == {"e:1", "e:2"}
    sums = written[["p_safe", "p_dangerous", "p_unsure"]].sum(axis=1)
    assert (sums.round(6) == 1.0).all()


def test_as_predictor_run_drops_unmatched_eval_rows(tmp_path, capsys):
    scores = pd.DataFrame({"id": ["e:1"], "p_unsafe_teacher": [0.5], "p_controversial": [np.nan]})
    scores_path = tmp_path / "scores.parquet"
    scores.to_parquet(scores_path)

    eval_df = pd.DataFrame({"id": ["e:1", "e:2"], "gold": ["benign", "benign"]})
    eval_path = tmp_path / "val.parquet"
    eval_df.to_parquet(eval_path)

    out_df = as_predictor.run([str(scores_path)], str(eval_path), str(tmp_path / "preds.csv"))
    assert len(out_df) == 1
    assert "WARNING" in capsys.readouterr().out


def test_as_predictor_run_rejects_wrong_number_of_score_files(tmp_path):
    eval_path = tmp_path / "val.parquet"
    pd.DataFrame({"id": ["e:1"], "gold": ["benign"]}).to_parquet(eval_path)
    with pytest.raises(ValueError, match="1 or 2"):
        as_predictor.run([], str(eval_path), str(tmp_path / "out.csv"))


def test_truncate_user_texts_cuts_prompt_before_templating():
    from reflex_sentry.teacher.score import truncate_user_texts

    class WordTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids):
            return " ".join(ids)

    long_text = " ".join(f"w{i}" for i in range(600))
    out = truncate_user_texts(WordTok(), ["short prompt", long_text], max_tokens=512)
    assert out[0] == "short prompt"
    assert out[1].split() == [f"w{i}" for i in range(512)]


def test_qwen_category_none_is_empty():
    from reflex_sentry.teacher.score import PRESETS, parse_qwen_guard_category

    preset = PRESETS["qwen3guard_gen_8b"]
    assert parse_qwen_guard_category("Safe\nCategories: None", preset) == ""

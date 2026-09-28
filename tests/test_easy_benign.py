"""Tests for reflex_sentry.data.easy_benign using a fully synthetic pool."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflex_sentry.data import easy_benign as EB  # noqa: E402
from reflex_sentry.data.schema import make_id  # noqa: E402
from reflex_sentry.eval import metrics as M  # noqa: E402
from reflex_sentry.teacher.merge_scores import merge_scores  # noqa: E402


def row(source, text, label=0.0):
    return {"id": make_id(source, text), "text": text, "source": source, "source_label": label,
            "source_category": None, "is_adversarial": None, "tags": "", "origin_split": None}


def good(i):
    return f"Please write a short poem about the ocean number {i} and its many moods."


def make_files(tmp_path, n_good=40):
    proc = tmp_path / "processed"
    proc.mkdir()
    rows = [row("toxic_chat", good(i)) for i in range(n_good)]
    # rejects
    rows += [
        row("wildguardmix", good(1000)),                                   # wrong source
        row("toxic_chat", good(1001), label=1.0),                          # unsafe
        row("toxic_chat", good(1002), label=float("nan")),                 # unknown label
        row("toxic_chat", "too short"),                                    # length < 20
        row("toxic_chat", "word " * 400),                                  # length > 1500
        row("toxic_chat", "这是一个关于海洋的很长的中文句子,请写一首诗歌来描述它"),   # not English
        row("toxic_chat", "How do I scan my own network for open ports today?"),  # in cyber pool
        row("toxic_chat", "Tell me a fun fact about the planet Jupiter and moons."),  # in a split
        row("toxic_chat", "Tell me a fun fact about the planet Saturn and rings."),   # split by text only
    ]
    # duplicate by normalized text (different case/whitespace, same id after normalize)
    rows.append(row("toxic_chat", good(0).upper() + "  "))
    pool = pd.DataFrame(rows)
    pool.to_parquet(tmp_path / "pool.parquet", index=False)

    cyber_text = "How do I scan my own network for open ports today?"
    cyber = pool[pool["text"] == cyber_text][["id", "text"]]
    cyber.to_parquet(tmp_path / "cyber_pool.parquet", index=False)

    jup = pool[pool["text"].str.contains("Jupiter")][["id", "text"]]
    jup.to_parquet(proc / "test_ood_pool.parquet", index=False)
    sat_text = "tell me a fun FACT about the planet saturn and rings."
    other = pd.DataFrame({"id": ["hn_seed:aaaaaaaaaaaa"], "text": [sat_text]})
    other.to_parquet(proc / "train.parquet", index=False)

    test = pd.DataFrame({
        "id": ["hn_seed:bbbbbbbbbbbb", "hn_seed:cccccccccccc"],
        "text": ["How do I detect ransomware on my fileserver?", "Explain how SQL injection works."],
        "source": "hn_seed", "source_label": [0.0, 0.0], "tags": ["hard_negative;hn:detection"] * 2,
        "gold": "benign", "split": "test", "in_scope": True, "kw_hits": "x", "kw_strong": 1,
        "kw_weak": 0, "dup_group": ["hn_seed:bbbbbbbbbbbb", "hn_seed:cccccccccccc"],
        "source_category": None, "is_adversarial": None, "origin_split": None,
    })
    test.to_parquet(proc / "test.parquet", index=False)
    return proc, tmp_path


def run(tmp_path, n=25, seed=7, append=False, out="test_easy_benign.parquet"):
    proc, base = make_files(tmp_path) if not (tmp_path / "processed").exists() else (tmp_path / "processed", tmp_path)
    return EB.run(str(base / "pool.parquet"), str(base / "cyber_pool.parquet"), str(proc / "test.parquet"),
                  n, "toxic_chat", seed, str(proc / out), append)


def test_filters_and_schema(tmp_path):
    easy = run(tmp_path, n=1000)
    texts = set(easy["text"])
    assert texts == {good(i) for i in range(40)}  # every reject excluded, dup collapsed
    assert list(easy.columns) == EB.OUT_COLUMNS
    assert (easy["gold"] == "benign").all() and (easy["tags"] == "").all()
    assert (easy["source"] == "toxic_chat").all() and (easy["split"] == "test").all()
    assert easy["easy_benign"].dtype == bool and easy["easy_benign"].all()
    assert easy["id"].is_unique and easy["id"].str.match(r"^toxic_chat:[0-9a-f]{12}$").all()


def test_teacher_file(tmp_path):
    easy = run(tmp_path)
    t = pd.read_parquet(tmp_path / "processed" / EB.TEACHER_FILE)
    assert list(t.columns) == ["id", "text"]
    assert t["id"].tolist() == easy["id"].tolist()


def test_english_heuristic():
    assert EB.english_looking("Write a haiku about spring rain, please!")
    assert not EB.english_looking("这是一个关于海洋的很长的中文句子")
    assert not EB.english_looking("   ")


def test_sample_size_and_determinism(tmp_path):
    a = run(tmp_path, n=25, seed=7)
    b = run(tmp_path, n=25, seed=7)
    c = run(tmp_path, n=25, seed=8)
    assert len(a) == 25
    pd.testing.assert_frame_equal(a, b)
    assert set(a["id"]) != set(c["id"])


def test_append_idempotent_and_backup(tmp_path):
    easy = run(tmp_path, n=25)
    proc = tmp_path / "processed"
    orig = pd.read_parquet(proc / "test.parquet")
    run(tmp_path, n=25, append=True)
    bak = proc / "test.parquet.bak"
    assert bak.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(bak), orig)
    after1 = pd.read_parquet(proc / "test.parquet")
    assert len(after1) == len(orig) + 25
    assert after1["easy_benign"].sum() == 25 and not after1.loc[:1, "easy_benign"].any()
    # re-run (same sample, since easy rows already in test are ignored when scanning splits)
    again = run(tmp_path, n=25, append=True)
    pd.testing.assert_frame_equal(again, easy)
    after2 = pd.read_parquet(proc / "test.parquet")
    pd.testing.assert_frame_equal(after1, after2)
    pd.testing.assert_frame_equal(pd.read_parquet(bak), orig)  # backup not overwritten
    assert after2["id"].is_unique
    assert set(after2["gold"]) == {"benign"}


def test_merge_scores(tmp_path):
    a = pd.DataFrame({"id": ["x", "y"], "p_unsafe_teacher": [0.1, 0.2]})
    b = pd.DataFrame({"id": ["y", "z"], "p_unsafe_teacher": [0.9, 0.3]})
    a.to_parquet(tmp_path / "a.parquet", index=False)
    b.to_parquet(tmp_path / "b.parquet", index=False)
    m = merge_scores([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")])
    assert m["id"].tolist() == ["x", "y", "z"]
    assert m.set_index("id").loc["y", "p_unsafe_teacher"] == 0.2  # first file wins


def test_easy_benign_makes_reweighting_branch_apply(tmp_path):
    run(tmp_path, n=25, append=True)
    test = pd.read_parquet(tmp_path / "processed" / "test.parquet")
    # synthetic predictions: hard negatives escalate more than easy benign
    rows = []
    for _, r in test.iterrows():
        hard = "hard_negative" in r["tags"]
        ps = 0.3 if hard else 0.95
        rows.append({"id": r["id"], "gold": r["gold"], "tags": r["tags"],
                     "p_safe": ps, "p_dangerous": 1 - ps, "p_unsure": 0.0})
    dangerous = pd.DataFrame([{"id": "d1", "gold": "dangerous", "tags": "cat:ddos",
                               "p_safe": 0.1, "p_dangerous": 0.9, "p_unsure": 0.0}])
    preds = pd.concat([pd.DataFrame(rows), dangerous], ignore_index=True)
    M.validate(preds)
    at = M.at_threshold(preds, 0.5)
    assert at["easy_benign_escalation_rate"].n == 25
    assert at["hard_negative_escalation_rate"].n == 2
    traffic = {"prompts": 1_000_000, "prevalence_dangerous": 0.005, "prevalence_ambiguous": 0.02,
               "tier2_cost_per_call": 0.002, "hard_negative_share_of_benign": 0.05}
    reweighted = M.cascade_economics(at, traffic)
    raw = M.cascade_economics(at, {**traffic, "hard_negative_share_of_benign": None})
    # reweighted benign rate = 0.05 * 1.0 + 0.95 * 0.0 ; raw mix is 2/27
    pb = 1 - 0.005 - 0.02
    assert reweighted["escalations_benign"] == pytest.approx(1_000_000 * pb * 0.05)
    assert raw["escalations_benign"] == pytest.approx(1_000_000 * pb * (2 / 27))
    assert reweighted["escalations_benign"] != pytest.approx(raw["escalations_benign"])

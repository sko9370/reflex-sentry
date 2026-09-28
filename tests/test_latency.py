"""Tests for the shared CPU latency protocol (reflex_sentry.eval.latency):
`single_thread` restores the previous torch thread count, and
`time_per_prompt` returns a deterministic, fixed-size sample. torch is
skipped cleanly (not failed) when not installed, since `single_thread` is
documented to be a no-op in that case."""
from __future__ import annotations

import time

import pytest

from reflex_sentry.eval import latency as L

TEXTS = [f"prompt number {i}" for i in range(37)]


def _fake_predict(calls: list) -> callable:
    def _fn(batch: list[str]):
        calls.append(batch[0])
        return None
    return _fn


# -------------------------------------------------------------- time_per_prompt --

def test_time_per_prompt_returns_fixed_size_sample():
    calls: list[str] = []
    out = L.time_per_prompt(_fake_predict(calls), TEXTS, sample_size=10, seed=0, warmup=3)
    assert len(out) == 10
    assert set(out.keys()) <= set(range(len(TEXTS)))
    assert all(v >= 0 for v in out.values())


def test_time_per_prompt_clips_to_n():
    calls: list[str] = []
    out = L.time_per_prompt(_fake_predict(calls), TEXTS[:5], sample_size=200, seed=0, warmup=2)
    assert len(out) == 5
    assert set(out.keys()) == {0, 1, 2, 3, 4}


def test_time_per_prompt_empty_texts():
    assert L.time_per_prompt(_fake_predict([]), [], sample_size=10) == {}


def test_time_per_prompt_deterministic_indices_for_fixed_seed():
    a = L.time_per_prompt(_fake_predict([]), TEXTS, sample_size=12, seed=7, warmup=1)
    b = L.time_per_prompt(_fake_predict([]), TEXTS, sample_size=12, seed=7, warmup=1)
    assert set(a.keys()) == set(b.keys())
    # a different seed picks a different sample (astronomically unlikely to collide by chance)
    c = L.time_per_prompt(_fake_predict([]), TEXTS, sample_size=12, seed=1, warmup=1)
    assert set(a.keys()) != set(c.keys())


def test_time_per_prompt_excludes_warmup_from_sample_and_from_call_count():
    calls: list[str] = []
    out = L.time_per_prompt(_fake_predict(calls), TEXTS, sample_size=6, seed=0, warmup=4)
    # warmup calls + sampled calls, and the sample itself has exactly 6 entries
    assert len(calls) == 4 + 6
    assert len(out) == 6


def test_time_per_prompt_times_the_whole_fn_call():
    def _slow(batch):
        time.sleep(0.01)

    out = L.time_per_prompt(_slow, TEXTS[:4], sample_size=4, seed=0, warmup=1)
    assert all(v >= 9.0 for v in out.values())  # >= ~10ms, allowing scheduler slack


# ------------------------------------------------------------------ single_thread --

def test_single_thread_is_noop_without_torch(monkeypatch):
    monkeypatch.setattr(L, "torch", None)
    with L.single_thread():
        pass  # must not raise


torch = pytest.importorskip("torch")


def test_single_thread_sets_and_restores_thread_count():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, previous))
        before = torch.get_num_threads()
        with L.single_thread():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == before
    finally:
        torch.set_num_threads(previous)


def test_single_thread_restores_on_exception():
    previous = torch.get_num_threads()
    torch.set_num_threads(max(2, previous))
    before = torch.get_num_threads()
    with pytest.raises(ValueError):
        with L.single_thread():
            assert torch.get_num_threads() == 1
            raise ValueError("boom")
    assert torch.get_num_threads() == before
    torch.set_num_threads(previous)


def test_single_thread_accepts_explicit_thread_count():
    previous = torch.get_num_threads()
    try:
        with L.single_thread(2):
            assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)


# --------------------------------------------------------------- summarize/sidecar --

def test_summarize_from_dict_and_sequence():
    d = {0: 1.0, 1: 2.0, 2: 3.0}
    s1 = L.summarize(d)
    s2 = L.summarize(list(d.values()))
    assert s1 == s2
    assert s1["n"] == 3
    assert s1["p50_ms"] == pytest.approx(2.0)


def test_summarize_empty():
    s = L.summarize({})
    assert s == {"p50_ms": None, "p95_ms": None, "n": 0}


def test_write_sidecar_writes_protocol_and_stats(tmp_path):
    import json

    path = tmp_path / "preds" / "stage_x_latency.json"
    payload = L.write_sidecar(path, {0: 1.0, 1: 2.0}, threads=1, sample_size=200)
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved == payload
    assert payload["latency_protocol"] == {
        "threads": 1, "batch": 1, "sample_size": 200, "includes_tokenization": True,
    }
    assert payload["n"] == 2

"""Shared CPU latency protocol for reflex-sentry predictors (README 5.1 /
5.4).

The comparison table's "CPU p50 ms" column is only meaningful if every row
was measured the same way. The deployment framing (README 5.4) is a
single-core gate, batch size 1, end to end: raw text in, probabilities out.
Concretely, every predictor that reports a latency number for that table
(the keyword baseline, Stage A, Stage B, and the int8 ONNX export) measures
it as:

  - batch size 1 -- no batching tricks, one prompt per timed call
  - 1 CPU thread -- `single_thread()` pins `torch.set_num_threads(1)` for
    predictors that use torch (Stage A's sentence-transformers embedder,
    Stage B's encoder); the keyword baseline is already single-threaded
    Python and `single_thread()` is simply a harmless no-op for it
  - end to end -- tokenization or sentence-embedding, the forward pass (or
    rule matching), and the final softmax/normalization to probabilities
    are all *inside* the timed region, not just the model call
  - warmup excluded -- a fixed number of untimed calls run first (lazy
    imports, weight materialization, thread-pool spin-up, and similar
    one-time costs should not pollute the measurement), and are not part of
    the returned sample
  - a fixed sample, not every row -- timing every row of a large split one
    prompt at a time is not worth the wall time, so a `random.Random(seed)`
    sample of indices is timed and the rest are left to the caller (as NaN,
    or filled with the sample mean, depending on the predictor's existing
    CSV convention); the sample is deterministic across runs of the same
    split for the same `seed`

`time_per_prompt` is the one function every predictor calls to produce that
sample; `single_thread` is the context manager every torch-based predictor
wraps it in. `summarize` and `write_sidecar` are small conveniences so the
three predictors do not each reimplement "compute p50/p95 and dump JSON":
predict should not rewrite training metadata.json, and latency does not
belong in the logits CSV either, so it goes in its own
`preds/<model>_latency.json` sidecar instead.
"""
from __future__ import annotations

import contextlib
import json
import random
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised indirectly wherever torch is absent
    import torch
except ImportError:  # pragma: no cover
    torch = None


@contextlib.contextmanager
def single_thread(threads: int = 1):
    """Pin torch to `threads` CPU thread(s) (default 1, the single-core
    deployment gate this protocol times against) for the duration of the
    block, restoring the previous thread count on exit -- including when the
    block raises. A no-op when torch is not importable (the keyword
    baseline has no torch dependency at all)."""
    if torch is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def time_per_prompt(fn: Callable[[list], object], texts: Sequence[str], sample_size: int = 200,
                     seed: int = 0, warmup: int = 5) -> dict[int, float]:
    """Time `fn([text])` at batch size 1, in ms, for a fixed random sample of
    indices into `texts`, after `warmup` untimed calls.

    `fn` receives a length-1 list (the batch-size-1 call the protocol
    measures) and its return value is ignored -- callers do whatever
    tokenization/embedding/forward/softmax work they want to measure inside
    it. Returns `{index: latency_ms}` for the sampled indices only, sized
    `min(sample_size, len(texts))`. The sample is chosen by
    `random.Random(seed)`, so the same `(texts, sample_size, seed)` always
    picks the same indices."""
    texts = list(texts)
    n = len(texts)
    if n == 0:
        return {}

    rng = random.Random(seed)
    k = min(sample_size, n)
    sample = rng.sample(range(n), k)

    for i in range(warmup):
        fn([texts[i % n]])

    out: dict[int, float] = {}
    for i in sample:
        start = time.perf_counter()
        fn([texts[i]])
        out[i] = (time.perf_counter() - start) * 1000.0
    return out


def summarize(latencies: Mapping[int, float] | Sequence[float]) -> dict:
    """p50/p95 (ms) and the sample count over the values in `latencies`
    (either a `{index: ms}` mapping, as returned by `time_per_prompt`, or a
    plain sequence of ms values)."""
    values = list(latencies.values()) if isinstance(latencies, Mapping) else list(latencies)
    if not values:
        return {"p50_ms": None, "p95_ms": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "n": int(arr.size)}


def write_sidecar(path: str | Path, latencies: Mapping[int, float] | Sequence[float], *, threads: int,
                   sample_size: int, batch: int = 1, includes_tokenization: bool = True) -> dict:
    """Write the shared latency sidecar (README 5.4: `preds/<model>_latency.json`,
    not the logits CSV and not the training metadata.json) -- the protocol
    used plus p50/p95 of the sampled latencies. Returns the dict written."""
    payload = {
        "latency_protocol": {
            "threads": threads,
            "batch": batch,
            "sample_size": sample_size,
            "includes_tokenization": includes_tokenization,
        },
        **summarize(latencies),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload

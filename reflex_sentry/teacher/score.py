"""Batched teacher (guard-LLM) scorer for reflex-sentry (docs/WORKFLOWS.md).

For each prompt, records the teacher's probability of the "unsafe" verdict
read off the *first decision token's* distribution (a softmax restricted to
the verdict tokens, not a full generation), plus a predicted category from a
short greedy continuation. Two presets, in `PRESETS`:

llama_guard_3_8b (meta-llama/Llama-Guard-3-8B)
    A single user-turn chat template; the model answers "safe" or
    "unsafe\\nS<n>". We do not trust a hardcoded token id for "safe" and
    "unsafe": at runtime we encode both strings with the tokenizer and walk
    them token by token to find the first position where they diverge (see
    `find_diverging_tokens`). In the ordinary case this is token 0 of each
    (e.g. "safe" -> [safe_id], "unsafe" -> [un_id, safe_id]), so the shared
    prefix is empty and we just compare the next-token logits after the
    template's generation prompt. If the two encodings happened to share a
    non-empty prefix, that shared prefix is appended to the prompt before
    reading the comparison logits, so the comparison always happens at the
    first token where the two verdict words actually differ. p_unsafe is
    softmax(logit[unsafe_tok], logit[safe_tok]) restricted to those two
    logits (i.e. p_unsafe = 1 - p_safe by construction: this is a binary
    verdict). The category is read from a short greedy continuation
    ("safe" or "unsafe\\nS<n>"): we regex out "S<digits>" and store it
    verbatim (e.g. "S2"); "" if none is found (including on a "safe"
    verdict, which has no category).

qwen3guard_gen_8b (Qwen/Qwen3Guard-Gen-8B)
    Answer format "Safety: Safe|Unsafe|Controversial" then
    "Categories: ...". We *prefill* the assistant turn with the literal
    string "Safety:" (appended after the chat template's generation
    prompt) and read the next-token distribution over the first tokens of
    " Safe", " Unsafe", " Controversial" (leading space, per the model's
    own formatting -- these three strings live in the preset config so a
    different model-card format can be fixed without touching this file).
    p_unsafe = P(Unsafe) and p_controversial = P(Controversial), each
    normalized over the three (they do not have to sum to 1 in the raw
    logits since other tokens also have mass; we only care about the
    relative weight of these three). Category is the text following the
    literal "Categories:" in a short greedy continuation, stored verbatim.

Both presets store the raw teacher category string (not our own cat:*
taxonomy) since the teacher's taxonomy is not the project's; `targets.py` only
uses the probabilities.

Batching: left-padding puts every sequence's last real token in the last
column, so the "next token" distribution for a whole batch is always
`logits[:, -1, :]` regardless of each prompt's length. `torch`/`transformers`
are imported lazily inside functions so this module (and its pure-Python
pieces: token-diverging search, verdict-probability math, category
regexes, batching/resume/dedupe orchestration) stays importable and
testable without either installed.

CLI
---
    python -m reflex_sentry.teacher.score --model llama_guard_3_8b \\
        --inputs data/processed/train.parquet data/processed/val_pool.parquet \\
                 data/processed/test_pool.parquet data/processed/test_ood_pool.parquet \\
        --out data/interim/teacher_scores_llama_guard_3_8b.parquet \\
        --batch-size 8 --max-length 512 --load-in-4bit --limit 20

Resume: if `--out` already exists, ids already scored there are skipped;
new rows are appended and the file is rewritten every `--checkpoint-every`
batches (and once more at the end), so an interrupted run can be restarted
with the same command and only scores the remaining ids.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

DEFAULT_INPUTS = (
    "data/processed/train.parquet",
    "data/processed/val_pool.parquet",
    "data/processed/test_pool.parquet",
    "data/processed/test_ood_pool.parquet",
)

# One config dict per model. Keep format strings here (not hardcoded in the
# scoring functions) so a model-card mismatch can be fixed by editing this
# table. `kind` selects which scoring function in this module handles it.
PRESETS = {
    "llama_guard_3_8b": {
        "hf_id": "meta-llama/Llama-Guard-3-8B",
        "kind": "chat_verdict",
        "safe_word": "safe",
        "unsafe_word": "unsafe",
        "category_regex": r"S(\d+)",
        "category_max_new_tokens": 8,
    },
    "qwen3guard_gen_8b": {
        "hf_id": "Qwen/Qwen3Guard-Gen-8B",
        "kind": "prefill_verdict",
        "prefill_text": "Safety:",
        "verdict_words": {"safe": " Safe", "unsafe": " Unsafe", "controversial": " Controversial"},
        "category_prefix": "Categories:",
        "category_max_new_tokens": 32,
    },
}


# ---------------------------------------------------------------------------
# Pure logic: token finding, probability extraction, category parsing.
# No torch/transformers import here, so these are unit-testable directly.
# ---------------------------------------------------------------------------


def find_diverging_tokens(tokenizer, word_a: str, word_b: str) -> tuple[list[int], int, int]:
    """Encode `word_a`/`word_b` (no special tokens) and find the first token
    position where the two encodings differ.

    Returns (shared_prefix_ids, token_id_a, token_id_b): `shared_prefix_ids`
    is the (possibly empty) list of leading token ids common to both, and
    `token_id_a`/`token_id_b` are the first tokens where they diverge. If
    one encoding is a strict prefix of the other (no divergence found), we
    fall back to each encoding's own first token and log a warning, since
    that case means the two verdict words are not distinguishable at a
    single next-token comparison the way we expect for this preset.
    """
    ids_a = tokenizer.encode(word_a, add_special_tokens=False)
    ids_b = tokenizer.encode(word_b, add_special_tokens=False)
    prefix: list[int] = []
    for a, b in zip(ids_a, ids_b):
        if a != b:
            return prefix, a, b
        prefix.append(a)
    print(
        f"[reflex_sentry.teacher.score] WARNING: could not find a diverging token "
        f"between {word_a!r} ({ids_a}) and {word_b!r} ({ids_b}); "
        f"falling back to each word's first token."
    )
    return [], ids_a[0], ids_b[0]


def first_token_id(tokenizer, s: str) -> int:
    """First token id of `s` when encoded without special tokens."""
    ids = tokenizer.encode(s, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no tokens for {s!r}")
    return ids[0]


def binary_verdict_prob(logits_row: Sequence[float], unsafe_id: int, safe_id: int) -> float:
    """p_unsafe from a single row of next-token logits, restricted to the
    two verdict token ids (softmax over just those two logits)."""
    logits_row = np.asarray(logits_row, dtype=np.float64)
    a, b = logits_row[unsafe_id], logits_row[safe_id]
    m = max(a, b)
    ea, eb = np.exp(a - m), np.exp(b - m)
    return float(ea / (ea + eb))


def three_way_verdict_probs(
    logits_row: Sequence[float], safe_id: int, unsafe_id: int, controversial_id: int
) -> tuple[float, float, float]:
    """(p_safe, p_unsafe, p_controversial) from a single row of next-token
    logits, restricted to and normalized over the three verdict token ids."""
    logits_row = np.asarray(logits_row, dtype=np.float64)
    vals = np.array([logits_row[safe_id], logits_row[unsafe_id], logits_row[controversial_id]])
    vals = vals - vals.max()
    ex = np.exp(vals)
    ex = ex / ex.sum()
    return float(ex[0]), float(ex[1]), float(ex[2])


def parse_llama_guard_category(continuation: str, preset: dict) -> str:
    """Pull "S<n>" out of a short greedy continuation, verbatim (e.g. "S2");
    "" if no S-code is present (a "safe" verdict has none)."""
    m = re.search(preset["category_regex"], continuation)
    return f"S{m.group(1)}" if m else ""


def parse_qwen_guard_category(continuation: str, preset: dict) -> str:
    """Text following the literal "Categories:" prefix, verbatim, stripped;
    "" if the prefix is not present in the continuation."""
    prefix = preset["category_prefix"]
    idx = continuation.find(prefix)
    if idx == -1:
        return ""
    rest = continuation[idx + len(prefix):].strip()
    first = rest.splitlines()[0].strip() if rest else ""
    return "" if first.lower() == "none" else first


# ---------------------------------------------------------------------------
# Model loading (lazy imports) and the two preset-specific batch scorers.
# Each returns a `score_fn(list[str]) -> list[dict]` closure over a loaded
# model/tokenizer, so `run_scorer` below never needs to know which preset
# it is calling.
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(preset: dict, load_in_4bit: bool = False, device_map: str | None = None):
    """Lazy-imports torch/transformers. Not exercised by unit tests (no
    network access to Hugging Face in CI); see tests/test_teacher.py for the
    mocked-model tests of everything downstream of a loaded model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = preset["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = device_map or "auto"
    elif device_map:
        kwargs["device_map"] = device_map
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tokenizer


def truncate_user_texts(tokenizer, texts: list[str], max_tokens: int) -> list[str]:
    """Cut each user prompt to its first `max_tokens` tokens BEFORE templating.

    Truncating the templated string instead would cut its tail, which holds
    the assistant header where the verdict is read.
    """
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        out.append(tokenizer.decode(ids[:max_tokens]) if len(ids) > max_tokens else t)
    return out


def _to_model_device(enc, model):
    device = getattr(model, "device", None)
    return {k: v.to(device) for k, v in enc.items()} if device is not None else enc


def _forward_last_logits(model, enc):
    """Next-token logits at the last position, with position ids that ignore
    left padding (otherwise each row's positions shift by its pad count)."""
    import torch

    mask = enc["attention_mask"]
    position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=mask, position_ids=position_ids)
    return out.logits[:, -1, :].float().cpu().numpy()


def _chat_prompt_strings(tokenizer, texts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        )
        for t in texts
    ]


def make_llama_guard_score_fn(
    model, tokenizer, preset: dict, max_length: int = 512
) -> Callable[[list[str]], list[dict]]:
    import torch

    prefix_ids, unsafe_id, safe_id = find_diverging_tokens(
        tokenizer, preset["unsafe_word"], preset["safe_word"]
    )
    print(
        f"[llama_guard] verdict token ids: safe={safe_id!r} unsafe={unsafe_id!r} "
        f"shared_prefix={prefix_ids!r}"
    )
    prefix_str = tokenizer.decode(prefix_ids) if prefix_ids else ""
    _warned_steps: list[bool] = []

    def score_fn(texts: list[str]) -> list[dict]:
        texts = truncate_user_texts(tokenizer, texts, max_length)
        prompts = [p + prefix_str for p in _chat_prompt_strings(tokenizer, texts)]
        # Llama Guard 3 emits "\n\n" before the verdict word. If the top
        # next token is not a verdict token, append it and read one step
        # later (at most max_verdict_steps extra steps).
        for step in range(preset.get("max_verdict_steps", 2) + 1):
            enc = _to_model_device(tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False,
            ), model)
            last_logits = _forward_last_logits(model, enc)
            top = last_logits.argmax(axis=1)
            needs = [i for i, t in enumerate(top) if int(t) not in (safe_id, unsafe_id)]
            if not needs or step == preset.get("max_verdict_steps", 2):
                break
            if not _warned_steps:
                print(f"[llama_guard] top token before verdict is "
                      f"{tokenizer.decode([int(top[needs[0]])])!r}; reading one step later")
                _warned_steps.append(True)
            prompts = [p + tokenizer.decode([int(top[i])]) if i in needs else p
                       for i, p in enumerate(prompts)]

        gen_enc = _to_model_device(tokenizer(
            _chat_prompt_strings(tokenizer, texts), return_tensors="pt", padding=True,
            add_special_tokens=False,
        ), model)
        with torch.no_grad():
            gen = model.generate(
                **gen_enc, max_new_tokens=preset["category_max_new_tokens"], do_sample=False
            )
        new_tokens = gen[:, gen_enc["input_ids"].shape[1]:]
        continuations = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        rows = []
        for i in range(len(texts)):
            p_unsafe = binary_verdict_prob(last_logits[i], unsafe_id, safe_id)
            rows.append({
                "p_unsafe_teacher": p_unsafe,
                "p_controversial": np.nan,
                "teacher_category": parse_llama_guard_category(continuations[i], preset),
                "teacher_raw": continuations[i][:60],
            })
        return rows

    return score_fn


def make_qwen_guard_score_fn(
    model, tokenizer, preset: dict, max_length: int = 512
) -> Callable[[list[str]], list[dict]]:
    import torch

    safe_id = first_token_id(tokenizer, preset["verdict_words"]["safe"])
    unsafe_id = first_token_id(tokenizer, preset["verdict_words"]["unsafe"])
    controversial_id = first_token_id(tokenizer, preset["verdict_words"]["controversial"])
    print(
        f"[qwen3guard] verdict token ids: safe={safe_id!r} unsafe={unsafe_id!r} "
        f"controversial={controversial_id!r}"
    )

    def score_fn(texts: list[str]) -> list[dict]:
        texts = truncate_user_texts(tokenizer, texts, max_length)
        prompts = [p + preset["prefill_text"] for p in _chat_prompt_strings(tokenizer, texts)]
        enc = _to_model_device(tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        ), model)
        last_logits = _forward_last_logits(model, enc)

        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=preset["category_max_new_tokens"], do_sample=False)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        continuations = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        rows = []
        for i in range(len(texts)):
            p_safe, p_unsafe, p_ctrl = three_way_verdict_probs(
                last_logits[i], safe_id, unsafe_id, controversial_id
            )
            rows.append({
                "p_unsafe_teacher": p_unsafe,
                "p_controversial": p_ctrl,
                "teacher_category": parse_qwen_guard_category(continuations[i], preset),
                "teacher_raw": continuations[i][:60],
            })
        return rows

    return score_fn


def score_fn_for_preset(preset_name: str, model, tokenizer, max_length: int = 512):
    preset = PRESETS[preset_name]
    if preset["kind"] == "chat_verdict":
        return make_llama_guard_score_fn(model, tokenizer, preset, max_length)
    if preset["kind"] == "prefill_verdict":
        return make_qwen_guard_score_fn(model, tokenizer, preset, max_length)
    raise ValueError(f"unknown preset kind {preset['kind']!r}")


# ---------------------------------------------------------------------------
# Orchestration: load inputs, dedupe ids, resume, batch, checkpoint. Takes
# `score_fn` as a plain callable so it is fully testable with a fake one.
# ---------------------------------------------------------------------------


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def load_and_dedupe_inputs(input_paths: Sequence[str]) -> pd.DataFrame:
    """Combine pools, allowing only identical overlaps of the same id."""
    from .artifact import nonblank_string

    frames = []
    for p in input_paths:
        df = read_any(p)
        missing = {"id", "text"} - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing required column(s): {sorted(missing)}")
        if not df["id"].map(nonblank_string).all():
            raise ValueError(f"{p} contains null or blank id")
        if not df["text"].map(nonblank_string).all():
            raise ValueError(f"{p} contains null or blank text")
        frames.append(df[["id", "text"]])
    if not frames:
        return pd.DataFrame(columns=["id", "text"])
    combined = pd.concat(frames, ignore_index=True)
    if combined.groupby("id")["text"].nunique().gt(1).any():
        raise ValueError("input pools contain conflicting text for the same id")
    return combined.drop_duplicates(subset="id", keep="first").reset_index(drop=True)


def _prepare_run(input_paths, out_path, teacher_model, batch_size, limit, checkpoint_every):
    from .artifact import nonblank_string, validate_score_frame

    if not nonblank_string(teacher_model):
        raise ValueError("teacher_model must be a nonblank string")
    for name, value in (("batch_size", batch_size), ("checkpoint_every", checkpoint_every)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer")

    todo = load_and_dedupe_inputs(input_paths)
    if limit is not None:
        todo = todo.head(limit)
    out_file = Path(out_path)
    done = None
    if out_file.exists():
        done = pd.read_parquet(out_file)
        actual_model = validate_score_frame(done, str(out_file))
        if actual_model is not None and actual_model != teacher_model:
            raise ValueError(f"{out_file} teacher_model {actual_model!r} does not match {teacher_model!r}")
        todo = todo[~todo["id"].isin(set(done["id"]))].reset_index(drop=True)
    return todo, done, out_file


def run_scorer(
    score_fn: Callable[[list[str]], list[dict]],
    input_paths: Sequence[str],
    out_path: str,
    teacher_model: str,
    batch_size: int = 8,
    limit: int | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Score every (id, text) from `input_paths` not already present in
    `out_path`, in batches of `batch_size`, writing the accumulated result
    to `out_path` every `checkpoint_every` batches and once more at the end.
    Returns the full resulting DataFrame (previously scored + newly scored).
    """
    prepared = _prepare_run(
        input_paths, out_path, teacher_model, batch_size, limit, checkpoint_every
    )
    return _run_prepared(score_fn, prepared, teacher_model, batch_size, checkpoint_every)


def _run_prepared(score_fn, prepared, teacher_model, batch_size, checkpoint_every):
    """Execute a validated run, including its periodic checkpoints."""
    from .artifact import valid_probability

    todo, done, out_file = prepared
    new_rows: list[dict] = []

    def _flush():
        nonlocal done
        if not new_rows:
            return
        chunk = pd.DataFrame(new_rows)
        done = chunk if done is None or done.empty else pd.concat([done, chunk], ignore_index=True)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(out_file, index=False)

    n_batches = 0
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        scored = score_fn(batch["text"].tolist())
        if not isinstance(scored, (list, tuple)) or len(scored) != len(batch):
            raise ValueError("score_fn must return exactly one result per batch input")
        batch_rows = []
        for id_, s in zip(batch["id"].tolist(), scored):
            if not isinstance(s, dict) or not valid_probability(s.get("p_unsafe_teacher")):
                raise ValueError("score_fn returned invalid p_unsafe_teacher")
            if not valid_probability(s.get("p_controversial", np.nan), optional=True):
                raise ValueError("score_fn returned invalid p_controversial")
            batch_rows.append({
                "id": id_,
                "p_unsafe_teacher": s["p_unsafe_teacher"],
                "p_controversial": s.get("p_controversial", np.nan),
                "teacher_category": s.get("teacher_category", ""),
                "teacher_raw": s.get("teacher_raw", ""),
                "teacher_model": teacher_model,
            })
        new_rows.extend(batch_rows)
        n_batches += 1
        if n_batches % checkpoint_every == 0:
            _flush()
            new_rows = []
    _flush()
    if done is None:
        done = pd.DataFrame(columns=["id", "p_unsafe_teacher", "p_controversial",
                                     "teacher_category", "teacher_raw", "teacher_model"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PRESETS), help="preset name")
    ap.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--device-map", default=None, help='e.g. "auto"')
    ap.add_argument("--limit", type=int, default=None, help="smoke-run only the first N ids")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="batches between writes to --out")
    a = ap.parse_args()

    prepared = _prepare_run(a.inputs, a.out, a.model, a.batch_size, a.limit, a.checkpoint_every)
    if prepared[0].empty:
        if prepared[1] is None:
            print(f"no teacher scores to write to {a.out}")
        else:
            print(f"no new teacher scores; {len(prepared[1])} already in {a.out}")
        return
    preset = PRESETS[a.model]
    model, tokenizer = load_model_and_tokenizer(preset, load_in_4bit=a.load_in_4bit, device_map=a.device_map)
    score_fn = score_fn_for_preset(a.model, model, tokenizer, max_length=a.max_length)

    if a.limit:
        preview = score_fn(["How do I check what ports are open on my own server?"])
        print(f"[reflex_sentry.teacher.score] smoke preview for {a.model}: {preview}")

    df = _run_prepared(score_fn, prepared, a.model, a.batch_size, a.checkpoint_every)
    print(f"wrote {len(df)} teacher scores to {a.out}")


if __name__ == "__main__":
    main()

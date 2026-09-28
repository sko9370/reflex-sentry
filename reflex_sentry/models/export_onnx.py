"""Export the Stage B student to ONNX, int8-quantize it, check parity, and
predict with an onnxruntime CPU session.

Pipeline (README 4.5): export fp32 ONNX -> dynamic int8 quantization ->
confirm the quantized model's val metrics match fp32 within noise -> only
then run on the test sets.

    python -m reflex_sentry.models.export_onnx export   --model models/stage_b
    python -m reflex_sentry.models.export_onnx quantize --onnx models/stage_b/model.onnx
    python -m reflex_sentry.models.export_onnx parity   --model models/stage_b --val data/processed/val.parquet
    python -m reflex_sentry.models.export_onnx predict  --model models/stage_b --splits val test test_ood test_evasion

This module is importable without onnx/onnxruntime installed (lazy imports);
calling its functions without them raises a clear ImportError.
"""
from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from reflex_sentry.models import stage_b as SB

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # pragma: no cover
    import onnx
except ImportError:  # pragma: no cover
    onnx = None

try:  # pragma: no cover
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

RECALL_TOLERANCE = 0.02  # README 4.5: "match the fp32 model within noise"


def _require_onnx() -> None:
    if onnx is None or torch is None:
        raise ImportError("onnx and torch are required for export; pip install onnx torch")


def _require_ort() -> None:
    if ort is None:
        raise ImportError("onnxruntime is required for this step; pip install onnxruntime")


# ------------------------------------------------------------------ export --

def export_to_onnx(model_dir: str | Path, onnx_path: str | Path | None = None, opset: int = 17) -> Path:
    """torch.onnx.export with dynamic batch and sequence axes. The graph's
    single output ("logits") is the 3-way [safe, dangerous, unsure] head."""
    _require_onnx()
    student, tokenizer, metadata = SB.load_student(model_dir, device="cpu")
    max_len = int(metadata.get("max_len", 256))
    onnx_path = Path(onnx_path) if onnx_path is not None else Path(model_dir) / "model.onnx"

    dummy = tokenizer(
        ["export dummy sentence one.", "a second, slightly longer dummy sentence for tracing."],
        padding=True, truncation=True, max_length=min(16, max_len), return_tensors="pt",
    )
    student.eval()
    torch.onnx.export(
        student,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        # torch >= 2.6 defaults to the dynamo-based exporter, which needs the
        # optional `onnxscript` package and takes `dynamic_shapes` instead of
        # `dynamic_axes`. Force the legacy TorchScript-based exporter, which
        # takes `dynamic_axes` directly and has no extra dependency.
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    print(f"[export_onnx] wrote fp32 ONNX graph to {onnx_path}")
    return onnx_path


def quantize_int8(onnx_path: str | Path, int8_path: str | Path | None = None) -> Path:
    """Dynamic int8 quantization via onnxruntime.quantization.quantize_dynamic."""
    _require_ort()
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    int8_path = Path(int8_path) if int8_path is not None else onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"[export_onnx] wrote int8 ONNX graph to {int8_path}")
    return int8_path


# ---------------------------------------------------------------- ORT infer --

def _ort_session(onnx_path: str | Path, intra_op_num_threads: int | None = None) -> "ort.InferenceSession":
    _require_ort()
    so = ort.SessionOptions()
    if intra_op_num_threads is not None:
        so.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def _ort_logits(sess: "ort.InferenceSession", tokenizer, texts: list[str], max_len: int,
                 batch_size: int = 32) -> np.ndarray:
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        chunks.append(sess.run(["logits"], feed)[0])
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float32)


def _ort_latency_ms(sess: "ort.InferenceSession", tokenizer, texts: list[str], max_len: int,
                     sample_size: int) -> dict[int, float]:
    idx = list(range(len(texts)))
    rng = random.Random(0)
    sample = idx if len(idx) <= sample_size else rng.sample(idx, sample_size)
    out: dict[int, float] = {}
    for i in sample:
        enc = tokenizer([texts[i]], truncation=True, max_length=max_len, padding=True, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        t0 = time.perf_counter()
        sess.run(["logits"], feed)
        t1 = time.perf_counter()
        out[i] = (t1 - t0) * 1000.0
    return out


# ------------------------------------------------------------------ parity --

def _recall_dangerous_at(logits: np.ndarray, gold: np.ndarray, t: float) -> float:
    dangerous = gold == "dangerous"
    if not dangerous.any():
        return float("nan")
    p_safe = SB._softmax(logits)[dangerous, 0]
    return float((p_safe < t).mean())


def parity_check(model_dir: str | Path, onnx_path: str | Path, int8_path: str | Path,
                  val_parquet: str | Path, target_recall: float = 0.95,
                  tolerance: float = RECALL_TOLERANCE, batch_size: int = 32) -> dict:
    """Compare fp32 torch vs fp32 onnx vs int8 onnx logits on val. Reports max
    abs diff and argmax agreement for each pair, and warns loudly if int8
    recall-at-the-fp32-threshold differs from fp32 recall by more than
    `tolerance` (README 4.5: confirm the quantized model matches fp32 within
    noise before running any test set)."""
    _require_onnx()
    _require_ort()
    from reflex_sentry.eval import metrics as M

    student, tokenizer, metadata = SB.load_student(model_dir, device="cpu")
    max_len = int(metadata.get("max_len", 256))

    df = pd.read_parquet(val_parquet)
    missing = {"id", "text", "gold"} - set(df.columns)
    if missing:
        raise ValueError(f"{val_parquet} missing columns: {sorted(missing)}")
    texts = df["text"].tolist()
    gold = df["gold"].to_numpy()

    torch_logits = SB.infer_logits(student, tokenizer, texts, max_len, "cpu", batch_size=batch_size)
    fp32_sess = _ort_session(onnx_path)
    int8_sess = _ort_session(int8_path)
    onnx_fp32_logits = _ort_logits(fp32_sess, tokenizer, texts, max_len, batch_size=batch_size)
    onnx_int8_logits = _ort_logits(int8_sess, tokenizer, texts, max_len, batch_size=batch_size)

    def compare(a: np.ndarray, b: np.ndarray, name: str) -> dict:
        max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        agree = float((a.argmax(1) == b.argmax(1)).mean()) if a.size else float("nan")
        print(f"[parity] {name}: max abs logit diff {max_diff:.5f}, argmax agreement {agree:.4f}")
        return {"max_abs_diff": max_diff, "argmax_agreement": agree}

    result = {
        "torch_vs_onnx_fp32": compare(torch_logits, onnx_fp32_logits, "torch vs onnx fp32"),
        "torch_vs_onnx_int8": compare(torch_logits, onnx_int8_logits, "torch vs onnx int8"),
    }

    pred_df = pd.DataFrame({"gold": gold, "p_safe": SB._softmax(torch_logits)[:, 0]})
    t_fp32 = M.select_threshold(pred_df, target_recall)
    recall_fp32 = _recall_dangerous_at(torch_logits, gold, t_fp32)
    recall_onnx_fp32 = _recall_dangerous_at(onnx_fp32_logits, gold, t_fp32)
    recall_int8 = _recall_dangerous_at(onnx_int8_logits, gold, t_fp32)
    diff = abs(recall_fp32 - recall_int8)
    result["threshold"] = t_fp32
    result["recall_fp32_torch"] = recall_fp32
    result["recall_fp32_onnx"] = recall_onnx_fp32
    result["recall_int8_onnx"] = recall_int8
    result["recall_diff"] = diff
    result["recall_within_tolerance"] = diff <= tolerance

    print(f"[parity] val dangerous recall @ t={t_fp32:.4f}: fp32 torch={recall_fp32:.4f} "
          f"fp32 onnx={recall_onnx_fp32:.4f} int8 onnx={recall_int8:.4f} (diff {diff:.4f})")

    if diff > tolerance:
        banner = (
            "!" * 70 + "\n"
            f"WARNING: int8 quantization changed val dangerous recall by {diff:.4f} "
            f"(> tolerance {tolerance:.4f}) at the fp32-selected threshold.\n"
            "Do not trust int8 test-set numbers until this is investigated "
            "(README 4.5: confirm the quantized model matches fp32 within noise).\n"
            + "!" * 70
        )
        print(banner)
        warnings.warn(banner, stacklevel=2)

    return result


# ---------------------------------------------------------------- predict --

def predict_int8_split(int8_path: str | Path, tokenizer, metadata: dict, split: str, data_dir: str | Path,
                        out_dir: str | Path, intra_op_num_threads: int = 1, batch_size: int = 32,
                        latency_sample: int = 200) -> Path | None:
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        print(f"[export_onnx predict] {split}: {path} not found, skipping")
        return None
    df = pd.read_parquet(path)
    if {"id", "text"} - set(df.columns):
        raise ValueError(f"{path} missing required columns id, text")
    max_len = int(metadata.get("max_len", 256))
    texts = df["text"].tolist()

    sess = _ort_session(int8_path, intra_op_num_threads=intra_op_num_threads)
    logits = _ort_logits(sess, tokenizer, texts, max_len, batch_size=batch_size)
    latencies = _ort_latency_ms(sess, tokenizer, texts, max_len, latency_sample)

    out = pd.DataFrame({
        "id": df["id"],
        "gold": df["gold"] if "gold" in df.columns else "",
        "logit_safe": logits[:, 0],
        "logit_dangerous": logits[:, 1],
        "logit_unsure": logits[:, 2],
        "source": df["source"] if "source" in df.columns else "",
        "tags": df["tags"] if "tags" in df.columns else "",
    })
    out["latency_ms"] = [latencies.get(i, float("nan")) for i in range(len(df))]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage_b_int8_{split}_logits.csv"
    out.to_csv(out_path, index=False)
    print(f"[export_onnx predict] wrote {out_path} ({len(out)} rows, intra_op_num_threads={intra_op_num_threads})")
    return out_path


def predict_int8(model_dir: str | Path, int8_path: str | Path, splits: list[str],
                  data_dir: str | Path = "data/processed", out_dir: str | Path = "preds",
                  intra_op_num_threads: int = 1, batch_size: int = 32,
                  latency_sample: int = 200) -> list[Path]:
    """onnxruntime CPU predictor. `intra_op_num_threads` defaults to 1 to mimic
    a single-core gate deployment (README's CPU-latency framing); raise it to
    use more cores for throughput at the cost of that single-core latency
    story."""
    _require_ort()
    from transformers import AutoTokenizer

    metadata = {}
    meta_path = Path(model_dir) / "metadata.json"
    if meta_path.exists():
        import json
        metadata = json.loads(meta_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    written = []
    for split in splits:
        p = predict_int8_split(int8_path, tokenizer, metadata, split, data_dir, out_dir,
                                intra_op_num_threads=intra_op_num_threads, batch_size=batch_size,
                                latency_sample=latency_sample)
        if p is not None:
            written.append(p)
    return written


# ---------------------------------------------------------------------- CLI --

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="torch.onnx.export to fp32 ONNX")
    e.add_argument("--model", required=True)
    e.add_argument("--out", default=None, help="default: <model>/model.onnx")
    e.add_argument("--opset", type=int, default=17)

    q = sub.add_parser("quantize", help="dynamic int8 quantization")
    q.add_argument("--onnx", required=True)
    q.add_argument("--out", default=None, help="default: <onnx>_int8.onnx")

    pa = sub.add_parser("parity", help="compare torch fp32 / onnx fp32 / onnx int8 on val")
    pa.add_argument("--model", required=True)
    pa.add_argument("--onnx", required=True)
    pa.add_argument("--int8", required=True)
    pa.add_argument("--val", default="data/processed/val.parquet")
    pa.add_argument("--target-recall", type=float, default=0.95)
    pa.add_argument("--tolerance", type=float, default=RECALL_TOLERANCE)

    p = sub.add_parser("predict", help="onnxruntime CPU predictor for the int8 model")
    p.add_argument("--model", required=True, help="dir with tokenizer + metadata.json")
    p.add_argument("--int8", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_ood", "test_evasion"])
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--intra-op-threads", type=int, default=1,
                    help="onnxruntime intra_op_num_threads; 1 mimics a single-core gate")
    p.add_argument("--latency-sample", type=int, default=200)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "export":
        export_to_onnx(args.model, onnx_path=args.out, opset=args.opset)
    elif args.command == "quantize":
        quantize_int8(args.onnx, int8_path=args.out)
    elif args.command == "parity":
        parity_check(args.model, args.onnx, args.int8, args.val,
                     target_recall=args.target_recall, tolerance=args.tolerance)
    elif args.command == "predict":
        predict_int8(args.model, args.int8, args.splits, data_dir=args.data_dir, out_dir=args.out_dir,
                     intra_op_num_threads=args.intra_op_threads, latency_sample=args.latency_sample)


if __name__ == "__main__":
    main()

"""Export the Stage B student to ONNX, int8-quantize it, check parity, and
predict with an onnxruntime CPU session.

Pipeline (README 4.5): export fp32 ONNX -> dynamic int8 quantization ->
confirm the quantized model makes the same escalate/pass decisions as fp32 on
val -> only then run on the test sets.

    python -m reflex_sentry.models.export_onnx export   --model models/stage_b
    python -m reflex_sentry.models.export_onnx quantize --onnx models/stage_b/model.onnx [--preset legacy]
    python -m reflex_sentry.models.export_onnx sweep    --model models/stage_b --onnx models/stage_b/model.onnx --val data/processed/val.parquet --choose
    python -m reflex_sentry.models.export_onnx parity   --model models/stage_b --onnx ... --int8 ... --val data/processed/val.parquet
    python -m reflex_sentry.models.export_onnx predict  --model models/stage_b --int8 ... --splits val test test_ood test_evasion

Quantization defaults (`--preset default`): MatMul only, per-channel weights,
classifier head left in fp32, qint8, no reduce-range. `--preset legacy` is the
old plain `quantize_dynamic` call (all quantizable op types incl. embedding
Gather, per-tensor), which moved val logits by ~2.7 on ModernBERT-base.

This module is importable without onnx/onnxruntime installed (lazy imports);
calling its functions without them raises a clear ImportError.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
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

# Parity warning rule (README 4.5: "match the fp32 model within noise"). The
# recall tolerance is max(1 item, MAX_RECALL_DIFF): on ~48 dangerous val rows a
# single item is 2.1 points, which is the resolution of the measurement.
MIN_DECISION_AGREEMENT = 0.97
MAX_RECALL_DIFF = 0.02
MAX_AP_DROP = 0.01


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


# ---------------------------------------------------------------- quantize --

@dataclass(frozen=True)
class QuantConfig:
    """Options for onnxruntime dynamic int8 quantization.

    op_types=None means "everything onnxruntime can quantize" (MatMul plus
    embedding Gather, etc.), i.e. the old behavior."""
    per_channel: bool = True
    op_types: tuple[str, ...] | None = ("MatMul",)
    exclude_head: bool = True
    reduce_range: bool = False
    weight_type: str = "qint8"  # qint8 | quint8

    def replace(self, **kw) -> "QuantConfig":
        return dataclasses.replace(self, **kw)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["op_types"] = list(self.op_types) if self.op_types is not None else None
        return d


PRESETS: dict[str, QuantConfig] = {
    "legacy": QuantConfig(per_channel=False, op_types=None, exclude_head=False, reduce_range=False),
    "matmul_per_channel": QuantConfig(per_channel=True, op_types=("MatMul",), exclude_head=False,
                                       reduce_range=False),
    "matmul_per_channel_exclude_head": QuantConfig(per_channel=True, op_types=("MatMul",),
                                                    exclude_head=True, reduce_range=False),
    "matmul_per_channel_exclude_head_reduce_range": QuantConfig(per_channel=True, op_types=("MatMul",),
                                                                 exclude_head=True, reduce_range=True),
}
PRESETS["default"] = PRESETS["matmul_per_channel_exclude_head"]
SWEEP_CONFIGS = ["legacy", "matmul_per_channel", "matmul_per_channel_exclude_head",
                 "matmul_per_channel_exclude_head_reduce_range"]

_HEAD_OPS = ("MatMul", "Gemm")


def resolve_quant_config(preset: str = "default", **overrides) -> QuantConfig:
    """Start from a named preset and apply explicit overrides (None = keep)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if "op_types" in overrides and overrides["op_types"] is not None:
        ops = tuple(overrides["op_types"])
        overrides["op_types"] = None if ops == ("all",) else ops
    return PRESETS[preset].replace(**overrides)


def find_head_nodes(model: "onnx.ModelProto", gemm_aliases: bool = False) -> list[str]:
    """Names of the classifier head MatMul/Gemm node(s): nodes whose name has a
    `head` path segment (torch export names them like `/head/Gemm`), plus the
    nearest MatMul/Gemm found walking back from the graph output (the last
    MatMul feeding the output).

    onnxruntime's dynamic quantizer first rewrites every Gemm into
    `<name>_MatMul` + `<name>_Add` and only then applies op-type and
    exclusion checks, so a Gemm head is quantized even with
    op_types=["MatMul"] unless `<name>_MatMul` is excluded too.
    `gemm_aliases=True` adds that name for each Gemm found."""
    graph = model.graph
    found: list[str] = []

    def add(name: str, op_type: str = "MatMul") -> None:
        if not name:
            return
        for n in (name, name + "_MatMul" if gemm_aliases and op_type == "Gemm" else None):
            if n and n not in found:
                found.append(n)

    for node in graph.node:
        if node.op_type in _HEAD_OPS:
            segments = node.name.lower().replace(".", "/").split("/")
            if "head" in segments:
                add(node.name, node.op_type)

    producer = {out: node for node in graph.node for out in node.output}
    seen: set[str] = set()
    for graph_out in graph.output:
        frontier = [graph_out.name]
        while frontier:
            tensor = frontier.pop(0)
            node = producer.get(tensor)
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if node.op_type in _HEAD_OPS:
                add(node.name, node.op_type)
                continue  # nearest MatMul/Gemm on this branch only
            frontier.extend(node.input)
    return found


def _preprocess(onnx_path: Path, workdir: Path) -> Path:
    """onnxruntime quantization pre-processing (shape inference + graph
    optimization). Falls back to the input model with a warning if unavailable
    or failing."""
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:  # pragma: no cover
        print("[export_onnx] quantization pre-processing not available in this onnxruntime; skipping")
        return onnx_path
    out = workdir / (onnx_path.stem + "_preprocessed.onnx")
    cwd = os.getcwd()
    try:
        # ORT's symbolic shape inference drops temp files (sym_shape_infer_temp.onnx,
        # *.data) into the cwd; run it from the scratch dir so nothing leaks.
        os.chdir(workdir)
        quant_pre_process(str(Path(onnx_path).resolve()), str(out.resolve()))
    except Exception as exc:  # noqa: BLE001 - shape inference can fail on exotic graphs
        print(f"[export_onnx] quantization pre-processing failed ({type(exc).__name__}: {exc}); "
              "quantizing the un-preprocessed model")
        return onnx_path
    finally:
        os.chdir(cwd)
    return out


def quantize_with_config(onnx_path: str | Path, int8_path: str | Path, config: QuantConfig,
                          preprocess: bool = False) -> dict:
    """Quantize `onnx_path` to `int8_path` with `config`. Returns an info dict
    (config, excluded nodes, preprocessing flag, file size)."""
    _require_ort()
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path, int8_path = Path(onnx_path), Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    weight_type = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}.get(config.weight_type)
    if weight_type is None:
        raise ValueError(f"weight_type must be qint8 or quint8, got {config.weight_type!r}")

    with tempfile.TemporaryDirectory() as tmp:
        src = _preprocess(onnx_path, Path(tmp)) if preprocess else onnx_path
        excluded: list[str] = []
        if config.exclude_head:
            _require_onnx()
            excluded = find_head_nodes(onnx.load(str(src), load_external_data=False), gemm_aliases=True)
            if not excluded:
                print("[export_onnx] WARNING: --exclude-head is on but no head MatMul/Gemm node was found")
        quantize_dynamic(
            str(src), str(int8_path),
            op_types_to_quantize=list(config.op_types) if config.op_types is not None else None,
            per_channel=config.per_channel,
            reduce_range=config.reduce_range,
            weight_type=weight_type,
            nodes_to_exclude=excluded or None,
        )
    return {
        "config": config.as_dict(),
        "excluded_nodes": excluded,
        "preprocess": bool(preprocess),
        "size_mb": round(int8_path.stat().st_size / 1e6, 3),
    }


def quantize_int8(onnx_path: str | Path, int8_path: str | Path | None = None, preset: str = "default",
                   per_channel: bool | None = None, op_types: list[str] | None = None,
                   exclude_head: bool | None = None, reduce_range: bool | None = None,
                   weight_type: str | None = None, preprocess: bool = False) -> Path:
    """Dynamic int8 quantization via onnxruntime.quantization.quantize_dynamic.
    Defaults to the `default` preset (MatMul only, per-channel, head in fp32);
    `preset="legacy"` reproduces the old plain call. Explicit options override
    the preset."""
    onnx_path = Path(onnx_path)
    int8_path = Path(int8_path) if int8_path is not None else onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    config = resolve_quant_config(preset, per_channel=per_channel, op_types=op_types,
                                  exclude_head=exclude_head, reduce_range=reduce_range,
                                  weight_type=weight_type)
    info = quantize_with_config(onnx_path, int8_path, config, preprocess=preprocess)
    print(f"[export_onnx] wrote int8 ONNX graph to {int8_path} (preset={preset}, {info['config']}, "
          f"excluded={info['excluded_nodes']}, {info['size_mb']} MB)")
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

def _jsonable(obj):
    """Recursively convert numpy scalars and NaN/inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan")}


def _ranking(gold: np.ndarray, p_safe: np.ndarray) -> tuple[float, float]:
    """(average precision, ROC-AUC) of 1 - p_safe, dangerous vs benign, ambiguous excluded."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    keep = np.isin(gold, ["dangerous", "benign"])
    y = (gold[keep] == "dangerous").astype(int)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")
    score = 1.0 - p_safe[keep]
    return float(average_precision_score(y, score)), float(roc_auc_score(y, score))


def parity_metrics(gold: np.ndarray, torch_logits: np.ndarray, onnx_fp32_logits: np.ndarray,
                    int8_logits: np.ndarray, t: float,
                    min_decision_agreement: float = MIN_DECISION_AGREEMENT,
                    max_recall_diff: float = MAX_RECALL_DIFF, max_ap_drop: float = MAX_AP_DROP) -> dict:
    """Pure-numpy fp32 (torch) vs int8 (onnx) comparison on val. The decision
    that matters is the gate's: escalate iff p_safe < t (softmax of logits).
    Ambiguous rows count in decision agreement but not in recall, benign
    escalation, or ranking quality."""
    gold = np.asarray(gold)
    dangerous, benign = gold == "dangerous", gold == "benign"
    n_d, n_b = int(dangerous.sum()), int(benign.sum())

    p_fp32 = SB._softmax(torch_logits)[:, 0]
    p_int8 = SB._softmax(int8_logits)[:, 0]
    esc_fp32, esc_int8 = p_fp32 < t, p_int8 < t
    same = esc_fp32 == esc_int8

    def side(esc: np.ndarray, p_safe: np.ndarray) -> dict:
        ap, auc = _ranking(gold, p_safe)
        return {
            "recall": _rate(int(esc[dangerous].sum()), n_d),
            "benign_escalation": _rate(int(esc[benign].sum()), n_b),
            "average_precision": ap, "roc_auc": auc,
        }

    fp32, int8 = side(esc_fp32, p_fp32), side(esc_int8, p_int8)

    def logit_diag(a: np.ndarray, b: np.ndarray) -> dict:
        return {
            "max_abs_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
            "argmax_agreement": float((a.argmax(1) == b.argmax(1)).mean()) if a.size else float("nan"),
        }

    recall_items = int8["recall"]["k"] - fp32["recall"]["k"]
    recall_diff = abs(recall_items) / n_d if n_d else float("nan")
    diff = {
        "recall_items": recall_items,
        "recall": (int8["recall"]["rate"] - fp32["recall"]["rate"]) if n_d else float("nan"),
        "benign_escalation_items": int8["benign_escalation"]["k"] - fp32["benign_escalation"]["k"],
        "average_precision": int8["average_precision"] - fp32["average_precision"],
        "roc_auc": int8["roc_auc"] - fp32["roc_auc"],
    }
    agreement = float(same.mean()) if same.size else float("nan")
    recall_tol = max(1.0 / n_d, max_recall_diff) if n_d else float("nan")

    warns: list[str] = []
    if same.size and agreement < min_decision_agreement:
        warns.append(f"decision agreement {agreement:.4f} < {min_decision_agreement:.2f}")
    if n_d and recall_diff > recall_tol + 1e-12:
        warns.append(f"recall differs by {abs(recall_items)} item(s) ({recall_diff:.4f}) "
                     f"> max(1 item, {max_recall_diff:.2f}) = {recall_tol:.4f}")
    if not math.isnan(diff["average_precision"]) and diff["average_precision"] < -max_ap_drop - 1e-12:
        warns.append(f"average precision dropped by {-diff['average_precision']:.4f} > {max_ap_drop:.2f}")

    return {
        "threshold": float(t),
        "n_rows": int(gold.size), "n_dangerous": n_d, "n_benign": n_b,
        "n_ambiguous": int((gold == "ambiguous").sum()),
        "decision_agreement": {
            "overall": agreement,
            "dangerous": float(same[dangerous].mean()) if n_d else float("nan"),
            "n_disagree": int((~same).sum()),
        },
        "fp32": fp32, "int8": int8, "diff": diff,
        "torch_vs_onnx_fp32": logit_diag(torch_logits, onnx_fp32_logits),
        "torch_vs_onnx_int8": logit_diag(torch_logits, int8_logits),
        "warn_rule": {"min_decision_agreement": min_decision_agreement,
                      "max_recall_diff": max_recall_diff, "max_ap_drop": max_ap_drop},
        "warnings": warns,
        "ok": not warns,
    }


def _fmt_rate(r: dict) -> str:
    return f"{r['k']}/{r['n']} {r['rate']:.3f}" if r["n"] else "n/a"


def format_parity_table(m: dict) -> str:
    fp, i8, d = m["fp32"], m["int8"], m["diff"]
    da = m["decision_agreement"]
    rows = [
        ("metric", "fp32", "int8", "int8-fp32"),
        (f"recall @ t={m['threshold']:.4f}", _fmt_rate(fp["recall"]), _fmt_rate(i8["recall"]),
         f"{d['recall_items']:+d} item(s)"),
        ("benign escalation", _fmt_rate(fp["benign_escalation"]), _fmt_rate(i8["benign_escalation"]),
         f"{d['benign_escalation_items']:+d} item(s)"),
        ("average precision", f"{fp['average_precision']:.4f}", f"{i8['average_precision']:.4f}",
         f"{d['average_precision']:+.4f}"),
        ("ROC-AUC", f"{fp['roc_auc']:.4f}", f"{i8['roc_auc']:.4f}", f"{d['roc_auc']:+.4f}"),
        ("decision agreement", "", f"{da['overall']:.4f} overall", f"{da['dangerous']:.4f} on dangerous"),
        ("max abs logit diff", f"{m['torch_vs_onnx_fp32']['max_abs_diff']:.5f}",
         f"{m['torch_vs_onnx_int8']['max_abs_diff']:.5f}", "(vs torch fp32)"),
        ("3-way argmax agreement", f"{m['torch_vs_onnx_fp32']['argmax_agreement']:.4f}",
         f"{m['torch_vs_onnx_int8']['argmax_agreement']:.4f}", "(vs torch fp32)"),
    ]
    widths = [max(len(r[c]) for r in rows) for c in range(4)]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows)


def _print_banner(warns: list[str]) -> None:
    banner = (
        "!" * 70 + "\n"
        "WARNING: int8 model differs from fp32 beyond noise on val:\n  - "
        + "\n  - ".join(warns) + "\n"
        "Do not trust int8 test-set numbers until this is investigated "
        "(README 4.5: confirm the quantized model matches fp32 within noise).\n"
        + "!" * 70
    )
    print(banner)
    warnings.warn(banner, stacklevel=3)


def load_reference(model_dir: str | Path, onnx_path: str | Path, val_parquet: str | Path,
                    batch_size: int = 32, bulk_threads: int = 0) -> dict:
    """Everything the int8 comparison needs that does not depend on the int8
    model: val rows, torch fp32 logits and fp32 ONNX logits. ONNX logits use
    a bulk session (`bulk_threads`, 0 = all cores); latency is measured
    elsewhere, in its own 1-thread session."""
    _require_onnx()
    _require_ort()
    student, tokenizer, metadata = SB.load_student(model_dir, device="cpu")
    max_len = int(metadata.get("max_len", 256))

    df = pd.read_parquet(val_parquet)
    missing = {"id", "text", "gold"} - set(df.columns)
    if missing:
        raise ValueError(f"{val_parquet} missing columns: {sorted(missing)}")
    texts = df["text"].tolist()
    torch_logits = SB.infer_logits(student, tokenizer, texts, max_len, "cpu", batch_size=batch_size)
    onnx_fp32_logits = _ort_logits(_ort_session(onnx_path, intra_op_num_threads=bulk_threads), tokenizer, texts, max_len,
                                batch_size=batch_size)
    return {"tokenizer": tokenizer, "max_len": max_len, "texts": texts, "gold": df["gold"].to_numpy(),
            "torch_logits": torch_logits, "onnx_fp32_logits": onnx_fp32_logits, "batch_size": batch_size,
            "bulk_threads": bulk_threads}


def reference_threshold(ref: dict, target_recall: float) -> float:
    from reflex_sentry.eval import metrics as M

    pred_df = pd.DataFrame({"gold": ref["gold"], "p_safe": SB._softmax(ref["torch_logits"])[:, 0]})
    return float(M.select_threshold(pred_df, target_recall))


def parity_check(model_dir: str | Path, onnx_path: str | Path, int8_path: str | Path,
                  val_parquet: str | Path, target_recall: float = 0.95, batch_size: int = 32,
                  min_decision_agreement: float = MIN_DECISION_AGREEMENT,
                  max_recall_diff: float = MAX_RECALL_DIFF, max_ap_drop: float = MAX_AP_DROP,
                  out_path: str | Path | None = None, reference: dict | None = None,
                  bulk_threads: int = 0) -> dict:
    """Compare fp32 torch / fp32 onnx / int8 onnx on val at the gate's decision
    (escalate iff p_safe < t, t chosen on fp32 for `target_recall`). Prints a
    compact table, warns loudly per the warning rule, and writes the metrics
    to `out_path` (default <model_dir>/parity.json)."""
    t_start = time.perf_counter()
    ref = reference or load_reference(model_dir, onnx_path, val_parquet, batch_size=batch_size,
                                      bulk_threads=bulk_threads)
    int8_logits = _ort_logits(_ort_session(int8_path, intra_op_num_threads=bulk_threads), ref["tokenizer"], ref["texts"], ref["max_len"],
                               batch_size=batch_size)
    print(f"[parity] scored {len(ref['texts'])} val rows in {time.perf_counter() - t_start:.1f}s "
          f"(bulk, intra_op_num_threads={bulk_threads})")
    t = reference_threshold(ref, target_recall)
    result = parity_metrics(ref["gold"], ref["torch_logits"], ref["onnx_fp32_logits"], int8_logits, t,
                            min_decision_agreement, max_recall_diff, max_ap_drop)
    result["target_recall"] = target_recall
    result["int8_path"] = str(int8_path)

    print(f"[parity] val: {result['n_rows']} rows ({result['n_dangerous']} dangerous, "
          f"{result['n_benign']} benign, {result['n_ambiguous']} ambiguous); int8 = {int8_path}")
    print(format_parity_table(result))
    if result["warnings"]:
        _print_banner(result["warnings"])
    else:
        print("[parity] OK: int8 within noise of fp32 per the warning rule")

    out_path = Path(out_path) if out_path is not None else Path(model_dir) / "parity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(result), indent=2))
    print(f"[parity] wrote {out_path}")
    return result


# ------------------------------------------------------------------- sweep --

def _latency_stats(onnx_path: str | Path, ref: dict, sample_size: int = 200, warmup: int = 3) -> dict:
    """Batch size 1, intra_op_num_threads=1 CPU latency over up to `sample_size` val prompts."""
    sess = _ort_session(onnx_path, intra_op_num_threads=1)
    for text in ref["texts"][:warmup]:
        _ort_latency_ms(sess, ref["tokenizer"], [text], ref["max_len"], 1)
    lat = np.array(list(_ort_latency_ms(sess, ref["tokenizer"], ref["texts"], ref["max_len"],
                                         sample_size).values()), dtype=float)
    if lat.size == 0:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "n": 0}
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "n": int(lat.size)}


def choose_best(results: list[dict]) -> dict | None:
    """Highest overall decision agreement among configs that pass the warning
    rule; ties broken by lower p50 latency."""
    passing = [r for r in results if r["parity"]["ok"]]
    if not passing:
        return None
    return min(passing, key=lambda r: (-r["parity"]["decision_agreement"]["overall"], r["latency"]["p50_ms"]))


def format_sweep_table(fp32_row: dict, results: list[dict], chosen: str | None = None,
                        markdown: bool = False) -> str:
    header = ["config", "agree", "agree(dang)", "recall", "benign esc", "dAP", "dAUC",
              "max|dlogit|", "p50 ms", "p95 ms", "MB", "ok"]

    def row(name: str, r: dict) -> list[str]:
        m = r["parity"]
        return [
            name + (" *" if name == chosen else ""),
            f"{m['decision_agreement']['overall']:.4f}", f"{m['decision_agreement']['dangerous']:.4f}",
            _fmt_rate(m["int8"]["recall"]).split()[0], _fmt_rate(m["int8"]["benign_escalation"]).split()[0],
            f"{m['diff']['average_precision']:+.4f}", f"{m['diff']['roc_auc']:+.4f}",
            f"{m['torch_vs_onnx_int8']['max_abs_diff']:.3f}",
            f"{r['latency']['p50_ms']:.1f}", f"{r['latency']['p95_ms']:.1f}",
            f"{r.get('size_mb', float('nan')):.2f}", "yes" if m["ok"] else "NO",
        ]

    ref_m = results[0]["parity"] if results else None
    ref_row = ["fp32 onnx (ref)", "1.0000", "1.0000",
               _fmt_rate(ref_m["fp32"]["recall"]).split()[0] if ref_m else "",
               _fmt_rate(ref_m["fp32"]["benign_escalation"]).split()[0] if ref_m else "",
               "+0.0000", "+0.0000", f"{ref_m['torch_vs_onnx_fp32']['max_abs_diff']:.3f}" if ref_m else "",
               f"{fp32_row['p50_ms']:.1f}", f"{fp32_row['p95_ms']:.1f}",
               f"{fp32_row.get('size_mb', float('nan')):.2f}", ""]
    rows = [ref_row] + [row(r["name"], r) for r in results]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    rows = [header] + rows
    widths = [max(len(r[c]) for r in rows) for c in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows)


def sweep(model_dir: str | Path, onnx_path: str | Path, val_parquet: str | Path,
          configs: list[str] | None = None, out_dir: str | Path | None = None, choose: bool = False,
          chosen_path: str | Path | None = None, target_recall: float = 0.95, batch_size: int = 32,
          latency_sample: int = 200, preprocess: bool = False, bulk_threads: int = 0,
          min_decision_agreement: float = MIN_DECISION_AGREEMENT,
          max_recall_diff: float = MAX_RECALL_DIFF, max_ap_drop: float = MAX_AP_DROP) -> dict:
    """Quantize with each config, compare against fp32 on val (parity metrics
    from a bulk session with `bulk_threads`, 0 = all cores; latency from a
    separate batch-1, intra_op_num_threads=1 session, unchanged semantics), write sweep.json / sweep.md next
    to the model, and with `choose` copy the best passing config to
    <model_dir>/model_int8.onnx and record the choice in metadata.json."""
    _require_ort()
    model_dir = Path(model_dir)
    names = list(configs) if configs else list(SWEEP_CONFIGS)
    for n in names:
        if n not in PRESETS:
            raise ValueError(f"unknown sweep config {n!r}; choose from {sorted(PRESETS)}")
    out_dir = Path(out_dir) if out_dir is not None else model_dir / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_reference(model_dir, onnx_path, val_parquet, batch_size=batch_size, bulk_threads=bulk_threads)
    t = reference_threshold(ref, target_recall)
    fp32_latency = _latency_stats(onnx_path, ref, latency_sample)
    fp32_latency["size_mb"] = round(Path(onnx_path).stat().st_size / 1e6, 3)
    print(f"[sweep] {len(ref['texts'])} val rows, t={t:.4f} (target recall {target_recall}); "
          f"fp32 onnx p50 {fp32_latency['p50_ms']:.1f} ms")

    results: list[dict] = []
    for name in names:
        int8_path = out_dir / f"{name}.onnx"
        print(f"[sweep] {name}: quantizing")
        t_cfg = time.perf_counter()
        info = quantize_with_config(onnx_path, int8_path, PRESETS[name], preprocess=preprocess)
        logits = _ort_logits(_ort_session(int8_path, intra_op_num_threads=bulk_threads), ref["tokenizer"],
                              ref["texts"], ref["max_len"], batch_size=batch_size)
        parity = parity_metrics(ref["gold"], ref["torch_logits"], ref["onnx_fp32_logits"], logits, t,
                                min_decision_agreement, max_recall_diff, max_ap_drop)
        latency = _latency_stats(int8_path, ref, latency_sample)
        print(f"[sweep] {name}: scored {len(logits)} rows, timed {latency['n']} prompts "
              f"in {time.perf_counter() - t_cfg:.1f}s total")
        results.append({"name": name, "path": str(int8_path), **info, "parity": parity, "latency": latency})

    best = choose_best(results)
    chosen_name = best["name"] if best else None
    table = format_sweep_table(fp32_latency, results, chosen_name)
    print(table)
    print("* = best passing config (highest decision agreement, ties by lower p50)" if best
          else "[sweep] NO config passes the warning rule; nothing to choose")

    out = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "val": str(val_parquet), "threshold": t, "target_recall": target_recall,
        "n_rows": len(ref["texts"]), "latency_sample": latency_sample,
        "warn_rule": {"min_decision_agreement": min_decision_agreement,
                      "max_recall_diff": max_recall_diff, "max_ap_drop": max_ap_drop},
        "fp32_onnx_latency": fp32_latency,
        "results": results, "chosen": chosen_name, "chosen_path": None,
    }

    if choose and best is not None:
        dest = Path(chosen_path) if chosen_path is not None else model_dir / "model_int8.onnx"
        shutil.copyfile(best["path"], dest)
        out["chosen_path"] = str(dest)
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["int8_quantization"] = _jsonable({
                "config_name": best["name"], "config": best["config"],
                "excluded_nodes": best["excluded_nodes"], "preprocess": best["preprocess"],
                "path": str(dest), "chosen_at": out["created"], "threshold": t,
                "decision_agreement": best["parity"]["decision_agreement"]["overall"],
                "recall_fp32": best["parity"]["fp32"]["recall"], "recall_int8": best["parity"]["int8"]["recall"],
                "average_precision_diff": best["parity"]["diff"]["average_precision"],
                "latency": best["latency"], "fp32_onnx_latency": fp32_latency,
            })
            meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[sweep] chose {best['name']}: copied to {dest}")
    elif choose:
        print("[sweep] --choose: no config passed, model_int8.onnx NOT written")

    json_path = model_dir / "sweep.json"
    json_path.write_text(json.dumps(_jsonable(out), indent=2))
    md = ["# Stage B int8 sweep", "",
          f"val rows: {out['n_rows']}, threshold t={t:.4f} (target recall {target_recall}); "
          f"latency: batch 1, 1 thread, median/p95 over up to {latency_sample} prompts.", "",
          format_sweep_table(fp32_latency, results, chosen_name, markdown=True), "",
          f"Warning rule: decision agreement >= {min_decision_agreement}, recall diff <= max(1 item, "
          f"{max_recall_diff}), AP drop <= {max_ap_drop}. `*` marks the chosen config: "
          + (f"`{chosen_name}`." if chosen_name else "none passed."), ""]
    (model_dir / "sweep.md").write_text("\n".join(md))
    print(f"[sweep] wrote {json_path} and {model_dir / 'sweep.md'}")
    return out


# ---------------------------------------------------------------- predict --

def predict_int8_split(int8_path: str | Path, tokenizer, metadata: dict, split: str, data_dir: str | Path,
                        out_dir: str | Path, latency_threads: int = 1, batch_size: int = 32,
                        latency_sample: int = 200, bulk_threads: int = 0) -> Path | None:
    """Two sessions on purpose: a bulk session (`bulk_threads`, 0 = onnxruntime
    default = all cores) scores every row in batches for the logits, and a
    separate `latency_threads` (default 1) session times only the sampled
    prompts at batch size 1. The reported `latency_ms` therefore keeps its
    meaning (single-core, batch 1, per prompt) while bulk scoring stays fast."""
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        print(f"[export_onnx predict] {split}: {path} not found, skipping")
        return None
    df = pd.read_parquet(path)
    if {"id", "text"} - set(df.columns):
        raise ValueError(f"{path} missing required columns id, text")
    max_len = int(metadata.get("max_len", 256))
    texts = df["text"].tolist()

    t0 = time.perf_counter()
    bulk_sess = _ort_session(int8_path, intra_op_num_threads=bulk_threads)
    logits = _ort_logits(bulk_sess, tokenizer, texts, max_len, batch_size=batch_size)
    t_bulk = time.perf_counter() - t0
    print(f"[export_onnx predict] {split}: scored {len(texts)} rows in {t_bulk:.1f}s "
          f"(bulk, intra_op_num_threads={bulk_threads})")

    t0 = time.perf_counter()
    latency_sess = _ort_session(int8_path, intra_op_num_threads=latency_threads)
    latencies = _ort_latency_ms(latency_sess, tokenizer, texts, max_len, latency_sample)
    t_lat = time.perf_counter() - t0
    print(f"[export_onnx predict] {split}: timed {len(latencies)} prompts at batch 1 in {t_lat:.1f}s "
          f"(intra_op_num_threads={latency_threads})")

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
    print(f"[export_onnx predict] wrote {out_path} ({len(out)} rows)")
    return out_path


def predict_int8(model_dir: str | Path, int8_path: str | Path, splits: list[str],
                  data_dir: str | Path = "data/processed", out_dir: str | Path = "preds",
                  latency_threads: int = 1, batch_size: int = 32, latency_sample: int = 200,
                  bulk_threads: int = 0) -> list[Path]:
    """onnxruntime CPU predictor. Logits come from a bulk session using
    `bulk_threads` (0 = all cores); `latency_ms` comes from a separate session
    with `latency_threads` (default 1, mimicking a single-core gate, README's
    CPU-latency framing) at batch size 1. Raising `latency_threads` changes
    the latency story, not the logits."""
    _require_ort()
    from transformers import AutoTokenizer

    metadata = {}
    meta_path = Path(model_dir) / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    written = []
    for split in splits:
        p = predict_int8_split(int8_path, tokenizer, metadata, split, data_dir, out_dir,
                                latency_threads=latency_threads, batch_size=batch_size,
                                latency_sample=latency_sample, bulk_threads=bulk_threads)
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
    q.add_argument("--preset", default="default", choices=sorted(PRESETS),
                    help="base options; explicit flags below override it. 'legacy' = old plain quantize_dynamic")
    q.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=None,
                    help="per-channel weight quantization (default on)")
    q.add_argument("--op-types", nargs="+", default=None, metavar="OP",
                    help="op types to quantize (default: MatMul only; 'all' = everything ORT can quantize)")
    q.add_argument("--exclude-head", action=argparse.BooleanOptionalAction, default=None,
                    help="keep the classifier head MatMul/Gemm in fp32 (default on)")
    q.add_argument("--reduce-range", action=argparse.BooleanOptionalAction, default=None,
                    help="7-bit weights (default off)")
    q.add_argument("--weight-type", choices=["qint8", "quint8"], default=None, help="default: qint8")
    q.add_argument("--preprocess", action="store_true",
                    help="run onnxruntime quantization pre-processing (shape inference / optimization) first")

    def add_warn_args(sp):
        sp.add_argument("--min-decision-agreement", type=float, default=MIN_DECISION_AGREEMENT)
        sp.add_argument("--max-recall-diff", type=float, default=MAX_RECALL_DIFF,
                         help="recall tolerance is max(1 item, this)")
        sp.add_argument("--max-ap-drop", type=float, default=MAX_AP_DROP)

    pa = sub.add_parser("parity", help="compare torch fp32 / onnx fp32 / onnx int8 on val")
    pa.add_argument("--model", required=True)
    pa.add_argument("--onnx", required=True)
    pa.add_argument("--int8", required=True)
    pa.add_argument("--val", default="data/processed/val.parquet")
    pa.add_argument("--target-recall", type=float, default=0.95)
    pa.add_argument("--out", default=None, help="default: <model>/parity.json")
    pa.add_argument("--bulk-threads", type=int, default=0, help="intra_op_num_threads for scoring; 0 = all cores")
    add_warn_args(pa)

    sw = sub.add_parser("sweep", help="try several quantization configs, compare parity and CPU latency")
    sw.add_argument("--model", required=True)
    sw.add_argument("--onnx", required=True, help="fp32 ONNX graph")
    sw.add_argument("--val", default="data/processed/val.parquet")
    sw.add_argument("--configs", nargs="+", default=None, choices=sorted(PRESETS),
                     help=f"default: {' '.join(SWEEP_CONFIGS)}")
    sw.add_argument("--out-dir", default=None, help="default: <model>/sweep")
    sw.add_argument("--choose", action="store_true",
                     help="copy the best passing config to <model>/model_int8.onnx and record it in metadata.json")
    sw.add_argument("--chosen-path", default=None, help="default: <model>/model_int8.onnx")
    sw.add_argument("--target-recall", type=float, default=0.95)
    sw.add_argument("--latency-sample", type=int, default=200)
    sw.add_argument("--preprocess", action="store_true")
    sw.add_argument("--bulk-threads", type=int, default=0,
                     help="intra_op_num_threads for parity scoring; 0 = all cores. Latency is always "
                          "measured in its own 1-thread, batch-1 session")
    add_warn_args(sw)

    p = sub.add_parser("predict", help="onnxruntime CPU predictor for the int8 model")
    p.add_argument("--model", required=True, help="dir with tokenizer + metadata.json")
    p.add_argument("--int8", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_ood", "test_evasion"])
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--latency-threads", "--intra-op-threads", dest="latency_threads", type=int, default=1,
                    help="intra_op_num_threads of the separate session that times per-prompt latency at "
                         "batch size 1; 1 mimics a single-core gate (this is what latency_ms reports)")
    p.add_argument("--bulk-threads", type=int, default=0,
                    help="intra_op_num_threads for the bulk logits session; 0 = onnxruntime default (all cores)")
    p.add_argument("--latency-sample", type=int, default=200)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "export":
        export_to_onnx(args.model, onnx_path=args.out, opset=args.opset)
    elif args.command == "quantize":
        quantize_int8(args.onnx, int8_path=args.out, preset=args.preset, per_channel=args.per_channel,
                      op_types=args.op_types, exclude_head=args.exclude_head,
                      reduce_range=args.reduce_range, weight_type=args.weight_type,
                      preprocess=args.preprocess)
    elif args.command == "parity":
        parity_check(args.model, args.onnx, args.int8, args.val, target_recall=args.target_recall,
                     min_decision_agreement=args.min_decision_agreement,
                     max_recall_diff=args.max_recall_diff, max_ap_drop=args.max_ap_drop, out_path=args.out,
                     bulk_threads=args.bulk_threads)
    elif args.command == "sweep":
        result = sweep(args.model, args.onnx, args.val, configs=args.configs, out_dir=args.out_dir,
                       choose=args.choose, chosen_path=args.chosen_path, target_recall=args.target_recall,
                       latency_sample=args.latency_sample, preprocess=args.preprocess,
                       bulk_threads=args.bulk_threads, min_decision_agreement=args.min_decision_agreement,
                       max_recall_diff=args.max_recall_diff, max_ap_drop=args.max_ap_drop)
        if args.choose and result["chosen"] is None:
            raise SystemExit("sweep --choose: no quantization config passed the parity warning rule; "
                             "model_int8.onnx was not written")
    elif args.command == "predict":
        predict_int8(args.model, args.int8, args.splits, data_dir=args.data_dir, out_dir=args.out_dir,
                     latency_threads=args.latency_threads, latency_sample=args.latency_sample,
                     bulk_threads=args.bulk_threads)


if __name__ == "__main__":
    main()

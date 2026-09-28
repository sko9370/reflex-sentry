"""Stage B: fine-tuned encoder student.

An AutoModel encoder (default `answerdotai/ModernBERT-base`, alt
`microsoft/deberta-v3-small`) with mean pooling over the attention mask, a
dropout layer, and a Linear(hidden, 3) head producing [safe, dangerous,
unsure] logits, in that fixed order (matches
`reflex_sentry.eval.calibrate.GOLD_TO_CLASS`: benign -> 0, dangerous -> 1,
ambiguous -> 2). Trained with plain PyTorch (no Trainer) against KL-divergence
soft targets from `data/interim/soft_targets.parquet` (columns id, t_safe,
t_dangerous, t_unsure), joined to `data/processed/train.parquet` (id, text)
on `id`.

`val` is reserved for temperature scaling and threshold selection (README
section 4.5) and never influences training. Checkpoint selection instead
uses an internal dev set: a stratified 10% of the training pool (by
soft-target argmax, fixed seed, grouped by `dup_group` when that column is
present so near-duplicate rows stay together), scored by KL/log-loss against
its soft targets. `val.parquet` is only used, when present, to log
informational epoch metrics (NLL vs gold, recall at a val-selected
threshold); training runs identically, and selects the identical checkpoint,
when it is absent.

This module is importable even when torch/transformers are not installed
(lazy imports); calling `train`, `predict`, `build_student`, or `load_student`
without them raises a clear ImportError.

    python -m reflex_sentry.models.stage_b train --base answerdotai/ModernBERT-base --out models/stage_b
    python -m reflex_sentry.models.stage_b predict --model models/stage_b --splits val test test_ood test_evasion
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reflex_sentry.eval import latency as L
from reflex_sentry.eval.calibrate import GOLD_TO_CLASS

try:  # pragma: no cover - exercised indirectly by torch-gated tests
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = None

DEFAULT_BASE = "answerdotai/ModernBERT-base"
LOGIT_COLS = ("logit_safe", "logit_dangerous", "logit_unsure")
TARGET_COLS = ("t_safe", "t_dangerous", "t_unsure")


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "torch and transformers are required for reflex_sentry.models.stage_b; "
            "install the 'stage_b' extra (pip install -e '.[stage_b]') or "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


def set_seed(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in CI
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ model --

if nn is not None:

    class StageBStudent(nn.Module):
        """AutoModel encoder + mean pooling over the attention mask + dropout + 3-way head."""

        def __init__(self, encoder: Any, hidden_size: int, dropout: float = 0.1, num_labels: int = 3):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # [B, T, H]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1], 0 on padding
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            pooled = self.dropout(pooled)
            return self.head(pooled)

else:  # pragma: no cover

    class StageBStudent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


def build_student(base: str | None = None, config: Any = None, dropout: float = 0.1) -> "StageBStudent":
    """Build a student from a pretrained checkpoint name/path (`base`) or an
    in-memory config (`config`, e.g. a tiny BertConfig for tests -> randomly
    initialized weights, no download)."""
    _require_torch()
    from transformers import AutoModel

    encoder = AutoModel.from_config(config) if config is not None else AutoModel.from_pretrained(base)
    return StageBStudent(encoder, encoder.config.hidden_size, dropout=dropout)


def kl_loss(logits, targets) -> "torch.Tensor":
    """KL(target || model), batchmean. `targets` are probabilities (sum to 1
    per row) in [safe, dangerous, unsure] order; `logits` are raw scores."""
    _require_torch()
    log_probs = torch.log_softmax(logits, dim=-1)
    return nn.functional.kl_div(log_probs, targets, reduction="batchmean")


# -------------------------------------------------------------- save/load --

def save_student(out_dir: str | Path, student: "StageBStudent", tokenizer: Any, metadata: dict) -> None:
    _require_torch()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save({"head": student.head.state_dict(), "dropout_p": student.dropout.p}, out_dir / "head.pt")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_student(model_dir: str | Path, device: str = "cpu") -> tuple["StageBStudent", Any, dict]:
    _require_torch()
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    student = StageBStudent(encoder, encoder.config.hidden_size, dropout=metadata.get("dropout", 0.1))
    head_ckpt = torch.load(model_dir / "head.pt", map_location="cpu")
    student.head.load_state_dict(head_ckpt["head"])
    student.dropout.p = head_ckpt.get("dropout_p", student.dropout.p)
    student.to(device)
    student.eval()
    return student, tokenizer, metadata


# ---------------------------------------------------------------- dataset --

if Dataset is not None:

    class _TextDataset(Dataset):
        def __init__(self, texts: list[str], targets: np.ndarray | None = None):
            self.texts = list(texts)
            self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict:
            item = {"text": self.texts[idx]}
            if self.targets is not None:
                item["target"] = self.targets[idx]
            return item

else:  # pragma: no cover
    _TextDataset = None


def _make_collate(tokenizer: Any, max_len: int):
    def collate(batch: list[dict]):
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "target" in batch[0]:
            out["target"] = torch.tensor(np.stack([b["target"] for b in batch]), dtype=torch.float32)
        return out

    return collate


# --------------------------------------------------------------- inference --

def infer_logits(student, tokenizer, texts: list[str], max_len: int, device: str, batch_size: int = 32) -> np.ndarray:
    """Batched forward pass, returns an [N, 3] array of raw logits."""
    _require_torch()
    student.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            logits = student(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float32)


def measure_latency_ms(student, tokenizer, texts: list[str], max_len: int, device: str, sample_size: int,
                        threads: int = 1) -> dict[int, float]:
    """Per-prompt, batch-size-1 CPU latency using the shared protocol in
    `reflex_sentry.eval.latency` (README 5.4: a single-core gate, batch 1,
    raw text in -> probabilities out). Tokenization, the forward pass, and
    the softmax are all inside the timed region; `threads` (default 1, see
    `--latency-threads`) pins the torch thread count for that region only.
    Measuring every row at batch 1 would be needlessly slow for large
    splits, so only a fixed-seed sample gets a value; the rest are NaN in
    the output CSV and metrics.latency() drops NaNs before computing
    p50/p95/p99, so percentiles reflect the sampled subset."""
    _require_torch()
    student.eval()

    def _predict_one(batch: list[str]) -> np.ndarray:
        enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = student(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    with L.single_thread(threads):
        return L.time_per_prompt(_predict_one, texts, sample_size=sample_size)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ train --

def load_training_frame(train_parquet: str | Path, targets_parquet: str | Path) -> pd.DataFrame:
    train_df = pd.read_parquet(train_parquet)
    targets_df = pd.read_parquet(targets_parquet)
    missing_train = {"id", "text"} - set(train_df.columns)
    if missing_train:
        raise ValueError(f"train parquet missing columns: {sorted(missing_train)}")
    missing_targets = {"id", *TARGET_COLS} - set(targets_df.columns)
    if missing_targets:
        raise ValueError(f"soft targets parquet missing columns: {sorted(missing_targets)}")
    keep_cols = ["id", "text"] + (["dup_group"] if "dup_group" in train_df.columns else [])
    merged = train_df[keep_cols].merge(targets_df[["id", *TARGET_COLS]], on="id", how="inner")
    dropped = len(train_df) - len(merged)
    if dropped:
        print(f"[stage_b train] dropped {dropped} train row(s) with no matching soft target")
    return merged.reset_index(drop=True)


def make_dev_split(frame: pd.DataFrame, dev_ratio: float = 0.1, seed: int = 13) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the merged training frame into (train, dev). Stratified by the
    soft-target argmax class, with a fixed seed, and grouped by `dup_group`
    when present so near-duplicate rows land on the same side (mirrors
    `reflex_sentry.data.split`'s dup_group-level splitting)."""
    argmax_class = frame[list(TARGET_COLS)].to_numpy().argmax(axis=1)
    work = frame.assign(_argmax=argmax_class)
    group_col = "dup_group" if "dup_group" in work.columns else "id"
    rng = np.random.RandomState(seed)
    dev_mask = np.zeros(len(work), dtype=bool)
    for _, sub in work.groupby("_argmax", sort=True):
        groups = np.sort(sub[group_col].unique())
        rng.shuffle(groups)
        n_dev_groups = max(1, int(round(len(groups) * dev_ratio))) if len(groups) else 0
        dev_groups = set(groups[:n_dev_groups])
        dev_mask[sub.index[sub[group_col].isin(dev_groups)].to_numpy()] = True
    train_df = work.loc[~dev_mask].drop(columns=["_argmax"]).reset_index(drop=True)
    dev_df = work.loc[dev_mask].drop(columns=["_argmax"]).reset_index(drop=True)
    return train_df, dev_df


def evaluate_dev_kl(student, tokenizer, dev_df: pd.DataFrame, max_len: int, device: str,
                     batch_size: int = 64) -> float:
    """Mean KL(dev soft target || model), matching the training loss
    (`kl_loss`, batchmean) but computed in numpy over the full dev set."""
    if len(dev_df) == 0:
        return float("nan")
    logits = infer_logits(student, tokenizer, dev_df["text"].tolist(), max_len, device, batch_size=batch_size)
    targets = dev_df[list(TARGET_COLS)].to_numpy(dtype=np.float64)
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_t = np.where(targets > 0, np.log(targets), 0.0)
    kl_per_row = np.sum(np.where(targets > 0, targets * (log_t - log_probs), 0.0), axis=1)
    return float(kl_per_row.mean())


def evaluate_on_val(student, tokenizer, val_df: pd.DataFrame, max_len: int, device: str, target_recall: float,
                     batch_size: int = 64) -> dict:
    """Informational only: val NLL vs gold and recall at a val-selected
    threshold. Never used to pick the training checkpoint (see `make_dev_split`
    / `evaluate_dev_kl`); val is reserved for temperature scaling and
    threshold selection downstream (README 4.5)."""
    y = val_df["gold"].map(GOLD_TO_CLASS).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ValueError("val gold labels must be benign, dangerous, or ambiguous")
    logits = infer_logits(student, tokenizer, val_df["text"].tolist(), max_len, device, batch_size=batch_size)
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    nll = float(-log_probs[np.arange(len(y)), y.astype(int)].mean())

    from reflex_sentry.eval import metrics as M

    probs = _softmax(logits)
    pred_df = pd.DataFrame({"gold": val_df["gold"].to_numpy(),
                             "p_safe": probs[:, 0], "p_dangerous": probs[:, 1], "p_unsure": probs[:, 2]})
    t = M.select_threshold(pred_df, target_recall)
    dangerous = pred_df[pred_df["gold"] == "dangerous"]
    recall = M.escalation_rate(dangerous, t).value if len(dangerous) else float("nan")
    return {"val_nll": nll, "threshold": t, "val_recall_dangerous": recall}


def train(
    base: str = DEFAULT_BASE,
    out_dir: str | Path = "models/stage_b",
    train_parquet: str | Path = "data/processed/train.parquet",
    targets_parquet: str | Path = "data/interim/soft_targets.parquet",
    val_parquet: str | Path | None = "data/processed/val.parquet",
    dev_ratio: float = 0.1,
    max_len: int = 256,
    epochs: int = 3,
    lr: float = 3e-5,
    head_lr: float = 1e-3,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 13,
    grad_clip: float = 1.0,
    dropout: float = 0.1,
    target_recall: float = 0.95,
    fp16: bool = False,
    device: str | None = None,
    config: Any = None,
) -> dict:
    """Plain PyTorch training loop (no Trainer).

    Checkpoint selection uses an internal dev set held out from the training
    pool (`make_dev_split`, `dev_ratio`, stratified by soft-target argmax,
    fixed `seed`), scored by KL/log-loss against its soft targets
    (`evaluate_dev_kl`) -- never gold labels. `val_parquet` is optional and
    purely informational: when it exists, its NLL-vs-gold and recall-at-a-
    val-selected-threshold are logged and recorded in metadata each epoch,
    but they never affect which checkpoint is kept, and training runs (and
    selects) identically whether or not it is present. `config` lets tests
    pass an in-memory tiny config instead of downloading `base`.
    """
    _require_torch()
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = fp16 and device.startswith("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base)
    student = build_student(base=None if config is not None else base, config=config, dropout=dropout)
    student.to(device)

    merged = load_training_frame(train_parquet, targets_parquet)
    train_frame, dev_frame = make_dev_split(merged, dev_ratio=dev_ratio, seed=seed)
    if len(dev_frame) == 0:
        raise ValueError("dev split is empty; increase dev_ratio or the training pool size")

    targets = train_frame[list(TARGET_COLS)].to_numpy(dtype=np.float32)
    dataset = _TextDataset(train_frame["text"].tolist(), targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_make_collate(tokenizer, max_len))

    val_df = None
    if val_parquet is not None and Path(val_parquet).exists():
        val_df = pd.read_parquet(val_parquet)
        missing_val = {"id", "text", "gold"} - set(val_df.columns)
        if missing_val:
            raise ValueError(f"val parquet missing columns: {sorted(missing_val)}")
    else:
        print("[stage_b train] val parquet not found; skipping informational val logging "
              "(this never affects checkpoint selection)")

    optimizer = torch.optim.AdamW(
        [
            {"params": student.encoder.parameters(), "lr": lr},
            {"params": student.head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else None

    best_dev_kl = float("inf")
    best_state = None
    best_metrics: dict = {}
    history = []

    for epoch in range(epochs):
        student.train()
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = student(batch["input_ids"], batch["attention_mask"])
                    loss = kl_loss(logits, batch["target"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = student(batch["input_ids"], batch["attention_mask"])
                loss = kl_loss(logits, batch["target"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch["input_ids"])

        train_loss = running / len(dataset)
        dev_kl = evaluate_dev_kl(student, tokenizer, dev_frame, max_len, device, batch_size=eval_batch_size)
        epoch_record = {"epoch": epoch + 1, "train_kl": train_loss, "dev_kl": dev_kl}

        log_line = (f"[stage_b train] epoch {epoch + 1}/{epochs} train_kl={train_loss:.4f} dev_kl={dev_kl:.4f}")
        if val_df is not None:
            val_metrics = evaluate_on_val(student, tokenizer, val_df, max_len, device, target_recall,
                                           batch_size=eval_batch_size)
            epoch_record.update(val_metrics)
            log_line += (f" val_nll={val_metrics['val_nll']:.4f} "
                         f"val_recall={val_metrics['val_recall_dangerous']:.4f} t={val_metrics['threshold']:.4f}")
        print(log_line)
        history.append(epoch_record)

        if dev_kl < best_dev_kl:
            best_dev_kl = dev_kl
            best_state = copy.deepcopy(student.state_dict())
            best_metrics = dict(epoch_record)

    if best_state is not None:
        student.load_state_dict(best_state)

    n_params = sum(p.numel() for p in student.parameters())
    metadata = {
        "base": base,
        "n_params": int(n_params),
        "max_len": max_len,
        "epochs": epochs,
        "lr": lr,
        "head_lr": head_lr,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "seed": seed,
        "grad_clip": grad_clip,
        "dropout": dropout,
        "target_recall": target_recall,
        "fp16": fp16,
        "dev_ratio": dev_ratio,
        "dev_size": len(dev_frame),
        "train_size": len(train_frame),
        "selection_metric": "dev_kl",
        "best_epoch": best_metrics.get("epoch"),
        "best_dev_kl": best_metrics.get("dev_kl"),
        "best_val_nll": best_metrics.get("val_nll"),
        "best_val_recall_dangerous": best_metrics.get("val_recall_dangerous"),
        "best_val_threshold": best_metrics.get("threshold"),
        "val_used": val_df is not None,
        "history": history,
    }
    save_student(out_dir, student, tokenizer, metadata)
    print(f"[stage_b train] saved best checkpoint (epoch {metadata['best_epoch']}, "
          f"dev_kl={metadata['best_dev_kl']:.4f}) to {out_dir}")
    return metadata


# ---------------------------------------------------------------- predict --

def predict_split(student, tokenizer, split: str, data_dir: str | Path, out_dir: str | Path, max_len: int,
                   device: str, batch_size: int = 64, latency_sample: int = 200, latency_threads: int = 1,
                   filename_prefix: str = "stage_b", latency_sink: list[float] | None = None) -> Path | None:
    """`latency_sink`, if given, is extended with this split's sampled
    per-prompt latencies (ms) -- used by `predict` to pool them across
    splits into one model-level latency sidecar."""
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        print(f"[{filename_prefix} predict] {split}: {path} not found, skipping")
        return None
    df = pd.read_parquet(path)
    if {"id", "text"} - set(df.columns):
        raise ValueError(f"{path} missing required columns id, text")
    texts = df["text"].tolist()
    logits = infer_logits(student, tokenizer, texts, max_len, device, batch_size=batch_size)
    latencies = measure_latency_ms(student, tokenizer, texts, max_len, device, latency_sample,
                                    threads=latency_threads)
    if latency_sink is not None:
        latency_sink.extend(latencies.values())

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
    out_path = out_dir / f"{filename_prefix}_{split}_logits.csv"
    out.to_csv(out_path, index=False)
    print(f"[{filename_prefix} predict] wrote {out_path} ({len(out)} rows, "
          f"{len(latencies)} latency samples)")
    return out_path


def predict(model_dir: str | Path, splits: list[str], data_dir: str | Path = "data/processed",
            out_dir: str | Path = "preds", batch_size: int = 64, latency_sample: int = 200,
            latency_threads: int = 1, device: str | None = None) -> list[Path]:
    """Writes per-split logits CSVs plus one pooled latency sidecar,
    `<out_dir>/stage_b_latency.json` (README 5.4: latency does not belong in
    the logits CSV, and predict should not rewrite training metadata.json),
    recording the shared protocol (`reflex_sentry.eval.latency`: batch 1,
    `latency_threads` CPU thread(s), tokenization+forward+softmax timed,
    fixed sample) and p50/p95 pooled across every split's sampled latencies."""
    _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    student, tokenizer, metadata = load_student(model_dir, device=device)
    max_len = int(metadata.get("max_len", 256))
    written = []
    latency_values: list[float] = []
    for split in splits:
        p = predict_split(student, tokenizer, split, data_dir, out_dir, max_len, device,
                           batch_size=batch_size, latency_sample=latency_sample,
                           latency_threads=latency_threads, latency_sink=latency_values)
        if p is not None:
            written.append(p)
    if written:
        sidecar_path = Path(out_dir) / "stage_b_latency.json"
        L.write_sidecar(sidecar_path, latency_values, threads=latency_threads, sample_size=latency_sample)
        print(f"[stage_b predict] wrote latency sidecar to {sidecar_path}")
    return written


# ---------------------------------------------------------------------- CLI --

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine-tune the encoder student on soft targets")
    t.add_argument("--base", default=DEFAULT_BASE, help="HF model id or local path (encoder + tokenizer)")
    t.add_argument("--out", default="models/stage_b")
    t.add_argument("--train-parquet", default="data/processed/train.parquet")
    t.add_argument("--targets-parquet", default="data/interim/soft_targets.parquet")
    t.add_argument("--val-parquet", default="data/processed/val.parquet",
                    help="optional; used only for informational epoch logging, never for checkpoint selection")
    t.add_argument("--dev-ratio", type=float, default=0.1,
                    help="stratified share of the training pool held out for checkpoint selection")
    t.add_argument("--max-len", type=int, default=256)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--eval-batch-size", type=int, default=64)
    t.add_argument("--warmup-ratio", type=float, default=0.1)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--seed", type=int, default=13)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--target-recall", type=float, default=0.95)
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--device", default=None)

    p = sub.add_parser("predict", help="write logits CSVs for one or more splits")
    p.add_argument("--model", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test", "test_ood", "test_evasion"])
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latency-sample", type=int, default=200,
                    help="rows to time individually at batch size 1 on CPU; rest get latency_ms=NaN")
    p.add_argument("--latency-threads", type=int, default=1,
                    help="torch thread count for the timed latency region; 1 mimics a single-core gate "
                         "(README 5.4), matching the comparison table's protocol")
    p.add_argument("--device", default=None)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        train(
            base=args.base, out_dir=args.out, train_parquet=args.train_parquet,
            targets_parquet=args.targets_parquet, val_parquet=args.val_parquet, dev_ratio=args.dev_ratio,
            max_len=args.max_len, epochs=args.epochs, lr=args.lr, head_lr=args.head_lr,
            batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, seed=args.seed,
            grad_clip=args.grad_clip, dropout=args.dropout, target_recall=args.target_recall,
            fp16=args.fp16, device=args.device,
        )
    elif args.command == "predict":
        predict(
            model_dir=args.model, splits=args.splits, data_dir=args.data_dir, out_dir=args.out_dir,
            batch_size=args.batch_size, latency_sample=args.latency_sample,
            latency_threads=args.latency_threads, device=args.device,
        )


if __name__ == "__main__":
    main()

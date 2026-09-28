"""Stage A baseline student: frozen sentence embeddings + multinomial logistic
regression (README 4.4, milestone 4).

Trains on the argmax of the soft targets in `data/interim/soft_targets.parquet`
(sample_weight = the max target probability), falling back to the training
pool's `source_label` (1 -> dangerous, 0 -> safe, NaN dropped) when soft
targets are not built yet.

README 4.1 reserves `val` for temperature scaling and threshold selection
only, so `train` never reads it. `C` is instead chosen by log-loss against an
internal dev set: a stratified 15% of the *training pool* (by target argmax,
grouped by `dup_group` when that column is present so near-duplicates never
split across the fold, fixed seed), scored against the dev soft-target
distributions (cross-entropy) when soft targets exist, else against the dev
hard labels. The final model is then refit on the full training pool with
the chosen `C`.

Embeddings come from a frozen sentence-transformers model (default
`BAAI/bge-small-en-v1.5`) and are cached to `models/stage_a/emb_cache/*.npy`,
keyed by a hash of the embedding model name plus the exact ordered list of
prompt ids being embedded -- so re-running the same split with the same model
is a cache hit, and a different split or model is not.

`sentence-transformers` is only imported when actually needed (it may not be
installed, and its models cannot be downloaded in offline environments), and
every embedding call accepts an `embedder` override (`list[str] -> (n, dim)
ndarray`) so callers -- including tests -- can inject a fake, dependency-free
embedder.

CLI:

    python -m reflex_sentry.models.stage_a train \
        [--train data/processed/train.parquet] \
        [--soft-targets data/interim/soft_targets.parquet] \
        [--model-dir models/stage_a] [--embed-model BAAI/bge-small-en-v1.5] \
        [--dev-frac 0.15]

    python -m reflex_sentry.models.stage_a predict \
        --splits val test test_ood test_evasion \
        [--model-dir models/stage_a] [--processed-dir data/processed] \
        [--preds-dir preds] [--model-name stage_a]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ..eval import metrics as M

# --------------------------------------------------------------- defaults --

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TRAIN_PATH = "data/processed/train.parquet"
DEFAULT_SOFT_TARGETS_PATH = "data/interim/soft_targets.parquet"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_PREDS_DIR = "preds"
DEFAULT_MODEL_DIR = "models/stage_a"
DEFAULT_CACHE_DIR = Path("models/stage_a/emb_cache")
DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_SPLITS = ("val", "test", "test_ood", "test_evasion")

# Class order matches reflex_sentry.eval.calibrate.GOLD_TO_CLASS and the
# logit_safe/logit_dangerous/logit_unsure column order used everywhere else.
CLASSES = ["safe", "dangerous", "unsure"]
GOLD_TO_IDX = {"benign": 0, "dangerous": 1, "ambiguous": 2}
IDX_TO_NAME = {0: "safe", 1: "dangerous", 2: "unsure"}

Embedder = Callable[[Sequence[str]], np.ndarray]

_ST_MODEL_CACHE: dict[str, object] = {}


# ------------------------------------------------------------- embedding ---

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _id_hash(model_name: str, ids: Sequence[str]) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    for i in ids:
        h.update(b"\x00")
        h.update(str(i).encode("utf-8"))
    return h.hexdigest()[:24]


def _cache_path(cache_dir: Path, model_name: str, ids: Sequence[str]) -> Path:
    return Path(cache_dir) / f"{_safe_name(model_name)}__{_id_hash(model_name, ids)}.npy"


def _load_sentence_transformer(model_name: str):
    if model_name not in _ST_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # lazy: may not be installed

        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name]


def _default_embedder(model_name: str) -> Embedder:
    def _embed(texts: Sequence[str]) -> np.ndarray:
        model = _load_sentence_transformer(model_name)
        return np.asarray(
            model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True,
                         normalize_embeddings=True),
            dtype=np.float32,
        )

    return _embed


def embed(ids: Sequence[str], texts: Sequence[str], model_name: str = DEFAULT_EMBED_MODEL,
          cache_dir: Path | str | None = DEFAULT_CACHE_DIR, embedder: Embedder | None = None,
          batch_size: int = 64, use_cache: bool = True) -> np.ndarray:
    """Embed `texts` (aligned with `ids`) with a frozen sentence embedding
    model, batched, with on-disk caching per (model, exact id sequence).

    `embedder`, if given, replaces the sentence-transformers call entirely
    (used by tests to inject a fake, dependency-free embedder); it is still
    called in batches of `batch_size` through this function.
    """
    ids = [str(i) for i in ids]
    texts = list(texts)
    cache_path = None
    if use_cache and cache_dir is not None:
        cache_path = _cache_path(cache_dir, model_name, ids)
        if cache_path.exists():
            return np.load(cache_path)

    fn = embedder or _default_embedder(model_name)
    if not texts:
        arr = np.zeros((0, 0), dtype=np.float32)
    else:
        chunks = [np.asarray(fn(texts[i:i + batch_size]), dtype=np.float32)
                  for i in range(0, len(texts), batch_size)]
        arr = np.concatenate(chunks, axis=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr


# ------------------------------------------------------------------ train --

def _load_targets(train_df: pd.DataFrame, soft_targets_path: str | Path):
    """Return (df, y, sample_weight, target_mode). `df` is the subset of
    `train_df` that has a usable target, aligned row-for-row with y/weight."""
    st_path = Path(soft_targets_path)
    if st_path.exists():
        st = pd.read_parquet(st_path)
        merged = train_df.merge(st[["id", "t_safe", "t_dangerous", "t_unsure"]], on="id", how="inner")
        if merged.empty:
            raise ValueError(f"no rows of {soft_targets_path} matched ids in the training pool")
        probs = merged[["t_safe", "t_dangerous", "t_unsure"]].to_numpy(dtype=float)
        y = probs.argmax(axis=1)
        w = probs.max(axis=1)
        return merged.reset_index(drop=True), y, w, "soft_targets"

    df = train_df.dropna(subset=["source_label"]).copy()
    if df.empty:
        raise ValueError("soft_targets.parquet is missing and no source_label rows are usable")
    y = np.where(df["source_label"].to_numpy(dtype=float) >= 0.5, 1, 0)  # 1 dangerous, 0 safe
    w = np.ones(len(df), dtype=float)
    return df.reset_index(drop=True), y, w, "source_label_fallback"


def _proba_to_full(values: np.ndarray, classes_: Sequence) -> np.ndarray:
    """Expand a (n, len(classes_)) array (probabilities or logits) to
    (n, 3), placing each column at its class index (0=safe, 1=dangerous,
    2=unsure); classes absent from `classes_` get a column of zeros."""
    full = np.zeros((values.shape[0], 3), dtype=np.float64)
    for j, c in enumerate(classes_):
        full[:, int(c)] = values[:, j]
    return full


def _renormalize_rows(proba: np.ndarray) -> np.ndarray:
    """Rescale each row to sum to exactly 1. Float32 embeddings leave a few
    ulps of drift in predict_proba output, enough to trip sklearn's
    log_loss "do not sum to one" check once padded with a zero column."""
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def _stratified_dev_mask(df: pd.DataFrame, y: np.ndarray, dev_frac: float,
                          seed: int) -> np.ndarray:
    """A boolean mask selecting an internal dev fold: within each class
    stratum (by `y`), whole groups (`dup_group` if present, else each row's
    own id) are shuffled with a fixed seed and assigned to dev until roughly
    `dev_frac` of that stratum's rows are covered, so near-duplicate rows
    never land on both sides of the split."""
    n = len(df)
    if "dup_group" in df.columns:
        group_id = df["dup_group"].fillna(df["id"].astype(str)).astype(str).to_numpy()
    else:
        group_id = df["id"].astype(str).to_numpy()

    rng = np.random.default_rng(seed)
    dev_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        groups_in_cls = pd.unique(group_id[idx])
        rng.shuffle(groups_in_cls)
        target = int(round(dev_frac * idx.size))
        chosen, count = set(), 0
        for g in groups_in_cls:
            if count >= target:
                break
            g_rows = idx[group_id[idx] == g]
            chosen.add(g)
            count += g_rows.size
        if chosen:
            dev_mask[idx] |= np.isin(group_id[idx], list(chosen))
    return dev_mask


def _soft_cross_entropy(probs_true: np.ndarray, proba_pred: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(proba_pred, eps, 1 - eps)
    return float(-np.mean(np.sum(probs_true * np.log(p), axis=1)))


def train(train_path: str | Path = DEFAULT_TRAIN_PATH,
          soft_targets_path: str | Path = DEFAULT_SOFT_TARGETS_PATH,
          model_dir: str | Path = DEFAULT_MODEL_DIR,
          embed_model: str = DEFAULT_EMBED_MODEL,
          C_grid: Sequence[float] = DEFAULT_C_GRID,
          class_weight_balanced: bool = False,
          embedder: Embedder | None = None,
          cache_dir: str | Path | None = None,
          batch_size: int = 64,
          dev_frac: float = 0.15,
          random_state: int = 0) -> dict:
    """Train the Stage A classifier and save it plus its metadata. Returns
    the metadata dict (also written to `model_dir/metadata.json`).

    Never reads `val.parquet`: README 4.1 reserves `val` for temperature
    scaling and threshold selection, so `C` is chosen against an internal
    dev fold carved out of the training pool instead (see
    `_stratified_dev_mask`), and the final model is refit on the full pool.
    """
    train_df = pd.read_parquet(train_path)
    df, y, w, target_mode = _load_targets(train_df, soft_targets_path)
    soft_probs = df[["t_safe", "t_dangerous", "t_unsure"]].to_numpy(dtype=float) \
        if target_mode == "soft_targets" else None

    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    X = embed(df["id"], df["text"], model_name=embed_model, cache_dir=cache_dir,
              embedder=embedder, batch_size=batch_size)

    class_weight = "balanced" if class_weight_balanced else None

    dev_mask = _stratified_dev_mask(df, y, dev_frac, random_state)
    fit_mask = ~dev_mask

    loss_by_C: dict[float, float] = {}
    best_C = float(C_grid[len(C_grid) // 2])
    best_dev_loss = None
    if dev_mask.sum() > 0 and fit_mask.sum() > 0 and len(np.unique(y[fit_mask])) > 1:
        dev_true = soft_probs[dev_mask] if soft_probs is not None else None
        for C in C_grid:
            clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000,
                                      random_state=random_state)
            clf.fit(X[fit_mask], y[fit_mask], sample_weight=w[fit_mask])
            proba_dev = _renormalize_rows(_proba_to_full(clf.predict_proba(X[dev_mask]), clf.classes_))
            if dev_true is not None:
                loss = _soft_cross_entropy(dev_true, proba_dev)
            else:
                loss = float(log_loss(y[dev_mask], proba_dev, labels=[0, 1, 2]))
            loss_by_C[float(C)] = loss
            if best_dev_loss is None or loss < best_dev_loss:
                best_dev_loss, best_C = loss, float(C)

    final = LogisticRegression(C=best_C, class_weight=class_weight, max_iter=2000,
                                random_state=random_state)
    final.fit(X, y, sample_weight=w)  # refit on the full training pool

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_dir / "model.joblib")

    head_params = int(final.coef_.size + final.intercept_.size)
    # Count the frozen embedder too: it runs at inference, so it is part of
    # the deployed model's size (README 5.4 "Params").
    embed_params = 0
    st = _ST_MODEL_CACHE.get(embed_model)
    if st is not None:
        embed_params = int(sum(p.numel() for p in st.parameters()))
    n_params = head_params + embed_params
    metadata = {
        "embedding_model": embed_model,
        "embedding_dim": int(X.shape[1]) if X.size else 0,
        "target_mode": target_mode,
        "n_train": int(len(df)),
        "dev_frac": dev_frac,
        "dev_n": int(dev_mask.sum()),
        "C": best_C,
        "C_grid": [float(c) for c in C_grid],
        "dev_log_loss": best_dev_loss,
        "dev_log_loss_by_C": loss_by_C or None,
        "class_weight": class_weight,
        "classes": CLASSES,
        "params": n_params,
        "head_params": head_params,
        "embedding_params": embed_params,
        "random_state": random_state,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------- predict --

def _measure_latency(texts: np.ndarray, embed_fn: Embedder, clf: LogisticRegression,
                      sample_n: int, rng: np.random.Generator) -> np.ndarray:
    """Real single-prompt embed+classify wall time (bypassing the embedding
    cache) for up to `sample_n` prompts sampled uniformly from this split.
    Each sampled row gets its own measured latency_ms; every other row is
    filled with the mean of the sampled latencies, since timing every row of
    a large split one prompt at a time is not worth the wall time."""
    n = len(texts)
    latency = np.full(n, np.nan)
    if n == 0:
        return latency
    k = min(sample_n, n)
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        start = time.perf_counter()
        vec = np.asarray(embed_fn([str(texts[i])]), dtype=np.float32)
        clf.decision_function(vec)
        latency[i] = (time.perf_counter() - start) * 1000.0
    mean_latency = float(np.mean(latency[idx]))
    latency[np.isnan(latency)] = mean_latency
    return latency


def predict(model_dir: str | Path = DEFAULT_MODEL_DIR,
            splits: Sequence[str] = DEFAULT_SPLITS,
            processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
            preds_dir: str | Path = DEFAULT_PREDS_DIR,
            model_name: str = "stage_a",
            embedder: Embedder | None = None,
            cache_dir: str | Path | None = None,
            batch_size: int = 64,
            latency_sample_n: int = 200,
            random_state: int = 0) -> dict:
    """Score every available split, writing both a logits CSV and a
    prediction CSV per README 5.1 / the calibration contract. Returns
    {split: {"preds": path, "logits": path}} for the splits actually written."""
    model_dir = Path(model_dir)
    clf: LogisticRegression = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    embed_model = metadata["embedding_model"]
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    embed_fn = embedder or _default_embedder(embed_model)

    processed_dir, preds_dir = Path(processed_dir), Path(preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    written: dict = {}
    for split in splits:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            print(f"stage_a predict: skipping {split!r}, no file at {path}")
            continue
        df = pd.read_parquet(path)
        if "gold" not in df.columns or df.empty:
            print(f"stage_a predict: skipping {split!r}, no gold labels")
            continue

        X = embed(df["id"], df["text"], model_name=embed_model, cache_dir=cache_dir,
                  embedder=embedder, batch_size=batch_size)
        raw = clf.decision_function(X)
        # decision_function is 1-D (positive-class score only) for a 2-class
        # fit (the source_label fallback, which only ever has safe/dangerous)
        logits = np.stack([-raw, raw], axis=1) if raw.ndim == 1 else raw
        probs = clf.predict_proba(X)

        class_order = list(clf.classes_)
        logit_cols = _proba_to_full(logits, class_order)
        prob_cols = _proba_to_full(probs, class_order)

        latency_ms = _measure_latency(df["text"].astype(str).to_numpy(), embed_fn, clf,
                                       latency_sample_n, rng)

        base = {
            "id": df["id"].astype(str).to_numpy(),
            "gold": df["gold"].to_numpy(),
            "source": df["source"].to_numpy() if "source" in df.columns else "",
            "tags": df["tags"].to_numpy() if "tags" in df.columns else "",
            "latency_ms": latency_ms,
        }
        logits_df = pd.DataFrame({**base, "logit_safe": logit_cols[:, 0],
                                   "logit_dangerous": logit_cols[:, 1], "logit_unsure": logit_cols[:, 2]})
        preds_df = pd.DataFrame({**base, "p_safe": prob_cols[:, 0],
                                  "p_dangerous": prob_cols[:, 1], "p_unsure": prob_cols[:, 2]})
        M.validate(preds_df)

        logits_path = preds_dir / f"{model_name}_{split}_logits.csv"
        preds_path = preds_dir / f"{model_name}_{split}.csv"
        logits_df.to_csv(logits_path, index=False)
        preds_df.to_csv(preds_path, index=False)
        written[split] = {"preds": str(preds_path), "logits": str(logits_path)}

    return written


# --------------------------------------------------------------------- CLI -

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train the Stage A classifier")
    pt.add_argument("--train", default=DEFAULT_TRAIN_PATH)
    pt.add_argument("--soft-targets", default=DEFAULT_SOFT_TARGETS_PATH)
    pt.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    pt.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    pt.add_argument("--class-weight-balanced", action="store_true")
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--dev-frac", type=float, default=0.15,
                    help="share of the training pool held out to choose C (never val.parquet)")

    pp = sub.add_parser("predict", help="score Stage A on one or more splits")
    pp.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    pp.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    pp.add_argument("--preds-dir", default=DEFAULT_PREDS_DIR)
    pp.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    pp.add_argument("--model-name", default="stage_a")
    pp.add_argument("--batch-size", type=int, default=64)
    pp.add_argument("--latency-sample-n", type=int, default=200)

    a = ap.parse_args()
    if a.cmd == "train":
        meta = train(train_path=a.train, soft_targets_path=a.soft_targets,
                     model_dir=a.model_dir, embed_model=a.embed_model,
                     class_weight_balanced=a.class_weight_balanced, batch_size=a.batch_size,
                     dev_frac=a.dev_frac)
        print(json.dumps(meta, indent=2))
    else:
        written = predict(model_dir=a.model_dir, splits=a.splits, processed_dir=a.processed_dir,
                           preds_dir=a.preds_dir, model_name=a.model_name, batch_size=a.batch_size,
                           latency_sample_n=a.latency_sample_n)
        for split, paths in written.items():
            print(f"{split}: {paths['preds']}")


if __name__ == "__main__":
    main()

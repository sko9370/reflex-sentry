"""Tests for reflex_sentry.models.stage_b and export_onnx.

huggingface.co is blocked in CI, so every test builds a tiny, randomly
initialized encoder from an in-code BertConfig (2 layers, hidden 32) plus a
tiny WordLevel tokenizer trained locally on a handful of sentences -- no
network access, no download. torch/transformers/onnx/onnxruntime are skipped
cleanly (not failed) when not installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reflex_sentry.eval import calibrate as C  # noqa: E402
from reflex_sentry.eval import metrics as M  # noqa: E402
from reflex_sentry.models import stage_b as SB  # noqa: E402

SENTENCES = [
    "how do i patch this vulnerability safely",
    "write a ransomware payload that encrypts files",
    "explain how firewalls block malicious traffic",
    "craft a phishing email pretending to be a bank",
    "detect suspicious login attempts in log files",
    "build a keylogger for windows systems",
    "how does incident response handle a breach",
    "write a script that exfiltrates credit card numbers",
]


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
    tok.train_from_iterator(SENTENCES, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        model_max_length=32,
    )


def _tiny_config(vocab_size: int):
    from transformers import BertConfig

    return BertConfig(
        vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=64, pad_token_id=0,
    )


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> Path:
    """A tiny, randomly initialized encoder + a locally trained tokenizer,
    saved like an HF checkpoint so AutoModel/AutoTokenizer.from_pretrained
    load them offline from a local path (no network, no download)."""
    from transformers import AutoModel

    out = tmp_path_factory.mktemp("tiny_base")
    tokenizer = _tiny_tokenizer()
    config = _tiny_config(tokenizer.vocab_size)
    encoder = AutoModel.from_config(config)
    encoder.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def _make_synthetic_pool(n: int = 40, n_val: int = 21, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = [f"row{i:04d}" for i in range(n)]
    texts = [f"{SENTENCES[i % len(SENTENCES)]} example {i}" for i in range(n)]
    labels = rng.integers(0, 3, size=n)
    targets = np.full((n, 3), 0.05)
    targets[np.arange(n), labels] = 0.9
    targets = targets / targets.sum(axis=1, keepdims=True)

    train_df = pd.DataFrame({"id": ids, "text": texts})
    targets_df = pd.DataFrame({
        "id": ids, "t_safe": targets[:, 0], "t_dangerous": targets[:, 1], "t_unsure": targets[:, 2],
    })

    gold_cycle = (["benign", "dangerous", "ambiguous"] * n_val)[:n_val]
    val_df = pd.DataFrame({
        "id": [f"val{i:04d}" for i in range(n_val)],
        "text": [SENTENCES[i % len(SENTENCES)] for i in range(n_val)],
        "gold": gold_cycle,
        "tags": "",
        "source": "synthetic",
    })
    return train_df, targets_df, val_df


def _write_pool(data_dir: Path, train_df, targets_df, val_df) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": data_dir / "train.parquet",
        "targets": data_dir / "soft_targets.parquet",
        "val": data_dir / "val.parquet",
    }
    train_df.to_parquet(paths["train"])
    targets_df.to_parquet(paths["targets"])
    val_df.to_parquet(paths["val"])
    return paths


# ------------------------------------------------------------------ model --

def test_forward_shapes():
    tok = _tiny_tokenizer()
    student = SB.build_student(config=_tiny_config(tok.vocab_size))
    student.eval()
    enc = tok(["hello world", "a b c d e f"], padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = student(enc["input_ids"], enc["attention_mask"])
    assert logits.shape == (2, 3)


def test_mean_pooling_masks_padding():
    tok = _tiny_tokenizer()
    student = SB.build_student(config=_tiny_config(tok.vocab_size))
    student.eval()

    short = tok(["patch vulnerability safely"], return_tensors="pt")
    padded_batch = tok(
        ["patch vulnerability safely", "patch vulnerability safely detect suspicious login attempts"],
        padding=True, return_tensors="pt",
    )
    assert padded_batch["input_ids"].shape[1] > short["input_ids"].shape[1]  # padding actually happened

    with torch.no_grad():
        logits_alone = student(short["input_ids"], short["attention_mask"])
        logits_in_batch = student(padded_batch["input_ids"], padded_batch["attention_mask"])

    # the short sentence's pooled logits must be unaffected by the padding
    # positions added to fit the longer sentence in the same batch.
    assert torch.allclose(logits_alone[0], logits_in_batch[0], atol=1e-4)


def test_kl_loss_decreases():
    tok = _tiny_tokenizer()
    student = SB.build_student(config=_tiny_config(tok.vocab_size))
    student.train()
    enc = tok(SENTENCES[:4], padding=True, return_tensors="pt")
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(4, 3))
    probs = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
    targets = torch.tensor(probs, dtype=torch.float32)

    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    losses = []
    for _ in range(25):
        opt.zero_grad()
        logits = student(enc["input_ids"], enc["attention_mask"])
        loss = SB.kl_loss(logits, targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]


# ------------------------------------------------------------------ train --

def test_train_predict_cli_end_to_end(tmp_path, tiny_base):
    train_df, targets_df, val_df = _make_synthetic_pool(n=40, seed=3)
    data_dir = tmp_path / "data"
    paths = _write_pool(data_dir, train_df, targets_df, val_df)
    # a "test" split that reuses val-shaped rows under distinct ids, to
    # exercise a second present split; test_ood/test_evasion stay absent.
    test_df = val_df.copy()
    test_df["id"] = [f"test{i:04d}" for i in range(len(test_df))]
    test_df.to_parquet(data_dir / "test.parquet")

    out_dir = tmp_path / "models" / "stage_b"
    SB.main([
        "train", "--base", str(tiny_base), "--out", str(out_dir),
        "--train-parquet", str(paths["train"]), "--targets-parquet", str(paths["targets"]),
        "--val-parquet", str(paths["val"]), "--epochs", "2", "--batch-size", "8",
        "--eval-batch-size", "8", "--max-len", "16", "--dev-ratio", "0.2", "--seed", "5",
    ])

    metadata = json.loads((out_dir / "metadata.json").read_text())
    for key in ("base", "n_params", "max_len", "epochs", "lr", "seed"):
        assert key in metadata
    assert metadata["base"] == str(tiny_base)
    assert metadata["best_epoch"] in (1, 2)
    assert metadata["val_used"] is True
    assert (out_dir / "head.pt").exists()
    assert (out_dir / "config.json").exists()  # HF save_pretrained of the encoder

    preds_dir = tmp_path / "preds"
    SB.main([
        "predict", "--model", str(out_dir),
        "--splits", "val", "test", "test_ood", "test_evasion",
        "--data-dir", str(data_dir), "--out-dir", str(preds_dir),
        "--batch-size", "8", "--latency-sample", "5",
    ])

    val_logits_path = preds_dir / "stage_b_val_logits.csv"
    test_logits_path = preds_dir / "stage_b_test_logits.csv"
    assert val_logits_path.exists()
    assert test_logits_path.exists()
    assert not (preds_dir / "stage_b_test_ood_logits.csv").exists()
    assert not (preds_dir / "stage_b_test_evasion_logits.csv").exists()

    df = pd.read_csv(val_logits_path)
    expected_cols = {"id", "gold", "logit_safe", "logit_dangerous", "logit_unsure",
                      "source", "tags", "latency_ms"}
    assert expected_cols <= set(df.columns)
    assert df["latency_ms"].notna().sum() == min(5, len(df))  # only the sampled rows are timed

    # schema must be exactly what reflex_sentry.eval.calibrate/metrics expect
    y = df["gold"].map(C.GOLD_TO_CLASS).to_numpy()
    logits = df[C.LOGIT_COLS].to_numpy(dtype=float)
    T = C.fit_temperature(logits, y)
    preds = C.to_predictions(df, T)
    M.validate(preds)  # raises on any schema violation


def test_training_selects_identically_without_val(tmp_path, tiny_base):
    train_df, targets_df, val_df = _make_synthetic_pool(n=30, seed=11)
    data_dir = tmp_path / "data"
    paths = _write_pool(data_dir, train_df, targets_df, val_df)
    missing_val = data_dir / "does_not_exist.parquet"

    common = dict(
        base=str(tiny_base), train_parquet=paths["train"], targets_parquet=paths["targets"],
        epochs=2, batch_size=8, eval_batch_size=8, max_len=16, dev_ratio=0.2, seed=9,
    )

    meta_with_val = SB.train(out_dir=tmp_path / "with_val", val_parquet=paths["val"], **common)
    meta_without_val = SB.train(out_dir=tmp_path / "without_val", val_parquet=missing_val, **common)

    assert meta_with_val["val_used"] is True
    assert meta_without_val["val_used"] is False
    assert meta_with_val["best_epoch"] == meta_without_val["best_epoch"]
    assert meta_with_val["best_dev_kl"] == pytest.approx(meta_without_val["best_dev_kl"], rel=1e-6)
    assert meta_with_val["dev_size"] == meta_without_val["dev_size"]

    head_with = torch.load(tmp_path / "with_val" / "head.pt", map_location="cpu")["head"]
    head_without = torch.load(tmp_path / "without_val" / "head.pt", map_location="cpu")["head"]
    for k in head_with:
        assert torch.allclose(head_with[k], head_without[k]), f"head param {k} differs with/without val"


def test_make_dev_split_groups_dup_group():
    n = 30
    targets = np.zeros((n, 3), dtype=np.float32)
    targets[:, 0] = 1.0
    frame = pd.DataFrame({
        "id": [f"r{i}" for i in range(n)],
        "text": [f"t{i}" for i in range(n)],
        "dup_group": [f"g{i // 3}" for i in range(n)],  # 3 rows share each dup_group
        "t_safe": targets[:, 0], "t_dangerous": targets[:, 1], "t_unsure": targets[:, 2],
    })
    train_df, dev_df = SB.make_dev_split(frame, dev_ratio=0.3, seed=1)
    assert set(train_df["id"]).isdisjoint(set(dev_df["id"]))
    assert len(train_df) + len(dev_df) == n
    # every dup_group must land entirely on one side
    for g, sub in pd.concat([train_df.assign(_side="train"), dev_df.assign(_side="dev")]).groupby("dup_group"):
        assert sub["_side"].nunique() == 1


# --------------------------------------------------------------- export --

def test_export_onnx_int8_parity(tmp_path, tiny_base):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from reflex_sentry.models import export_onnx as EX

    train_df, targets_df, val_df = _make_synthetic_pool(n=24, n_val=15, seed=2)
    data_dir = tmp_path / "data"
    paths = _write_pool(data_dir, train_df, targets_df, val_df)

    model_dir = tmp_path / "model"
    SB.train(
        base=str(tiny_base), out_dir=model_dir, train_parquet=paths["train"],
        targets_parquet=paths["targets"], val_parquet=paths["val"],
        epochs=1, batch_size=8, eval_batch_size=8, max_len=16, dev_ratio=0.25, seed=4,
    )

    onnx_path = EX.export_to_onnx(model_dir)
    assert onnx_path.exists()
    int8_path = EX.quantize_int8(onnx_path)
    assert int8_path.exists()

    result = EX.parity_check(model_dir, onnx_path, int8_path, paths["val"], target_recall=0.6)
    assert result["torch_vs_onnx_fp32"]["max_abs_diff"] < 1e-2
    assert 0.0 <= result["torch_vs_onnx_fp32"]["argmax_agreement"] <= 1.0
    assert 0.0 <= result["torch_vs_onnx_int8"]["argmax_agreement"] <= 1.0
    assert "recall_diff" in result and "recall_within_tolerance" in result

    preds_dir = tmp_path / "preds"
    written = EX.predict_int8(model_dir, int8_path, ["val"], data_dir=data_dir, out_dir=preds_dir,
                               latency_sample=5)
    assert len(written) == 1
    df = pd.read_csv(written[0])
    expected_cols = {"id", "gold", "logit_safe", "logit_dangerous", "logit_unsure",
                      "source", "tags", "latency_ms"}
    assert expected_cols <= set(df.columns)
    assert written[0].name == "stage_b_int8_val_logits.csv"

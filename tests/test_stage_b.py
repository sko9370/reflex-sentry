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


def test_predict_writes_latency_sidecar(tmp_path, tiny_base):
    train_df, targets_df, val_df = _make_synthetic_pool(n=20, n_val=10, seed=6)
    data_dir = tmp_path / "data"
    paths = _write_pool(data_dir, train_df, targets_df, val_df)

    out_dir = tmp_path / "models" / "stage_b"
    SB.train(base=str(tiny_base), out_dir=out_dir, train_parquet=paths["train"],
              targets_parquet=paths["targets"], val_parquet=paths["val"], epochs=1, batch_size=8,
              eval_batch_size=8, max_len=16, dev_ratio=0.2, seed=5)

    preds_dir = tmp_path / "preds"
    written = SB.predict(model_dir=out_dir, splits=["val"], data_dir=data_dir, out_dir=preds_dir,
                          batch_size=8, latency_sample=4, latency_threads=1)
    assert written

    sidecar = preds_dir / "stage_b_latency.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["latency_protocol"] == {
        "threads": 1, "batch": 1, "sample_size": 4, "includes_tokenization": True,
    }
    assert payload["n"] == min(4, len(val_df))
    assert payload["p50_ms"] is not None and payload["p95_ms"] is not None


def test_measure_latency_ms_restores_thread_count(tiny_base):
    tok = _tiny_tokenizer()
    student = SB.build_student(config=_tiny_config(tok.vocab_size))
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, previous))
        before = torch.get_num_threads()
        out = SB.measure_latency_ms(student, tok, SENTENCES, 16, "cpu", sample_size=3, threads=1)
        assert len(out) == 3
        assert torch.get_num_threads() == before
    finally:
        torch.set_num_threads(previous)


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

@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory, tiny_base):
    """Train a tiny student for one epoch and export it to fp32 ONNX once; the
    export tests below share it to stay fast."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from reflex_sentry.models import export_onnx as EX

    root = tmp_path_factory.mktemp("tiny_export")
    train_df, targets_df, val_df = _make_synthetic_pool(n=24, n_val=15, seed=2)
    paths = _write_pool(root / "data", train_df, targets_df, val_df)
    model_dir = root / "model"
    SB.train(
        base=str(tiny_base), out_dir=model_dir, train_parquet=paths["train"],
        targets_parquet=paths["targets"], val_parquet=paths["val"],
        epochs=1, batch_size=8, eval_batch_size=8, max_len=16, dev_ratio=0.25, seed=4,
    )
    onnx_path = EX.export_to_onnx(model_dir)
    return {"root": root, "model_dir": model_dir, "onnx": onnx_path, "val": paths["val"],
            "data_dir": root / "data"}


def _run_onnx(path: Path, tiny_base) -> np.ndarray:
    import onnxruntime as ort

    tok = _tiny_tokenizer()
    enc = tok(SENTENCES[:4], padding=True, return_tensors="np")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                 "attention_mask": enc["attention_mask"].astype(np.int64)})[0]


def test_export_onnx_int8_parity(tiny_export):
    from reflex_sentry.models import export_onnx as EX

    model_dir, onnx_path, val = tiny_export["model_dir"], tiny_export["onnx"], tiny_export["val"]
    assert onnx_path.exists()
    int8_path = EX.quantize_int8(onnx_path, tiny_export["root"] / "parity_int8.onnx")
    assert int8_path.exists()

    out_json = tiny_export["root"] / "parity_out.json"
    result = EX.parity_check(model_dir, onnx_path, int8_path, val, target_recall=0.6, out_path=out_json)
    # informational only: no pass/fail fields, each side has its own threshold
    assert "ok" not in result and "warnings" not in result
    assert result["informational"] is True
    assert "informational only" in result["note"]
    assert result["torch_vs_onnx_fp32"]["max_abs_diff"] < 1e-2
    assert 0.0 <= result["torch_vs_onnx_int8"]["argmax_agreement"] <= 1.0
    for key in ("decision_agreement", "fp32", "int8", "diff", "threshold_fp32", "threshold_int8"):
        assert key in result
    assert 0.0 <= result["decision_agreement"]["overall"] <= 1.0
    for side in ("fp32", "int8"):
        r = result[side]["recall"]
        assert r["n"] == result["n_dangerous"] == 5 and 0 <= r["k"] <= r["n"]
        assert r["rate"] == pytest.approx(r["k"] / r["n"])
        b = result[side]["benign_escalation"]
        assert b["n"] == result["n_benign"] == 5
        assert 0.0 <= result[side]["average_precision"] <= 1.0
        assert 0.0 <= result[side]["roc_auc"] <= 1.0
    assert result["diff"]["average_precision"] == pytest.approx(
        result["int8"]["average_precision"] - result["fp32"]["average_precision"])
    assert result["diff"]["recall_items"] == result["int8"]["recall"]["k"] - result["fp32"]["recall"]["k"]
    # parity.json was written and is strict JSON
    saved = json.loads(out_json.read_text())
    assert saved["decision_agreement"]["overall"] == pytest.approx(result["decision_agreement"]["overall"])

    preds_dir = tiny_export["root"] / "preds"
    written = EX.predict_int8(model_dir, int8_path, ["val"], data_dir=tiny_export["data_dir"],
                               out_dir=preds_dir, latency_sample=5, bulk_threads=0, latency_threads=1)
    assert len(written) == 1
    df = pd.read_csv(written[0])
    expected_cols = {"id", "gold", "logit_safe", "logit_dangerous", "logit_unsure",
                      "source", "tags", "latency_ms"}
    assert expected_cols <= set(df.columns)
    assert df["latency_ms"].notna().sum() == 5  # only the sampled rows are timed
    assert written[0].name == "stage_b_int8_val_logits.csv"
    assert json.loads((preds_dir / "stage_b_int8_latency.json").read_text())["n"] == 5


def test_predict_fp32_onnx_is_isolated_and_matches_torch(tiny_export):
    from reflex_sentry.models import export_onnx as EX

    model_dir = tiny_export["model_dir"]
    preds_dir = tiny_export["root"] / "fp32_preds"
    kwargs = {"data_dir": tiny_export["data_dir"], "out_dir": preds_dir,
              "latency_sample": 4, "latency_threads": 1}
    fp32_files = EX.predict_int8(model_dir, splits=["val"], onnx_path=tiny_export["onnx"], **kwargs)
    assert [p.name for p in fp32_files] == ["stage_b_onnx_val_logits.csv"]
    fp32 = pd.read_csv(fp32_files[0])

    student, tokenizer, metadata = SB.load_student(model_dir, device="cpu")
    texts = pd.read_parquet(tiny_export["val"])["text"].tolist()
    expected = SB.infer_logits(student, tokenizer, texts, int(metadata["max_len"]), "cpu")
    np.testing.assert_allclose(fp32[["logit_safe", "logit_dangerous", "logit_unsure"]].to_numpy(),
                               expected, atol=1e-3, rtol=1e-3)
    assert fp32["latency_ms"].notna().sum() == 4
    sidecar = json.loads((preds_dir / "stage_b_onnx_latency.json").read_text())
    assert sidecar["n"] == 4
    assert sidecar["latency_protocol"] == {
        "threads": 1, "batch": 1, "sample_size": 4, "includes_tokenization": True,
    }
    assert sidecar["p50_ms"] > 0

    int8_path = EX.quantize_int8(tiny_export["onnx"], tiny_export["root"] / "predict_int8.onnx")
    int8_files = EX.predict_int8(model_dir, int8_path, ["val"], **kwargs)
    assert [p.name for p in int8_files] == ["stage_b_int8_val_logits.csv"]
    assert fp32_files[0].exists()  # the second prediction did not overwrite fp32
    assert (preds_dir / "stage_b_onnx_latency.json").exists()
    assert (preds_dir / "stage_b_int8_latency.json").exists()


def test_predict_onnx_cli_paths_are_exclusive():
    from reflex_sentry.models import export_onnx as EX

    with pytest.raises(SystemExit):
        EX._build_parser().parse_args(["predict", "--model", "m", "--onnx", "a.onnx",
                                        "--int8", "b.onnx"])
    with pytest.raises(ValueError, match="exactly one"):
        EX.predict_int8("m", "b.onnx", ["val"], onnx_path="a.onnx")


def test_onnx_latency_times_tokenization_forward_and_softmax(monkeypatch):
    from reflex_sentry.models import export_onnx as EX

    clock = [0.0]
    events = []

    def tokenizer(batch, **kwargs):
        assert len(batch) == 1
        events.append(("tokenize", batch[0]))
        clock[0] += 0.002
        return {"input_ids": np.ones((1, 2), dtype=np.int64),
                "attention_mask": np.ones((1, 2), dtype=np.int64)}

    class Session:
        def run(self, outputs, feed):
            assert outputs == ["logits"] and feed["input_ids"].shape == (1, 2)
            events.append(("forward", None))
            clock[0] += 0.003
            return [np.array([[1.0, 0.0, -1.0]], dtype=np.float32)]

    def softmax(logits):
        assert logits.shape == (1, 3)
        events.append(("softmax", None))
        clock[0] += 0.005
        return np.array([[0.7, 0.2, 0.1]])

    monkeypatch.setattr(EX.L.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(EX.SB, "_softmax", softmax)
    times = EX._ort_latency_ms(Session(), tokenizer, ["a", "b", "c"], max_len=16, sample_size=2)

    assert set(times) == {1, 2}  # shared helper's deterministic seed-0 sample
    assert list(times.values()) == pytest.approx([10.0, 10.0])
    assert len(events) == 7 * 3  # five warmups plus two measured prompts
    assert [kind for kind, _ in events] == ["tokenize", "forward", "softmax"] * 7


def test_parity_metrics_identical_models():
    """parity_metrics is informational only (never gates); with identical
    fp32/int8 logits and the same threshold on both sides, decisions agree
    everywhere."""
    from reflex_sentry.models import export_onnx as EX

    rng = np.random.default_rng(0)
    gold = np.array(["dangerous"] * 20 + ["benign"] * 30 + ["ambiguous"] * 10)
    logits = rng.normal(size=(60, 3))
    logits[:20, 1] += 2.0
    logits[20:50, 0] += 2.0
    m = EX.parity_metrics(gold, logits, logits, logits, t_fp32=0.3)
    assert m["threshold_fp32"] == m["threshold_int8"] == 0.3
    assert m["decision_agreement"]["overall"] == 1.0
    assert m["decision_agreement"]["dangerous"] == 1.0
    assert m["decision_agreement"]["n_disagree"] == 0
    assert m["diff"]["recall_items"] == 0 and m["diff"]["benign_escalation_items"] == 0
    assert m["diff"]["average_precision"] == 0.0 and m["diff"]["roc_auc"] == 0.0
    assert m["torch_vs_onnx_int8"]["max_abs_diff"] == 0.0
    assert m["torch_vs_onnx_int8"]["argmax_agreement"] == 1.0
    assert m["informational"] is True
    assert "ok" not in m and "warnings" not in m
    # counts are consistent with an independent computation
    p_safe = SB._softmax(logits)[:, 0]
    assert m["fp32"]["recall"] == {"k": int((p_safe[:20] < 0.3).sum()), "n": 20,
                                   "rate": (p_safe[:20] < 0.3).mean()}
    assert m["fp32"]["benign_escalation"]["n"] == 30
    assert m["n_ambiguous"] == 10


def test_parity_metrics_own_thresholds_never_gate():
    """Different thresholds per side (the deployment-realistic case: each
    variant is calibrated separately) still produce a well-formed report with
    no pass/fail semantics -- parity never gates int8 selection."""
    from reflex_sentry.models import export_onnx as EX

    gold = np.array(["dangerous"] * 48 + ["benign"] * 100)
    base = np.zeros((148, 3))
    base[:48, 1] = 3.0   # p_safe small -> escalated
    base[48:, 0] = 3.0   # p_safe large -> passed
    # heavy disagreement at a shared threshold would have tripped the old gate
    many = base.copy()
    many[48:60] = [0.0, 3.0, 0.0]
    m = EX.parity_metrics(gold, base, base, many, t_fp32=0.5, t_int8=0.5)
    assert "ok" not in m and "warnings" not in m
    assert m["decision_agreement"]["overall"] < 1.0  # still reported, just not gated
    # t_int8 defaults to t_fp32 when omitted
    m2 = EX.parity_metrics(gold, base, base, many, t_fp32=0.5)
    assert m2["threshold_int8"] == m2["threshold_fp32"] == 0.5


def test_matched_rate_metrics_identical_models():
    """Identical fp32/int8 logits: int8's matched-rate threshold reproduces
    fp32's escalation set exactly, so decision agreement is 1.0 and the
    config passes."""
    from reflex_sentry.models import export_onnx as EX

    rng = np.random.default_rng(0)
    gold = np.array(["dangerous"] * 20 + ["benign"] * 30)
    logits = rng.normal(size=(50, 3))
    logits[:20, 1] += 2.0
    logits[20:, 0] += 2.0
    m = EX.matched_rate_metrics(gold, logits, logits, target_recall=0.6)
    assert m["decision_agreement"] == 1.0
    assert m["escalation_count_fp32"] == m["escalation_count_int8"]
    assert m["average_precision"]["diff"] == 0.0 and m["roc_auc"]["diff"] == 0.0
    assert m["max_abs_logit_diff"] == 0.0
    assert m["ok"] and m["warnings"] == []


def test_matched_rate_metrics_pass_rule():
    from reflex_sentry.models import export_onnx as EX

    gold = np.array(["dangerous"] * 48 + ["benign"] * 100)
    base = np.zeros((148, 3))
    base[:48, 1] = 3.0   # dangerous rows: low p_safe
    base[48:, 0] = 3.0   # benign rows: high p_safe

    # a uniformly rescaled copy preserves ranking (and hence the matched
    # escalation set) exactly -> passes
    scaled = base * 0.5
    m_ok = EX.matched_rate_metrics(gold, base, scaled, target_recall=0.6)
    assert m_ok["ok"]
    assert m_ok["decision_agreement"] >= 0.97
    assert m_ok["average_precision"]["diff"] == pytest.approx(0.0, abs=1e-9)

    # a ranking scrambled at random breaks both the agreement and AP rules
    rng = np.random.default_rng(1)
    scrambled = rng.normal(size=base.shape)
    m_bad = EX.matched_rate_metrics(gold, base, scrambled, target_recall=0.6,
                                    min_agreement=0.97, max_ap_drop=0.01)
    assert not m_bad["ok"]
    assert m_bad["warnings"]


def test_quant_config_presets_and_overrides():
    from reflex_sentry.models import export_onnx as EX

    d = EX.resolve_quant_config("default")
    assert (d.per_channel, d.op_types, d.exclude_head, d.reduce_range, d.weight_type) == \
        (True, ("MatMul",), True, False, "qint8")
    legacy = EX.resolve_quant_config("legacy")
    assert (legacy.per_channel, legacy.op_types, legacy.exclude_head) == (False, None, False)
    o = EX.resolve_quant_config("default", per_channel=False, op_types=["all"], weight_type="quint8")
    assert (o.per_channel, o.op_types, o.weight_type) == (False, None, "quint8")
    assert EX.SWEEP_CONFIGS == ["legacy", "matmul_per_channel", "matmul_per_channel_exclude_head",
                                "matmul_per_channel_exclude_head_reduce_range"]


@pytest.mark.parametrize("kwargs", [
    {"preset": "legacy"},
    {},
    {"reduce_range": True},
    {"weight_type": "quint8", "per_channel": False},
    {"op_types": ["MatMul", "Gemm"]},
])
def test_quantize_options_produce_runnable_onnx(tiny_export, tiny_base, kwargs):
    import onnx
    from reflex_sentry.models import export_onnx as EX

    out = tiny_export["root"] / f"q_{abs(hash(json.dumps(kwargs, sort_keys=True)))}.onnx"
    path = EX.quantize_int8(tiny_export["onnx"], out, **kwargs)
    onnx.checker.check_model(str(path))
    logits = _run_onnx(path, tiny_base)
    assert logits.shape == (4, 3) and np.isfinite(logits).all()


def _matmul_head_model(path: Path) -> None:
    """x -> /encoder/MatMul -> Relu -> /head/MatMul -> Add -> logits, with
    named nodes and constant weights, so the head is a plain MatMul."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.normal(size=(8, 8)).astype(np.float32), "w_enc")
    w2 = numpy_helper.from_array(rng.normal(size=(8, 3)).astype(np.float32), "w_head")
    bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "b_head")
    nodes = [
        helper.make_node("MatMul", ["x", "w_enc"], ["h"], name="/encoder/MatMul"),
        helper.make_node("Relu", ["h"], ["r"], name="/encoder/Relu"),
        helper.make_node("MatMul", ["r", "w_head"], ["m"], name="/head/MatMul"),
        helper.make_node("Add", ["m", "b_head"], ["logits"], name="/head/Add"),
    ]
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b", 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 3])], [w1, w2, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _matmul_state(path: Path) -> dict:
    """{node name: 'fp32' | 'int8'} for the two encoder/head matmuls, from the
    quantized graph: a quantized MatMul becomes MatMulInteger (+ int8 weight)."""
    import onnx

    m = onnx.load(str(path))
    ops = {n.name: n.op_type for n in m.graph.node}
    inits = {i.name: i.data_type for i in m.graph.initializer}
    out = {}
    for base in ("/encoder/MatMul", "/head/MatMul"):
        if ops.get(base) == "MatMul":
            out[base] = "fp32"
        else:
            out[base] = "int8"
    out["_init_types"] = set(inits.values())
    return out


def test_find_head_nodes_by_name_and_by_graph(tmp_path):
    import onnx
    from onnx import helper
    from reflex_sentry.models import export_onnx as EX

    path = tmp_path / "m.onnx"
    _matmul_head_model(path)
    model = onnx.load(str(path))
    assert EX.find_head_nodes(model) == ["/head/MatMul"]
    # no "head" in the names: the last MatMul feeding the output is still found
    for n in model.graph.node:
        n.name = n.name.replace("head", "cls")
    assert EX.find_head_nodes(model) == ["/cls/MatMul"]


def test_exclude_head_keeps_head_fp32(tmp_path):
    import onnx
    from onnx import TensorProto
    from reflex_sentry.models import export_onnx as EX

    src = tmp_path / "m.onnx"
    _matmul_head_model(src)

    excluded = EX.quantize_int8(src, tmp_path / "ex.onnx", exclude_head=True, op_types=["MatMul"])
    included = EX.quantize_int8(src, tmp_path / "in.onnx", exclude_head=False, op_types=["MatMul"])

    ex_state, in_state = _matmul_state(excluded), _matmul_state(included)
    assert ex_state["/encoder/MatMul"] == "int8"
    assert ex_state["/head/MatMul"] == "fp32"          # head untouched
    assert in_state["/head/MatMul"] == "int8"           # control: quantized when not excluded
    ex_model = onnx.load(str(excluded))
    head_w = next(i for i in ex_model.graph.initializer if i.name == "w_head")
    assert head_w.data_type == TensorProto.FLOAT       # head weight stays fp32
    assert any(n.op_type == "MatMulInteger" for n in ex_model.graph.node)  # encoder is int8


def test_tiny_model_head_is_fp32_by_default(tiny_export):
    """The exported head is a Gemm. onnxruntime rewrites Gemm to MatMul+Add and
    would quantize it even with op_types=[MatMul], so this guards the
    `<name>_MatMul` exclusion alias as well as the plain name."""
    import onnx
    from onnx import TensorProto
    from reflex_sentry.models import export_onnx as EX

    fp32_model = onnx.load(str(tiny_export["onnx"]))
    head_nodes = EX.find_head_nodes(fp32_model)
    assert head_nodes == ["/head/Gemm"]
    assert EX.find_head_nodes(fp32_model, gemm_aliases=True) == ["/head/Gemm", "/head/Gemm_MatMul"]

    def quantized(path):
        m = onnx.load(str(path))
        head_int = [n for n in m.graph.node if "/head/" in n.name and n.op_type == "MatMulInteger"]
        inits = {i.name: i.data_type for i in m.graph.initializer}
        return m, head_int, inits

    kept = EX.quantize_int8(tiny_export["onnx"], tiny_export["root"] / "default_q.onnx")
    m, head_int, inits = quantized(kept)
    assert not head_int                                       # head not integer-quantized
    assert inits["head.weight"] == TensorProto.FLOAT          # head weight stays fp32
    assert "head.weight_quantized" not in inits
    assert any(n.op_type == "MatMulInteger" for n in m.graph.node)  # encoder matmuls are int8
    # embeddings stay fp32 with the default op-type restriction: no quantized Gather data
    assert not any(i.endswith("embeddings.word_embeddings.weight_quantized") for i in inits)

    # control: without exclusion the head is quantized
    m2, head_int2, inits2 = quantized(
        EX.quantize_int8(tiny_export["onnx"], tiny_export["root"] / "nohead_q.onnx", exclude_head=False))
    assert head_int2 and inits2["head.weight_quantized"] == TensorProto.INT8


def test_quantize_preprocess_flag(tiny_export, tiny_base):
    from reflex_sentry.models import export_onnx as EX

    path = EX.quantize_int8(tiny_export["onnx"], tiny_export["root"] / "pre_q.onnx", preprocess=True)
    logits = _run_onnx(path, tiny_base)
    assert logits.shape == (4, 3) and np.isfinite(logits).all()


def test_sweep_end_to_end_and_choose(tiny_export):
    import shutil
    from reflex_sentry.models import export_onnx as EX

    # work on a copy of the model dir so metadata.json edits do not leak into other tests
    model_dir = tiny_export["root"] / "model_sweep"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(tiny_export["model_dir"], model_dir)
    onnx_path = model_dir / "model.onnx"
    train_parquet = tiny_export["data_dir"] / "train.parquet"
    targets_parquet = tiny_export["data_dir"] / "soft_targets.parquet"

    # lenient rule so the random tiny model deterministically passes and --choose has something to pick
    EX.main(["sweep", "--model", str(model_dir), "--onnx", str(onnx_path),
             "--train-parquet", str(train_parquet), "--targets-parquet", str(targets_parquet),
             "--target-recall", "0.6", "--latency-sample", "6", "--choose",
             "--min-matched-agreement", "0.0", "--max-ap-drop", "1.0"])

    for name in EX.SWEEP_CONFIGS:
        assert (model_dir / "sweep" / f"{name}.onnx").exists()
    assert (model_dir / "sweep.md").exists()
    sw = json.loads((model_dir / "sweep.json").read_text())
    # selection ran on the dev fold, not val
    assert "val" not in sw and sw["train_parquet"] == str(train_parquet)
    assert [r["name"] for r in sw["results"]] == EX.SWEEP_CONFIGS
    for r in sw["results"]:
        assert r["latency"]["p50_ms"] > 0 and r["latency"]["p95_ms"] >= r["latency"]["p50_ms"]
        assert 0.0 <= r["metrics"]["decision_agreement"] <= 1.0
        assert r["metrics"]["ok"]
    assert sw["fp32_onnx_latency"]["p50_ms"] > 0
    assert sw["chosen"] in EX.SWEEP_CONFIGS

    # chosen = highest agreement, ties by lower p50
    best = min(sw["results"], key=lambda r: (-r["metrics"]["decision_agreement"], r["latency"]["p50_ms"]))
    assert sw["chosen"] == best["name"]

    chosen_file = model_dir / "model_int8.onnx"
    assert chosen_file.exists()
    assert chosen_file.read_bytes() == (model_dir / "sweep" / f"{sw['chosen']}.onnx").read_bytes()
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["int8_quantization"]["config_name"] == sw["chosen"]
    assert meta["int8_quantization"]["selected_on"] == "dev_fold"
    assert "config" in meta["int8_quantization"] and "latency" in meta["int8_quantization"]
    assert "base" in meta  # original metadata preserved


def test_sweep_choose_refuses_when_nothing_passes(tiny_export):
    import shutil
    from reflex_sentry.models import export_onnx as EX

    model_dir = tiny_export["root"] / "model_sweep_fail"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(tiny_export["model_dir"], model_dir)
    train_parquet = tiny_export["data_dir"] / "train.parquet"
    targets_parquet = tiny_export["data_dir"] / "soft_targets.parquet"
    result = EX.sweep(model_dir, model_dir / "model.onnx", train_parquet, targets_parquet, configs=["legacy"],
                      choose=True, target_recall=0.6, latency_sample=3, min_agreement=1.01)
    assert result["chosen"] is None
    assert not (model_dir / "model_int8.onnx").exists()
    with pytest.raises(SystemExit):
        EX.main(["sweep", "--model", str(model_dir), "--onnx", str(model_dir / "model.onnx"),
                 "--train-parquet", str(train_parquet), "--targets-parquet", str(targets_parquet),
                 "--configs", "legacy", "--choose", "--latency-sample", "3",
                 "--target-recall", "0.6", "--min-matched-agreement", "1.01"])


def test_sweep_dev_fold_never_reads_val(tiny_export, monkeypatch):
    """The int8 sweep selects on the training dev fold (README 4.1: val is
    reserved for temperature scaling and threshold selection). Monkeypatch
    pd.read_parquet to record every path it is asked to read and assert none
    of them is val.parquet."""
    import shutil
    from reflex_sentry.models import export_onnx as EX

    model_dir = tiny_export["root"] / "model_sweep_no_val"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(tiny_export["model_dir"], model_dir)
    train_parquet = tiny_export["data_dir"] / "train.parquet"
    targets_parquet = tiny_export["data_dir"] / "soft_targets.parquet"

    seen: list[str] = []
    real_read_parquet = pd.read_parquet

    def spy(path, *a, **kw):
        seen.append(str(path))
        return real_read_parquet(path, *a, **kw)

    monkeypatch.setattr(pd, "read_parquet", spy)

    EX.sweep(model_dir, model_dir / "model.onnx", train_parquet, targets_parquet, configs=["legacy"],
             choose=False, target_recall=0.6, latency_sample=3)

    assert seen, "sweep should have read at least the train/targets parquet files"
    assert not any(Path(p).name == "val.parquet" for p in seen)
    assert any(Path(p).name == "train.parquet" for p in seen)
    assert any(Path(p).name == "soft_targets.parquet" for p in seen)

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
    assert result["torch_vs_onnx_fp32"]["max_abs_diff"] < 1e-2
    assert 0.0 <= result["torch_vs_onnx_int8"]["argmax_agreement"] <= 1.0
    for key in ("decision_agreement", "fp32", "int8", "diff", "warnings", "ok", "threshold"):
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


def test_parity_metrics_identical_models():
    from reflex_sentry.models import export_onnx as EX

    rng = np.random.default_rng(0)
    gold = np.array(["dangerous"] * 20 + ["benign"] * 30 + ["ambiguous"] * 10)
    logits = rng.normal(size=(60, 3))
    logits[:20, 1] += 2.0
    logits[20:50, 0] += 2.0
    m = EX.parity_metrics(gold, logits, logits, logits, t=0.3)
    assert m["decision_agreement"]["overall"] == 1.0
    assert m["decision_agreement"]["dangerous"] == 1.0
    assert m["decision_agreement"]["n_disagree"] == 0
    assert m["diff"]["recall_items"] == 0 and m["diff"]["benign_escalation_items"] == 0
    assert m["diff"]["average_precision"] == 0.0 and m["diff"]["roc_auc"] == 0.0
    assert m["torch_vs_onnx_int8"]["max_abs_diff"] == 0.0
    assert m["torch_vs_onnx_int8"]["argmax_agreement"] == 1.0
    assert m["ok"] and m["warnings"] == []
    # counts are consistent with an independent computation
    p_safe = SB._softmax(logits)[:, 0]
    assert m["fp32"]["recall"] == {"k": int((p_safe[:20] < 0.3).sum()), "n": 20,
                                   "rate": (p_safe[:20] < 0.3).mean()}
    assert m["fp32"]["benign_escalation"]["n"] == 30
    assert m["n_ambiguous"] == 10


def test_parity_warning_rule():
    from reflex_sentry.models import export_onnx as EX

    gold = np.array(["dangerous"] * 48 + ["benign"] * 100)
    base = np.zeros((148, 3))
    base[:48, 1] = 3.0   # p_safe small -> escalated
    base[48:, 0] = 3.0   # p_safe large -> passed
    # flip exactly one dangerous row to "pass" in int8: 1 item on 48 is within tolerance
    one = base.copy()
    one[0] = [3.0, 0.0, 0.0]
    m1 = EX.parity_metrics(gold, base, base, one, t=0.5)
    assert m1["diff"]["recall_items"] == -1
    assert not any("recall" in w for w in m1["warnings"])
    assert m1["decision_agreement"]["overall"] == pytest.approx(147 / 148)
    # two items exceed max(1 item, 2 points)
    two = one.copy()
    two[1] = [3.0, 0.0, 0.0]
    m2 = EX.parity_metrics(gold, base, base, two, t=0.5)
    assert any("recall" in w for w in m2["warnings"]) and not m2["ok"]
    # heavy disagreement trips the agreement rule
    many = base.copy()
    many[48:60] = [0.0, 3.0, 0.0]
    m3 = EX.parity_metrics(gold, base, base, many, t=0.5)
    assert any("decision agreement" in w for w in m3["warnings"])


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

    # lenient rule so the random tiny model deterministically passes and --choose has something to pick
    EX.main(["sweep", "--model", str(model_dir), "--onnx", str(onnx_path), "--val", str(tiny_export["val"]),
             "--target-recall", "0.6", "--latency-sample", "6", "--choose",
             "--min-decision-agreement", "0.0", "--max-recall-diff", "1.0", "--max-ap-drop", "1.0"])

    for name in EX.SWEEP_CONFIGS:
        assert (model_dir / "sweep" / f"{name}.onnx").exists()
    assert (model_dir / "sweep.md").exists()
    sw = json.loads((model_dir / "sweep.json").read_text())
    assert [r["name"] for r in sw["results"]] == EX.SWEEP_CONFIGS
    for r in sw["results"]:
        assert r["latency"]["p50_ms"] > 0 and r["latency"]["p95_ms"] >= r["latency"]["p50_ms"]
        assert 0.0 <= r["parity"]["decision_agreement"]["overall"] <= 1.0
        assert r["parity"]["ok"]
    assert sw["fp32_onnx_latency"]["p50_ms"] > 0
    assert sw["chosen"] in EX.SWEEP_CONFIGS

    # chosen = highest agreement, ties by lower p50
    best = min(sw["results"], key=lambda r: (-r["parity"]["decision_agreement"]["overall"],
                                             r["latency"]["p50_ms"]))
    assert sw["chosen"] == best["name"]

    chosen_file = model_dir / "model_int8.onnx"
    assert chosen_file.exists()
    assert chosen_file.read_bytes() == (model_dir / "sweep" / f"{sw['chosen']}.onnx").read_bytes()
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["int8_quantization"]["config_name"] == sw["chosen"]
    assert "config" in meta["int8_quantization"] and "latency" in meta["int8_quantization"]
    assert "base" in meta  # original metadata preserved


def test_sweep_choose_refuses_when_nothing_passes(tiny_export):
    import shutil
    from reflex_sentry.models import export_onnx as EX

    model_dir = tiny_export["root"] / "model_sweep_fail"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(tiny_export["model_dir"], model_dir)
    result = EX.sweep(model_dir, model_dir / "model.onnx", tiny_export["val"], configs=["legacy"],
                      choose=True, target_recall=0.6, latency_sample=3, min_decision_agreement=1.01)
    assert result["chosen"] is None
    assert not (model_dir / "model_int8.onnx").exists()
    with pytest.raises(SystemExit):
        EX.main(["sweep", "--model", str(model_dir), "--onnx", str(model_dir / "model.onnx"),
                 "--val", str(tiny_export["val"]), "--configs", "legacy", "--choose", "--latency-sample", "3",
                 "--target-recall", "0.6", "--min-decision-agreement", "1.01"])

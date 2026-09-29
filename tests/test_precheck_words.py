"""Local word-list generation from a small synthetic Hunspell dictionary."""

import hashlib
import json

from reflex_sentry.models import build_precheck_words as builder


def test_build_is_deterministic_and_filters_dictionary(tmp_path):
    dictionary = tmp_path / "sample.dic"
    dictionary.write_text("7\nalpha/AB\nBeta/M\nAAA/M\nbeta\na\nfoo-bar\nphishing\n", encoding="utf-8")
    output = tmp_path / "words.txt"

    first = builder.build(dictionary, output)
    content = output.read_bytes()
    words = content.decode("utf-8").splitlines()
    assert words == sorted(set(words))
    assert content.endswith(b"\n")
    assert {"alpha", "beta", "phishing", "authentication", "the"} <= set(words)
    assert "foo-bar" not in words
    assert "Beta" not in words
    assert "AAA" not in words
    assert first["dictionary_words"] == 3
    assert first["output_sha256"] == hashlib.sha256(content).hexdigest()
    assert first["dictionary_sha256"] == hashlib.sha256(dictionary.read_bytes()).hexdigest()

    assert builder.build(dictionary, output) == first
    assert output.read_bytes() == content
    assert json.loads((tmp_path / "words.meta.json").read_text()) == first


def test_main_uses_environment_output(monkeypatch, tmp_path):
    dictionary = tmp_path / "sample.dic"
    dictionary.write_text("1\nhello\n", encoding="utf-8")
    output = tmp_path / "custom" / "common_words.txt"
    monkeypatch.setenv(builder.OUTPUT_ENV, str(output))
    assert builder.main(["--dictionary", str(dictionary)]) == 0
    assert output.exists()
    assert output.with_name("common_words.meta.json").exists()

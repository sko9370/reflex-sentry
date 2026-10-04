"""Teacher labeling for reflex-sentry (docs/WORKFLOWS.md).

`score.py` runs a guard-LLM preset over prompts and records per-prompt
verdict probabilities plus a predicted category. `as_predictor.py` turns
one or two teacher score files into harness prediction CSVs so a teacher
can be scored as the quality-reference row (docs/STATUS.md).
"""

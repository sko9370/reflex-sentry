"""Teacher labeling for reflex-sentry (README 4.2, milestone 3).

`score.py` runs a guard-LLM preset over prompts and records per-prompt
verdict probabilities plus a predicted category. `as_predictor.py` turns
one or two teacher score files into harness prediction CSVs so a teacher
can be scored as the upper-reference row (README 5.4).
"""

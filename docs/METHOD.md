# Method and evaluation

This page describes the implemented model and evaluation contracts. File
schemas and split ownership are in [DATA_CONTRACT.md](DATA_CONTRACT.md);
the label taxonomy and judgment rules are in the
[labeling guide](../configs/labeling_guide.md).

## Targets and students

The teacher score is the probability of its unsafe first token. The two
configured teacher presets are Llama Guard 3 8B and Qwen3Guard-Gen-8B.
`reflex_sentry/targets.py` combines teacher evidence and any binary source
label into `[safe, dangerous, unsure]` training targets. In the original
single-teacher formulation, with teacher unsafe probability `p` and source
label `y`, disagreement is `d = |p-y|` (zero when `y` is absent), and

```text
u = 0.50 * clip(2 * (d - 0.25), 0, 1) + 0.25 * (1 - |2p - 1|)
target = [(1-p) * (1-u), p * (1-u), u]
```

With two teachers, `p` is their mean and `u` also receives a disagreement
term (`0.25 * clip(2 * |p1-p2|, 0, 1)`) and a controversial-score term
(`0.25 * nanmean(p_controversial)`, zero when absent). It is capped at 0.9
and targets are renormalized. Two-teacher targets use the intersection of
teacher IDs and the train pool. `--source-only` instead smooths known source
labels by 0.05 and assigns unknown labels `p=0.5, u=0.5`.

Stage A trains multinomial logistic regression on frozen sentence
embeddings. Stage B fine-tunes a ModernBERT encoder with a three-way head and
KL loss against soft targets; its default maximum length is 256 tokens.
The Stage B training dev fold selects checkpoints, without gold validation
labels.

## Calibration and decisions

Each student variant with logits fits temperature on validation logits.
The runner selects the smallest validation threshold `t` meeting target
dangerous recall (default 0.95) from its prediction scores; keyword and
teacher probability rows are not temperature-fitted there. The decision is
`force_escalate OR p_safe < t`; absent
`force_escalate` means false. The optional deterministic pre-check flags
encoded blobs, character substitution, low English-word coverage, spacing,
and mixed-script tokens. It changes routing only. Base and pre-check variants
share the same probabilities, temperature, and model threshold. The override
must still work at `t = 0`; legacy pre-check files that overwrote probabilities
must be regenerated.

Dynamic int8 presets are compared on Stage B's **training dev fold**, not the
gold validation set. The sweep picks a fp32 dev threshold for 0.95 recall,
then chooses an int8 threshold that matches fp32's escalation count. It
compares decision agreement at that matched rate, AP/ROC-AUC deltas, maximum
logit difference, CPU latency, and size. The passing rule is agreement at
least 0.97 and AP drop at most 0.01; `--choose` favors agreement, then lower
p50 latency. The validation parity command is informational. On the gold
validation set, fp32 and int8 are calibrated and thresholded separately.

## Metrics and protocol

The runner requires exact prediction IDs for each split, with no missing,
extra, duplicate, or null IDs. A
prediction row has `id`, `gold`, `p_safe`, `p_dangerous`, `p_unsure`; optional
fields include `source`, `tags`, `latency_ms`, `force_escalate`,
`precheck_flag`, and `precheck_reasons`. Probabilities sum to one within
1e-3. Wrapped evasion IDs append the wrapper name to the base ID.

Average precision, ROC-AUC, ECE, Brier score, and reliability plots describe
the unchanged model probabilities. Dangerous recall, benign and hard-negative
escalation, ambiguous escalation, source/category/evasion slices, error lists,
tradeoff curves, and cascade economics use final routed decisions. Rates
include Wilson intervals. AP and ROC-AUC exclude ambiguous rows and score
`1 - p_safe`; calibration uses `p_dangerous` on dangerous vs benign rows.
The report also shows model-only and policy-added escalations.

Cascade calculations reweight benign traffic using the configured
hard-negative share, because the evaluation set deliberately over-samples
legitimate security prompts. They combine assumed dangerous/ambiguous
prevalence, escalation rates, and tier-two price to estimate calls, cost,
queue precision, and missed dangerous prompts per million. These are scenario
estimates, not measured production costs.

Latency percentiles come from sampled, warmed-up, batch-one CPU measurements;
ONNX timing includes tokenization, inference, and softmax. Pre-check reports
add separately measured detector time to cached base timing samples and
preserve missing samples. See [STATUS.md](STATUS.md) for the limits of the
saved results.

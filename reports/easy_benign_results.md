# Expanded test results: easy-benign slice

## 2026-10-01 measurement correction

Pre-check routing now preserves model probabilities and uses a separate
`force_escalate` flag. Refreshed fp32 pre-check AP is **0.898** and int8
pre-check AP is **0.848**, matching their base models; calibration metrics
also match. These replace the policy-distorted AP values recorded in the
superseded measurement note below. This is a measurement correction, not a
change in model quality.
All 12 Stage B pre-check reports retain their previous thresholds and
operating counts. FP32 pre-check evasion recall remains 96.81%, and test
benign escalation remains 10.81%. Reports now distinguish model decisions
from additional policy escalations. Validation: 318 passed, 3 skipped.

## Evaluation population and current interpretation

Evaluated 2026-09-29. All model rows cover the same 586 test IDs: 509 benign, 47 dangerous, and 30 ambiguous. The 300 added benign prompts are held-out ToxicChat rows selected by the existing scope filter and seed 7. Their labels come from source annotations and filtering, not a new human review.

Both returned teacher files contain exactly the expected 300 unique IDs. Each merged teacher artifact now contains 6,497 unique IDs. Original score files and the original test file have local backups. No training, calibration-set change, or detector tuning was performed.

| Model | Dangerous recall | New benign escalated (of 300) | All benign escalated | Hard-negative escalated | AP | Test CPU p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| stage_a | 100.00% | 162/300 (54.00%) | 42.44% | 25.96% | 0.577 | 30.3 |
| stage_b | 97.87% | 39/300 (13.00%) | 10.81% | 7.69% | 0.898 | 92.6 |
| stage_b_onnx | 97.87% | 39/300 (13.00%) | 10.81% | 7.69% | 0.898 | 49.5 |
| stage_b_onnx_pc | 97.87% | 39/300 (13.00%) | 10.81% | 7.69% | 0.898 | 49.5 |
| stage_b_int8_pc | 93.62% | 62/300 (20.67%) | 15.32% | 7.69% | 0.848 | 19.6 |
| teacher_both | 93.62% | 7/300 (2.33%) | 2.16% | 1.92% | 0.688 | n/a |

## Interpretation

- The new prompts are not easy for the students: fp32 Stage B escalates 39/300 (13%), compared with 7/300 (2.33%) for the combined teachers and 162/300 (54%) for Stage A. The earlier expectation that adding these negatives would improve AP was wrong; ranking quality falls on this slice.
- Thresholds selected on the unchanged validation set are identical to the prior run. Dangerous recall is unchanged. OOD and evasion sets remain unchanged: fp32 Stage B has 85.12% OOD recall and 96.81% evasion recall with the pre-check (71.81% without).
- The pre-check flags one new benign row, already escalated by the base models, so it adds no test escalations at the selected threshold. The `_pc` predictions preserve model probabilities and carry a separate `force_escalate` routing flag. Their AP and calibration therefore match the respective base models; operating counts use the routed decisions.
- Source labels warrant care: Llama assigns unsafe >=0.5 to 17 new rows, Qwen to 3, either to 19, and both to 1. Qwen assigns controversial >=0.5 to 25. These are disagreements, not adjudicated label errors, and labels were not changed.
- The harness easy-benign rate uses 301 rows because the original test already contained one benign row without a hard-negative tag. The new-slice column above isolates exactly the 300 added rows.
- Fresh CPU timings use 200 sampled prompts from the expanded test. Changes from earlier timings reflect the changed prompt mix and a separate measurement run, not a model speedup. Pre-check timings add measured overhead to base timings rather than timing a deployed short-circuit cascade.

## Illustrative cascade load

Using `configs/eval.yaml` assumptions: 1,000,000 prompts, 0.5% dangerous, 2% ambiguous, 5% hard negatives among benign traffic, and $0.002 per tier-two call. Costs below include tier-two calls only, excluding gate inference and infrastructure. Prior Stage A and teacher reports used the evaluation set's benign mix instead; prior Stage B economics had just one easy-benign example. Old and new cost estimates should not be treated as a controlled cost comparison.

| Model | Estimated escalations per million | Estimated tier-two cost |
|---|---:|---:|
| stage_a | 528,170 | $1,056.34 |
| stage_b_onnx_pc | 134,656 | $269.31 |
| stage_b_int8_pc | 205,887 | $411.77 |
| teacher_both | 31,159 | $62.32 |

These are scenario estimates from small evaluation slices, not measured production cost or guarantees. The fp32 easy-benign escalation Wilson interval is about 9.6%–17.2% on the 301-row harness slice. The combined-teacher row represents quality reference predictions; its own GPU inference cost is not included.

## Validation

All nine reported model variants have exactly the same 586 test IDs and matching gold labels. Validation thresholds and dangerous recall match the previous run. The evaluation runner now rejects missing, extra, duplicate, or null prediction IDs before writing reports, including validation inputs used only for threshold selection. Full suite: 296 passed, 3 skipped.

The separate 2026-10-01 routing correction passed 318 tests with 3 skipped;
the 2026-10-04 repository baseline repeated that count (Playwright/Chromium
unavailable). These suite runs do not constitute a new model evaluation.

## Superseded measurement (2026-09-29)

The original report overwrote flagged model probabilities with a routing
score. It consequently showed fp32 pre-check AP 0.829 and int8 pre-check AP
0.781. Those values ranked altered policy scores and are **not** current
model AP. The corrected AP values above are 0.898 and 0.848, while routed
recall/escalation counts remain the saved operating results.

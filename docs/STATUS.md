# Project status — 2026-10-04

The data pipeline, gold-label workflow, keyword baseline, two teacher scorers,
Stage A embedding classifier, Stage B ModernBERT student, fp32 ONNX and dynamic
int8 export, calibration, pre-check, and evaluation harness are implemented.
This is a research proof of concept for single-turn English cyber prompts, not
a deployed guard. Generated models, predictions, and most reports are local,
ignored artifacts; a fresh clone does not contain them.

## Saved evaluation

The expanded test has 586 rows: 509 benign, 47 dangerous, and 30 ambiguous.
Three hundred benign rows were added from held-out ToxicChat ordinary prompts.
Their labels are source-derived and were not newly hand-reviewed. The other
gold sets were largely drafted by a model in two passes and reviewed by the
owner; review of a visible draft is not an independent blind annotation.
The owner reviewed the 800 sampled gold rows, which does not establish that
the entire AI-assisted seed corpus was reviewed.

The saved fp32 ONNX plus pre-check result has 97.87% dangerous recall and
10.81% benign escalation on the expanded test, 85.12% OOD dangerous recall,
96.81% evasion recall, and test average precision 0.898. Its test AP equals
the base fp32 model's AP because the pre-check changes routing, not model
probabilities. The evasion improvement followed inspection of base-model
failures, so it is post-hoc evidence. See the dated
[expanded-test analysis](../reports/easy_benign_results.md) and the
[pre-check audit](../reports/precheck_audit.md) for populations and caveats.

During the earlier cleanup pass on 2026-10-04, the local suite passed **318
tests, with 3 skipped** because Playwright/Chromium was unavailable
(5 existing export warnings). The smoke
harness and offline editable installation passed. The CI Stage B/ONNX step
passed 29 tests with no skips. In an integration comparison against commit
`9d5305d`, all 35 regenerated report metrics JSON/CSV files and the nine-model
comparison were exactly equal; original prediction files were untouched.
This checks cleanup parity, not a new model evaluation.

## Artifact integrity checks — 2026-10-04

The shared evaluation runner now compares IDs and gold labels with processed
splits before calibration or reporting, including validation-only references.
Teacher scoring validates input overlap, saved model identity and probabilities,
and complete batch output. Score merging rejects conflicting overlaps while
allowing identical rows and enrichment of missing optional fields. Earlier valid
checkpoints survive a later invalid batch. The final consistency pass removed
the public scorer's validation-bypass parameter and verified that no-work CLI
runs neither load a model nor claim to have written an output file.

The final artifact-integrity and consistency pass superseded that test count: **359 tests
passed, 3 browser tests skipped**, with the same 5 existing export warnings;
the smoke harness passed. All 51 existing
prediction/logit CSVs match their processed gold mappings, and all four saved
teacher score files pass the new schema checks. Four model recalibrations and
35 regenerated reports retain the existing results. Merging the already-scored
300-row slice into each teacher and resuming on already-scored test inputs both
preserve all 6,497 rows. Original prediction and teacher files remain unchanged.
These checks did not retrain models or revise labels. The file-only report and
calibration commands still have no processed reference for cross-file label checks.

## Measurement and limits

The saved CPU latency figures use sampled, warmed-up, batch-one inference;
ONNX timing includes tokenization and softmax. Pre-check timing adds measured
detector overhead to cached base samples, so it is an estimate of combined
cost, not a fresh deployed-cascade benchmark. Scenario cost uses assumed
traffic prevalence and tier-two price from `configs/eval.yaml`.

The model sees one prompt without conversation history, covers one harm
domain, and was developed for English. Heuristic language buckets in the
pre-check audit are not validated language labels. OOD recall, ambiguous
routing, false positives on legitimate security work, source-label noise,
model-draft anchoring, and adaptive evasion need independent study.

## Remaining work

The completed maintenance pass is recorded in [CLEANUP_PLAN.md](CLEANUP_PLAN.md).
The forward research agenda is in [PLAN.md](PLAN.md). Verify dataset licenses
against current cards before redistribution; [SOURCES.md](../data/SOURCES.md)
marks unverified license notes explicitly.

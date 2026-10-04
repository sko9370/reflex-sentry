# Repository consolidation plan

Prepared 2026-10-04 against commit `9d5305d`. Status: completed 2026-10-04.
The stages below preserve the agreed plan; the execution record records the
implemented scope and decisions on conditional candidates.

## Execution record

- **Documentation:** shortened README, separated STATUS/WORKFLOWS/METHOD,
  reconciled DATA_CONTRACT and provenance notes, replaced the stale roadmap,
  archived dated decisions and the completed Kaggle procedure, and corrected
  report snapshot interpretation. METHOD preserves the algorithm explanations
  previously embedded in README.
- **Tooling:** pyproject is the dependency source, pytest lives in the test extra,
  requirements is a thin package-install entry point, notebooks use package
  extras, and CI has an explicit Stage B/ONNX job that rejects skips.
- **Removed:** unused comparison arguments; the uncalled internal
  `predict_int8_split` wrapper (the prediction CLI remains); historical labeling
  input defaults; redundant `notebooks/.gitkeep`; and the checked-in generated
  Label Studio XML. The XML generator remains canonical, exports to the same
  default path, and produces byte-identical content. That output path is ignored.
- **Retained deliberately:** the draft merger, recommendation helper, and review
  UI support future labeling batches; their retirement would remove useful
  capabilities. Explicit input paths replace batch-specific defaults. Teacher
  score merging remains a small separate tool. Quantization presets, baseline
  models, pre-prediction evasion preparation, and evaluation checks remain.
  Pre-check runtime/audit code and different JSON serializers were not split
  merely to rearrange code.
- **Validation:** before and after cleanup, 318 tests passed and 3 browser tests
  skipped because Playwright/Chromium was unavailable; 5 existing ONNX/Torch
  warnings remain. The end-to-end smoke script passed. Offline editable package
  installation succeeded; the CI Stage B/ONNX check passed 29 tests with no skips,
  and its missing-dependency failure path was checked separately.
  All relative Markdown links resolve, and 19 documented CLI invocations were
  checked against their actual argument parsers without executing model work.
  Notebook setup paths were reviewed and notebook JSON validated; remote Kaggle
  GPU jobs were not rerun. The teacher notebook now installs a writable copy when
  its source is on a read-only Kaggle input mount.
- **Output parity:** regenerated 35 reports from identical cached predictions
  with original `9d5305d` code and cleaned code in scratch directories. Every
  metrics JSON and report CSV matched exactly, as did the nine-model comparison.
  Original prediction files were verified unchanged. The labeling XML also
  matches the original byte for byte. No training or detector tuning occurred.
- **Review:** subagents owned documentation, dependency/CI, and bounded model/eval
  cleanup; the primary agent integrated and verified. An additional read-only
  integration review found no code/tooling regression. User data, models, and
  the unexplained untracked `:memory:.ses` were left alone.

Follow-up behavioral changes remain in the forward roadmap. Cleanup is complete;
optional removals were assessed rather than treated as automatic deletions.

## Goal and boundaries

Make the repository easier to understand, reproduce, and extend by reconciling
documentation and removing demonstrated redundancy. Prefer deletion and focused
simplification over new abstractions. This is a maintenance pass: model training,
labels, split membership, thresholds, detector rules, and metric definitions stay
unchanged. Research improvements belong in the subsequent roadmap.

Three read-only subagent audits covered documentation, model/evaluation code, and
data/labeling/dependencies. No large subsystem was proven obsolete. Line-count
reduction is useful only when it reduces maintenance without losing reproducibility.

## Baseline to preserve

- Stage A, Stage B, teachers, ONNX/int8 exports, calibration, and evaluation are
  implemented. The old M1–M6 checklist does not reflect their current state.
- Expanded test: 586 rows, comprising 509 benign, 47 dangerous, and 30 ambiguous.
- FP32 ONNX with pre-check: 97.87% dangerous test recall, 10.81% benign escalation,
  85.12% OOD recall, 96.81% evasion recall, and AP 0.898. These describe the saved
  evaluation, not a new validation run or deployment guarantee.
- Pre-check flags control routing without changing probabilities. Base and
  pre-check variants share model thresholds, ranking, and calibration metrics.
- The recorded 2026-10-01 suite result is 318 passed, 3 skipped. Establish a fresh
  baseline during implementation; do not present this historical count as rerun.
- Gold labels largely came from model drafts reviewed by the owner. The appended
  ordinary-benign labels are source-derived. Evasion improvements are post-hoc.
- `reports/comparison.*`, model artifacts, predictions, and most generated reports
  are ignored. A fresh clone cannot depend on those local files being present.

## Stage 1 — Capture the baseline and classify files

1. Record Git status and inventory tracked docs, source, scripts, notebooks,
   fixtures, and committed report snapshots. Leave the unexplained untracked
   `:memory:.ses` and all user data/model artifacts untouched.
2. Run the existing suite and smoke script, recording dependency-related skips.
   Save validation outputs in a separate scratch directory when possible.
3. Capture current prediction IDs, labels, probabilities, routing decisions,
   thresholds, and deterministic metrics for later parity checks. Use copies of
   cached logits/predictions; never overwrite the sole original artifacts.
4. Record each proposed removal with its callers, notebook/CLI/doc references,
   replacement (if any), and reason. Search dynamic entry points as well as imports.

Done when there is a reproducible baseline and every proposed deletion has evidence.

## Stage 2 — Give each kind of documentation one home

| Destination | Owns | Action |
|---|---|---|
| `README.md` | Project purpose, current scope, installation, minimal quickstart, navigation | Replace the proposal-style tutorial and empty results table with a concise entry point. |
| `docs/STATUS.md` (new) | Dated milestone status, compact verified results, limitations, remaining work | Use saved metrics as evidence; distinguish recorded test runs and timing protocols. |
| `docs/WORKFLOWS.md` (new) | Runnable local/Kaggle procedures and artifact flow | Consolidate commands from README, PLAN, and scripts; clearly separate rerunning reports from retraining. |
| `docs/DATA_CONTRACT.md` | Schemas, producers/consumers, split and provenance rules | Correct stale ownership statements and cover logits, probabilities, routing fields, and timing sidecars by reference where appropriate. |
| `docs/PLAN.md` | Short forward roadmap | Replace completed milestone scaffolding with links to STATUS and this maintenance plan; retain future research separately. |
| `docs/history/DECISIONS.md` (new) | Dated experiment decisions and superseded approaches | Move useful chronological entries from PLAN; mark obsolete behavior as historical. |
| `docs/history/KAGGLE_EASY_BENIGN.md` | Completed one-off procedure | Move the existing runbook, use historical tense, and fix all inbound links. |
| Existing provenance/reference docs | Labeling rules, dataset sources, seed provenance, vocabulary setup | Keep their current locations and reconcile facts; link instead of duplicating. |

Specific corrections:

- Replace the single-source OOD description with the four-source configuration.
  Explain the later ordinary-benign ToxicChat addition to test as a separate slice.
- Remove claims that model-assisted gold is an independent blind human annotation.
- Distinguish review of sampled seed rows from review of the entire seed corpus;
  do not infer corpus-wide review from the 800-item gold review.
- Replace incomplete `gold.sample/export/ingest/agreement` command examples with
  valid invocations using actual required arguments/subcommands.
- Update package layout, completed milestones, model names, and pre-check semantics.
- Correct `data/SOURCES.md` and config comments that call AI-assisted seeds
  hand-written or still describe the singular OOD default. Preserve license
  verification uncertainty; do not silently upgrade it to a verified claim.
- Reconcile `reports/easy_benign_results.md`: present corrected current AP and
  routing interpretation together; retain superseded values only in a clearly
  labeled historical section, not as competing current results.
- Label `reports/precheck_audit.md` with its actual evaluated population: its saved
  test counts predate the expanded test. Either retain it as a dated snapshot or
  regenerate it explicitly and record the new provenance.
- Keep `reports/example_synthetic/` clearly synthetic. Keep generated comparisons
  ignored and link to STATUS for readers without local artifacts.
- Review notebook markdown, module docstrings, shell comments, labeling guide,
  seed README, and vocabulary README alongside Markdown files. Replace fragile
  numbered README references after moving sections.

Done when current facts have one authoritative home, commands are complete, and
historical text cannot be mistaken for pending work or current model behavior.

## Stage 3 — Simplify dependencies and development setup

- Make `pyproject.toml` the dependency source of truth. Move pytest to a test extra;
  make `requirements.txt` a thin install entry point or remove it after migrating
  every documented/CI/notebook consumer. Remove stale commented dependency lists.
- Update CI to install the package itself, exercising packaging metadata. Keep a
  lightweight test job and add an explicit Stage B/ONNX job using existing offline
  tiny-model fixtures so important tests do not silently disappear behind skips.
- Document optional browser-review testing separately if the review UI is retained.
- Resolve `configs/label_studio.xml` versus the template generated by
  `gold/export.py`: choose one authoritative representation after inspecting the
  dynamic fields, then ensure export produces the same intended interface.
- Remove redundant tracked `notebooks/.gitkeep` if confirmed unnecessary. Do not
  purge ignored directories or broaden ignore rules to hide unexplained files.

Done when installation instructions and CI use the same dependency definitions,
and required test coverage is distinguishable from optional skips.

## Stage 4 — Remove proven redundancy, then assess larger candidates

### First pass: small, behavior-preserving changes

| Candidate | Evidence and decision | Validation |
|---|---|---|
| `eval/run_all.py::build_comparison` | `processed_dir` and `preds_dir` are unused; one repository caller. Remove the arguments and update the caller after rechecking references. | Comparison output parity and run-all tests. |
| `models/export_onnx.py::predict_int8_split` | Thin compatibility wrapper; no Python, shell, or notebook callers found in the audit. Remove only if repository API/documentation history establishes no supported compatibility obligation; otherwise retain the small wrapper. | ONNX/int8 prediction and CLI tests. |
| Obsolete comments/imports/locals | Remove only after call-site and behavior inspection; stale explanatory text is the clearest broad cleanup opportunity. | Import checks and relevant existing tests. |

### Second pass: bounded candidates, not an automatic deletion list

- `gold/draft_merge.py`, `gold/suggest.py`, and `tools/review.html` contain sizeable
  support for the completed model-assisted labeling workflow. Decide whether they
  remain a supported future labeling path. If retained, remove batch-specific
  defaults and document their role; if retired, preserve a Git revision and exact
  reproduction recipe, then remove implementation and exclusively associated tests
  together. Do not move dead code to an archive directory merely to retain it twice.
- Keep `teacher/merge_scores.py` unless another existing operation demonstrably
  preserves its merge behavior. Its small size does not justify coupling it to
  teacher inference. Keep core sampling/export/ingest/agreement and teacher tools.
- Consider moving the audit/encoding/report helpers out of `models/precheck.py`
  only if it materially clarifies the runtime boundary. Preserve the existing CLI;
  this is organization, not a claimed reduction in code.
- Avoid a generic utility framework for a few local read/serialization helpers.
  The JSON serializers in report and ONNX code differ semantically; sharing them
  without schema tests risks changing nonfinite values and report structures.

Explicitly retain the documented quantization presets, necessary pre-prediction
evasion preparation in scripts, baseline models, confidence intervals, split-ID
checks, strict routing-flag validation, and dictionary provenance checks.

Done when each removal has an explained benefit and preserved supported behavior.
Do not set an arbitrary percentage or line-count deletion target.

## Stage 5 — Integrate and verify

1. Run targeted tests for each changed subsystem, then the full required suite and
   end-to-end smoke test once integration is complete. Record actual skips/reasons.
2. Validate documented CLI commands with `--help`/argument inspection and exercise
   representative workflows on fixtures. Check relative links, moved anchors, and
   notebook command references. No full Kaggle rerun is needed for this cleanup.
3. Re-evaluate copied cached predictions into scratch reports. Require unchanged
   IDs, gold labels, probabilities, thresholds, routed counts, ranking/calibration,
   and comparison values. Ignore only expected paths, timestamps, and freshly
   measured timing variation. Do not use image-byte equality for plot correctness.
4. For any inference refactor, compare tiny-fixture logits within the existing
   numerical tolerance. Do not retrain or regenerate labels to validate cleanup.
5. Preserve these invariants explicitly: validation selects temperature/threshold;
   the training dev fold selects quantization; routing never overwrites model
   probabilities; `force_escalate` works at threshold zero; missing timings remain
   missing; base/pre-check pairs use the same model threshold.
6. Review the final diff for unintended data/report churn. Update STATUS with the
   actual verification outcome and summarize files removed, consolidated, retained,
   and deferred. Keep a short maintenance record for future contributors.

## Execution and delegation

Use small reviewable change groups in this order: baseline, documentation,
dependency/CI cleanup, proven code cleanup, optional historical-tool retirement,
final integration. Avoid mixing behavioral fixes into structural changes.

- Documentation subagent: README, docs, provenance prose, report snapshot wording.
- Tooling subagent: pyproject/requirements/CI and install references, after the
  documentation outline and command conventions are agreed within the team.
- Code subagent: bounded source/test changes, one subsystem at a time; coordinate
  shared files explicitly. Separate history-tool retirement from core eval work.
- Primary agent: artifact baseline, cross-file contract review, integration,
  final tests, and current-status accuracy. Use GPT-6 Sol for routine bounded work.

## Follow-ups deliberately outside the cleanup

Track discovered behavior issues separately: teacher resume/merge validation of
model identity, conflicting IDs, score ranges, and complete batch output; curve
computation optimization; new inference packaging; and research work on OOD recall,
ambiguity, false positives, and independent holdouts. They warrant their own tests
and decisions, and must not be smuggled into a behavior-preserving refactor.

The teacher resume/merge validation and evaluation gold-consistency follow-ups
were subsequently implemented as a separate change; see [STATUS.md](STATUS.md)
for validation results and [DATA_CONTRACT.md](DATA_CONTRACT.md) for their scope.

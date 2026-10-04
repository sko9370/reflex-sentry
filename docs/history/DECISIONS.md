# Dated development decisions

This is a compact record of decisions and measurements, not an active
checklist. Current procedures are in [WORKFLOWS.md](../WORKFLOWS.md), current
results in [STATUS.md](../STATUS.md), and the original detailed development
plan remains available at pre-cleanup commit `9d5305d`.

## 2026-09-27: identity, sources, and labeling

The project was renamed from `s1gate` to `reflex-sentry`, including the Python
package. `data/` stays ignored except `data/SOURCES.md`; raw inputs, gold,
teacher scores, and model artifacts are local/private.

The seed hard negatives were drafted with AI assistance. This may differ
from actual defender phrasing, so the seed corpus needs separate realism
review. The owner later reviewed the sampled gold rows, including sampled
seeds, but that does not establish full seed-corpus review.

ToxicChat alone yielded only 78 in-scope OOD candidates (about 16 unsafe).
The OOD holdout therefore became four sources: ToxicChat, Aegis 2, HH
red-team attempts, and BeaverTails. Together they were under 4% of train
rows in the first build. The keyword scope filter was changed to require
one strong hit or two distinct weak hits after ordinary words admitted
non-cyber OR-Bench prompts and casual account-takeover phrasing was missed.

The gold sample sizes became 300 validation, 300 test, and 200 OOD, drawn
from candidate pools. Two Opus passes drafted labels in different orders;
all 30 pass disagreements were adjudicated. Five of 16 batches stopped
early due to a safety classifier, without retry. Of 800 sampled rows, 717
had at least one model label. Pass A/B agreement on doubly labeled rows
was 0.93 (kappa 0.82) on val, 0.94 (kappa 0.81, n=55) on test, and 0.91
(kappa 0.73) on OOD. These are agreement between model drafts, not between
independent human annotators.

## 2026-09-28: owner review and model selection

The owner reviewed all 800 sampled gold rows. Opus-drafted labels were
accepted without change for 300 val, 278 test, and 200 OOD rows. Sixteen
test rows were labeled directly by the owner; 40 more with non-model
suggestions were reviewed, with 34 accepted and 6 changed. Visible draft
labels can anchor review, so this is not a blind human holdout.

Stage B checkpoint selection uses a dev fold from train. Int8 configuration
selection moved from gold val to that same fold after initial int8 variants
shifted logits: AP deltas were within 0.01, but agreement under a shared
fp32 threshold was only 0.925–0.940. Current comparison matches fp32 and
int8 escalation counts using separate thresholds; it requires agreement
at least 0.97 and AP drop at most 0.01. Gold val remains for temperature
and operational threshold selection; parity there is informational.

The deterministic pre-check was integrated after base-model evasion failures
were inspected. It raised saved fp32 evasion recall from 71.81% to 96.81%,
with 97.87% test dangerous recall and 85.12% OOD recall. This is a post-hoc
refinement, not independent confirmation. An ordinary-benign audit flagged
16/2000 rows; a heuristic Spanish/Portuguese bucket flagged 6/33. The
pre-check does not resolve OOD shift. The historical implementation briefly
overwrote flagged probabilities; the 2026-10-01 decision below supersedes
that behavior.

Fp32 ONNX direct inference was added beside PyTorch and int8. Saved ONNX
logits differed from PyTorch by at most 0.000029 over four splits. The then
measured test median was 62.1 ms for ONNX versus 92.9 ms for PyTorch in
separate local runs, and 22.9 ms for int8 under the corrected end-to-end
timing protocol. These are historical timings on the smaller test population,
not a controlled speedup claim.

## 2026-09-29: ordinary-benign expansion

The original in-scope benign gold rows all had hard-negative tags, leaving
no ordinary-benign slice for cascade reweighting. Three hundred held-out,
out-of-scope, source-labeled ToxicChat safe prompts were appended to test.
They were English-looking, 20–1500 characters, and deduplicated; they were
not newly hand-reviewed. Both teacher files were scored on the new IDs and
merged, yielding 6,497 unique IDs each. The new test has 586 rows: 509
benign, 47 dangerous, and 30 ambiguous. Validation, OOD, evasion, model
weights, and detector thresholds were unchanged.

The new slice escalated 39/300 for fp32 Stage B, 62/300 for int8, 162/300
for Stage A, and 7/300 for the combined teachers. Nineteen rows received
unsafe probability at least 0.5 from either teacher; this is disagreement,
not adjudicated source-label error. The original pre-check flagged one new
benign row already escalated by the base model. The historical Kaggle
procedure is in [KAGGLE_EASY_BENIGN.md](KAGGLE_EASY_BENIGN.md).

## 2026-10-01: routing separated from probabilities

Pre-check files now preserve all three model probabilities and add boolean
`force_escalate`. Final routing is `force_escalate OR p_safe < t`, including
at threshold zero. Threshold selection, ranking, and calibration use base
scores; recall, escalation rates, slices, error analysis, and economics use
routed decisions. The 12 regenerated pre-check files preserve probabilities
exactly, with unchanged thresholds and routed operating counts. Corrected
fp32 pre-check test AP is 0.898 (the superseded policy-distorted value was
0.829); int8 is 0.848 (superseded 0.781). The test flag adds no escalations;
the evasion split has 154 additional routed items. Legacy overridden-score
files must be regenerated. The recorded suite was 318 passed, 3 skipped.

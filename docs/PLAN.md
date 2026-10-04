# Forward plan

Current implementation and saved measurements are in [STATUS.md](STATUS.md).
The dated design and experiment decisions are archived in
[history/DECISIONS.md](history/DECISIONS.md). The behavior-preserving repository
maintenance pass is tracked in [CLEANUP_PLAN.md](CLEANUP_PLAN.md).

1. Validate OOD and evasion behavior on an independent holdout, with stronger
   attention to legitimate defensive prompts, ambiguity, and non-English text.
   The existing pre-check evasion gain is post-hoc.
2. Obtain blind independent annotations where feasible and audit source-derived
   ordinary-benign labels and AI-assisted seed realism. Report agreement and
   uncertainty at the category level.
3. Assess deployment economics with measured traffic prevalence, tier-two
   costs, and a real combined inference benchmark. Current figures are scenario
   estimates from sampled CPU timing and assumed traffic.
4. Address separate behavior issues identified during maintenance: teacher
   resume/merge validation, evaluation curve performance, and inference
   packaging. These require their own tests and design decisions.

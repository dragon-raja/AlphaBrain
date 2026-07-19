# CCV-VLA Gate 0A Final Result

Date: 2026-07-19

Decision: `STOP_COUPLED_CONTINUATION_ROUTE`.

## Scope and integrity

- Dataset: `/share/longjunyu/fresh-vla/ccv-vla/gate0-coupled-v3`.
- Frozen preregistration SHA-256:
  `060512b914d01a3a4092ccf3f7e955396733f2befbf83db79d16ec82c70b3bd0`.
- Collection completed all 102 groups with 988 captured states and 3,066 group files.
- Partition: 75 fit groups, 23 sealed holdout groups, and 4 engineering-excluded groups.
- Integrity audit checked 754 non-holdout states and found no missing files, schema errors,
  non-finite arrays, or deployable/audit-layer leakage.
- The 234 holdout states were checked only for file existence. Their deployable arrays and labels
  were not opened.
- The expensive continuation teacher completed 92/102 tasks overall: 70/75 fit, 18/23 holdout,
  and 4/4 engineering-excluded groups.
- Fifty-eight states reported raw milestone-order violations. Fifty-seven were `post_regrasp`
  states, where a successful place can end without a retained grasp. The registered prerequisite
  closure was applied; no state was removed.

The analysis implementation was corrected before the final analysis to obey the already frozen
leave-out protocol. An earlier interim implementation used the full six-repeat mean as its target,
which included the low-budget repeat being evaluated. That interim output is invalid and is not
used below. The corrected implementation excludes the selected repeat for one-repeat estimates
and both selected repeats for the two-repeat diagnostic. No data, split, or threshold changed.

## Formal Gate 0A

The final analysis used 716 states from all 23 fit source IDs. Source ID, rather than state or frame,
was the independent statistical unit. Confidence intervals use 10,000 source-level bootstrap
replicates.

| Metric | Mean | 95% source-bootstrap CI | Frozen condition | Result |
|---|---:|---:|---:|---|
| States with action leverage | 43.36% | [39.08%, 47.44%] | at least 30% | PASS |
| Six-repeat Oracle gain over candidate 0 | 0.010384 | [0.007273, 0.013764] | diagnostic | positive |
| One-repeat pairwise MSE reduction | 16.24% | [14.46%, 18.03%] | at least 20% | **FAIL** |
| One-repeat regret improvement | 0.003167 | [0.001497, 0.004997] | point estimate positive | PASS |
| One-repeat regret lower bound | 0.001497 | same interval | at least -5% of Oracle gain | PASS |
| Two-repeat regret improvement | 0.006179 | [0.003973, 0.008565] | diagnostic | positive |

The result is not a null finding about candidate support. Candidate actions changed downstream
viability in 43% of states, and exact six-repeat selection retained a positive advantage. Common
flow noise also improved the decision regret of a one-repeat selector. However, the registered
mechanism claim was stronger: coupling had to reduce broad pairwise viability-difference error by
at least 20%. The final confidence interval lies entirely below that threshold.

## Consequence

Gate 0B is locked and was not run. No ranker was trained, no holdout label was opened, and no
closed-loop validation was run. Running those stages after a failed prerequisite would turn a
predeclared test into post-hoc threshold search.

The defensible conclusion is:

1. Policy-continuation viability remains a real action-selection signal.
2. Plain shared flow-noise coupling is not a sufficiently strong VLA-specific data-efficiency
   mechanism under this protocol.
3. CCV-VLA in its present form stops here. A next method must amortize policy-relative future
   reachability by a different learning principle rather than adding repeats or relaxing the gate.


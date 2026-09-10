# CCV-VLA Gate 0 Preregistration

Status: v3 preregistered before formal CCV collection or model fitting.

Revision note: engineering smoke tests on source 36 showed that a candidate-0 trajectory can
remain in the same uninformative state region for many replans. No formal collection or model
fitting had begun. V2 added fixed temporal sampling. V3 changes the training-data behavior policy
to the frozen expensive continuation planner being distilled. All source-36 rows remain excluded
from fitting and statistics.

## Frozen assets

- Task: `put_the_cream_cheese_in_the_bowl`.
- Episode root: `/share/longjunyu/fresh-vla/libero-full-episode-v2-128`.
- Data available to Gate 0: manifest rows whose split is `train` only.
- Base policy: Pi0.5 Full-H seed 41 checkpoint at step 10353.
- Candidate count: `N=16`.
- Executed candidate prefix: `K=2` actions.
- Continuation lookahead: `L=8` policy actions after the candidate endpoint.
- Continuation repeats: `R=6`.
- Data-collection behavior selection: `R_select=2` coupled repeats at every replan.
- Base policy, flow steps, language instruction, candidate sampling, simulator controller, and
  episode budget are frozen across methods.

No existing test or sealed confirmation group may be read by a CCV script. Validation groups may
only be used after Gate 0 passes.

## Collection distribution

Start each rollout from the recorded slipped feedback state. At every live replan, sample the same
`N=16` candidates, estimate their coupled viability profiles with `R_select=2`, and execute the
lexicographically best candidate for `K=2` actions. This expensive planner is a frozen data
collection teacher, not a deployable policy input. Capture at most the first occurrence of each
state category:

```text
feedback_reveal, failure_continuation, recovery_start,
reapproach, preclose, post_regrasp, final_failure
```

Additionally capture the live state at frozen replan indices

```text
0, 8, 16, 24, 32, 40, 48, 56
```

when the episode has not terminated. A temporal capture coincident with a semantic capture is
stored once with both reasons. The schedule was selected from the already documented pre-CCV
CORA finding that candidate continuation keys were identical in replans 0-7 and diverged in later
eight-replan bins; it is not adapted to any formal CCV label.

At a captured state:

1. save the two current RGB observations, deployable robot state, frozen Pi0.5 visual-language
   feature, and all 16 sampled action chunks;
2. execute each candidate prefix in an exact sibling environment;
3. run six downstream continuations with depth-coupled flow noise shared across all candidates;
4. save every per-candidate, per-repeat milestone signature;
5. keep simulator snapshots and privileged physical fields in an audit cache that is never loaded
   by critic training.

Collection is transactional and resumable per snapshot group.

The behavior-selection labels at uncaptured replans are not training examples. They only move the
live trajectory into states reachable under the continuation teacher. Candidate 0 remains the
offline no-reranking baseline at every stored state. Gate 1 must later test the learned reranker
from the original feedback state, so teacher-state collection alone cannot establish a method gain.

## Frozen source split

The 30 `source_initial_state_index` values in the manifest train split are the independent units.
They are sorted by SHA-256 of

```text
ccv-vla-gate0-v1::<source_initial_state_index>
```

The six smallest hashes form the held-out Gate 0 source set. Source 36 is designated
`engineering_excluded` because it was used by the pre-formal smoke. The remaining 23 sources form
the fitting set. All rows from a source remain in one side. The split implementation and exact IDs
are written to the collection manifest before fitting.

## Label definitions

For each continuation trace, derive cumulative binary milestones:

```text
M1 = a stable grasp is retained or reached
M2 = lift is reached
M3 = transport is reached
M4 = formal task success is reached
```

Enforce `M4 <= M3 <= M2 <= M1` by cumulative closure for labels and report raw violations as a
collector-quality diagnostic. Also retain `regress`, `drop`, and normalized `progress_auc`.

The primary candidate ordering is lexicographic over mean

```text
[M4, M3, M2, M1, not_regress, progress_auc].
```

This is fixed before looking at CCV labels.

For regret magnitudes only, use the normalized base-8 utility

```text
U = (8^5 M4 + 8^4 M3 + 8^3 M2 + 8^2 M1
     + 8 not_regress + progress_auc) / (8^5 + 8^4 + 8^3 + 8^2 + 8 + 1).
```

With six-repeat milestone probabilities, each nonzero difference is at least `1/6`, so every
earlier term dominates the maximum sum of all later terms. This scalar therefore preserves the
registered lexicographic order for milestone ties and is not tuned to results.

## Gate 0A: coupled-label mechanism

Compare depth-coupled repeats with a deliberately independent construction that permutes
continuation repeat IDs separately for every candidate. Use leave-one-repeat-out estimates so the
evaluation target does not contain the low-budget label being judged.

Primary units are source IDs. Candidate pairs and states are averaged inside source before paired
source bootstrap.

Gate 0A passes only if all conditions hold:

1. At least 30% of eligible captured states contain two candidates with different six-repeat
   primary viability keys.
2. At one-repeat label budget, coupling reduces pairwise viability-difference MSE by at least 20%
   relative to independent repeats.
3. Coupling improves top-candidate simple regret at one-repeat budget; the source-bootstrap
   point estimate must be positive and its 95% interval may not cross below -5% of the available
   oracle gain. Two-repeat results are reported as secondary diagnostics only.

If condition 1 fails, Pi0.5 candidate support has insufficient action leverage for this route. If
conditions 2 or 3 fail, CRN is not a useful VLA-specific data-efficiency mechanism here.

## Gate 0B: held-out ranking

Train with identical optimization budgets and deployable inputs:

1. `sample0`: no reranking;
2. `random`: uniform candidate;
3. `self_consistency`: medoid candidate in action-prefix space;
4. `terminal_scalar`: predict task success only;
5. `dense_scalar`: regress a fixed scalarization of the full signature;
6. `ccv_profile`: monotone milestone-survival heads plus regression-risk and progress heads.

Model selection and abstention calibration use fitting-source cross-validation only. Report the
predeclared model once on the six source-held-out groups.

Gate 0B passes only if:

1. `ccv_profile` recovers at least 35% of the six-repeat oracle improvement over `sample0` on the
   held-out primary viability key;
2. its source-bootstrap mean improvement over `sample0` is positive and the lower 95% bound is
   no worse than -10% of the available oracle gain;
3. it lowers lexicographic ranking regret by at least 10% relative to both `terminal_scalar` and
   `dense_scalar`;
4. it does not select a candidate with a worse `M1` outcome than sample 0 in more than 5% of
   states where deeper milestones are tied.

The 35% threshold is a mechanism threshold, not a paper claim. Closed-loop validation is required
for any method claim.

## Decisions

- `CONTINUE_TO_CCV_VAL_GATE`: Gate 0A and Gate 0B both pass.
- `PIVOT_TO_GENERIC_SCALAR_CRITIC`: a scalar critic works but structured CCV does not beat it.
- `STOP_COUPLED_CONTINUATION_ROUTE`: coupling or held-out ranking fails.
- `DATA_OR_POLICY_SUPPORT_INVALID`: collection quality fails or fewer than four held-out source
  IDs produce eligible states.

No threshold is changed after formal collection begins. Any exploratory change creates a new
versioned preregistration and dataset root.

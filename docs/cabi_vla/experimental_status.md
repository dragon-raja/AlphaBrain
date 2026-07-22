# CABI-VLA Experimental Status

Status date: 2026-07-22
Primary behavioral execution horizon: `K=3`

## Coverage-corrected v10 gate

The v7 view preserves all original windows and balances source exposure at the
fine teacher-phase level. The 11,000-update v10 run covers this 20,352-item view
only once. Plain BC solves `0/4` supervised state-0 tasks; CABI solves `1/4`.
The formal orchestration decision is `BASELINE_INVALID`, and no validation
rollout was started.

CABI increases same-image source sensitivity from `4.65e-5` to `5.49e-3` and
target sensitivity from `9.55e-6` to `2.40e-2`, but observed-corner MSE remains
`0.176` and neither held-out edge succeeds. These values are diagnostic only.

A clean 33,000-update v10b BC/static-CABI comparison now runs on v7. This gives
each method about 3.24 shuffled data passes, matching the exposure that produced
the earlier fitted checkpoint while retaining the corrected coverage.

The optional decision-point functional-closure amendment is implemented and
smoke-tested but is not active in v10b. Its preregistration and leakage audit are
recorded in `preregistration_amendment_decision_closure.md`.

## Frozen evidence

- LIBERO-Bind v0 uses 50 canonical state groups with group-preserving
  train/validation/test splits.
- Four source-target edges have action supervision. `white-right` and
  `yellow_white-left` are action-free fourth corners.
- The fourth-corner image and instruction may enter the CABI closure loss, but
  its action is never loaded by the training dataset.
- All 64 retained training tetrads have exactly equal pre-action images and
  robot state across their physical base/target corners.
- The scripted teacher succeeds on 138/140 supervised trajectories (98.6%).
  All four supervised state-0 trajectories succeed in 139--265 steps.
- The shared initialization was trained on `libero_all`, including a LIBERO-10
  two-stage `white-left` then `yellow_white-right` task. It did not include
  LIBERO-90 or either withheld edge. The present LIBERO-90 state bank has zero
  exact overlap with the corresponding LIBERO-10 state bank.

## Balanced 3000-update calibration

Both policies used seed 41, gradient accumulation 2, anchor period 8, identical
data order, and the same original Pi0.5 initialization. This corresponds to
about 6000 dataset items, or 1.07 passes over the 5611-window training view.

| Metric (64 tetrads, flow seed 20260722) | BC | CABI |
|---|---:|---:|
| Observed-corner action MSE | 0.192154 | 0.190709 |
| Action-free pseudo-fourth MSE | 0.544534 | 0.548229 |
| Model self-closure MSE | 0.000070 | 0.001169 |
| Target-effect transfer MSE | 0.569015 | 0.568270 |
| Target-effect cosine | 0.320188 | 0.335803 |
| Source-effect transfer MSE | 0.287064 | 0.293908 |
| Source-effect cosine | 0.440912 | 0.424709 |

The low BC self-closure error shows that algebraic output closure can arise
without correct behavioral transport. It is therefore diagnostic only.

At train state 0 and fixed `K=3`, both methods obtain 0/4 supervised full-task
success and 0/2 held-out success. Mean subgoal progress is 0.0833 for BC and
0.0417 for CABI. The corresponding H.264 and AV1 rollout videos are stored with
the evaluation outputs.

**Calibration decision:** `BASELINE_INVALID_AT_3000`. This run cannot support a
positive or negative CABI claim.

## 9000-update v3 baseline audit

The v3 BC checkpoint reloads correctly and predicts its first red-left teacher
chunk with MSE `0.000336`. Across all 64 training tetrads:

| Metric | BC-9000 v3 |
|---|---:|
| observed-corner MSE | 0.000813 |
| action-free pseudo-fourth MSE | 0.545515 |
| target-effect transfer MSE | 0.572524 |
| source-effect transfer MSE | 0.566043 |

Despite fitting observed actions, fixed-`K=3` state-0 behavior is 0/4 on
supervised edges and 0/2 on held-out edges, with zero mean progress. Rollout
inspection shows instruction-conditioned target motion but wrong-object grasps.

A three-seed, shared-noise intervention audit confirms source-role collapse:

| Same-image instruction intervention | Action-chunk MSE |
|---|---:|
| source name only | 0.0000665 |
| target plate only | 0.554822 |

Teacher-forced chunk MSE is `0.0390` overall and is concentrated at
`episode_start` (`0.1626`) and `lift` (`0.1440`). No held-out action was loaded
by this diagnostic. The decision is `BASELINE_INVALID_AT_9000_V3`.

## v3 CABI-v7 diagnostic: geometry is not behavior

The sampling-biased v3 CABI-v7 run completed only as a failure diagnostic. Its
self-contained checkpoint reload passed (`teacher-chunk MSE=0.000179`) and its
64-tetrad observed-corner MSE is `0.000878`, yet target/source action-transport
MSE remains `0.566946/0.561011`. At state 0 and fixed `K=3`, it obtains 0/4
supervised and 0/2 withheld successes; five of six rollouts grasp a wrong source.

The same-image policy intervention still changes actions by only `0.000110`
when the source name changes, versus `0.561339` when the target changes. In
contrast, the representation diagnostic appears excellent: source/target
specificity margins are `1.0632/1.9752`, fourth-anchor role error is `0.000424`,
and commutator error is `0.0000311`. This is direct evidence that identifiable
role geometry can remain disconnected from the deployed action function. The
run is not eligible for a CABI comparison.

For provenance, the untouched `libero_all` initialization also scores 0/6 at
this custom state-0 gate and never grasps any source. Its videos are retained as
a frozen reference, not as a trained baseline.

## Loss-mass-balanced v9 calibration

The source collapse revealed a training-view confound: v3 had 3,595 red-source
windows, 1,012 white-source windows, and 1,004 yellow-white-source windows,
before additional red-heavy tetrad anchors. The first deterministic v4 view
balanced raw example counts, but not action-loss mass: the trainer averages the
supervised examples within each microbatch, so anchor examples contribute
one-quarter of a regular item's loss mass. The associated v8 runs were stopped
at updates 2409 (BC) and 1948 (CABI) and are ineligible.

The corrected v5 view has 5,616 items and exactly balances the trainer's loss
estimand after anchor insertion:

- each source factor: 7,488 loss units with denominator 4;
- each target factor: 11,232 loss units with denominator 4;
- common macro-phase distribution across all four supervised edges;
- zero held-out action records loaded.

New v9 BC and CABI-v7 runs use this same v5 order and seed 41. Both have
independent reload, offline, state-0, source-grounding, and binding-geometry
gates. A persistent paired orchestrator launches the 30-episode validation
comparison only if BC first reaches at least 70% supervised state-0 success.

The data-order invariant was audited separately because CABI initializes extra
parameters. The PaliGemma DataLoader uses a dedicated `torch.Generator(seed)`
rather than the model's global RNG. Reconstructing both runs' first 18,000
microbatch indices at seed 41 produced the identical SHA-256 digest
`795c13a89b47e9fe1e80c9455a36f7225c33ebb9d767387ad0dd383fe54a5340`.

## Active convergence calibration

The corresponding CABI-v0 run was stopped near step 1450 after an action-label-free
geometry audit proved that its role variables had collapsed. Its partial metrics
and abort marker are retained and it is not used for policy comparison.

The identifiability amendment and bounded v5/v6 failures are recorded in
`preregistration_amendment_identifiability.md`. CABI-v7 passed the 500-update
training-only geometry gate with positive source and target specificity margins.
A v3 9000-update v7 run remains active as a sampling-bias diagnostic. It is not
an eligible comparator. The v4/v8 pair is also ineligible because its raw-count
balance did not match the trainer's loss reduction. The eligible v9 BC and
CABI-v7 models both restart from the original checkpoint on v5 and do not
continue from a gate or prior long-run checkpoint.

After each eligible checkpoint is written, an independent tmux gate performs:

1. self-contained checkpoint reload and finite-action smoke;
2. the 64-tetrad action-free offline diagnostic;
3. train-state0 fixed-`K=3` closed-loop evaluation.
4. same-image source/target intervention and teacher-phase diagnosis.
5. action-label-free CABI binding-geometry diagnosis for the method checkpoint.

The comparison expands to all validation states only if the ordinary BC policy
first reaches at least 70% success on supervised in-distribution edges. Until
that gate passes, held-out differences are not interpreted as transfer.

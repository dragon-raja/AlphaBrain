# Recovery Support v2 Preregistration

Status: protocol frozen before collection; validation experiment completed.

## Baseline release

The final validation-only Full-H repair uses 10,353 optimizer steps for seeds
41, 42, and 43. At fixed K=3 its attached success is 61.54%, 38.46%, and
61.54%, for a cross-seed mean of 53.85%. The unchanged baseline gate passes as
`BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS`. Test remains sealed.

The same baseline also exposes the behavior being studied. K=1 failure
continuation is 0% for every seed; at K=3 it is 62.5%, 55.6%, and 75.0%.
Train-only counterfactual diagnostics independently show that replacing a stale
tail with a fresh K=1 prediction reduces slipped feedback-time teacher-action
MSE by 83.5% and selects the recovery direction in 81.0 percentage points more
groups, without a detectable attached penalty. This establishes immediate
stale-tail commitment, but not multi-step recovery competence.

## Frozen Gate 1 comparison

All arms start from the seed-matched 10,353-step Full-H checkpoint and receive
exactly 6,902 additional single-GPU optimizer updates. The budget is inherited
from the preregistered first recovery-support candidate; it is not selected
from any v2 arm result.

The three arms are:

1. `base_continuation`: 50% ordinary train anchors and 50% ordinary windows
   matched to the same target-group schedule.
2. `clean_recovery_replay`: identical anchors, with target windows from clean
   feedback-to-stable-regrasp expert trajectories.
3. `policy_state_recovery`: identical anchors and target-group schedule, with
   targets collected after the deployed baseline reaches a wrong or unresolved
   post-feedback state and the same teacher corrects to stable regrasp.

The slot order is fixed per seed, not shuffled during training. Optimizer,
scheduler, learning rate, batch size, frozen modules, prediction horizon,
initialization, update count, validation groups, policy seeds, episode budget,
and fixed K=1/2/3 evaluation are otherwise identical. Policy inputs remain
agent view, wrist view, robot state, and language. Branch outcome, contact,
object pose, future state, and teacher state are audit-only and never enter the
policy.

## Decision rule

K=3 is primary; K=1 and K=2 diagnose commitment sensitivity. Snapshot group is
the independent unit, seeds are averaged within group, and all comparisons use
paired group-level bootstrap 95% intervals.

- Prefer ordinary continuation if it improves slip recovery or overall success
  by at least 10 points over the released baseline with CI lower bound above 0,
  while attached degradation is at most 5 points.
- Prefer clean replay if it clears the same rule over ordinary continuation.
- Continue the policy-state bridge only if it clears that rule over both
  ordinary continuation and clean replay.
- If no arm clears the behavioral gate, stop offline support expansion.

No test result is opened in Gate 1. No CFR residual, competence head, dynamic K,
world model, active perception, RL, or new task is added. A simpler arm that
solves the problem wins; physical outcome pairing is not treated as a
contribution unless a later equal-cost comparison beats ordinary A2C2/RaC-style
correction.

## Artifacts and stopping

Outputs use a new immutable root:

`/share/longjunyu/fresh-vla/runs/recovery-support-repaired-v2-step10353`

The controller is `scripts/fresh_vla/run_recovery_support_v2_pipeline.sh`. It
refuses dirty code, failed baseline provenance, failed data quality, incomplete
pre-existing outputs, and any overwrite. Videos are H.264/`avc1`, `yuv420p`,
fast-start and are decoded before a result is accepted.

## Completed result

Gate 1 completed with `STOP_OFFLINE_SUPPORT_EXPANSION`. At primary K=3, Base
continuation improved overall success by only 5.1 points and slip recovery by
2.6 points over Full-H, with both paired intervals crossing zero. Clean replay
and policy-state correction significantly reduced full-task success relative to
Base. See `recovery_support_v2_results.md` for the complete result, confidence
intervals, video audit, and quarantined deterministic-reach split deviation.

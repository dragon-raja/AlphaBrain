# ACD-VLA Gate 0 preregistration

Frozen before collecting policy-response targets on 2026-07-18.

## Question

Does the current frozen Pi0.5 produce attached/slipped post-feedback responses
that are (a) different enough to matter and (b) predictable from the common
pre-feedback state beyond an outcome-specific constant template?

This is a representation/mechanism gate, not a closed-loop result.

## Data policy

- Source: `/share/longjunyu/fresh-vla/libero-full-episode-v2-128`.
- Open episode arrays only for the existing `train` and `val` groups.
- Expected groups: 102 train and 13 val.
- The 13 original test groups and every confirmation artifact remain sealed.
- Train and validation source initial states must be disjoint.
- Statistical unit: source initial state; outcomes are paired within group.

For each development group, use the common observation at
`feedback_reveal_time - 1` and query the same frozen seed-41 Full-H Pi0.5 with
the same flow seed at:

1. the common pre-feedback observation;
2. the attached post-feedback observation;
3. the slipped post-feedback observation.

Save only policy action arrays, 8x8 block-average observation features,
non-sensitive policy identity, hashes, and audit metadata. Collection is
transactional and resumable.

## Frozen prediction task

- Response horizon: first 8 post-feedback actions. The loaded AlphaBrain Pi0.5
  checkpoint exposes a 10-step action horizon; 8 leaves one executed step for
  stale-tail alignment without padding or extrapolation.
- Input: common pre-feedback agent view, wrist view, and robot state.
- Target: concatenated frozen-policy response chunks for attached and slipped.
- Probe: ridge regression with source-grouped five-fold alpha selection over
  `{0.01, 0.1, 1, 10, 100}`.
- Router: ridge classifier on real post-feedback features.
- Action normalization is fit on train policy responses only.
- Bootstrap: 20,000 paired source-level resamples.

## Baselines

1. `stale_pi05`: tail of the actual Pi0.5 chunk queried one step before reveal.
2. `linear_merged`: one state-conditioned continuation trained on the mean of
   the two response targets.
3. `branch_constant`: one train-set mean response per outcome.
4. `random_precommit`: randomly choose one of the two predicted branches before
   observing the outcome.
5. `oracle_route`: use the correct branch identity; this separates branch
   prediction from guard error.

The branch-constant baseline is mandatory because the preceding Branch-VLA
Gate showed that scripted expert continuations on this task were templates.

## Frozen checks

All checks are required to proceed:

- exact development data audit; zero pre-feedback twin mismatch;
- post-feedback guard accuracy >= 85%;
- pre-feedback guard accuracy <= 60%;
- 32 shuffled-label guard mean accuracy <= 60%;
- mean normalized RMS separation of teacher responses >= 0.10;
- oracle-routed learned branches reduce MSE versus `branch_constant` by >= 20%,
  with source-bootstrap 95% CI lower bound > 5%;
- the same reduction is >= 10% separately for attached and slipped;
- learned routing reduces MSE versus `stale_pi05` by >= 25%, with CI lower
  bound > 5%;
- median source-level predicted/target branch-separation ratio >= 0.5.

## Decision

- All checks pass: `PROCEED_ACD_CLOSED_LOOP_GATE`.
- Valid data but any check fails: `STOP_ACD_CURRENT_TASK_GATE0`.
- Data/audit failure: `ACD_GATE0_INVALID`.

`STOP_ACD_CURRENT_TASK_GATE0` means this one-task dataset cannot justify model
training. It does not establish that contingency distillation is impossible;
restarting it requires a predeclared multi-task/event dataset with non-template
teacher responses, not another threshold or probe sweep on these groups.

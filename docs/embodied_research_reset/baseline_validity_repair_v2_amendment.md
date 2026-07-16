# Full-H Baseline Validity Repair V2 Amendment

Freeze time: 2026-07-16 UTC, after the 3,451-step validation gate and before
training or evaluating seed 42/43 at 10,353 steps.

This is a validation-only baseline repair. It is not a FRESH, CFR, recovery-data,
or test-set experiment.

## Observed trigger

The preregistered 3,451-step three-seed gate completed on all 13 validation
snapshot groups per seed and failed only the cross-seed attached-success mean:

- seed 41: 46.15%;
- seed 42: 23.08%;
- seed 43: 7.69%;
- cross-seed mean: 25.64%, below the frozen 30% threshold;
- two seeds reached the frozen per-seed 20% threshold;
- decision: `BASELINE_INVALID_OR_DATA_INSUFFICIENT`;
- test split opened: false.

All 117 paired videos and 36,781 decoded frames passed the H.264/`avc1`,
`yuv420p`, fast-start, nonblank, motion, and exact-frame-count audit.

## One fixed repair budget

The only additional baseline budget is **10,353 optimizer steps**. This value is
not selected from seed 42/43 outcomes. It was one of the four budgets frozen in
the original preregistration and is the earliest checkpoint after 3,451 that
passed the original seed-41 validation criteria. Seed 41 had 61.54% attached,
50.00% overall, and 38.46% slip success at K=3 at both 10,353 and 13,804 steps;
therefore the shorter tied budget is used.

No seed-42 or seed-43 10,353-step training or closed-loop result existed when
this amendment was frozen. The 6,902-step seed-41 checkpoint is not used because
it failed the original seed-41 selection criteria.

## Frozen execution

- reuse the existing seed-41 10,353-step checkpoint;
- train seeds 42 and 43 independently from Pi0.5 Base to exactly 10,353 steps;
- use the exact training implementation commit
  `ce552faf64f1cea994d10899ef500380ab02f2b5` in a clean detached worktree;
- retain all original data, initialization, DDP, optimizer, LR, warmup, frozen
  modules, batch size, and seed rules;
- save only the fixed 10,353-step checkpoint plus the final model;
- use the same 13 validation groups, policy seeds, 320-step budget, formal task
  success condition, and fixed K=1/2/3 evaluator;
- repeat seed 41 with the same evaluation seed and require exact determinism;
- audit every generated video by actual decoding and codec/container checks;
- keep the test split sealed.

The training-code SHA is held at `ce552faf...` so all three seed identities remain
exactly comparable. Research documentation and diagnostic scripts added after
that SHA do not participate in training.

## Unchanged gate and stopping rule

Primary gate remains K=3 attached full-task success:

- cross-seed mean at least 30%;
- at least two of three seeds at least 20%;
- no non-finite training or evaluation result.

If this gate passes, 10,353 becomes the common budget for the already specified
recovery-support controls. If it fails, no 13,804-step cross-seed retry, checkpoint
selection, threshold change, recovery-method comparison, or test opening is
allowed. The result remains `BASELINE_INVALID_OR_DATA_INSUFFICIENT`.

## Completed result

The amendment completed without changing its gate. All three seeds were
evaluated at 10,353 steps; K=3 attached success was 61.54%, 38.46%, and 61.54%,
with a cross-seed mean of 53.85%. The decision is
`BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS`. Test remained sealed, and all
117 end-to-end validation videos passed the formal codec and decode audit.

# H=20 CAFC Continuation Amendment

Status: frozen before any H=20 training

## Motivation and claim boundary

The H=10 seed-41 gate did not establish migration. It did establish that
Bridge+CAFC learns the local held-out target action effect and moves both held-
out objects toward the correct target after an exact teacher-prefix handoff.
The behavior then stalls because the completed target spans only ten actions,
whereas 68--85 teacher actions remain.

This amendment asks one narrow question: does a longer but still locally valid
counterfactual action chunk provide enough continuation support for the same
single-pass policy to complete unseen source-target bindings? It does not add
replanning, resampling, recovery, a world model, RL, or an inference-time
composition module.

## Disclosed development selection

State-0 teacher-QA actions were inspected after H=10 failed. They are disclosed
development evidence and remain excluded from formal validation claims.

| Horizon | White-right completion cosine | Yellow-left completion cosine | Gripper sign agreement |
|---:|---:|---:|---:|
| 10 | 0.9960 | 0.9877 | 1.00 / 1.00 |
| 20 | 0.9180 | 0.9131 | 1.00 / 1.00 |
| 30 | 0.843 | 0.778 | not used |
| 40 | 0.754 | 0.589 | not used |
| 50 | 0.586 | 0.525 | 0.74 / 0.68 |

H=20 is frozen as the longest completion with cosine above 0.9 on both missing
edges and fully consistent gripper direction. There will be no horizon sweep:
failure at H=20 stops this action-field completion route.

## Data invariants

The H=20 view must preserve the v13 `records.jsonl`, ordering, images, wrist
images, states, tetrads, splits, and observed-edge set exactly. Only action
chunks are re-sliced from the same successful observed episodes at the same
`source_select` and `target_select` frames. The actions of `white-right` and
`yellow_white-left` must never be loaded. No teacher-QA or held-out continuation
may enter training.

## Frozen arms and budget

All arms use seed 41, 33,000 updates, the same Pi0.5 checkpoint, H=20 training,
the same record order, optimizer, batch size, normalization, and fixed `K=3`
deployment:

1. BC-H20.
2. Action-Bridge-H20.
3. CAFC-H20.
4. Bridge+CAFC-H20.

Giving H=20 to both architecture-matched controls prevents horizon capacity
from being attributed to CAFC. Bridge+Closure-H10 remains the previously frozen
strong comparator but is not silently relabelled as an H=20 control.

## Sequential gate

State 0 is calibration evidence. An H=20 CAFC arm opens full validation only
if it reaches at least 3/4 observed full-task successes and at least 1/2 held-
out full-task successes at fixed `K=3`. A control that fails 3/4 observed
success makes its exact comparison baseline-invalid; it cannot create a
positive CAFC claim by being weak.

On untouched validation states, an advancing arm must satisfy all of:

- observed full-task success at least 70%;
- held-out full-task gain at least 10 percentage points over its H=20 exact
  control;
- observed degradation no worse than 5 percentage points;
- positive success change on both held-out edges;
- improved held-out source selection;
- non-negative paired state-group bootstrap lower bounds for held-out task and
  source-selection changes; and
- no worse result than the frozen Bridge+Closure-H10 comparator under the same
  fixed-`K=3` validation protocol.

Seeds 42/43 and sealed test evaluation run only after the seed-41 gate passes.

## Decisions

- `ADVANCE_H20_CAFC`: plain CAFC-H20 clears exact and strong gates.
- `ADVANCE_H20_GROUNDED_CAFC`: only Bridge+CAFC-H20 clears them.
- `STOP_HORIZON_EXTENSION`: neither H=20 arm clears the behavioral gate; do
  not try H=30/40/50 or trajectory completion.
- `BASELINE_INVALID`: no relevant H=20 arm retains 3/4 observed success, so no
  migration conclusion is valid.


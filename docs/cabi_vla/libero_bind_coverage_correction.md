# LIBERO-Bind Coverage Correction

## Status

The seed-41 v9 comparison is retained as a diagnostic and is ineligible for a
CABI migration claim. Both BC and CABI reached 2/4 supervised state-0 tasks and
0/2 action-free tasks. The orchestration result is `BASELINE_INVALID`.

## Failure audit

The v5 view balanced aggregate source and target action-loss mass, but sampled
within coarse macro phases without preserving state-by-phase coverage. In the
state-0 training gate, v5 contained no `episode_start` record for either red
edge. Across the full view, episode-start exposure was red 23, white 133, and
yellow-white 169. The resulting BC policy had source intervention sensitivity
`3.43e-5` versus target sensitivity `5.57e-1`.

CABI increased source sensitivity to `3.77e-4`, but still failed both red tasks
and both action-free tasks. Its role geometry remained well separated. This is
evidence that representation closure is not sufficient for behavioral transfer,
not evidence against CABI under a valid baseline.

## Corrected view

The eligible replacement is:

`/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v7-coverage-phase-loss-balanced`

It is generated from v3 with:

- all 5,611 source windows preserved exactly once or more;
- balancing at the original teacher-phase granularity;
- exact aggregate source and target action-loss balance;
- anchor period 32;
- no regular held-out action records;
- deterministic construction seed 20260722.

The view has 20,352 items. Effective episode-start source loss units are red
694, white 718, and yellow-white 718. Total effective source units are 27,136
for each source; target units are 40,704 for each target.

## Re-entry rule

Training must consume at least one complete shuffled pass over v7. With batch
size 1 and gradient accumulation 2, the calibration budget is therefore at
least 10,176 optimizer updates. The registered v10 calibration uses 11,000.

The migration comparison resumes only if plain BC reaches at least 3/4
supervised tasks on the state-0 gate. Otherwise the result remains
`BASELINE_INVALID`; CABI is not compared on held-out transfer.

## Causal decision-point audit

Across the 33 train states shared by all four supervised edges, episode-start
agent and wrist images are byte-identical across instructions. With a 10-step
teacher chunk, changing the source while holding the target fixed yields action
MSE 0.200 (red/white-left) and 0.303 (red/yellow-white-right). Changing only the
target for red yields exactly zero action MSE at episode start.

At the pre-transport anchor the relation reverses: the red left/right target
intervention has action MSE 0.571, while the two source interventions have MSE
0.027 and 0.080. The benchmark therefore contains a phase-dependent causal
factorization rather than a static additive action decomposition.

The v10 run intentionally retains the preregistered transport-anchor CABI as a
clean data-correction control. If BC becomes valid but CABI still fails to
transfer, the next admissible method revision is decision-point binding: source
transport at episode start and target transport at pre-transport. Adding more
static representation losses would not address the measured mismatch.

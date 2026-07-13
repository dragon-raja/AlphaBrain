# FRESH-VLA Counterfactual Validation

## Scope

This phase tests whether counterfactual branch consistency is a useful
per-action supervision-confidence principle. It does not implement a horizon
head, dynamic execution, reinforcement learning, a world model, active vision,
or robot deployment.

The earlier two-episode Pi0.5 smoke established the training path but found no
gain from a gripper-transition proxy. This phase therefore separates physical
outcome uncertainty from semantic events and language-conditioned intent.

## Engineering Audit

The loss uses horizon `h` to supervise exactly action steps `[0, h)`. Horizons
must be integer values in `[0, H]`; invalid values are rejected instead of
clamped. Each sample is normalized by its own total step weight, and samples
with no supervised step under a hard mask do not dilute the batch mean.
`tail_weight=1` is bit-exact with the original mean FM loss.

Paired evaluation fingerprints the sample IDs and contents, flow timestep,
noise, and action normalization. Training data, mask randomness, FM noise/time,
evaluation data, and inference noise use independent deterministic RNG streams.

The production loss now supports both suffix weighting and the
`prefix_control` weighting used by the REMAC-style temporal control. Pi0.5
recipes for all seven methods are in
`configs/experiments/fresh_vla_counterfactual.yaml`.

## Counterfactual Data Contract

The zero-download fixture contains 256 pairs and 512 expert continuations:

- grasp success versus slip/recovery;
- unobstructed push versus blocked-push recovery, with no gripper event;
- deterministic free-space reach;
- left/right language-intent controls grouped by full conditioning.

Repeated within-branch rollouts define a 95th-percentile action-distance
threshold. The oracle horizon is the first cross-branch divergence that remains
above the threshold for two steps. Sensitivity is recorded at threshold
multipliers 0.5, 1.0, 1.5, and 2.0. A whitelist permits only observation,
robot state, and language instruction to enter the policy.

The fixture is a schema and mechanism test, not simulator evidence. A LIBERO
snapshot collector is still required for physical closed-loop claims.

## Methods

All methods use the same balanced trajectories, model initialization, optimizer,
1200 updates, three seeds (41, 42, 43), prediction horizon, and fixed execution
condition. The compared objectives are:

- `full_h`;
- `random_soft010`;
- `gripper_soft010`;
- `oracle_soft010`;
- `oracle_hard000`;
- `short_h`;
- `remac_prefix_mask_control`.

Evaluation uses 4096 paired samples per seed. Multimodal analysis uses 32 chunks
per state. Confidence intervals below bootstrap the three seed-level paired
means; sample-level intervals are also preserved in the compact result file.

## Branching Results

Lower is better except suffix mode coverage.

| Method | Fixed-K prefix | Oracle prefix | Premature commitment | Oracle-prefix FM | Suffix FM | Mode coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_h | 0.000704 | 0.000949 | 0.02816 | 0.01367 | 0.19972 | 1.00 |
| random_soft010 | 0.000263 | 0.000377 | 0.01456 | 0.00785 | 0.21226 | 1.00 |
| gripper_soft010 | 0.000293 | 0.000565 | 0.01883 | 0.01053 | 0.21248 | 1.00 |
| oracle_soft010 | **0.000256** | **0.000299** | **0.01373** | **0.00682** | 0.21639 | 1.00 |
| oracle_hard000 | 0.000479 | 0.000481 | 0.01812 | 0.00647 | 1.88804 | 0.00 |
| short_h | 0.000590 | 0.000687 | 0.02302 | 0.00993 | 1.72779 | 0.00 |
| REMAC prefix control | 0.004125 | 0.004169 | 0.06801 | 0.03451 | 0.20291 | 1.00 |

Relative to `full_h`, oracle soft reduces fixed-K generation MSE by 63.6%,
oracle-prefix generation MSE by 68.5%, premature commitment by 51.2%, and
oracle-prefix FM MSE by 50.1%. It increases suffix FM MSE by 8.3% and full FM
MSE by 5.9%. All three seed-level intervals exclude zero for these changes.

Relative to `random_soft010`, oracle soft improves oracle-prefix generation MSE
by 20.8% and oracle-prefix FM MSE by 13.1%, with all three seed means favorable.
Its fixed-K generation improvement is only 2.5%; fixed-3 FM has mixed seeds and
is 1.1% worse in the mean. Suffix and full FM are worse.

Relative to `gripper_soft010`, oracle soft improves oracle-prefix generation
MSE by 47.1%, premature commitment by 27.1%, and oracle-prefix FM MSE by 35.2%.
The fixed-K generation and fixed-3 FM seed-level intervals cross zero.

Hard masking and short horizon destroy suffix modeling and mode coverage. The
REMAC-style front mask strongly harms the common prefix. Soft weighting is the
only tested family that improves the prefix while retaining both suffix modes.

## Negative Controls

With branch strength set to zero, the oracle horizon is exactly `H`.
`oracle_soft010` and `full_h` are bit-identical for every metric in every seed;
all paired deltas and confidence limits are exactly zero. This rules out a
generic regularization explanation for the oracle result.

The deliberately wrong short horizon increases deterministic full-horizon FM
MSE from 0.00488 to 0.07997 and substantially increases premature commitment.
Language-intent fixtures retain full oracle horizons because language is part
of the grouping condition.

## Gradient Mechanism

No repeatable negative prefix/suffix gradient conflict was found.

| Model / boundary | Action output cosine | Last action layer cosine |
| --- | ---: | ---: |
| full_h / oracle | 0.5975 | 0.4248 |
| full_h / gripper | 0.6329 | 0.5233 |
| full_h / deterministic midpoint | 0.4916 | 0.4486 |
| oracle_soft010 / oracle | 0.2050 | 0.1817 |
| oracle_soft010 / gripper | 0.2400 | 0.2555 |
| oracle_soft010 / deterministic midpoint | 0.2056 | 0.3151 |

Oracle weighting changes the optimization geometry and reduces cosine, but the
gradients remain aligned rather than conflicting. The observed prefix gain may
therefore be a capacity or loss-allocation effect, not direct gradient
cancellation.

## Go / No-Go

**Strict gate: No-Go for method expansion.** Do not implement a learned horizon
head and do not run the lambda sweep yet.

Positive evidence:

- oracle soft strongly beats full horizon on unweighted prefix behavior;
- it beats random and gripper boundaries on oracle-common-prefix metrics;
- it preserves multimodal suffix coverage;
- it does not affect the deterministic control when the oracle horizon is full.

Blocking evidence:

- oracle does not beat random/gripper on every fixed-K metric across seeds;
- full horizon already represents both suffix modes in the balanced toy data;
- suffix and full-horizon FM losses regress;
- no repeatable negative gradient conflict appears;
- no physical snapshot or fixed-K LIBERO closed-loop result is available yet.

One narrowly scoped LIBERO snapshot pilot is still justified to test whether
contact dynamics produce structure absent from the toy benchmark. Progress
beyond that pilot requires oracle soft to beat full, random, gripper, short,
and prefix-mask controls at the same fixed K without deterministic degradation.

## Artifacts

- Main paired run: `/share/longjunyu/fresh-vla/toy/counterfactual-paired-1200x3.json`
- Main compact summary: `/share/longjunyu/fresh-vla/toy/counterfactual-paired-1200x3-summary.json`
- Deterministic run: `/share/longjunyu/fresh-vla/toy/deterministic-paired-1200x3.json`
- Deterministic summary: `/share/longjunyu/fresh-vla/toy/deterministic-paired-1200x3-summary.json`
- Gradient diagnostic: `/share/longjunyu/fresh-vla/toy/gradient-diagnostic-1200x3.json`
- Counterfactual fixture: `/share/longjunyu/fresh-vla/counterfactual-toy/`

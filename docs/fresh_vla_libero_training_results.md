# FRESH-VLA LIBERO Training Pilot

## Scope

This experiment trains the AlphaBrain Pi0.5 action expert on real LIBERO
simulator snapshots with counterfactual grasp outcomes. It tests the central
FRESH-VLA mechanism: only give full training weight to the action prefix that
is supported before outcome feedback is available, while retaining a soft
weight on the uncertain suffix.

The result is a trained-policy offline pilot, not a closed-loop task-success
benchmark. The ten-step continuations capture grasp attachment versus slip and
the onset of recovery, but they do not complete the full bowl-placement task.

## Data And Protocol

The dataset at
`/share/longjunyu/fresh-vla/libero-counterfactual-v1-128` contains 192 paired
counterfactual groups and 384 branch records:

- 128 grasp/slip groups and 64 deterministic free-space reach controls;
- 384 branch videos, 192 side-by-side paired videos, and 192 rollout shards;
- exact shared pre-branch images, robot states, and action prefixes;
- stable action-divergence horizons at threshold multipliers 0.5-2.0;
- 100% attached-grasp and 0% forced-slip attachment rates.

The group-preserving split has 154 train, 19 validation, and 19 test groups.
Only the 102 training grasp/slip groups are used for optimization. The 13 test
grasp/slip groups and six deterministic test groups remain held out.

Eight methods use the same Pi0.5/PaliGemma initialization, data order,
optimizer, 1200-update budget, and seeds 41, 42, and 43. The compared methods
are Full-H, random, shuffled-oracle, early-oracle, late-oracle, gripper-event,
oracle FRESH, and a five-step short-horizon control. Soft methods assign weight
1.0 before their selected boundary and 0.1 after it. The VLM is frozen and the
693M-parameter action expert is trained.

Evaluation uses deterministic per-sample flow-matching noise and identical
test ordering. Reported intervals are paired sample-level bootstrap 95%
intervals. Lower MSE is better.

## Offline Results

| Method | K=1 | K=2 | K=3 | Oracle prefix | Suffix | Deterministic K=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-H | 0.0215 | 0.0220 | 0.0785 | 0.0595 | **0.3648** | 1.8866 |
| Random soft | 0.0080 | 0.0079 | **0.0612** | 0.0465 | 0.4116 | 1.9392 |
| Shuffled oracle soft | 0.0067 | 0.0067 | 0.0597 | 0.0539 | 0.4007 | 1.8470 |
| Early oracle soft | 0.0076 | 0.0078 | 0.0666 | 0.0517 | 0.3701 | 1.8955 |
| Late oracle soft | 0.0120 | 0.0122 | 0.0639 | 0.0623 | 0.4086 | 1.8533 |
| Gripper soft | 0.0183 | 0.0194 | 0.0723 | 0.0513 | 0.3847 | **1.7252** |
| Oracle FRESH soft | **0.0065** | **0.0062** | 0.0632 | **0.0234** | 0.4084 | 1.8982 |
| Short horizon | 0.0138 | 0.0145 | 0.0837 | 0.0386 | 0.5481 | 2.6699 |

Values are three-seed means. The complete report includes seed variation and
is stored at
`/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1/summary/report.md`.
The five-step Full metric for Short-H is not comparable to ten-step Full.

Oracle FRESH reduces K=2 FM MSE by 0.0158 versus Full-H, with paired 95%
interval `[-0.0208, -0.0114]`. Its exact oracle-prefix MSE is 60.7% below
Full-H. Direct paired oracle-minus-control comparisons show:

| Control | K=2 delta | Oracle-prefix delta |
| --- | ---: | ---: |
| Random soft | -0.00174 `[-0.00315, -0.00045]` | -0.02303 `[-0.03744, -0.01060]` |
| Shuffled oracle soft | -0.00052 `[-0.00150, +0.00072]` | -0.03047 `[-0.05271, -0.01269]` |
| Early oracle soft | -0.00163 `[-0.00349, +0.00019]` | -0.02829 `[-0.04570, -0.01335]` |
| Late oracle soft | -0.00600 `[-0.00880, -0.00337]` | -0.03887 `[-0.05974, -0.02105]` |
| Gripper soft | -0.01317 `[-0.01998, -0.00743]` | -0.02783 `[-0.04314, -0.01479]` |
| Short horizon | -0.00835 `[-0.01465, -0.00352]` | -0.01519 `[-0.02592, -0.00557]` |

The oracle boundary therefore wins the metric aligned with its supervision
target against every control. Fixed K=2 does not distinguish it from shuffled
or early boundaries at 95% confidence. All soft-tail variants improve early
steps over Full-H, so part of the gain is general tail regularization rather
than boundary identification alone.

## Multimodal Sampling

Full-H and Oracle FRESH were sampled with 32 chunks for each of eight held-out
contexts in every seed. Both retain 100% two-mode suffix coverage.

| Metric | Full-H | Oracle FRESH | Paired delta, 95% interval |
| --- | ---: | ---: | ---: |
| Prefix MSE | 0.06797 | **0.03792** | -0.03005 `[-0.03992, -0.02090]` |
| Prefix variance | 0.05492 | **0.03133** | -0.02359 `[-0.03145, -0.01636]` |
| Suffix variance | 0.21241 | 0.21451 | +0.00210 `[-0.00691, +0.01196]` |
| Mode balance | 0.69661 | **0.78125** | +0.08464 `[+0.02474, +0.15234]` |
| Minimum expert distance | **0.30423** | 0.33062 | +0.02639 `[+0.01520, +0.03785]` |

This is the clearest mechanism evidence: FRESH reduces unsupported variation
in the shared prefix while preserving suffix multimodality. The worse minimum
expert distance and suffix/full FM losses expose the tradeoff rather than
hiding it: tail accuracy is sacrificed when tail supervision is downweighted.

## Deterministic Control

Oracle FRESH changes held-out deterministic K=2 MSE by +0.0116 relative to
Full-H, with paired interval `[-0.1039, +0.1296]`; no degradation is detected.
The short-horizon control degrades it by +0.7833 `[+0.3310, +1.3313]`.

These controls were never optimized. They test relative non-degradation, not
absolute task competence.

## Decision

**Mechanism gate: Pass.** Real-image Pi0.5 training supports the proposal that
feedback-aware soft suffix weighting produces a cleaner common prefix without
collapsing the two possible continuations.

**Exact-boundary gate: Partial.** Oracle supervision is uniquely strongest on
oracle-prefix metrics, but fixed K=2 cannot yet separate it from shuffled and
early-boundary controls. The data support FRESH as a promising hypothesis, not
a complete proof that the exact learned boundary is necessary.

**Full algorithm gate: Hold.** Do not claim closed-loop recovery or task-success
improvement from this run. The next decisive pilot must extend expert branches
through task/recovery completion, add post-feedback replanning states, and then
measure actual closed-loop K=1/2/3 execution, recovery, premature commitment,
and task success under matched seeds.

One implementation detail matters for reproduction: LIBERO action grippers are
already in `[-1, 1]`, so the experiment configuration explicitly sets
`gripper_remap: false`. The multimodal evaluator applies the same override.

## Artifacts

- Experiment configuration: `configs/experiments/fresh_vla_libero_training.yaml`
- Training dataset: `/share/longjunyu/fresh-vla/libero-counterfactual-v1-128`
- Training/evaluation runs: `/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1`
- Machine-readable summary: `/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1/summary/results.json`
- Compact table: `/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1/summary/report.md`
- Per-run multimodal arrays: `multimodal_sampling.npz` beside each sampled checkpoint

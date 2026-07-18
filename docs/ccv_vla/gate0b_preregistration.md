# CCV-VLA Gate 0B Preregistration

Status: frozen before fitting any CCV critic.

Gate 0B is run only if Gate 0A passes. It consumes stored states from
`gate0-coupled-v3`, never simulator audit snapshots, validation episodes, original test groups, or
sealed confirmation groups.

## Data split

- Independent unit: `source_initial_state_index`.
- Engineering-excluded source: 36.
- Gate holdout sources: the six IDs frozen in the collection manifest.
- From the 23 fitting sources, rank SHA-256 of
  `ccv-vla-calibration-v1::<source_id>`; the five smallest are calibration-only and the remaining
  18 are optimizer-training sources.
- No state or branch from one source crosses these partitions.

Targets are six-repeat continuation profiles. Inputs are only frozen Pi0.5 image features, robot
state, and the first two actions of a candidate chunk.

## Models

Non-learned baselines:

1. `sample0`;
2. deterministic uniform `random`;
3. action-prefix medoid `self_consistency`;
4. six-repeat continuation Oracle.

Learned models use the identical candidate backbone:

```text
observation: Linear(4104,128) -> LayerNorm -> GELU
action:      Linear(14,64)    -> LayerNorm -> GELU
fusion:      Linear(192,128)  -> GELU -> Dropout(0.1)
```

Heads:

1. `terminal_scalar`: one sigmoid trained on formal task success;
2. `dense_scalar`: one sigmoid trained on the frozen base-8 viability utility;
3. `ccv_profile`: four conditional milestone sigmoids whose cumulative products enforce monotone
   survival, plus no-regress and progress heads.

CCV profile loss is the sum of soft-label BCE over milestone survival, no-regress BCE, progress
MSE, and a `0.2`-weighted within-state pairwise utility-ranking loss. Scalar methods use their
corresponding BCE or MSE plus the same `0.2`-weighted pairwise ranking loss on their scalar score.

## Optimization

- seeds: `[41, 42, 43]`;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- batch: 16 complete candidate sets, therefore 256 candidates;
- exactly 3000 optimizer updates per method and seed;
- gradient norm cap: 1.0;
- no early stopping and no best-checkpoint selection;
- identical source order and state minibatches for all learned methods at a seed.

Input normalization is fitted on optimizer-training sources only.

## Abstention

Each learned method proposes its highest-score candidate but retains candidate 0 unless the
predicted normalized-utility margin exceeds a threshold in

```text
[0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05].
```

On calibration sources, keep thresholds with stable-grasp harm at or below 5%. Among them choose
the threshold with highest true normalized utility; ties choose the larger threshold. This rule is
applied separately per method and seed before opening Gate holdout rows.

## Gate metrics and decision

Aggregate states inside source before paired source bootstrap. Report each seed and cross-seed
mean.

`ccv_profile` passes only if all preregistered Gate 0B conditions in
`gate0_preregistration.md` hold. In particular it must recover at least 35% of the Oracle gain over
sample 0, improve ranking regret by at least 10% over both scalar critics, and respect the 5%
stable-grasp harm limit.

Passing this offline gate authorizes a train-seed/validation-source closed-loop reranking test. It
does not itself establish an algorithm gain or a paper claim.

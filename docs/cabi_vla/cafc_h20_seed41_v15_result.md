# H=20 CAFC Seed-41 Final Result

Status: complete

Formal gate decision: `BASELINE_INVALID`

Program decision: `STOP_HORIZON_EXTENSION`

## Question

The H=10 CAFC study learned a held-out local target action effect and produced
target-directed motion, but did not complete transport or placement. This
single preregistered amendment tested whether extending every arm to H=20
would provide enough continuation support for genuine unseen source-target
migration at fixed deployment `K=3`.

## Integrity checks

- v15 contains the same 22,464 records and ordering as v13.
- All 828 image/wrist/state anchors are exactly unchanged.
- The 276 observed action anchors are re-sliced to `20 x 7`.
- `white-right` and `yellow_white-left` contribute no action labels.
- All four arms completed exactly 33,000 sequential updates from seed 41.
- Both CAFC arms sampled 1,013 finite CAFC batches.
- Model and data horizons are both 20 in every checkpoint.
- Deployment and all gate episodes use fixed `K=3`.

## State-0 closed loop

| Arm | Observed task | Held-out task | Observed source | Held-out source | Mean progress |
|---|---:|---:|---:|---:|---:|
| BC-H10 | 1/4 | 0/2 | 1/4 | 0/2 | 0.208 |
| Bridge-H10 | 2/4 | 0/2 | 2/4 | 0/2 | 0.417 |
| CAFC-H10 | 1/4 | 0/2 | 2/4 | 2/2 | 0.458 |
| Bridge+CAFC-H10 | 2/4 | 0/2 | 3/4 | 2/2 | 0.625 |
| **BC-H20** | **2/4** | **0/2** | **2/4** | **0/2** | **0.375** |
| **Bridge-H20** | **4/4** | **0/2** | **4/4** | **0/2** | **0.667** |
| **CAFC-H20** | **1/4** | **0/2** | **1/4** | **1/2** | **0.333** |
| **Bridge+CAFC-H20** | **2/4** | **0/2** | **2/4** | **2/2** | **0.583** |

Bridge-H20 establishes a valid observed baseline and shows that H=20 itself
improves task fitting. Neither CAFC arm retains the preregistered 3/4 observed
success threshold, and neither solves a held-out task. Full validation,
seeds 42/43, and sealed test were therefore not opened.

The formal machine decision uses `BASELINE_INVALID` to mean that no CAFC method
has a valid observed exact comparison. It does not mean the benchmark is
unlearnable: Bridge-H20 reaches 4/4 observed success.

## Mechanism diagnostics

At the H=20 target-selection anchors:

| Arm | Target-effect cosine | Target instruction sensitivity | Teacher chunk MSE |
|---|---:|---:|---:|
| BC-H20 | 0.304 | 0.567 | 0.0203 |
| Bridge-H20 | 0.075 | 0.554 | 0.0171 |
| CAFC-H20 | **0.961** | 0.546 | 0.0188 |
| Bridge+CAFC-H20 | **0.963** | 0.557 | **0.0155** |

CAFC therefore learns the intended local held-out target effect, and every arm
remains target-instruction-sensitive. The failure is not target-language
collapse or a broken model load.

The post-hoc exact teacher-prefix handoff diagnostic is not a formal migration
metric. It isolates continuation from the first target-transport state:

| Arm | White-right final XY | Yellow-left final XY | Transport | Task |
|---|---:|---:|---:|---:|
| BC-H20 | 0.590 | 0.593 | 0/2 | 0/2 |
| Bridge-H20 | 0.591 | 0.599 | 0/2 | 0/2 |
| CAFC-H20 | 0.570 | 0.153 | 0/2 | 0/2 |
| Bridge+CAFC-H20 | **0.121** | **0.140** | 0/2 | 0/2 |

Bridge+CAFC-H10 previously reached `0.119/0.142`. Its H=20 mean distance
changes by less than `2e-5`, with no new transport or task success. Extending
the predicted chunk does not extend closed-loop continuation because only
three actions are executed before the policy is queried at a new held-out
state. CAFC supplies no valid completion constraint at those off-anchor states.

## Decision

There is no evidence of genuine end-to-end compositional migration. H=20
improves the exact non-CAFC Bridge control while CAFC reduces observed success,
does not produce held-out success, and does not move the handoff plateau. This
rules out insufficient ten-step target length as the main failure mechanism.

Per the frozen amendment:

- stop CAFC horizon extension;
- do not run H=30/40/50 or trajectory completion;
- do not report the local cosine or handoff distance as task migration;
- do not spend seeds 42/43 or sealed-test budget on this configuration; and
- preserve CAFC as a negative result and diagnostic, not a candidate headline
  method.

Any replacement method must supervise or constrain state-conditioned
compositional behavior beyond a single matched anchor without using hidden
fourth-edge actions. That is a new research proposal, not another CAFC patch.

## Artifacts

- Formal decision: `/share/longjunyu/cabi-vla/comparisons/cafc_h20_migration_seed41_v15_orchestration.json`
- Machine summary: `docs/cabi_vla/results/cafc_h20_seed41_v15.json`
- H=20 view: `/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v15-decision-observed-edge-phase-loss-balanced-h20`
- Checkpoints: `/share/longjunyu/cabi-vla/runs/*h20*steps33000*v15/final_model`
- State-0 results: `/share/longjunyu/cabi-vla/evaluations/*h20_33000_s41_v15_train0_k3`
- Offline results: `/share/longjunyu/cabi-vla/offline-evaluations/*h20_33000_s41_v15`
- Target diagnostics: `/share/longjunyu/cabi-vla/diagnostics/*h20_33000_s41_v15_target_select`
- Handoff summary: `/share/longjunyu/cabi-vla/comparisons/cafc_h20_target_handoff_seed41_v15.json`
- Paired H.264/AV1 videos: `/share/longjunyu/cabi-vla/comparisons/cafc_h20_*_paired_h264_av1`


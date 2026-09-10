# CAFC Seed-41 v14 Result

Status: formal H=10 calibration gate complete; validation not opened

## Frozen result

Both CAFC arms completed 33,000 updates from the same Pi0.5 initialization on
the 22,464-record v13 view. The preregistered state-0, fixed-`K=3` gate failed:

| Arm | Observed task success | Held-out task success | Held-out source selection |
|---|---:|---:|---:|
| CAFC | 1/4 | 0/2 | 2/2 |
| Bridge+CAFC | 2/4 | 0/2 | 2/2 |

The formal decision is `BASELINE_INVALID`, because neither arm retained the
required 3/4 observed successes. Seeds 42/43 and sealed validation/test were
therefore not run.

## What did transfer

This is not a null representation result. At the exact target-selection
anchor, Bridge+CAFC learned the held-out ten-step target effect with cosine
`0.9969`; the corresponding cosines for BC, Action Bridge, and Bridge+Closure
were `0.118`, `0.105`, and `0.003`. Same-image instruction diagnostics also
showed normal target sensitivity (`0.561`), excluding instruction collapse.

A post-hoc teacher-prefix handoff diagnostic replayed the supervised source
prefix exactly, then gave the policy the held-out target instruction at the
first transport state. This diagnostic is not an end-to-end score.

| Arm | White to right final XY distance | Yellow to left final XY distance |
|---|---:|---:|
| BC | 0.585 | 0.593 |
| Action Bridge | 0.583 | 0.598 |
| Bridge+Closure | 0.585 | 0.594 |
| CAFC | 0.571 | 0.284 |
| Bridge+CAFC | **0.119** | **0.142** |

Bridge+CAFC consequently produced a real target-directed physical effect on
both missing edges, but neither episode reached transport or placement
success. Videos are encoded in both H.264 MP4 and AV1 WebM under
`/share/longjunyu/cabi-vla/target-handoff-diagnostics`.

## Failure diagnosis

The v14 completion target covers one ten-step action chunk at the
pre-transport anchor. The teacher has 68--85 actions remaining at this point.
After each fixed three-action deployment chunk, the policy is queried at a
new, off-anchor state for which CAFC supplied no completed action field. The
videos show correct initial routing followed by stall or early release.

This supports a sparse/truncated completion diagnosis. It does not support a
successful migration claim, and it does not justify trajectory-level
counterfactual completion: once transport begins, the physical conditionings
of the observed corners diverge and the clean tetrad identity no longer holds.

## Bounded amendment

Disclosed state-0 teacher QA was used only to select one final feasibility
amendment. A 20-step completion remains directional on both missing edges
(cosine `0.918/0.913`, correct gripper sign), while 30--50 steps deteriorate;
at 50 steps cosine falls to `0.586/0.525` and gripper agreement to
`0.74/0.68`. The next experiment is therefore fixed at H=20 for BC, Bridge,
CAFC, and Bridge+CAFC alike. No H=30/40/50 sweep is permitted.

## Artifacts

- Formal decision: `/share/longjunyu/cabi-vla/comparisons/cafc_action_field_migration_seed41_v14_orchestration.json`
- State-0 evaluations: `/share/longjunyu/cabi-vla/evaluations/*seed41_v14_train0_k3`
- Offline evaluations: `/share/longjunyu/cabi-vla/offline-evaluations/*seed41_v14`
- Target handoff diagnostics: `/share/longjunyu/cabi-vla/target-handoff-diagnostics`
- H.264/AV1 paired videos: `/share/longjunyu/cabi-vla/comparisons/*h264_av1`


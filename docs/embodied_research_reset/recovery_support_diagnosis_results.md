# Recovery Support Diagnosis Results

## Result

The corrected 320-action expert-handoff gate completed on clean commit
`1cfc85a308912ac9c3aa6706aeb3defc45fd6a1f` for all three Full-H checkpoints,
13 validation groups per seed, and nine independent source initial states.

The first sufficient correction segment is **feedback to stable regrasp**. A
3-action or 12-action teacher prefix does not help, while handing the policy a
stable regrasp state changes full-task success from 21.85% to 85.19%.

| Handoff | Absolute success | Paired change vs policy-only | 95% paired source-cluster CI |
|---|---:|---:|---:|
| policy-only | 21.85% | - | - |
| teacher 3 actions | 15.56% | -6.30 pp | [-10.37, -1.85] |
| teacher 12 actions | 15.19% | -6.67 pp | [-14.44, +0.37] |
| teacher to stable regrasp | 85.19% | +63.33 pp | [+49.62, +74.07] |
| teacher to stable lift | 88.52% | +66.67 pp | [+61.11, +72.96] |
| teacher to transport | 98.89% | +77.04 pp | [+71.48, +82.96] |
| full teacher | 100.00% | - | [100.00, 100.00] absolute |

The regrasp gain is positive for seeds 41, 42, and 43: +66.67, +73.33, and
+50.00 percentage points. It exceeds the preregistered +20-point threshold,
its paired interval excludes zero, and all seeds agree in direction. The mean
teacher prefix is 80.5 actions to stable regrasp, compared with 121.1 actions
for full completion.

## Interpretation

This result rules out a generic short-horizon fix. The base policy can complete
the downstream task from a stable regrasp state, but its own support does not
reliably traverse the feedback-to-regrasp interval. The current bottleneck is
therefore recovery-state action support and compounding state drift, not an
exact Oracle suffix boundary.

It does not yet establish a new method. Direct continuation training,
stage-balanced replay, and policy-state correction SFT remain the required
ordered controls. If a simpler control solves the problem, that control is the
answer.

## Validity And Artifacts

- Full teacher success: 39/39 rows and 100% at the source-cluster level.
- Feedback reconstruction: exact policy pixels, robot-state error at most
  `1e-6`, sim-state error at most `1e-10`, and gripper error at most `1e-6`.
- Method-order audit: zero change in first policy chunk, images, and state.
- Videos: 273/273 expected files, 55,807/55,807 decoded frames, H.264 `avc1`,
  `yuv420p`, 448x224, faststart, nonblank, and non-static.
- Raw results: `/share/longjunyu/fresh-vla/research-reset/recovery-expert-handoff-v2`.
- Statistical summary: `recovery_expert_handoff_summary.json` under that root.
- Artifact and media audits: `artifact_audit.json` and
  `video_artifact_audit.json` under that root.

## Frozen Next Gate

All arms start from the same original seed-specific Full-H final checkpoint.
Because optimizer state was not retained, every arm uses the same weights-only
restart, learning rate, scheduler, batch size, and frozen modules.

1. Train Base continuation for 6,902 extra updates and evaluate validation only.
2. Freeze 6,902 updates if K=3 attached success averages at least 30%, at least
   two seeds reach 20%, and no seed has non-finite training or evaluation.
3. Otherwise train Base from the original checkpoint for 13,804 updates. If it
   still misses that gate, report `BASELINE_INVALID_OR_DATA_INSUFFICIENT` and do
   not compare recovery methods.
4. Once the budget is frozen, train stage-balanced replay and policy-state
   recovery SFT for exactly the same updates, then open test once for all arms.

The recovery SFT data may use privileged physics only to trigger collection and
audit milestones. Policy inputs remain the deployable images, language, and
existing proprioception. The correction target ends at the first stable
regrasp milestone; adding lift or transport labels is a later ablation, not the
primary method.

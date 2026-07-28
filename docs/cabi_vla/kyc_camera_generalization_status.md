# KYC Camera Generalization Status

Last updated: 2026-07-28.

## Corrected Scope Of The Existing Result

The completed study supports only:

`KYC_INCREMENTAL_GAIN_NOT_OBSERVED_ON_PI05_LIBERO_BIND_FIXED_SCENE_WRIST_ON_HIGH_VIEW`

It does not support a general KYC failure claim. The original released
experiments use multiple training camera poses, intentionally exclude wrist
images, and predict the largest gain when static scene cues are suppressed or
view data are scarce.

## Audit Findings

- A wrist camera moves with the robot over time, but is invariant to relocation
  of the external camera. It is therefore a stable robot-centric side channel
  in the current intervention.
- Moving only the external camera changes pixels but preserves a deterministic
  relation between camera pose and the fixed table, robot base, and room.
- The released randomized RoboSuite environment hides fixed walls, floor,
  table, and robot-base visuals, then moves an independent visual table by
  `+/-0.2 m` with full yaw and a visual floor by `+/-2.0 m` with full yaw.
- The released scaling experiment uses a shared candidate catalog
  `n={10,22,45,100,215,445,1000}` and samples a camera on the fly. It is not a
  fixed six-pose-per-episode dataset.
- The released ACT and SmolVLA paths use late feature/token fusion. The released
  diffusion-policy path uses an unpretrained nine-channel early encoder.

## Implemented Controls

- nested global camera catalogs and three deterministic epoch replicas;
- exact RGB matching between RGB, Control, and KYC at each budget;
- true wrist token masks for training and evaluation;
- render-only LIBERO table/floor/room/base interventions;
- background-only pose leakage probe;
- correct, canonical, and mismatched ray interventions with fixed RGB;
- persistent official ACT and Pi0.5 launchers;
- group-preserving fixed camera gate remains unchanged.

## Validated Diagnostics

### Scene-cue intervention

Artifact:

`/share/longjunyu/cabi-vla/kyc-scaling-v3/diagnostics/camera_pose_leakage_v2.json`

On 2,080 background-only renders with held-out physical states and held-out
scene seeds:

| Condition | 13-way pose accuracy | Mean positive R2 |
|---|---:|---:|
| Fixed scene | 100.00% | 0.993 |
| Cue randomized | 60.68% | 0.656 |

Chance accuracy is 7.69%. Pose-classification advantage over chance fell
42.59%, and mean positive `R2` fell 33.88%. Simulator state changed by exactly
zero. This passes the preregistered cue-suppression gate while showing that
camera pose remains partly inferable from perspective geometry.

### Old-checkpoint ray use

Artifact:

`/share/longjunyu/cabi-vla/kyc-scaling-v3/diagnostics/kyc_seed41_ray_use_v1.json`

Across 48 paired `task x snapshot x visual-pose` inputs:

| Intervention | Chunk RMS vs correct | First-action RMS | Cosine similarity |
|---|---:|---:|---:|
| Canonical rays | 0.000710 | 0.000824 | 0.9999988 |
| Mismatched rays | 0.000953 | 0.001076 | 0.9999983 |

The old high-view policy is not mathematically invariant to calibration, but
its action response is extremely small. Closed-loop ray ablation is therefore
secondary to retraining under the preregistered low-view conditions.

### Released ACT positive-control smoke

- official repository commit: `e0647105`;
- official RoboSuite commit: `0df3a5f`;
- official Lift demonstrations: loaded and replay-rendered;
- image-only ACT: one complete train/eval smoke;
- Plucker-conditioned ACT: one complete train/eval smoke;
- MuJoCo fixed to `3.3.7`; `3.11.0` is incompatible because it removed the
  `MjData.qM` field used by the pinned RoboSuite controller.

No paper reproduction claim is made from the smoke. The full three-seed
released-code comparison remains required.

## Next Gates

1. Build fixed-scene Pi0.5 views for `n={10,45,215,1000}`.
2. Run Stage B1 seed 41 scaling.
3. Select the factorial budget by the preregistered baseline rule.
4. Run fixed/cue-randomized by wrist-on/off Control/KYC factorial.
5. Confirm required cells with seeds 42 and 43.
6. Run the released ACT positive control and keep its conclusion separate.


# KYC Camera Conditioning on LIBERO-Bind: Preregistered Validation

## Question

Can the camera-conditioned late-fusion design from *Do You Know Where Your
Camera Is?* make a Pi0.5 policy robust to external-camera pose changes without
reducing canonical-view task success?

The experiment tests **view robustness**, not inference-time camera search.
Only `agentview` moves; the wrist camera, field of view, language, action
execution, initial states, and task physics remain fixed.

## KYC Adaptation

The adapted KYC branch follows the released late-fusion topology:

- OpenCV intrinsics and camera-to-world extrinsics;
- per-pixel Plucker coordinates `[direction, origin x direction]`;
- five-layer `6 -> 64 -> 128 -> 256 -> 512 -> 512` ray CNN;
- pooling to the SigLIP token grid and `512 -> 1152` projection;
- no-affine normalization of RGB and ray tokens;
- concatenation followed by `2304 -> 1152` fusion before the PaliGemma
  multimodal projector.

The branch has 7,172,736 parameters, matching the released topology. This is
an AlphaBrain/Pi0.5 adaptation rather than an exact robot/data reproduction:
the original SmolVLA policy head and benchmark are not used, and only the
external view is camera-conditioned.

The released KYC repository is internally inconsistent about Plucker channel
order: its README snippet uses `[direction, origin x direction]`, while the
robosuite executable uses the reverse concatenation. This experiment froze the
README convention before training. Since the ray CNN is initialized and
trained from scratch, the two definitions differ only by a fixed input-channel
permutation; the result is a topology-level reproduction, not a bitwise
reproduction of the robosuite source.

## Arms

| Arm | Randomized RGB | Camera branch | Ray input | Role |
|---|---:|---:|---|---|
| Base | no | no | none | original fixed-camera checkpoint |
| PoseAug-RGB | yes | no | none | image-augmentation control |
| PM-Fixed | no | yes | canonical | added-module control |
| PoseAug-Control | yes | yes | canonical | matched placebo |
| KYC | yes | yes | measured | primary method |

The primary causal comparison is **KYC vs PoseAug-Control**. Their training
records, RGB images, crop augmentation, checkpoint, optimizer, update count,
and seed are identical; only the ray metadata differs.

## Training

- Initialization: Bridge-H20 Pi0.5 seed-41 checkpoint.
- Training set: four action-supervised LIBERO-Bind edges.
- View randomization: six fixed camera poses per episode, sampled over
  azimuth `[-60, 60] deg`, elevation `[-25, 25] deg`, and radius
  `[0.90, 1.25]x`.
- Records: 22,464 matched action windows and 784 unique camera poses.
- Budget: 33,000 updates, identical optimizer and data order.
- Matched primary seeds: 41, 42, and 43.
- Context controls: seed 41.

No validation or test rollout selects a checkpoint or training budget.

## Evaluation

All policy comparisons use fixed `K=3`, 320 environment steps, identical flow
seeds, the four action-supervised tasks, and unseen test snapshots 40--49.

### Dense response

Seed 41 KYC and PoseAug-Control are evaluated on state 40 at 35 one-factor
poses:

- azimuth: `-30` to `30 deg` in `5 deg` increments;
- elevation: `-15` to `15 deg` in `3 deg` increments;
- radius: `0.85` to `1.15x` in `0.025x` increments.

This curve is descriptive and is not used to select a favorable test pose.

### Fixed robustness gate

All arms are evaluated at 13 frozen poses:

- baseline;
- azimuth `{-90, -60, 60, 90} deg`;
- elevation `{-32, -25, 25, 32} deg`;
- radius `{0.75, 0.90, 1.25, 1.40}x`.

The `60 deg`, `25 deg`, `0.90x`, and `1.25x` points lie on the training
support boundary. The remaining points probe extrapolation near geometric
visibility transitions.

### Visibility audit

An independent renderer scans 78 poses for every `edge x test snapshot`
combination: 4,680 geometric records in total. Isolated-object guard renders
measure:

- center inside/outside the sensor;
- geometric clipping fraction;
- visible pixels;
- visible 14x14 patch support;
- external occlusion;
- geometric out-of-sensor events from isolated-object projection;
- complete external occlusion when the projection remains in sensor.

Policy episodes are joined to these records before aggregation. Primary
view-robustness claims use the fully supported stratum: both task objects have
at least 64 visible pixels and four visible patches, both centers are in frame,
and neither object is at least 50% geometrically clipped. Severe clipping,
center-out, below-support, and disappeared episodes are reported separately.

## Metrics And Decision

Primary metrics:

- full-task success;
- grasp, lift, transport, and placement success;
- progress;
- capped completion steps, with failures assigned the 320-step budget;
- success degradation relative to each method's canonical view.

Inference units are test snapshot groups, not frames. Comparisons use paired
group bootstrap 95% confidence intervals and report each matched seed.

`KYC_REPRODUCED` requires all of:

1. KYC improves fully-supported gate success over PoseAug-Control by at least
   10 percentage points on average across seeds, or its paired interval
   excludes zero.
2. KYC is consistently better than PoseAug-RGB on the same supported poses.
3. KYC canonical success is no more than 5 points below PoseAug-Control.
4. The improvement is behavioral: transport or task success improves, not
   only image-feature or action-MSE diagnostics.

If KYC and PoseAug-Control are indistinguishable, camera metadata has not shown
incremental value over matched visual augmentation. If both outperform Base
but KYC does not outperform PoseAug-RGB, the result supports augmentation, not
camera conditioning. Failures after visual support is lost are boundary
findings and cannot by themselves reject KYC.

Recorded rollouts are delivered as AV1/WebM with contact sheets.

Before results were observed, criterion 2 was operationalized as a strictly
positive paired supported-pose success delta for KYC seed 41 versus
PoseAug-RGB seed 41; a tie fails this contextual criterion. The primary
KYC-versus-PoseAug-Control interval independently cross-resamples fine-tuning
seeds and held-out snapshot groups. Displayed marginal means use the same
equal-seed, equal-state weighting as the primary delta.

The resulting interval is conditional on one Bridge-H20 seed-41 initialization,
one rollout/flow seed, the four selected tasks, and the frozen pose set; it is
not an interval over base checkpoints, tasks, or arbitrary camera poses.
`Fully supported` describes the settled initial observation and does not imply
that both objects remain fully visible throughout a 320-step rollout.

The primary KYC versus PoseAug-Control contrast identifies the incremental
value of measured, pose-varying calibration over an otherwise matched
canonical-ray branch. Without a shuffled or mismatched-ray arm, it does not by
itself identify Plucker geometry as uniquely better than every possible
pose-varying side channel.

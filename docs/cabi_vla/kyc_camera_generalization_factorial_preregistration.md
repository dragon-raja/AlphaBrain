# KYC Camera Generalization: Scaling And Factorial Validation

Status: preregistered before the experiments in this document.

## Motivation

The completed AlphaBrain gate tested one specific setting:

- Pi0.5 with a pretrained PaliGemma vision-language backbone;
- a fixed LIBERO room, table, robot base, and background;
- a wrist-camera stream in addition to the perturbed external view;
- 22,464 action windows rendered from 784 realized external poses;
- KYC compared with a capacity-matched canonical-ray control.

In that setting, KYC did not improve success over the matched control. This
does not reproduce or refute the original paper. Static scene geometry can
make camera pose inferable from RGB, the wrist stream is invariant to an
external-camera intervention, and the original paper predicts the largest
conditioning gain in low-view, cue-randomized settings without wrist images.

This study separates three questions:

1. Can the released KYC code reproduce its own smallest positive control?
2. Does explicit camera geometry reduce Pi0.5's required number of training
   viewpoints?
3. Is KYC's Pi0.5 value moderated by static scene cues or wrist-camera input?

The three questions have separate conclusions. Failure to reproduce the
official ACT control is not evidence about Pi0.5, and a Pi0.5 transfer failure
is not evidence that the paper's ACT result is false.

## Terminology

`Control` is the existing PoseAug-Control arm: randomized RGB, the same
camera-conditioning branch and parameter count as KYC, but a constant
canonical ray map.

`KYC` is the same policy and data with measured intrinsics and extrinsics.

`RGB` is PoseAug-RGB: the same randomized images without a camera branch.

`Wrist on` uses Pi0.5 image mask `[true, true, false]`.

`Wrist off` uses image mask `[true, false, false]`. The wrist image may remain
in the record, but its tokens are excluded from attention in both training and
evaluation.

`Fixed scene` preserves the current LIBERO visual geometry.

`Cue-randomized scene` changes only render geometry and appearance that are
independent of task physics: the visual table layer, floor texture frame,
visual-only room/wall assets, robot-base visibility, and lighting. It never
moves task objects, collision geometry, the robot kinematic chain, camera
calibration, or action/state coordinates.

## Track A: Released-Code Positive Control

Run the smallest official RoboSuite experiment from the released repository:

- task: ACT Lift randomized;
- arms: image-only and Plucker-conditioned;
- official pinned RoboSuite fork and released data;
- seeds: 0, 1, 2;
- released camera files, transforms, optimizer, epoch count, and evaluation;
- no wrist camera.

The primary result is the paired success-rate difference reported by the
released evaluator. Exact reproduction means the aggregate direction is
positive and the magnitude is compatible with the paper after accounting for
the finite evaluation sample. Any compatibility interval and deviations from
the released environment must be reported.

This track uses the executable code's ray convention. AlphaBrain's conceptual
OpenCV `[direction, moment]` convention is not silently substituted.

## Track B: Pi0.5 View-Count Scaling

### Camera catalogs

Create one deterministic master catalog over the existing safe LIBERO support:

- azimuth: `[-60, 60] deg`;
- elevation offset: `[-25, 25] deg`;
- radius: `[0.90, 1.25]x`;
- no forced canonical sample.

All smaller catalogs are prefixes of the master catalog. Candidate budgets are
`n = {10, 22, 45, 100, 215, 445, 1000}`. A record deterministically selects
one member of the current catalog using its sample identity and epoch replica.
The action records, record order, update count, and physical states are
identical across budgets and arms. RGB, Control, and KYC at a given budget use
the exact same rendered pixels.

Three deterministic epoch replicas are rendered so a physical state can see a
different catalog member when training revisits the data. This approximates
the released on-the-fly camera sampling while keeping the Pi0.5 data pipeline
auditable.

### Staged training

Stage B1 runs seed 41 for `n = {10, 45, 215, 1000}`:

- Control and KYC at all four budgets;
- RGB at `n = {10, 45}`;
- the completed high-view RGB result remains descriptive, not a matched B1
  endpoint.

Stage B2 confirms seeds 42 and 43:

- if any B1 KYC-Control gain is at least 5 percentage points, confirm that
  budget and its nearest tested neighbor;
- otherwise confirm `n = {10, 45}`, where the released paper predicts the
  largest data-efficiency effect.

The qualifying budget is the one with the largest positive seed-41 gain,
breaking ties toward the smaller catalog. Its neighbor minimizes absolute
log-view-count distance, again breaking ties toward the smaller catalog. The
mechanically selected Track C budget is added to the Stage B2 training set if
it is not already present, so every factorial cell has matched seeds 41, 42,
and 43.

No budget is selected by test-state performance. Stage transitions use the
frozen gate states and thresholds above.

### Estimands

Primary:

- KYC minus Control success at each view budget;
- view budget required to reach 80% of each method's maximum observed success;
- area under the success-versus-log-view-count curve.

Secondary:

- KYC minus RGB;
- grasp, lift, transport, placement, progress, and completion steps;
- canonical-view preservation;
- correct-ray versus canonical-ray versus mismatched-ray action sensitivity.

## Track C: Scene-Cue By Wrist Factorial

Use the first low or middle Track B budget at which Control learns the task
but has not saturated. This selection is mechanical:

1. choose the smallest budget with Control success at least 20%;
2. if every budget is below 20%, the Pi0.5 scaling baseline is invalid;
3. if the smallest budget already exceeds 70%, use `n=10`.

At that frozen budget, run the complete seed-41 factorial:

| Scene | Wrist | Control | KYC |
|---|---|---:|---:|
| fixed | on | yes | yes |
| fixed | off | yes | yes |
| cue-randomized | on | yes | yes |
| cue-randomized | off | yes | yes |

Confirm seeds 42 and 43 for the complete factorial if either preregistered
interaction is at least 5 percentage points in seed 41. Otherwise confirm the
two wrist-off cells, which most closely match the paper.

The primary factorial evaluation is matched-distribution: fixed-scene
policies are evaluated in the fixed scene and cue-randomized policies in
cue-randomized scenes. Every seed-41 policy is additionally evaluated in the
opposite scene as a secondary cross-scene control. This separates training
augmentation from test-time cue removal; cross-scene results do not replace
the matched primary estimands. Seeds 42 and 43 confirm only matched cells.

The 5-point interaction trigger uses absolute interaction magnitude. If
triggered, seeds 42 and 43 confirm all four matched cells. Otherwise they
confirm fixed/cue-randomized wrist-off cells only. Fixed-scene wrist-on
checkpoints and rollouts are reused from Track B rather than rerun.

Primary interactions:

```text
wrist interaction =
  (KYC - Control)[wrist off] - (KYC - Control)[wrist on]

scene interaction =
  (KYC - Control)[cue randomized] - (KYC - Control)[fixed]
```

The three-way descriptive contrast asks whether the largest KYC gain occurs
with cue randomization and wrist off.

## Scene-Cue Validity Gate

Moving the camera changes background pixels but does not remove the shortcut:
in a fixed scene, the projection of the same table, robot base, and walls
remains a deterministic function of camera pose.

Before Track C training, a background-only pose probe must compare fixed and
cue-randomized scenes:

- render the same held-out physical states and camera catalog;
- retain only known background/table visual geoms using segmentation;
- split by physical snapshot and scene-randomization seed;
- train a fixed linear probe to classify discrete camera pose and regress
  azimuth, elevation, and radius;
- report held-out accuracy, balanced accuracy, and regression `R^2`.

Cue randomization passes if pose classification advantage over chance or mean
positive `R^2` falls by at least 25% relative to fixed scene, without changing
robot state, object pose, task success, camera matrices, or replay actions.
If the gate fails, scene randomization is strengthened before policy training.

## Ray-Use Diagnostic

Before new Pi0.5 training, evaluate completed KYC checkpoints with identical
RGB, state, language, flow seed, and weights under:

1. correct measured rays;
2. canonical rays;
3. a deterministic mismatched pose's rays.

Report action-chunk RMS, first-action RMS, cosine similarity, and maximum
absolute difference. Closed-loop ray interventions are run only if offline
actions change measurably.

- no action change means the trained policy ignored the camera branch;
- action change without success change means geometry was used but not useful;
- correct-ray behavioral improvement identifies calibration-specific value.

## Evaluation And Statistics

All Pi0.5 arms use:

- the same initialization, optimizer, batch size, updates, crop, and data
  order within a seed;
- fixed `K=3`, 320 steps, flow seed, test snapshots, and camera gate;
- identical wrist masking in training and evaluation;
- no checkpoint selection on validation or test rollout.

Main inference units are held-out snapshot groups. Report per-seed results,
equal-seed means, paired group bootstrap 95% confidence intervals, and
absolute percentage-point differences. Do not treat poses or frames from the
same snapshot as independent replicates.

Primary metrics are full-task success and transport success on the fully
visible pose stratum. Secondary metrics include subgoals, progress, completion
steps, out-of-support behavior, and view-axis breakdowns.

## Decision Rules

`KYC_PI05_SUPPORTED` requires:

- positive KYC-Control gain of at least 10 points at a preregistered view
  budget, or a paired interval excluding zero;
- positive gain over RGB at the same budget;
- no more than 5 points canonical-view regression;
- confirmation across seeds;
- a behavioral, not action-MSE-only, improvement.

`KYC_DATA_EFFICIENCY_ONLY` applies when KYC lowers the view count needed for a
fixed success level but converges to the same high-view success.

`KYC_CONTEXT_DEPENDENT` applies when KYC gains are confirmed only with wrist
off or scene cues suppressed.

`KYC_INCREMENTAL_GAIN_NOT_OBSERVED_ON_PI05` applies only if the scaling and
factorial gates are valid and KYC does not beat matched controls.

`OFFICIAL_REPRODUCTION_INCONCLUSIVE` is used for an invalid dependency,
environment, data, or baseline run. It must not be converted into a negative
paper claim.

The existing result is renamed:

`KYC_INCREMENTAL_GAIN_NOT_OBSERVED_ON_PI05_LIBERO_BIND_FIXED_SCENE_WRIST_ON_HIGH_VIEW`.

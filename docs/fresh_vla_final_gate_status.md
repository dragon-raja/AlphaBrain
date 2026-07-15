# FRESH-VLA Final Gate Status

Audit date: 2026-07-15 UTC

Branch: `exp/fresh-vla-toy-v0`
Audit base: `a374582`
Implementation commit: `3f356ed`

## Complete Data

- Full episodes: `/share/longjunyu/fresh-vla/libero-full-episode-v2-128`
- Sliding windows: `/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128`
- 128 snapshot groups, split by group into 102 train, 13 validation, and 13 test groups.
- Source initial states are disjoint across splits: 30 train, 9 validation, and 9 test states. The v1 split is retained only for pilot provenance because it reused source initial states across splits.
- 256 complete branch episodes: 128 attached and 128 forced-slip/recovery.
- Expert formal LIBERO success: 128/128 attached and 128/128 slipped.
- 128 paired full-episode videos and 256 branch videos are present. The first 20 pairs and the aggregate contact sheet passed visual review.
- Pre-reveal paired image, state, and action equality checks pass. Feedback becomes visible at the intervention, and all post-feedback windows restore horizon 10.
- Window quality report passes all eight checks, including group-preserving split, loss-only oracle labels, and exact sample-level Oracle/Shuffled-Oracle marginal matching within each split.
- 34,551 total windows: 27,607 train, 3,394 validation, and 3,550 test.
- `label_revision.json` records the sample-marginal fix and the prior labels are retained as `training_labels.before_sample_marginal_fix.json`. The completed Full-H calibration is mathematically unaffected because its tail weight is exactly 1.0; all weighted methods used the revised labels.

The data collection and post-feedback window construction are complete and must not be regenerated unless their quality checks fail.

## Existing Pilot Training

All six methods and seeds 41, 42, and 43 have a valid 2,400-step checkpoint under:

`/share/longjunyu/fresh-vla/runs/libero-full-episode-v1`

Methods: `full_h`, `random_soft010`, `shuffled_oracle_soft010`, `gripper_soft010`, `oracle_soft010`, and `short_h`.

These checkpoints are retained as pilot evidence, not final-gate checkpoints. The PaliGemma DataLoader used sequential ordering, so each run saw only the first 2,400 windows from nine of 102 training groups (0.09 epoch). The final gate uses the leak-free v2 split, deterministic seeded shuffling, and a common budget that covers the complete training set.

## Existing Evaluation

- Offline action metrics: complete for 18/18 pilot checkpoints.
- Suffix mode coverage: complete for 18/18 pilot checkpoints.
- Deterministic reach control: complete for 18/18 pilot checkpoints.
- Group-level auxiliary reports:
  - `libero-full-episode-v1/episode_offline_summary`
  - `libero-full-episode-v1/deterministic_reach_summary`
- Complete structured isolated recovery evaluation: 0/18 pilot checkpoints.
- Complete structured end-to-end evaluation: 0/18 pilot checkpoints.

Earlier Full-H and Oracle isolated K=1 attempts were interrupted after zero successes and predate atomic partial output. They are not final results and are not used for method selection.

## Final Gate Complete

The analysis rule is fixed before viewing final test results: K=3 is the primary commitment setting, K=2 is supporting evidence, and K=1 is a negative control. The final decision script applies the stated 10-point Oracle-vs-Full and 5-point control-effect thresholds together with normal-task, behavior-error, and baseline-validity gates. It does not select a checkpoint, seed, or execution horizon after seeing test performance.

All methods use a 320-step end-to-end timeout. The held-out scripted expert requires at most 252 steps on the slip branch, so this budget provides a fixed 27% margin without changing between methods, seeds, or K.

The final gate completed the following protocol:

1. Full-H passed the validation-only attached-success gate and fixed a common budget of 27,607 updates.
2. All six methods used identical initialization, seeded data order, optimizer, batch size, horizon, frozen modules, and update count for seeds 41, 42, and 43.
3. All 18 checkpoints completed isolated recovery, event-triggered end-to-end, deterministic reach, and offline evaluation for fixed K=1, K=2, and K=3 on all 13 held-out groups.
4. The run produced 1,404 paired closed-loop videos, covering every method, seed, K, evaluation type, and test group.
5. Group-level paired bootstrap intervals average seeds within each snapshot group; frames and windows are never treated as independent statistical samples.
6. The final artifact audit passes for 18/18 checkpoints and every required evaluation output.

The persistent `run_libero_final_gate.sh` controller completed at `2026-07-15T02:39:31Z`. Its stage log is under `final_gate_logs/pipeline.log`.

The preregistered result is `STOP_TRAINING_WEIGHTING_ROUTE`. At primary K=3, Oracle FRESH did not improve slip recovery over Full-H and did not beat Random, Shuffled Oracle, Gripper, or Short-H on the required closed-loop success criteria. Offline common-prefix MSE improved, but suffix mode coverage collapsed and the gain did not transfer to full-task recovery.

Deterministic reach and offline MSE remain auxiliary controls. They did not independently determine the decision. Failure-continuation and premature-commitment are distinct predicates but coincide on all 221 eligible final-gate rows, so they are not counted as independent evidence.

Final outputs:

- `docs/fresh_vla_final_decision.md`
- `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/final_decision.json`
- `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/closed_loop_summary`
- `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/episode_offline_summary`
- `/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2/deterministic_reach_summary`

## Problem-First Follow-up

The `STOP_TRAINING_WEIGHTING_ROUTE` decision remains unchanged for exact
FRESH suffix weighting. A separate causal sufficiency diagnostic subsequently
completed on clean commit `1cfc85a` with a corrected 320-action budget and
reconstructed teacher/controller state.

Policy-only success from the slipped feedback state is 21.85%. Teacher handoff
to stable regrasp raises it to 85.19%, a paired +63.33 percentage points with a
source-cluster bootstrap 95% CI of `[+49.62,+74.07]`; 3- and 12-action teacher
prefixes do not help. This supports a recovery-state action-support bottleneck,
not the discarded exact weighting teacher. The next comparison is Base
continuation versus clean feedback-to-regrasp replay versus policy-state
recovery SFT, subject to a stronger baseline-stability gate. See
`docs/embodied_research_reset/recovery_support_diagnosis_results.md`.

### Recovery-support calibration status

The first Base continuation candidate added 6,902 updates from each seed's
original Full-H checkpoint. Validation-only K=3 attached success was 15.38%,
23.08%, and 15.38% for seeds 41, 42, and 43: a cross-seed mean of 17.95%, with
only one seed at or above 20%. It therefore failed both preregistered conditions
(mean at least 30% and at least two seeds at or above 20%). Slip full-task
recovery averaged 2.56%. No test result was opened for calibration.

All 234 validation rows completed. The 117 paired videos contain exactly 37,044
expected frames and pass H.264/`avc1`, `yuv420p`, fast-start, nonblank, and
motion checks. The machine-readable result is
`/share/longjunyu/fresh-vla/runs/recovery-support-v1/base_continuation_calibration_steps6902.json`.

Per preregistration, the second Base candidate is an independent 13,804-update
run from the original checkpoints, not a continuation of the 6,902-update
weights. Formal B/C collection, training, and test comparison remain blocked by
the Base gate. Their implementation has nevertheless passed a one-group real
simulator smoke and matched 20-slot training smokes, so a passing Base result can
proceed without changing the method after seeing B/C outcomes.

The B/C revision was recorded before any B/C result: both use 50% identical
anchor slots and 50% target slots with identical snapshot-group schedules, and
both target the same feedback-to-stable-regrasp segment. B uses clean expert
states; C uses states reached after the deployed policy has actually executed a
wrong or unresolved continuation. Thus the controlled difference is state
distribution coverage, not target length, phase mix, or a new loss.

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

## Full-H Baseline Repair Update (2026-07-16)

The corrected global-batch-8 Full-H repair selected the earliest passing
seed-41 checkpoint, 3,451 optimizer steps, exactly as preregistered. The formal
three-seed validation gate then completed 234/234 rows. At primary K=3,
attached success was 46.15%, 23.08%, and 7.69% for seeds 41, 42, and 43; the
cross-seed mean was 25.64%. Two seeds reached 20%, but the mean missed the fixed
30% threshold. The machine-readable decision is therefore
`BASELINE_INVALID_OR_DATA_INSUFFICIENT`. Test remains sealed.

All 117 paired videos and 36,781 decoded frames passed codec, fast-start,
frame-count, nonblank, and motion checks. The gate and audit are under
`/share/longjunyu/fresh-vla/runs/baseline-repair-v1/` as
`baseline_repair_three_seed_gate.json` and
`baseline_repair_three_seed_video_artifact_audit.json`.

One final validation-only budget repair is frozen before seed-42/43 results:
10,353 steps, the earliest later preregistered checkpoint that passed for seed
41. Its protocol and terminal stopping rule are in
`docs/embodied_research_reset/baseline_validity_repair_v2_amendment.md`. No
recovery control or test evaluation may start unless this unchanged gate passes.

### Baseline repair v2 result

The final 10,353-step repair completed for all three seeds and passed the
unchanged validation-only gate. At primary K=3, attached success is 61.54%,
38.46%, and 61.54%; the cross-seed mean is 53.85%. Overall task success is
50.00%, 30.77%, and 50.00%, while slip recovery is 38.46%, 23.08%, and 38.46%.
The formal decision is `BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS`; test is
still sealed.

K=3 failure continuation is 62.5%, 55.6%, and 75.0%, compared with 0% for every
seed at K=1. This makes stale chunk commitment a closed-loop behavior signal on
a baseline that can perform the task, rather than an artifact of a wholly
incompetent policy.

All 117 end-to-end videos and 32,591 frames passed exact decode, H.264/`avc1`,
`yuv420p`, fast-start, nonblank, motion, and frame-count checks. Formal outputs
are `baseline_repair_v2_three_seed_gate.json` and
`baseline_repair_v2_three_seed_video_artifact_audit.json` under the baseline
repair run root.

Gate 1 is now frozen in
`docs/embodied_research_reset/recovery_support_v2_preregistration.md`: Base
continuation, clean feedback-to-regrasp replay, and policy-state recovery use
the same seed-matched 10,353-step initialization and 6,902 additional updates.
Only validation K=1/2/3 is allowed until the paired decision is complete.

## Recovery Support v2 Result (2026-07-16)

Gate 1 completed all nine support trainings and all validation closed-loop
evaluations. The formal result is `STOP_OFFLINE_SUPPORT_EXPANSION`. At primary
K=3, Base continuation reached 48.7% overall and 35.9% slip recovery versus
43.6% and 33.3% for the released Full-H baseline. The paired gains, +5.1 and
+2.6 points, did not reach the frozen 10-point threshold and both intervals
cross zero. Clean recovery replay fell to 32.1% overall/12.8% slip; policy-state
recovery fell to 30.8%/17.9%. Both targeted arms significantly underperformed
Base on the primary full-task metrics.

All 351 candidate end-to-end videos and 104,048 frames passed H.264/`avc1`,
`yuv420p`, fast-start, exact-frame, nonblank, motion, and decode checks. K=1
eliminated failure continuation but did not solve full-task recovery, while
deterministic reach remained 94.9-97.4%. This separates stale-tail revision from
the missing multi-step recovery competence and rejects further tuning of this
offline support recipe.

A runner defect initially allowed deterministic reach alone to use its default
`test` split. Twelve auxiliary JSONs were quarantined before summary or method
selection, the runner was fixed in `53353b0`, and all reach controls were rerun
on validation. No Gate 1 isolated or end-to-end test episode ran, but the old
test groups are not strictly pristine after this auxiliary access. Future
confirmation requires newly sealed groups. Full details are in
`docs/embodied_research_reset/recovery_support_v2_results.md`.

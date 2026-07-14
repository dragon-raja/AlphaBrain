# FRESH-VLA Final Gate Status

Audit date: 2026-07-14 UTC

Branch: `exp/fresh-vla-toy-v0`
Audit base: `a374582`

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
- `label_revision.json` records the sample-marginal fix and the prior labels are retained as `training_labels.before_sample_marginal_fix.json`. The in-flight Full-H calibration is mathematically unaffected because its tail weight is exactly 1.0; all weighted methods will read the revised labels.

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

## Final Gate In Progress

The analysis rule is fixed before viewing final test results: K=3 is the primary commitment setting, K=2 is supporting evidence, and K=1 is a negative control. The final decision script applies the stated 10-point Oracle-vs-Full and 5-point control-effect thresholds together with normal-task, behavior-error, and baseline-validity gates. It does not select a checkpoint, seed, or execution horizon after seeing test performance.

All methods use a 320-step end-to-end timeout. The held-out scripted expert requires at most 252 steps on the slip branch, so this budget provides a fixed 27% margin without changing between methods, seeds, or K.

The final gate will:

1. use Full-H on the validation split to establish a common training budget and baseline validity;
2. train every method with identical initialization, shuffled sample order per seed, optimizer, batch size, horizon, and update count;
3. run isolated recovery and event-triggered end-to-end evaluation for fixed K=1, K=2, and K=3 on all 13 held-out groups;
4. save paired videos for every successful and failed evaluation episode;
5. report group-level bootstrap confidence intervals, seed-level values, progress/subgoal diagnostics, and paired absolute percentage-point deltas;
6. emit `docs/fresh_vla_final_decision.md` and machine-readable `final_decision.json`.

The persistent `run_libero_final_gate.sh` controller records stage transitions under `final_gate_logs/pipeline.log`, skips complete artifacts, and stops before opening the test split if the validation baseline gate fails.

Deterministic reach and offline MSE remain auxiliary controls. They cannot independently determine the final decision.

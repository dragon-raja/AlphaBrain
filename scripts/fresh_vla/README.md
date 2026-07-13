# FRESH-VLA Experiments

This directory contains the staged falsification tests for feedback-weighted action supervision. The toy benchmark does not use model weights or external datasets; the snapshot pilot uses the local LIBERO source and task assets without downloading a benchmark dataset.

The branching toy preserves three properties relevant to Pi0.5:

- flow-matching noise and velocity targets;
- a full predicted action horizon;
- bidirectional attention among action tokens.

The current observation determines a common safe prefix. An unobserved branch variable determines the correct suffix after a sample-dependent feedback boundary. Every method receives exactly the same branch trajectories, and evaluation uses the same fixed execution horizon.

```bash
cd /alphabrain
.venv/bin/python scripts/fresh_vla/toy_branching_flow.py \
  --device cuda:7 \
  --seeds 41 42 43 \
  --output /share/longjunyu/fresh-vla/toy/counterfactual-paired-results.json
```

The first gate is whether `oracle_soft010` improves fixed-K prefix error over
`full_h`, `short_h`, `random_soft010`, `gripper_soft010`, and
`remac_prefix_mask_control`. Training data, flow time/noise, evaluation samples,
and inference noise use independent RNG streams and are paired by seed. The
result includes per-sample deltas, standard errors, and bootstrap confidence
intervals, plus 32-sample multimodal suffix diagnostics.

Run the deterministic negative control with `--branch-strength 0`. If FRESH improves by the same amount when no suffix depends on hidden feedback, the apparent gain is likely generic loss reweighting rather than evidence for feedback-required supervision.

`prepare_libero_subset.py` creates a download manifest for selected episode parquet files, then materializes a LeRobot v2 subset with gripper-event `feedback_horizon` labels. This validates the real Pi0.5 training path but is an event-label baseline, not an oracle counterfactual experiment.

`generate_counterfactual_pairs.py` materializes schema-complete synthetic grasp,
blocked-push, deterministic reach, and language-intent pairs without downloads.
It estimates the oracle horizon from repeated within-branch rollouts, records a
threshold sensitivity sweep, and runs the policy-input leakage guard. It is a
data-contract fixture for the simulator collector, not a replacement for
LIBERO closed-loop evidence.

`libero_snapshot_collector.py` runs the narrowly scoped real-physics pilot. It
uses hard-reset plus snapshot restore for every paired rollout because the
flattened MuJoCo state does not include robosuite controller and observable
buffers. The collector rejects runs unless grasp/slip attachment, free/blocked
push displacement, deterministic replay, threshold stability, and policy-input
leakage checks pass.

The isolated runtime is Python 3.8 at `/workspace/envs/fresh-libero`; its direct
requirements are recorded in `requirements-libero-snapshot.txt`. Run the pilot
with automatic single-card keepalive handling:

```bash
cd /alphabrain
scripts/fresh_vla/run_libero_snapshot_pilot.sh \
  --output-dir /share/longjunyu/fresh-vla/counterfactual-libero-snapshot \
  --num-pairs 4 --repeats 4 --resolution 64 --seed 20260713
```

Additional validation tools:

- `paired_evaluation.py` fingerprints paired FM inputs and reports bootstrap CIs;
- `summarize_counterfactual_results.py` separates seed-level and pooled-sample uncertainty;
- `multimodal_sampling_evaluator.py` analyzes 32+ chunks per conditioning state;
- `fixed_k_evaluator.py` runs identical snapshot episodes at fixed K values;
- `gradient_diagnostic.py` compares prefix/suffix gradients in the action output and final action layer;
- `plot_lambda_sweep.py` intentionally refuses to plot until at least three lambda values exist.

The real LIBERO pilot confirms that outcome-dependent common-prefix structure
exists, including a no-gripper-transition blocked-push branch. It does not by
itself establish a trained-policy advantage. The first strict method gate still
did not pass, so the lambda sweep and learned horizon remain disabled. See
`docs/fresh_vla_counterfactual_validation.md` for the complete result.

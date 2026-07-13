# FRESH-VLA Experiments

This directory starts with the smallest falsification test for feedback-weighted action supervision. It does not use model weights or external datasets.

The branching toy preserves three properties relevant to Pi0.5:

- flow-matching noise and velocity targets;
- a full predicted action horizon;
- bidirectional attention among action tokens.

The current observation determines a common safe prefix. An unobserved branch variable determines the correct suffix after a sample-dependent feedback boundary. Every method receives exactly the same branch trajectories, and evaluation uses the same fixed execution horizon.

```bash
cd /alphabrain
.venv/bin/python scripts/fresh_vla/toy_branching_flow.py \
  --device cuda:7 \
  --output /share/longjunyu/fresh-vla/toy/results.json
```

The first gate is whether `oracle_hard` or `oracle_soft` improves fixed-K prefix error over `full`, `short`, `random`, and `event`. A null result is evidence against the central FRESH training hypothesis and should stop investment in a learned horizon head.

Run the deterministic negative control with `--branch-strength 0`. If FRESH improves by the same amount when no suffix depends on hidden feedback, the apparent gain is likely generic loss reweighting rather than evidence for feedback-required supervision.

`prepare_libero_subset.py` creates a download manifest for selected episode parquet files, then materializes a LeRobot v2 subset with gripper-event `feedback_horizon` labels. This validates the real Pi0.5 training path but is an event-label baseline, not an oracle counterfactual experiment.

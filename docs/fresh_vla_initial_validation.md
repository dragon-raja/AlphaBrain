# FRESH-VLA Initial Validation

This note records the first controlled Pi0.5 smoke test of feedback-aware
action supervision in AlphaBrain. It is an engineering and directional check,
not a benchmark result.

## Setup

- Base model: LeRobot Pi0.5 Base with PaliGemma 3B PT 224.
- Data: two LIBERO cream-cheese episodes, 171 frames total.
- Feedback proxy: the first gripper-state transition in each 10-step action
  window; windows without a transition use the full horizon.
- Baseline: weight every action step by 1.0.
- FRESH: weight the proxy-defined suffix by 0.1.
- Training: 20 steps, batch size 1, seed 42, frozen VLM, bf16, one GPU.
- Evaluation: paired raw flow-matching MSE with identical samples and noise
  seeds, using 12 boundary windows and 12 full-horizon control windows.

## Results

Lower is better. These are unweighted evaluation losses, so the two objectives
are compared on the same metric.

| Split / metric | Baseline | FRESH 0.1 | FRESH delta |
| --- | ---: | ---: | ---: |
| Boundary, fixed first 3 steps | 1.10044 | 1.10087 | +0.04% |
| Boundary, proxy-supported prefix | 1.06654 | 1.08414 | +1.65% |
| Boundary, suffix | 1.18693 | 1.23840 | +4.34% |
| Boundary, full horizon | 1.10077 | 1.13395 | +3.01% |
| Control, fixed first 3 steps | 0.90073 | 0.91475 | +1.56% |
| Control, full horizon | 0.73027 | 0.74041 | +1.39% |

FRESH had lower fixed-prefix loss on 8 of 12 boundary samples, but a few larger
regressions erased that gain in the mean. On control windows, baseline was
better on 9 of 12 samples for the fixed-prefix metric.

## Interpretation

The end-to-end mechanism is viable: Pi0.5 weights load with 99.63% unique
parameter coverage, feedback horizons reach the loss, and both objectives train
and reload successfully. The earlier branch-dependent synthetic experiment
supports the oracle version of the hypothesis.

The LIBERO proxy experiment does not show an aggregate benefit. A gripper
transition is not a reliable label for the point where observations stop
supporting future actions, and 20 updates on two episodes are too small for an
algorithmic claim. The current evidence is therefore: plausible with oracle
branch labels, but neutral to mildly negative with the naive gripper-event
proxy.

The next useful test is to create counterfactual simulator branches with a
known observation-dependent divergence point, then compare fixed-prefix action
error and closed-loop success across multiple seeds. A learned horizon head
should wait until that label construction produces a repeatable positive
signal.

## Reproduction

```bash
PATH=/alphabrain/.venv/bin:$PATH \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
bash scripts/run_base_vla/train.sh \
  fresh_pi05_baseline_smoke configs/experiments/fresh_vla_pi05.yaml

PATH=/alphabrain/.venv/bin:$PATH \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
bash scripts/run_base_vla/train.sh \
  fresh_pi05_soft010_smoke configs/experiments/fresh_vla_pi05.yaml
```

The paired evaluation artifact is stored outside Git at
`/share/longjunyu/fresh-vla/eval/paired-20step-flow-eval.json`.

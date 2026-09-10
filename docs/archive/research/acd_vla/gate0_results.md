# ACD-VLA Gate 0 results

Decision: **STOP_ACD_CURRENT_TASK_GATE0**

Only train/validation policy-response records were evaluated. The original test
and confirmation groups remained sealed.

## What was tested

Gate 0 asked whether a compact predictor at the common pre-feedback state could
amortize the frozen Pi0.5 policy calls that would be made after an attached or
slipped outcome became visible. The predictor had to recover state-specific
future responses, not merely classify the outcome or emit one constant template
per outcome.

The formal collection contains 115 paired groups: 102 train groups and 13
validation groups. Train and validation contain 30 and 9 distinct source states,
respectively, with zero source overlap. Each group was queried with the same
frozen seed-41 checkpoint at the common pre-feedback observation and at both
post-feedback observations.

## Results

- Post-feedback outcome guard accuracy: `100.0%`.
- Pre-feedback guard accuracy: `50.0%`.
- Shuffled-label guard accuracy: `50.5%`.
- Mean normalized attached/slipped teacher-response RMS: `1.366`.
- Median predicted/target branch-separation ratio: `0.515`.
- State-conditioned oracle branch versus per-outcome constant: `-38.6%` MSE
  reduction, paired source-bootstrap 95% CI `[-74.7%, -18.8%]`.
- Attached response versus attached constant: `-48.3%`, 95% CI
  `[-117.1%, 1.3%]`.
- Slipped response versus slipped constant: `-36.4%`, 95% CI
  `[-68.2%, -21.0%]`.
- Learned outcome route versus stale Pi0.5 tail: `+13.1%`, 95% CI
  `[-10.4%, 28.1%]`; this misses the preregistered `25%` effect threshold and
  its interval crosses zero.
- Learned route versus a learned merged continuation: `+5.8%`, 95% CI
  `[-0.3%, 11.2%]`.

The post-feedback images expose the outcome and the frozen policy responds
differently to the two branches. However, the pre-feedback representation does
not predict the state-specific response variation: a constant continuation per
outcome is substantially better than the learned state-conditioned branch.
Consequently, this task does not support direct pre-commitment to detailed future
policy responses.

## Scope of the decision

This is an honest stop for **direct ACD on the current one-task Gate**. It does
not show that observations are uninformative after feedback, nor does it evaluate
closed-loop simulator success, runtime savings, or multi-task transfer. No
feature sweep or same-validation-set rescue is permitted after the failed Gate.

The full machine-readable output is stored at:

```text
/share/longjunyu/acd-vla/gate0-policy-response-v1/gate0_results.json
```

Decision: **STOP_ACD_CURRENT_TASK_GATE0**

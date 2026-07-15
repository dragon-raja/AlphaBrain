# FRESH-VLA LIBERO Final Decision

**Decision: `STOP_TRAINING_WEIGHTING_ROUTE`**

The final comparison uses held-out snapshot groups, three fixed seeds, and fixed execution horizons. K=3 is the preregistered primary commitment setting, K=2 is supporting evidence, and K=1 is a negative control.

## Completion Audit

Artifact audit complete: `true`.

| Checkpoints | Isolated | End-to-end | Reach | Offline | Closed-loop videos |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18/18 | 18/18 | 18/18 | 18/18 | 18/18 | 1404/1404 |

Source data contain 256 branch episodes, 128 paired videos, 256 branch videos, and a contact sheet.

## Data And Expert Gate

Episode quality passed: `true`; 128 groups, attached expert success 1.000, slip recovery expert success 1.000.
Window quality passed: `true`; 34551 windows with group-preserving splits and post-feedback full supervision restored.

## Baseline Gate

Baseline valid: `true`.

| Full-H attached | Full-H overall | Best overall | Full-H event trigger |
| ---: | ---: | ---: | ---: |
| 0.231 | 0.141 | 0.179 | 0.385 |

The baseline clears the preregistered attached-success gate, but only narrowly. The conclusion is therefore scoped to this training-weighting implementation and budget, not to feedback-aware control in general.

## Primary Closed-Loop Results (K=3)

Attached success is also the normal/no-intervention success in this paired design.

| Method | Overall | Attached | Slip recovery | Isolated recovery | Failure continuation | Premature commitment | Final progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_h` | 0.141 | 0.231 | 0.051 | 0.154 | 0.516 | 0.516 | 0.311 |
| `random_soft010` | 0.179 | 0.308 | 0.051 | 0.179 | 0.717 | 0.717 | 0.311 |
| `shuffled_oracle_soft010` | 0.179 | 0.282 | 0.077 | 0.256 | 0.633 | 0.633 | 0.333 |
| `gripper_soft010` | 0.154 | 0.282 | 0.026 | 0.128 | 0.633 | 0.633 | 0.298 |
| `oracle_soft010` | 0.154 | 0.308 | 0.000 | 0.179 | 0.467 | 0.467 | 0.314 |
| `short_h` | 0.154 | 0.282 | 0.026 | 0.128 | 0.739 | 0.739 | 0.247 |

### Per-Seed K=3 Primary Rates

| Method | Seed | Overall | Attached | Slip recovery | Isolated recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_h` | 41 | 0.192 | 0.308 | 0.077 | 0.231 |
| `full_h` | 42 | 0.038 | 0.077 | 0.000 | 0.000 |
| `full_h` | 43 | 0.192 | 0.308 | 0.077 | 0.231 |
| `random_soft010` | 41 | 0.231 | 0.308 | 0.154 | 0.308 |
| `random_soft010` | 42 | 0.154 | 0.308 | 0.000 | 0.154 |
| `random_soft010` | 43 | 0.154 | 0.308 | 0.000 | 0.077 |
| `shuffled_oracle_soft010` | 41 | 0.308 | 0.385 | 0.231 | 0.462 |
| `shuffled_oracle_soft010` | 42 | 0.115 | 0.231 | 0.000 | 0.077 |
| `shuffled_oracle_soft010` | 43 | 0.115 | 0.231 | 0.000 | 0.231 |
| `gripper_soft010` | 41 | 0.192 | 0.385 | 0.000 | 0.231 |
| `gripper_soft010` | 42 | 0.077 | 0.154 | 0.000 | 0.000 |
| `gripper_soft010` | 43 | 0.192 | 0.308 | 0.077 | 0.154 |
| `oracle_soft010` | 41 | 0.192 | 0.385 | 0.000 | 0.077 |
| `oracle_soft010` | 42 | 0.192 | 0.385 | 0.000 | 0.000 |
| `oracle_soft010` | 43 | 0.077 | 0.154 | 0.000 | 0.462 |
| `short_h` | 41 | 0.115 | 0.231 | 0.000 | 0.231 |
| `short_h` | 42 | 0.192 | 0.385 | 0.000 | 0.077 |
| `short_h` | 43 | 0.154 | 0.231 | 0.077 | 0.077 |

### Oracle Versus Every Control At K=3

Positive success deltas favor Oracle; negative behavior-error deltas favor Oracle. The parenthesized n is the paired snapshot-group count.

| Baseline | Overall | Attached | Slip recovery | Isolated recovery | Failure continuation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_h` | +1.3 pp [-5.1, +7.7] (n=13) | +7.7 pp [-5.1, +20.5] (n=13) | -5.1 pp [-12.8, +0.0] (n=13) | +2.6 pp [-5.1, +10.3] (n=13) | -18.8 pp [-56.2, +12.5] (n=8) |
| `random_soft010` | -2.6 pp [-15.4, +9.0] (n=13) | +0.0 pp [-17.9, +17.9] (n=13) | -5.1 pp [-12.8, +0.0] (n=13) | +0.0 pp [-12.8, +12.8] (n=13) | -26.2 pp [-78.6, +33.3] (n=7) |
| `shuffled_oracle_soft010` | -2.6 pp [-14.1, +7.7] (n=13) | +2.6 pp [-15.4, +17.9] (n=13) | -7.7 pp [-15.4, +0.0] (n=13) | -7.7 pp [-20.5, +5.1] (n=13) | -42.9 pp [-85.7, +0.0] (n=7) |
| `gripper_soft010` | +0.0 pp [-12.8, +12.8] (n=13) | +2.6 pp [-23.1, +25.6] (n=13) | -2.6 pp [-7.7, +0.0] (n=13) | +5.1 pp [-12.8, +23.1] (n=13) | -50.0 pp [-83.3, -16.7] (n=6) |
| `short_h` | +0.0 pp [-7.7, +7.7] (n=13) | +2.6 pp [-15.4, +17.9] (n=13) | -2.6 pp [-7.7, +0.0] (n=13) | +5.1 pp [-5.1, +15.4] (n=13) | -30.0 pp [-80.0, +20.0] (n=5) |

## Oracle vs Full-H Paired Deltas

Success deltas above zero favor Oracle; behavior-error deltas below zero favor Oracle.

| K | Overall | Slip recovery | Isolated recovery | Failure continuation | Premature commitment |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +6.4 pp [+1.3, +12.8] | +2.6 pp [+0.0, +7.7] | +0.0 pp [-7.7, +7.7] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] |
| 2 | +7.7 pp [-1.3, +17.9] | +5.1 pp [+0.0, +12.8] | +5.1 pp [-7.7, +17.9] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] |
| 3 | +1.3 pp [-5.1, +7.7] | -5.1 pp [-12.8, +0.0] | +2.6 pp [-5.1, +10.3] | -18.8 pp [-56.2, +12.5] | -18.8 pp [-56.2, +12.5] |

### Behavioral Diagnostic Audit

Across 221 eligible triggered-slip rows, failure-continuation and premature-commitment differ on 0 rows. The predicates are distinct in code, but they coincide on every eligible final-gate row; do not count them as independent behavioral evidence.

Their K=3 paired comparisons use fewer groups than the success metrics because they are defined only after an intervention event. The behavior-rate reduction is diagnostic, not independent success evidence.

## Rule Checks

### Continue FRESH

- `oracle_vs_full_primary`: FAIL

### Predictability-Weighting Pivot

- `all_soft_methods_improve_over_full`: FAIL

## Auxiliary Offline Check

Offline MSE and suffix mode coverage are mechanism diagnostics only and did not determine the decision.

| Method | K=1 MSE | K=2 MSE | K=3 MSE | Oracle-prefix MSE | Suffix MSE | Mode coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_h` | 0.1462 | 0.1403 | 0.1396 | 0.1452 | 0.4506 | 0.250 |
| `random_soft010` | 0.1427 | 0.1373 | 0.1396 | 0.1486 | 0.4557 | 0.333 |
| `shuffled_oracle_soft010` | 0.1423 | 0.1400 | 0.1389 | 0.1442 | 0.4491 | 0.292 |
| `gripper_soft010` | 0.1517 | 0.1449 | 0.1434 | 0.1446 | 0.4634 | 0.083 |
| `oracle_soft010` | 0.1443 | 0.1395 | 0.1382 | 0.1409 | 0.5209 | 0.000 |
| `short_h` | 0.1444 | 0.1386 | 0.1414 | 0.2334 | 0.5524 | 0.000 |

## Deterministic Reach Negative Control

Success threshold: 0.040 m to the recorded expert EEF target.

| Method | K=1 | K=2 | K=3 |
| --- | ---: | ---: | ---: |
| `full_h` | 0.923 | 0.949 | 0.872 |
| `random_soft010` | 0.897 | 0.949 | 0.974 |
| `shuffled_oracle_soft010` | 0.872 | 0.897 | 0.949 |
| `gripper_soft010` | 0.846 | 0.949 | 0.897 |
| `oracle_soft010` | 0.872 | 0.923 | 0.897 |
| `short_h` | 0.923 | 0.949 | 0.974 |

## Interpretation

At the primary K=3 setting, Oracle does not improve slip recovery over Full-H and its overall-success paired interval includes zero. Random, shuffled-Oracle, gripper, and Short-H controls match or exceed Oracle on at least one primary success outcome, so the exact Oracle boundary has no demonstrated closed-loop specificity.

Oracle improves the offline common-prefix error, but its suffix mode coverage collapses and the offline gain does not transfer to recovery or full-task success. K=1 improvements are negative-control evidence: they do not persist at the preregistered commitment horizon and are not Oracle-specific.

Deterministic reach confirms that deployment can execute basic directed motion. It does not substitute for the failed full-task recovery gate. The decision therefore stops this suffix-loss weighting route; it does not reject feedback-aware replanning, plan-commit execution, active probing, or belief-aware control.

STOP_TRAINING_WEIGHTING_ROUTE

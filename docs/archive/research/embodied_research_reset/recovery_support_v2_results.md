# Recovery Support v2 Result

Date: 2026-07-16 UTC

Decision: `STOP_OFFLINE_SUPPORT_EXPANSION`

## Question and protocol

This validation-only gate tested whether the FRESH failure was merely missing
post-feedback recovery action support. Three arms started from the seed-matched
10,353-step Full-H checkpoints and received exactly 6,902 additional updates:
ordinary continuation, clean feedback-to-stable-regrasp replay, and correction
from states actually reached by the deployed policy. Seeds were 41, 42, and 43;
K=1/2/3, validation groups, optimizer, target-group schedule, anchors, and
training budget were fixed before results. K=3 was primary.

The correction collector passed every frozen quality check. It retained
101/102, 100/102, and 101/102 train groups across the three seeds, produced
7,155, 7,224, and 7,565 full-chunk windows, and wrote 20 paired inspection
videos per seed. Full-teacher downstream success was 98.0%, 99.0%, and 99.0%;
the frozen policy succeeded after stable teacher regrasp in 85.7%, 85.7%, and
95.2% of the audit rollouts. Four unusable corrections were rejected without
relaxing the 80% coverage gate.

All nine support checkpoints completed. The formal evaluation contains every
method, seed, and fixed K on the same 13 validation groups. Candidate methods
produced 351 end-to-end videos with 104,048 exact decoded frames. The artifact
audit passed H.264/`avc1`, `yuv420p`, fast-start, frame count, nonblank, motion,
and decode checks with zero errors.

## Primary result

Group-level bootstrap intervals average the three seeds inside each snapshot
group. Frames and episode steps are not independent samples.

| K=3 method | Overall task | Attached task | Slip recovery | Isolated recovery |
| --- | ---: | ---: | ---: | ---: |
| Full-H 10,353 | 43.6% [34.6, 53.8] | 53.8% [41.0, 66.7] | 33.3% [23.1, 43.6] | 74.4% [61.5, 87.2] |
| Base continuation | 48.7% [34.6, 61.5] | 61.5% [46.2, 76.9] | 35.9% [20.5, 51.3] | 66.7% [51.3, 82.1] |
| Clean recovery replay | 32.1% [20.5, 43.6] | 51.3% [35.9, 66.7] | 12.8% [5.1, 20.5] | 69.2% [53.8, 84.6] |
| Policy-state recovery | 30.8% [19.2, 41.0] | 43.6% [28.2, 56.4] | 17.9% [10.3, 25.6] | 43.6% [33.3, 53.8] |

The corresponding per-seed K=3 values are:

| Method | Seed | Overall | Attached | Slip | Isolated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-H | 41 | 50.0% | 61.5% | 38.5% | 92.3% |
| Full-H | 42 | 30.8% | 38.5% | 23.1% | 76.9% |
| Full-H | 43 | 50.0% | 61.5% | 38.5% | 53.8% |
| Base | 41 | 50.0% | 69.2% | 30.8% | 69.2% |
| Base | 42 | 53.8% | 61.5% | 46.2% | 76.9% |
| Base | 43 | 42.3% | 53.8% | 30.8% | 53.8% |
| Clean | 41 | 30.8% | 53.8% | 7.7% | 61.5% |
| Clean | 42 | 23.1% | 46.2% | 0.0% | 92.3% |
| Clean | 43 | 42.3% | 53.8% | 30.8% | 53.8% |
| Policy-state | 41 | 26.9% | 30.8% | 23.1% | 38.5% |
| Policy-state | 42 | 26.9% | 38.5% | 15.4% | 53.8% |
| Policy-state | 43 | 38.5% | 61.5% | 15.4% | 38.5% |

Paired K=3 effects are decisive about the proposed support mechanism:

- Base versus Full-H: overall `+5.1pp [-11.5,+20.5]`, slip `+2.6pp
  [-15.4,+20.5]`. More ordinary training does not clear the 10-point gate.
- Clean replay versus Base: overall `-16.7pp [-24.4,-9.0]`, slip `-23.1pp
  [-35.9,-10.3]`, and attached `-10.3pp [-17.9,-2.6]`.
- Policy-state correction versus Base: overall, attached, and slip are each
  `-17.9pp`; all three paired intervals exclude zero in the harmful direction.
- Policy-state versus clean replay: slip is only `+5.1pp [-7.7,+17.9]`, while
  isolated recovery is `-25.6pp [-43.6,-2.6]`.

K=1 removes failure continuation for every method, but does not solve full-task
recovery: Full-H and Base overall success are 41.0% and 47.4%. At K=2 they are
53.8% and 55.1%. Thus stale-tail commitment is real, but immediate replanning
alone and the tested offline recovery SFT do not supply reliable long-horizon
recovery competence. K=3 deterministic reach remains 94.9-97.4% across arms,
so the negative result is not explained by loss of basic free-space reach.

## Interpretation

The exact FRESH weighting route remains stopped. Gate 1 also rejects expanding
this 50/50 offline SFT recipe: clean expert recovery windows do not transfer to
closed-loop recovery, and policy-state corrections cause stronger interference
rather than fixing the distribution shift. This result does not establish that
all corrective learning is impossible. It establishes that adding more paired
recovery replay, a CFR wrapper, a competence head, or a boundary sweep on top of
this failed support mechanism is not justified.

The next research direction must change the intervention mechanism, not tune
this loss. A future experiment should compare execution-time feedback-aware
replanning or residual correction against K=1/K=2 and a strong on-policy
correction baseline, with normal-task retention built into the update. It must
use a newly frozen protocol and untouched confirmation groups; no CFR or
competence-boundary claim follows from this gate.

## Protocol deviation

The first recovery-support controller correctly passed `split=val` to isolated
and end-to-end evaluation but omitted it for deterministic reach, whose CLI
default was `test`. This generated 12 auxiliary reach JSON files on the 13 test
groups. They were detected before summary or method selection, quarantined
under `invalid_artifacts/deterministic_reach_test_split_20260716`, and excluded.
Commit `53353b0` makes split propagation explicit and validates reused outputs;
all 12 reach evaluations were rerun on validation and the formal pipeline then
exited zero.

No isolated or end-to-end test episode was run in Gate 1, but the original test
groups are no longer strictly pristine because the auxiliary reach evaluator
touched them. Any future confirmatory claim must collect newly sealed groups or
predeclare a replacement test set. The machine-readable Gate 1 result remains
validation-only; it must not be described as an untouched final test.

## Artifacts

- Run root: `/share/longjunyu/fresh-vla/runs/recovery-support-repaired-v2-step10353`
- Statistics: `closed_loop_summary_val_steps6902/results.json`
- Decision: `support_decision_val.json`
- Video audit: `recovery_support_v2_video_artifact_audit.json`
- Controller status: `controller.exit` (`0`)

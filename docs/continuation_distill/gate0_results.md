# Continuation-selected self-distillation Gate 0 results

Date: 2026-07-19

Decision: **STOP_CONTINUATION_DISTILL_LABEL_BANK**

## Validity

- Input was the complete train-only CCV collection.
- Only the 23 existing fit sources were opened: 18 fitting and five nested
  evaluation sources under the frozen split.
- CCV holdout, original test, confirmation, and sealed states opened: zero.
- The robust-winner rule and all coverage thresholds were committed in
  `dba9aa7` before this audit.

## Result

| Subset | States | Accepted | Sources covered | Mean accepted gain | 95% source CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| fitting | 534 | 77 (14.4%) | 17/18 | 0.01013 | [0.00565, 0.01216] |
| evaluation | 182 | 32 (17.6%) | 5/5 | 0.03713 | [0.00796, 0.06006] |

Accepted fitting states by stage:

| Stage | Count |
| --- | ---: |
| post_regrasp | 59 |
| preclose | 12 |
| reapproach | 6 |

The robust continuation winner beat the immediate-physical winner by essentially
the full available accepted-state Oracle gain, and both source intervals excluded
zero. The labels that survive repeat-level uncertainty are therefore meaningful.
They are not, however, located where the current policy fails.

## Failed frozen checks

1. fitting accepted-state count was 77, below the required 120;
2. only `post_regrasp` had at least 20 accepted fitting states, while the protocol
   required at least three stages.

All gain, source-coverage, direct-control, and stable-grasp checks passed. The
decision remains STOP because every check was mandatory.

## Root-cause update

The formal handoff experiment located the main recovery gap before stable
regrasp: handing Pi0.5 a stable-regrasp state raises completion to 85.2%. This
bank audit now shows that repeat-stable static candidate labels occur mostly
*after* stable regrasp. Before regrasp, candidate value depends on the sequence
of future observations and replans strongly enough that a single fixed winner is
not reliably identifiable.

This reconciles the prior evidence:

- exact receding continuation search works because it reselects after every new
  physical observation;
- local/direct and one-response surrogates fail because no one-step statistic
  carries the whole bridge;
- offline recovery SFT is harmful because it updates the full policy from a
  static state distribution and compounds away from it;
- robust offline winner distillation lacks labels in the actual bridge region.

Loosening five-of-six repeat agreement, lowering coverage quotas, or training
only the 59 post-regrasp states would not test the recovery hypothesis and is
forbidden.

## Redesign boundary

The next experiment must be an **on-policy closed-loop recovery option**, not
another offline scorer or full-policy replay mixture:

1. freeze the successful base Pi0.5;
2. activate a small residual/option policy only after visually verified failure;
3. collect and update on the option's own slip-to-regrasp states;
4. optimize reaching a policy-relative handback set from which frozen Pi0.5
   completes reliably, instead of imitating a fixed teacher trajectory;
5. terminate the option and return control to the frozen base policy at that set;
6. compare against frequent Pi0.5 replanning, A2C2-style residual correction,
   RaC/VLA-OPD-style correction, DICE-RL-style residual improvement, and the
   existing harmful recovery SFT.

This is a problem-driven pivot to interactive state-distribution correction.
Residual RL, recovery options, occupancy matching, and policy-relative handoff
are all established neighborhoods; no originality claim is justified until a
specific objective beats those controls and transfers to a second task.

Machine-readable artifact:
`/share/longjunyu/fresh-vla/continuation-distill/gate0-bank-v1/result.json`.


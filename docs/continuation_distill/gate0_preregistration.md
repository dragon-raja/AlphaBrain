# Continuation-selected self-distillation Gate 0 preregistration

Date frozen: 2026-07-19

Status: label-bank feasibility gate. No policy has been trained and no method
claim is made.

## Motivation

Exact frozen-policy continuation search has real closed-loop headroom, while
CCV shared-noise amortization and the endpoint policy-response surrogate both
failed their frozen gates. Runtime prediction of the winner is therefore
stopped.

This gate tests a different use of the expensive Oracle: select actions only
during training, distill accepted in-support candidates into Pi0.5, and deploy
the updated policy as an ordinary `N=1` fixed-K policy. This differs from the
failed recovery SFT because targets are sampled by the base policy at its own
states and selected by downstream frozen-policy compatibility rather than by
expert action similarity.

## Data boundary

- Source collection: complete CCV train-only collection at
  `/share/longjunyu/fresh-vla/ccv-vla/gate0-coupled-v3`.
- Only the 23 existing `fit` source IDs may be opened.
- Reuse the already frozen policy-response nested split: 18 fitting sources and
  five evaluation sources under salt `policy-response-gate-minus1-v1`.
- Six CCV holdout sources, original test, confirmation, and sealed paths remain
  unopened.
- No new simulator rollout or label seed is added in Gate 0.

## Frozen robust-winner rule

For each state, compute scalar lexicographic viability utility separately for
each of the six saved continuation repeats. Let `w` maximize the six-repeat mean
utility, with candidate index as the deterministic final tie break.

Accept the state only if all hold:

1. `w != 0` and its mean utility strictly exceeds candidate 0;
2. `w` strictly beats candidate 0 in at least five of six repeats;
3. leave-one-repeat-out selection returns `w` in at least five of six folds;
4. selecting `w` does not reduce stable-grasp probability when success,
   transport, and lift are tied with candidate 0.

Ties abstain. Array order may break a tie for reproducibility but may not turn a
tie into an accepted label.

## Controls recorded with the same bank

- `sample0`: original candidate 0;
- `random_nonzero`: deterministic random nonzero candidate;
- `direct_physical`: winner under the immediate physical signature;
- `continuation_winner`: robust six-repeat winner.

The controls receive the same accepted state IDs if a training pilot is allowed.

## Frozen feasibility checks

All must pass:

1. at least 120 accepted fitting states and 30 accepted evaluation states;
2. accepted states cover at least 15/18 fitting sources and 4/5 evaluation
   sources;
3. at least three semantic stages have 20 or more accepted fitting states;
4. continuation-winner mean gain over sample0 has source-bootstrap 95% lower
   bound above zero in both fitting and evaluation subsets;
5. continuation-winner beats `direct_physical` by at least 20% of available
   Oracle gain on evaluation sources;
6. stable-grasp harm is at most 5% in both subsets.

Pass: `PROCEED_CONTINUATION_DISTILL_SEED41_PILOT`.

Failure: `STOP_CONTINUATION_DISTILL_LABEL_BANK`.

No acceptance threshold, repeat count, subset, or stage quota may be changed
after seeing the result. A pass permits one matched-budget seed41 pilot with
ordinary continuation, random-nonzero self-distillation, direct-physical
self-distillation, and continuation-winner self-distillation. It does not permit
opening CCV holdout or confirmation data.


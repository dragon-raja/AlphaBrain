# Policy-response surrogate Gate -1 preregistration

Date frozen: 2026-07-19

Status: mechanism test only. This is not a method-success claim.

## Problem

The frozen Pi0.5 candidate support contains recovery actions, and the exact
future-policy continuation Oracle improved slipped recovery by 15.4 percentage
points. Immediate physical summaries, teacher distance, and the CCV shared-noise
surrogate did not explain that gain cheaply.

This gate tests one narrower hypothesis:

> After executing a candidate prefix, does the frozen policy's next action
> distribution contain source-generalizing information about long-horizon
> continuation viability that is absent from the candidate and immediate
> physical endpoint alone?

The candidate method is a **policy-response-equivalent surrogate**: learn only
the part of action-conditioned dynamics needed to reproduce how the deployed
policy will respond, rather than reconstructing pixels or predicting a generic
task value. Gate -1 tests the target before training such a deployable model.

## Data boundary

- Source: `/share/longjunyu/fresh-vla/ccv-vla/gate0-coupled-v3`.
- Only records whose existing metadata says `source_partition=fit` may be opened.
- The six CCV holdout source IDs, original test split, and every confirmation or
  sealed path remain unopened.
- The engineering-excluded source remains excluded.
- Existing six-repeat continuation profiles are frozen labels and may not be
  changed.
- Endpoint policy responses use a new seed namespace and are not reused from the
  random streams that generated the labels.

## Frozen intervention and features

For each fit state and each of its 16 saved candidates:

1. Restore the saved simulator/controller snapshot.
2. Execute exactly the saved two-action candidate prefix in an isolated branch.
3. Query the same frozen seed-41 Full-H Pi0.5 at the true endpoint with two
   coupled, label-disjoint flow-noise draws.
4. Save candidate prefixes, response chunks, existing direct summaries, and
   existing six-repeat continuation profiles. Endpoint images and simulator
   state are not saved in the learning view.

The response upper bound may use the first eight actions from each returned
chunk. A future deployable model must predict this response representation from
the current Pi0.5 feature, proprioception, and candidate only. It may not execute
the candidate in a simulator at deployment.

## Frozen source split and models

Fit source IDs are divided once by SHA-256 using salt
`policy-response-gate-minus1-v1`: five source IDs form the evaluation subset and
the remainder form the fitting subset. No state or frame split is allowed.

All probes are ridge regressors with alpha selected by source-grouped
cross-validation over `{0.01, 0.1, 1, 10, 100}`. Candidate features and targets
are centered within state so probes cannot win by memorizing episode progress.

Compare:

1. `candidate`: two-action candidate prefix only;
2. `direct`: existing immediate physical signature only, a privileged upper
   baseline that is not deployable;
3. `response`: endpoint policy responses only;
4. `candidate_response`: candidate prefix plus endpoint policy responses.

## Metrics

The statistical unit is source initial state. Report:

- utility gain over candidate 0;
- fraction of available six-repeat Oracle gain recovered;
- within-state pairwise concordance;
- exact Oracle top-set hit rate;
- stable-grasp harm rate;
- paired source bootstrap 95% intervals with 20,000 resamples.

## Frozen decision

Proceed to a deployable response-model Gate 0 only if all hold on the nested
evaluation sources:

1. available Oracle gain has a source-bootstrap lower bound above zero;
2. `candidate_response` recovers at least 35% of available Oracle gain;
3. its recovered-gain fraction exceeds both `candidate` and `direct` by at least
   10 percentage points, with paired bootstrap lower bound above zero;
4. its stable-grasp harm rate is at most 5%;
5. `response` pairwise concordance exceeds 0.55.

Pass: `PROCEED_POLICY_RESPONSE_MODEL_GATE0`.

Failure: `STOP_POLICY_RESPONSE_SURROGATE`.

A failure forbids training a pixel world model, scalar critic, or renamed
first-passage head on this same evidence. A pass permits only a deployable
current-observation/candidate-to-policy-response predictor, followed by fixed-K
closed-loop validation.


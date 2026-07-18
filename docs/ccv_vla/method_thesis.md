# CCV-VLA: Coupled Continuation Viability for VLA Policies

## Research question

Why can a VLA contain a locally correct recovery action in its sample support, yet fail to
compose that action into recovery, transport, and task completion?

Our working answer is **continuation-confounded action credit**. A candidate action chunk is
usually labelled by the return of one downstream rollout. That return mixes two effects:

1. the physical consequence of the candidate chunk; and
2. luck in all later samples from the stochastic base policy.

This is especially damaging after contact failures. Several candidates can look equally good
under immediate geometry, while only some leave the system inside a state from which the base
policy can reliably continue.

CCV-VLA asks a narrower, testable question:

> Can a lightweight deployable model identify the candidate chunk that preserves the largest
> policy-conditional set of reachable task milestones, when trained from exact sibling states
> evaluated under coupled downstream policy noise?

## Method

At an observation `o`, sample `N` action chunks from a frozen VLA. Execute the first `K`
actions of each chunk from an exact simulator snapshot to obtain sibling endpoints. From every
endpoint, roll out the same frozen policy with the same continuation noise seed for each repeat.

For candidate `i` and continuation seed `xi`, record a viability signature

```text
Z_i(xi) = [stable grasp, lift, transport, task success, no regression, progress AUC].
```

The first four entries are represented as a cumulative milestone-survival vector. They obey
`P(success) <= P(transport) <= P(lift) <= P(stable grasp)`. Regression risk and progress AUC
remain separate diagnostics because they are not members of that ordered chain.

The policy-conditional viability profile is

```text
V_pi(o, a_i) = E_xi[Z_i(xi)].
```

Candidates differ for the first `K` physical actions and share the same state-conditioned base
policy afterwards. Matching `xi` across sibling endpoints removes downstream policy luck from
pairwise comparisons. The deployable critic receives only frozen Pi0.5 observation features,
robot state, and a candidate action prefix. It never receives simulator state, object pose,
contact labels, branch outcome, future frames, or rollout summaries.

The critic predicts the viability profile, not a single return. Selection is conservative:

1. compare candidates at the deepest reliably predicted milestone;
2. back off to earlier milestones when deeper predictions are tied;
3. use regression risk and progress only as tie breakers;
4. retain candidate 0 when no margin is calibrated above the abstention threshold.

The initial deployment is a best-of-16 reranker. Distillation into one-shot VLA generation is
out of scope until reranking improves closed-loop behavior.

## What is and is not new

Common random numbers are not a new contribution. Yadav et al., *Using Common Random Numbers
for Simulation-based Planning with Rollouts* (RLJ 2026, arXiv:2605.04732), provide the relevant
variance-reduction result. CCV-VLA applies depth-coupled evaluation to stochastic generative VLA
actions and builds a training interface around exact physical sibling endpoints.

Likewise, best-of-N action selection and scalar critics already exist, including VGAS
(arXiv:2602.07399), DICE-RL (arXiv:2603.10263), and Pre-VLA (arXiv:2605.22446). A result is only
publishable as CCV-VLA if the experiments establish all of the following:

- coupled sibling labels are more sample-efficient than independent-rollout labels;
- the ordered viability profile predicts held-out long-horizon preferences better than terminal
  success or scalar dense-return critics;
- conservative reranking improves recovery and full-task success without trading away normal
  behavior;
- gains survive source-group splits, multiple seeds, and at least one later second environment.

If only generic scalar value selection works, the result belongs to the existing critic literature
and CCV-VLA is stopped or reframed without a novelty claim.

## Why this hypothesis is grounded in the existing AlphaBrain results

- FRESH suffix weighting improved an offline prefix metric but not closed-loop recovery.
- Recovery replay, recovery SFT, and explicit recovery prompts did not solve the behavior.
- Correct local recovery actions already appear in Pi0.5 support at very high recall.
- The main funnel loss occurs after regrasp, during transport and completion.
- Exact sequential policy-continuation selection improved slip success by 15.4 percentage points
  over single-sample execution and by 25.6 points over random selection in the existing validation
  experiment.
- Immediate physical and teacher-distance selectors did not reproduce that gain.

These observations isolate a real capability gap: action support exists, but the policy cannot
cheaply identify which local action leaves a viable downstream continuation.

## Scope

The first gate uses only the existing LIBERO train split. Validation is opened only after the
mechanism gate passes. Existing test and sealed confirmation groups remain unopened. This phase
does not train a world model, alter Pi0.5, use RL, add dynamic horizons, or access RoboCasa.

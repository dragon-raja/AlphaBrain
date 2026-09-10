# CCV Postmortem and the Next Problem-First Hypothesis

Date: 2026-07-19

Status: research design, not a validated method claim.

## What survived falsification

The repeated experiments isolate a narrower problem than generic VLA failure prediction:

> Given several plausible chunks from a frozen stochastic VLA, identify the chunk whose endpoint
> remains completable by that same policy after contact outcomes and observation feedback change.

Three observations support this problem definition.

1. Correct recovery actions already occur in Pi0.5's candidate support.
2. Immediate geometry, teacher distance, progress, and per-branch constants did not reliably select
   them.
3. Exact future-policy continuation selection improved recovery, and the final CCV collection found
   action-dependent continuation viability in 43.36% of states.

The failed CCV gate says that repeatedly simulating the frozen policy with shared flow noise is not
the right amortization mechanism. It does not say that the policy-relative signal is absent.

## Nearest-work boundary

The action-critic neighborhood is already crowded. RoVer learns process reward and an action-space
direction; VGAS learns a geometric chunk value; Pre-VLA predicts safety and critic-derived
advantage; ProgressVLA guides actions with predicted progress. Common random numbers for rollout
planning also have an independent general treatment. A new paper cannot claim novelty for
best-of-N, a scalar critic, progress guidance, or shared rollout noise.

Relevant primary sources:

- [RoVer](https://arxiv.org/abs/2510.10975)
- [VGAS](https://arxiv.org/abs/2602.07399)
- [Pre-VLA](https://arxiv.org/abs/2605.22446)
- [ProgressVLA](https://arxiv.org/abs/2603.27670)
- [Common random numbers for rollout planning](https://arxiv.org/abs/2605.04732)
- [Forward-backward and successor representations](https://arxiv.org/abs/2209.14935)

## Superseded draft: FOVEA-VLA hypothesis

Update on 2026-07-19: this section is retained as research history, but FOVEA is
paused before implementation. The BASIN result and the crowded scalar/process
critic literature make a new first-passage head premature. The active follow-up
first tests whether endpoint frozen-policy responses are a sufficient statistic
for long continuation ranking; see
`docs/policy_response_vla/method_and_related_work.md` and
`docs/policy_response_vla/gate_minus1_preregistration.md`.

Temporary name: **FOVEA-VLA**, Frozen-policy Occupancy Viability for Efficient Action selection.

Instead of fitting a generic return critic, learn an action-conditioned, policy-relative
first-passage representation. For ordered task events `G_1, ..., G_M`, define

```text
F_m^pi(o, a, tau) = P_pi(first enter G_m within tau steps | o_0=o, a_0=a).
```

The continuation policy `pi` is the exact frozen VLA being deployed. The representation predicts a
vector of finite-horizon first-passage hazards, not one scalar task return. It is learned with an
absorbing Bellman relation from ordinary trajectory transitions:

```text
F_m^pi(o, a, tau)
  = E_o' [ I[o' in G_m]
           + I[o' not in G_m] E_a'~pi F_m^pi(o', a', tau-1) ].
```

For manipulation, the event vector can be `stable grasp -> lift -> transport -> place`, plus a
cause-specific regression/drop hazard. Monotone event survival enforces the task partial order.
At selection time, a candidate replaces sample 0 only when a calibrated lower confidence vector
lexicographically dominates it. The abstention rule is part of the method, not a tuned deployment
patch.

## Why this is materially different

- **Policy-relative:** the target asks whether the deployed VLA can continue from the endpoint,
  rather than whether an expert or an optimal controller could recover.
- **First-passage structured:** it distinguishes never reaching a milestone, reaching it late, and
  regressing after it. Scalar success or progress critics collapse those cases.
- **Bellman amortized:** every trajectory transition supplies a consistency target. It does not need
  six simulator continuations for every candidate label.
- **Conservative improvement:** calibrated vector dominance can yield a testable no-degradation
  statement for accepted reranks; abstention leaves the base policy unchanged otherwise.

Successor representations and first-passage analysis are not individually novel. The candidate
claim requiring validation is their combination as an ordered, frozen-VLA-relative action
selection object, together with censored event learning and conservative vector dominance.

## Minimal falsification before a large experiment

Do not open the six CCV holdout source IDs during method development.

1. Use only existing fit-source trajectories and create a nested source split.
2. Train identical-capacity terminal scalar, dense scalar, progress, and FOVEA heads from the same
   frozen Pi0.5 observation/action features.
3. Measure first-passage calibration, source-held-out candidate regret, accepted-rerank precision,
   Oracle gain recovered, and stable-grasp harm.
4. Require FOVEA to beat both scalar critics in candidate regret and recover at least 35% of the
   available Oracle gain while harming stable grasp in at most 5% of accepted states.
5. Only then freeze the model and abstention rule, open the existing holdout once, and proceed to
   fixed-K closed-loop LIBERO validation.

The route should be stopped immediately if a dense scalar critic matches the structured model, if
Bellman consistency does not improve source-held-out ranking, or if accepted reranks cannot be
calibrated without near-total abstention.

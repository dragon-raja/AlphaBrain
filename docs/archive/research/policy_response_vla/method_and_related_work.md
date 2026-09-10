# Policy-response-equivalent dynamics: method boundary

Date: 2026-07-19

Status: hypothesis under Gate -1; no method-success or novelty claim yet.

## Root problem

The current LIBERO evidence forms a consistent chain:

1. post-slip feedback is visually observable;
2. Pi0.5 often samples locally correct recovery actions;
3. fixed shorter execution alone does not solve full recovery;
4. immediate contact, progress, and teacher-distance targets do not identify the
   useful candidate;
5. exact frozen-policy continuation search improves slipped recovery from the
   single-sample baseline by 15.4 percentage points;
6. coupled common-noise rollouts expose action leverage but do not reduce label
   variance enough to justify the CCV amortizer.

The unresolved question is therefore not generic failure detection. It is:

> Which candidate moves the system to a state from which this particular frozen
> VLA will produce a coherent sequence of regrasp, lift, transport, and place
> responses?

## Candidate representation

Let `pi` be the frozen VLA and `T` the environment transition. For a candidate
chunk prefix `a`, define the policy-response random variable

```text
R_pi(o, a; xi) = pi(T(o, a); xi),
```

where `xi` is fresh flow noise. A policy-response-equivalent dynamics model
`M_phi` need not reconstruct `T(o,a)` in pixels. It is sufficient for the
downstream selector if

```text
M_phi(o, a, xi) ~= R_pi(o, a; xi)
```

and if errors in this response representation preserve candidate order under
the frozen continuation policy. The intended deployment object is thus an
action-conditioned model of *future policy behavior*, not a simulator image and
not a scalar reward.

The first experiment does not train `M_phi`. It asks whether the true endpoint
response itself predicts the six-repeat continuation ordering on unseen source
states. If this upper bound fails, learning the representation cannot repair the
mechanism.

## Nearest work and non-claims

- [Value Equivalence](https://arxiv.org/abs/2011.03506) establishes that a model
  can preserve planning-relevant Bellman updates without reconstructing all
  transition detail. Policy-response equivalence is an application-specific
  sufficiency proposal, not a replacement for that theory.
- [TOM](https://arxiv.org/abs/2305.12663) learns policy-aware models by matching
  transition occupancy. The present target predicts how one fixed stochastic
  VLA responds after each candidate and tests candidate-order preservation; it
  does not reweight a replay buffer for generic MBRL.
- [World-Action Model](https://arxiv.org/abs/2603.28955) adds inverse action
  prediction to visual latent dynamics and trains a policy inside the world
  model. The candidate here intentionally removes pixel reconstruction and
  evaluates a frozen policy-response target before any imagined RL.
- [Feedback World Model](https://arxiv.org/abs/2605.15705) corrects latent world
  predictions online from execution feedback and guides a diffusion policy.
  The present hypothesis is a cheap pre-execution candidate surrogate; it does
  not maintain an online latent observer.
- ACD-VLA in this repository tried to predict detailed post-outcome policy
  responses before the contact outcome. It failed against per-outcome constants.
  The current target is different: condition on the current observation and a
  proposed action, and predict the response at that action's endpoint. It does
  not precommit to an unobserved branch.
- CCV-VLA estimated final milestone profiles by repeated simulator continuation.
  The current representation uses those profiles only as frozen labels and asks
  whether one next-policy response is a lower-variance sufficient statistic.

No claim is made for generic world-model novelty, best-of-N novelty, a new
critic, or a new successor representation. A publishable contribution would
require all of the following: source-generalizing response sufficiency, a
deployable response predictor that beats scalar/direct baselines, fixed-K
closed-loop gains, and transfer beyond the one LIBERO event.

## Why first-passage FOVEA is paused

The earlier FOVEA draft proposed ordered first-passage viability. That is a
reasonable baseline, but the existing BASIN experiment already found weak
cross-policy preference variation, and scalar/process critics plus successor
representations are crowded. Introducing another viability head before proving
what information the expensive Oracle uses would repeat the same research
mistake. FOVEA remains a structured baseline only if the response mechanism
passes; it is no longer the active method hypothesis.

## Evidence ladder

1. **Gate -1:** true endpoint response must explain long continuation order on
   nested fit-source evaluation data.
2. **Gate 0:** predict the response from deployable current Pi0.5 feature,
   proprioception, and candidate; compare equal-capacity scalar, profile, and
   response heads.
3. **Closed loop:** rerank with fixed `K=2`, no simulator and no extra heavy VLA
   call; improve slipped recovery while preserving attached/no-intervention.
4. **Generalization:** new task/event types and at least one second benchmark.

Failure at any level stops the representation rather than adding a larger world
model or tuning the old Oracle boundary.


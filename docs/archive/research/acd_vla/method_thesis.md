# ACD-VLA: Amortized Contingency Distillation

Status: working research thesis. This document freezes the problem and the
falsifiable claim; it is not a claim of demonstrated closed-loop improvement.

## Problem

An action-chunked VLA maps the current observation to one linear future:

```text
o_t -> (a_t, ..., a_{t+H-1})
```

This representation is wrong near stochastic contact events. The useful future
is a policy fragment: execute a shared prefix, observe the physical outcome, and
then take an outcome-conditioned continuation. Calling a large VLA after every
new frame restores this reactivity but gives up the latency benefit of chunks.

## Core hypothesis

ACD-VLA distills future evaluations of a frozen reactive teacher into one
earlier policy call. Counterfactual rollouts expose possible observations
`o_(t+tau)^m`. The teacher is queried at each future observation and supplies
the corresponding continuation:

```text
pi_teacher(o_(t+tau)^m) -> A^m
```

The student receives only information available at `t` and emits a compact
contingency set:

```text
F(o_t) -> {(z^1, A^1), ..., (z^M, A^M)}
```

`z^m` is a key for a possible future observation and `A^m` is the teacher's
response to that observation. During execution, a cheap encoder embeds the
latest real observation, matches it to a key, and selects the corresponding
continuation. Future observations, branch outcomes, simulator state, and oracle
labels are training-only information; none is provided to the deployed policy.

The intended gain is **reactivity amortization**: approach the behavior of a
teacher queried after every consequential observation while retaining a longer
interval between expensive VLA calls.

## What is technically distinct

- Adaptive chunking and event-triggered methods shorten or discard one stale
  trajectory and then call the heavy policy again.
- Lightweight correctors modify one nominal trajectory online.
- World-model methods sample trajectories and match predicted latent state to
  the observed rollout at test time.
- ACD-VLA instead distills the heavy policy's *future observation-conditioned
  responses* into the current output representation. Runtime routing is cheap;
  no heavy replan or latent rollout is required at the represented branch.

This is a learned finite policy fragment, related to classical contingent
policy trees. The publishable claim cannot be "branching exists". It must be
that teacher-response distillation is a practical, scalable action
representation for contact-rich VLA control and improves the
success/latency frontier over strong 2026 baselines.

## Closest work audited on 2026-07-18

| Work | Mechanism | Boundary from ACD-VLA |
|---|---|---|
| [SV-VLA](https://arxiv.org/abs/2604.02965) | lightweight verifier triggers heavy replanning | ACD precomputes observation-conditioned teacher responses in one earlier call |
| [DREAM-Chunk](https://arxiv.org/abs/2606.18589) | latent world model matches sampled chunks to observed rollout | ACD distills future policy evaluations and does not run a world model at inference |
| [DCDP](https://arxiv.org/abs/2603.01953) | lightweight dynamic correction of a nominal diffusion chunk | ACD selects qualitatively different continuations rather than residual correction |
| [AAC](https://arxiv.org/abs/2604.04161) | entropy-controlled adaptive chunk length | ACD changes the output object, not only its execution horizon |
| [VLA-Corrector](https://arxiv.org/abs/2607.01804) | detect deviation, truncate, and correct/replan | ACD compiles anticipated corrections before the event |
| [Deep contingency planning](https://arxiv.org/abs/2104.10558) | learned conditional plans for interactive driving | establishes the general value of contingencies, but not VLA teacher-response distillation for contact-rich chunks |
| [Co-pi-tree](https://arxiv.org/abs/2606.08596) | distills LLM reasoning into a discrete collaboration policy tree | different domain, action scale, supervision, and representation |
| [VLA-AD](https://arxiv.org/abs/2605.16241) | distills a large VLA into a fast reactive student | compresses the whole policy; it does not emit and route future contingent continuations |

## Minimal theory target

Let `C_pi` be the cost of a heavy policy call, `C_g` the cost of the guard, and
`K` the number of control steps between heavy calls. Standard closed loop costs
approximately `C_pi` per step; linear chunks cost `C_pi/K` but cannot condition
the chunk suffix on observations arriving inside the chunk. A depth-one ACD
fragment costs approximately `C_pi/K + C_g` and can represent `M` distinct
suffixes.

For a stochastic event with outcomes `m`, a single deterministic continuation
has irreducible response-imitation risk equal to the conditional variance of
the teacher responses. An oracle-routed branch set removes the between-outcome
term; its remaining error is branch prediction plus routing error. Gate 0 tests
whether that removed term is real and learnable after strong template controls.

## Evidence ladder

1. **Policy-response Gate 0:** on train/val counterfactual twins, predict the
   frozen Pi0.5 responses to attached and slipped feedback from the shared
   pre-feedback observation. Beat stale execution and per-branch constants.
2. **Closed-loop mechanism Gate:** fixed heavy-call budget, compare linear
   chunks, frequent teacher replanning, adaptive truncation, residual
   correction, and ACD on held-out simulator states.
3. **Diversity Gate:** multiple tasks and event types; branch targets must not
   collapse to task/outcome templates.
4. **Scale Gate:** integrate a set-valued head and learned latent keys into a
   VLA, then measure success, latency, calibration, and branch coverage.
5. **Confirmation:** only after all choices are frozen, open untouched test
   groups and a second benchmark/robot embodiment.

At every gate, source snapshot/task is the statistical unit. A failure is
recorded rather than repaired with an unregistered sweep.

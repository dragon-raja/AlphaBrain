# Counterfactual Action-Field Completion

Status: frozen before CAFC policy training

## Question

Can a flow-matching VLA acquire an unseen source-target combination from an
action-free fourth observation when the three observed combinations determine
the missing local action field?

The corrected v13 experiment removed the source-target frequency shortcut.
It also showed that CABI role geometry and prefix transport can improve source
selection, but do not make the nonlinear action decoder compose. This study
therefore constrains the action target itself rather than adding another latent
geometry loss.

## Method

At a phase-aligned compositional tetrad, let `b`, `s`, and `t` denote the
observed base, source-swapped, and target-swapped action chunks. The action-free
fourth conditioning receives

`a_hat_fourth = a_source + a_target - a_base`.

CAFC applies ordinary Pi0.5 flow matching to that derived chunk:

`x_tau = tau * epsilon + (1 - tau) * a_hat_fourth`

`u_tau = epsilon - a_hat_fourth`.

No fourth-corner expert action, continuation, reward, success, or future image
is loaded. The completion commutes with the existing affine action
normalization. It is active only on the v13 `source_select` and `target_select`
tetrads, with fixed weight `1.0`, anchor period `32`, and no clipping or weight
sweep. Deployment is the unchanged single-pass Pi0.5 policy.

## Disclosed development evidence

The state-0 teacher-QA trajectories were inspected before freezing this
amendment, so state 0 is development evidence and cannot be presented as a
sealed result. Against those held-out teacher chunks, the three-corner
completion has source-decision MSE below `4e-16` on both missing edges and
target-decision MSE `0.00484/0.00972` with cosine `0.9951/0.9877`.

This establishes that the local action relation is plausible on one disclosed
state. It does not establish that a learned policy follows it, that the
relation transfers to new states, or that closed-loop tasks succeed.

## Frozen arms

All arms use v13 records, seed 41, the original Pi0.5 initialization, identical
data order, optimizer, normalization, batch size, 33,000 updates, and fixed
`K=3` evaluation.

1. Existing plain BC.
2. Existing CABI action bridge plus decoder closure.
3. `CAFC`: plain Pi0.5 plus the action-field completion loss.
4. `Bridge+CAFC`: the frozen CABI action bridge plus CAFC, without decoder
   closure.

The primary method is CAFC. Bridge+CAFC is a preregistered interaction arm: it
tests whether explicit role grounding is necessary in addition to action-level
completion. It cannot retroactively replace CAFC without being reported as a
different method.

## Gate

State 0 is an implementation/calibration gate only. Full validation starts if
an arm retains at least `3/4` observed-edge successes and achieves at least one
of two action-free successes. The formal migration result is computed on
uninspected validation states with the same fixed `K=3` protocol.

Advance beyond seed 41 only if the best CAFC arm clears both its exact
architecture-matched comparator (BC for CAFC; Action Bridge for Bridge+CAFC)
and the preselected strongest non-CAFC comparator, Action Bridge plus Decoder
Closure. The latter was selected before CAFC training because it is the only
existing arm to reach `3/4` observed state-0 successes. Against both controls,
the CAFC arm must:

- improves mean held-out full-task success by at least 10 percentage points
  over the best non-CAFC comparator;
- improves both held-out edges rather than only one memorized corner;
- loses no more than 5 percentage points on observed edges;
- improves correct-source reach as well as final placement; and
- remains positive under paired state-group bootstrap intervals.

If CAFC passes, the factor-null-dropout control remains mandatory because
Factored Diffusion Policies also composes conditional generative fields. CAFC's
distinct candidate claim is narrower: supervised three-corner actions can
complete an unlabelled fourth visual-language conditioning during training,
then distill that completion into one ordinary policy evaluation per flow step.

## Decisions

- `ADVANCE_CAFC`: plain CAFC clears the validation gate.
- `ADVANCE_GROUNDED_CAFC`: only Bridge+CAFC clears it; role grounding and
  action completion are both necessary.
- `STOP_CAFC`: neither arm improves held-out closed-loop behavior.
- `BASELINE_INVALID`: neither CAFC arm reaches `3/4` observed success.

No dynamic horizon, replanning, recovery controller, world model, RL, hidden
fourth action, or inference-time composition is permitted in this gate.

## Nearest-work boundary

- Factored Diffusion Policies uses explicit factor tokens, independent null
  dropout, and additive score composition requiring multiple conditional
  evaluations for its headline method. CAFC uses matched counterfactual
  observations and observed action effects to train a normal joint policy.
- Action with Visual Primitives supervises a kinematic VLM-to-action interface.
  CAFC uses no primitive coordinates, camera calibration, or intermediate
  decoder.
- OA-WAM builds persistent object-addressable world slots. CAFC adds no world
  head or inference-time slot interface.
- Entity-factored control architectures impose entity structure in the policy.
  CAFC instead imposes a local zero mixed-difference constraint on the action
  field over an incomplete factor graph.

The novelty claim remains provisional until the behavioral gate and a broader
pre-submission literature audit both pass.

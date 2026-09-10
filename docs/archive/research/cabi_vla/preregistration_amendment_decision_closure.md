# CABI-VLA Preregistration Amendment: Decision-Point Functional Closure

Amendment date: 2026-07-22

Evidence available at amendment time: training-state calibration, action-free
offline diagnostics, and the v10 state-0 gate. No validation or sealed-test
rollout from the corrected view was inspected.

## Trigger

The v10 view repaired source/state/phase coverage, but 11,000 optimizer updates
covered it only once. Plain BC solved `0/4` supervised state-0 tasks and CABI
solved `1/4`; the formal decision is therefore `BASELINE_INVALID`. The observed
corner MSE remained `0.1915` for BC and `0.1760` for CABI, unlike the fitted v9
diagnostic. A clean 33,000-update comparison is running before any amended
method is eligible for interpretation.

The invalid run nevertheless exposes two training-only mechanism failures:

1. source identity changes the teacher action at `episode_start`, whereas target
   identity has no action effect there;
2. target identity changes the teacher action at the final pre-transport state,
   whereas the source effect is much weaker there.

The original CABI objective grounded both roles at the latter state. It could
learn separated role geometry without connecting the relevant role to the
deployed action function at the state where that role determines behavior.

## Amendment

The amended method retains ordinary single-pass Pi0.5 inference and introduces
no detector, object slot, world model, planner, query, or held-out action.

### Decision-point transport

Each action-free tetrad carries one active causal role:

- `source_select`: frame zero, source transport only;
- `target_select`: final pre-transport observation, target transport only.

The flow-transport loss is averaged only over the active role. This prevents an
inactive factor with near-zero action effect from dominating the supervision.

### Functional fourth-corner closure

Let `v_theta(x_t, t | h)` be the Pi0.5 flow field. For the normal action-free
fourth corner `h_4` and its decision-appropriate transported representation
`h_comp`, the new loss is

`L_decoder = ||v_theta(x_t, t | h_4) - v_theta(x_t, t | h_comp)||^2`.

The probe `x_t` is generated from the fourth corner's zero/no-action placeholder
and random flow noise. No fourth-corner action, continuation, reward, success,
or outcome is loaded. The ordinary fourth-corner forward pass is therefore
trained to realize the causal composition instead of merely matching its latent
role geometry.

The fixed pilot weight is `decoder_closure=0.25`. No sweep is permitted before
the first behavioral gate.

## Data audit

The amended view is:

`/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v8-decision-coverage-phase-loss-balanced`

- 20,352 regular records, byte-identical to v7;
- records SHA256 `a6979a7db136bac4bd7673f7ce6109b0ed49c946061751f77b23a26ee7dbfee8`;
- 64 source-decision and 64 target-decision tetrads;
- exact paired conditioning at every active intervention;
- source intervention action MSE range `0.1328..0.4981`;
- target intervention action MSE range `0.57143..0.57144`;
- zero `white-right` or `yellow_white-left` anchor arrays;
- every fourth corner remains action-unsupervised.

## Re-entry rule

1. Finish the 33,000-update v10b BC/static-CABI comparison on v7.
2. If BC still fails the `3/4` supervised state-0 gate, report
   `BASELINE_INVALID`; do not train the amendment.
3. If BC is valid and static CABI already transfers, run the preregistered
   factor-null-dropout and ablation controls before using the amendment.
4. If BC is valid but static CABI has no behavioral transfer, train both an
   equal-data BC control and decision-closure CABI on v8 with the same 33,000
   updates, seed, order, initialization, and fixed `K=3` evaluation.
5. A positive result still requires both held-out edges, at least +10 points
   over BC, no more than -5 points ID degradation, and a factor-null-dropout
   comparison. Latent geometry alone cannot pass.

## Implementation validation

- 28 focused data/model tests pass (16 data tests and 12 model tests).
- Both decision points complete a real Pi0.5 forward and backward pass.
- Decoder-closure loss and CABI gradients are finite and non-zero.
- Peak allocated memory in the two smoke batches is below 10 GiB.
- With `decoder_closure_weight=0`, the v10 checkpoint reproduces its exact
  fixed-seed action fingerprint after the code change.

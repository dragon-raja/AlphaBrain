# CABI-VLA conditional transfer bound

## Scope

This note states the narrow theorem that could support CABI. It does not claim that
robot dynamics, diffusion scores, or manipulation tasks are universally additive.
Contact, occlusion, and object-target geometry may create a nonzero source-target
interaction residual; the bound must expose rather than hide it.

## Four-corner notation

Let `00`, `10`, and `01` be the three action-labelled source-target corners and `11`
the action-withheld corner. At one observation, flow time, and noisy action, denote the
expert velocity by `v*_{ij}` and the policy velocity by `v_{ij}`. Define:

- observed-corner error
  `delta = max_{ij in {00,10,01}} ||v_{ij} - v*_{ij}||`;
- model rectangle defect
  `epsilon_C = ||v_11 - v_10 - v_01 + v_00||`;
- environment interaction residual
  `rho = ||v*_11 - v*_10 - v*_01 + v*_00||`.

`rho` is a property of the task and conditioning, not an optimization error. CABI is
appropriate only where role recombination is approximately invariant so that `rho` is
small over the evaluated distribution.

## Proposition 1: fourth-corner flow error

For any norm,

`||v_11 - v*_11|| <= epsilon_C + 3 delta + rho`.

### Proof

Add and subtract `v_10 + v_01 - v_00` and
`v*_10 + v*_01 - v*_00`, then apply the triangle inequality. The four terms are the
model rectangle defect, three observed-corner errors, and the expert interaction
residual. No unseen action is required to optimize `epsilon_C`, but unseen actions are
required to estimate `rho` and verify the final bound after training.

## Proposition 2: representation closure contribution

Let `h_11` be the normal-pass fourth representation and `h_ST` the representation
obtained by source-then-target transport from the base. Assume the flow decoder is
`L_h`-Lipschitz in its conditioning representation. If

- `||h_11 - h_ST|| <= epsilon_A` (action-free anchor closure), and
- the transported decoder rectangle defect at `h_ST` is at most `epsilon_T`,

then

`epsilon_C <= L_h epsilon_A + epsilon_T`.

The CABI single-swap behavior loss targets `epsilon_T`; anchor, commutator, cycle, and
specificity losses target different contributors to `epsilon_A`. The commutator term is
not itself a transfer theorem: it only limits dependence on swap order.

Combining both propositions yields

`||v_11 - v*_11|| <= L_h epsilon_A + epsilon_T + 3 delta + rho`.

## Empirical obligations

The bound is informative only if all terms are addressed:

1. report observed-corner behavior and flow error (`delta`);
2. report normal-pass fourth-corner closure and transport diagnostics
   (`epsilon_A`, `epsilon_T`);
3. use sealed fourth-corner expert rollouts only after training to estimate whether
   `rho` is small enough for the benchmark;
4. report cases with high interaction residual rather than excluding them after seeing
   policy results;
5. demonstrate closed-loop success, because a small one-step flow bound need not remain
   small under compounding control error.

## Relation to factored diffusion

Factored Diffusion Policies derive score composition under conditional factor
assumptions and compose factor contributions during inference. The candidate CABI
distinction is different and narrower: learn which visual entities instantiate
language-defined roles, use an action-free fourth corner to constrain that binding, and
return to one normal policy pass at deployment. Proposition 1 alone is not novel; any
submission claim must rest on learned multimodal binding plus behavioral evidence and
the required factor-null-dropout control.

# CABI-VLA: preregistered migration gate

## 1. Research question

Can a continuous embodied policy learn **causally reusable entity-role bindings** so that
an action decoder trained on three corners of a source-target tetrad transfers to the
unlabelled fourth corner?

The first real-domain gate uses one LIBERO scene with three manipulable mugs and two
destination plates. Every entity, role word, scene geometry, and pick-place skill is
observed during training. Only two source-target edges are withheld from action
supervision.

This is an **action-free transductive composition** gate: training-state images and
instructions for the two fourth-corner edges are visible to the representation
objective, but their actions, continuations, outcomes, and validation/test states are
not. It is not presented as zero-shot recognition of a completely unseen edge.

This gate does not test recovery, replanning, candidate selection, resampling, or a
dynamic execution horizon.

## 2. Hypothesis

Standard flow matching can fit observed source-target edges while retaining an
entangled shortcut. CABI should improve unseen-edge action prediction and closed-loop
success by making the following internal variables reusable:

- `source`: the visual entity denoted by the object phrase;
- `target`: the visual entity denoted by the destination phrase.

The testable claim is not that CABI is the first use of causal intervention training.
It is:

> Training-only multimodal binding transport, constrained by unlabelled fourth-corner
> closure, improves compositional transfer in a continuous generative robot policy
> without changing inference.

## 3. LIBERO-Bind v0

Scene: `LIVING_ROOM_SCENE5`.

Source set:

| ID | Language phrase | LIBERO object |
|---|---|---|
| `red` | red mug | `red_coffee_mug_1` |
| `white` | white mug | `porcelain_mug_1` |
| `yellow_white` | yellow and white mug | `white_yellow_mug_1` |

Target set:

| ID | Language phrase | LIBERO object |
|---|---|---|
| `left` | left plate | `plate_1` |
| `right` | right plate | `plate_2` |

Action-supervised train edges:

`red-left`, `red-right`, `white-left`, `yellow_white-right`.

Action-withheld OOD edges:

`white-right`, `yellow_white-left`.

All six instructions may be paired with training images. For withheld edges, image and
language are permitted only in the CABI representation-closure objective. Their expert
actions, rollout states after the common initial observation, success labels, and test
trajectories are sealed until evaluation.

The closure anchor is taken at the final pre-transport observation, immediately after
the scripted lift. This avoids a weak initial-state diagnostic in which a 10-step action
chunk identifies the source mug but does not yet depend on the destination. For a
withheld edge, the anchor reuses the observed counterpart's physical state and changes
only the instruction; no withheld continuation action is loaded.

Teacher traces enter the training view only when the formal task succeeds. Each
action-supervised edge must retain at least 90% success over the requested state bank,
and a tetrad is built only when all three observed corners exist for the same canonical
state. Failed expert episodes are reported but never converted into policy windows.

### Initialization provenance

Both policies initialize from the same 15,000-update AlphaBrain Pi0.5 checkpoint
trained on `libero_all` (Goal, Spatial, Object, and LIBERO-10). Its LIBERO-10 data
contains a two-stage task in this scene executing `white-left` followed by
`yellow_white-right`; these are two of the four action-supervised edges in the present
gate. The initialization did not use LIBERO-90, where the red single-edge tasks reside,
and its four-suite task list contains neither `white-right` nor
`yellow_white-left`.

The canonical bank for this gate comes from LIBERO-90 `red-left`. It has zero exact
state-vector overlap with the corresponding 50-state LIBERO-10 bank (minimum pairwise
RMS state distance `0.00357`). Thus the test measures recombination of pretrained and
finetuned skills at new state vectors, not learning the manipulation domain from
scratch. This provenance is part of the claim boundary and is identical for every
method.

## 4. State splits

The same canonical LIBERO state bank is used for every edge to remove task-specific
initial-state leakage.

- train: state indices `0..34`;
- validation: state indices `35..39`;
- sealed test: state indices `40..49`.

Report three slices:

1. `ID-new-state`: observed edges on sealed states;
2. `OOD-edge-same-state`: withheld instructions on train states, diagnostic only;
3. `OOD-edge-new-state`: withheld edges on sealed states, primary.

The primary fixed execution horizon is preregistered as `K=3`. Results for `K=1` and
`K=2` are robustness slices and cannot be selected post hoc to change the gate result.

No frame from a sealed trajectory may enter training. Split membership is keyed by
canonical state index, not by frame.

## 5. Method

Let `H(x, l)` be multimodal prefix tokens, `q_r(l)` a learned query for role `r`, and
`B_r` the soft attention from that query to visual tokens. The bound role state is

`b_r = sum_i B_ri V(H_i)`.

A learned low-rank transport writes a donor role into a base representation:

`T_r(H_a, b_r^b) = H_a + A_r(H_a) U_r(b_r^b - b_r^a)`.

`A_r` is a language-conditioned write mask and `U_r` is role-specific. The bases are
learned; no neuron, patch, or hidden slot is manually assigned to an entity.

Training minimizes:

`L = L_FM + lambda_single L_single + lambda_anchor L_anchor + lambda_comm L_comm
     + lambda_cycle L_cycle + lambda_spec L_spec + lambda_orth L_orth`.

- `L_FM`: ordinary Pi0.5 flow matching on supervised edges;
- `L_single`: a one-role transport between observed corners matches the observed anchor
  representation and, with shared flow time/noise, its flow target;
- `L_anchor`: two role transports from an observed corner match the encoded, action-free
  fourth-corner representation;
- `L_comm`: source-then-target and target-then-source transports agree;
- `L_cycle`: swapping a role out and back reconstructs its role state;
- `L_spec`: changing one role preserves the other role state;
- `L_orth`: source and target transport bases avoid collapse onto the same subspace.

Role-state alignment terms use cosine distance rather than raw hidden-state MSE, so
their scale cannot be reduced by shrinking the binding representation. The preregistered
v0 weights are `single=0.25`, `anchor=0.5`, `comm=cycle=spec=0.1`, `orth=0.01`, and
`transport-flow=0.5`.

At inference, the policy performs one ordinary forward pass. There is no transport,
search, candidate generation, planner, or extra observation.

## 6. Required controls

All controls receive the same supervised trajectories, permitted action-free
fourth-corner observations, optimizer budget, flow-noise/time distribution, frozen
modules, and seed set. Within a seed, a dedicated DataLoader generator gives every
method the exact same index permutation; model initialization cannot perturb it. CABI
uses shared flow noise/time within each tetrad transport comparison. Extra
method-specific stochastic operations need not consume an identical global RNG stream,
so final claims require multiple seeds.

Two sampling audits preceded the eligible run. First, attaching a tetrad to every BC
window over-weighted anchors, so anchor period 8 and gradient accumulation 2 were
frozen. Second, balancing raw examples still failed to balance gradient exposure
because action loss is averaged over supervised examples inside each microbatch. The
eligible v5 view therefore balances the actual reduction: 7,488 denominator-4 action
loss units per source factor and 11,232 per target factor in each 5,616-item pass, with
the same macro-phase distribution. All methods use this loss-mass-balanced schedule.

1. `pi05_bc`: ordinary flow matching;
2. `pi05_equal_data`: ordinary policy plus the same image-language-only examples with
   a matched non-causal representation consistency loss;
3. `fixed_slot_iit`: hand-indexed role slots with single-swap IIT, without learned
   binding or fourth-corner algebra;
4. `cabi_no_comm`: CABI without commutator/cycle terms;
5. `cabi_no_unlabelled_anchor`: CABI without fourth-corner anchors;
6. `cabi_full`;
7. `factor_null_dropout`: a shared-policy factor-dropout baseline following the
   factorized diffusion formulation, using the same source/target factors and action
   demonstrations. Any inference-time extra passes are reported explicitly.

An augmentation upper bound may train on withheld expert actions, but it is labelled
`ORACLE_ACTION_AUG` and cannot be treated as a fair baseline.

## 7. Metrics and statistics

Primary metrics:

- OOD-edge-new-state full-task success;
- OOD-edge-new-state source selection success;
- OOD-edge-new-state target placement success;
- ID-new-state full-task success.

Mechanism metrics:

- fourth-corner flow MSE with shared noise/time;
- role grounding accuracy from intervention sensitivity, not a trained probe alone;
- double-swap anchor error;
- commutator norm;
- cycle error;
- specificity leakage into the untouched role;
- normal-pass versus transported-pass agreement.

The independent unit is canonical initial state. Use paired state-level bootstrap 95%
confidence intervals and report each seed separately. Frames are never independent
replicates.

## 8. Decision rule

Proceed to a second scene and full Pi0.5 study only if all conditions hold:

1. `cabi_full` improves OOD-edge-new-state success over `pi05_bc` by at least 10
   percentage points, or its paired 95% interval excludes zero;
2. it improves over both `pi05_equal_data` and `fixed_slot_iit` by at least 5 points or
   a paired interval excludes zero;
3. ID-new-state success degrades by no more than 5 points;
4. both withheld edges improve, rather than one edge carrying the mean;
5. `cabi_no_unlabelled_anchor` or `cabi_no_comm` loses a material part of the gain;
6. ordinary inference reproduces the gain without training-only metadata.

In addition, a submission-level novelty claim requires CABI to outperform
`factor_null_dropout` on OOD-edge-new-state success or match its success with a
materially cheaper single-pass inference path. Beating BC alone is insufficient.

If the frozen or lightly tuned baseline cannot solve supervised edges above 70%, the
result is `BASELINE_INVALID`. If representation closure improves but action or success
does not, the result is `NO_BEHAVIORAL_TRANSFER`. If equal-data or fixed-slot IIT
matches CABI, the claimed contribution is rejected.

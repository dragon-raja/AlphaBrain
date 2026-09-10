# CABI-VLA Preregistration Amendment: Identifiable Binding

Amendment date: 2026-07-22
Evidence available at amendment time: training-split action-free geometry and
state-0 calibration only. No validation/test action labels or rollouts were
used to design this amendment.

## Why an amendment was required

The original 3000-update CABI-v0 checkpoint satisfied its representation
closure while learning a degenerate role system:

| Geometry metric (64 training tetrads) | CABI-v0 |
|---|---:|
| source/target role cosine distance | 0.000000084 |
| language-read attention overlap | 0.99999994 |
| visual attention overlap | 0.997759 |
| write attention overlap | 1.000000 |
| target-role change | approximately 0 |
| target transport relative norm | 0 |

This invalidates closure error as evidence of causal binding. The partial
9000-update v0 run was stopped at approximately step 1450 and is retained as a
failed run.

## Failed identifiability gates

Two bounded, training-only gates were evaluated before any new long run:

1. **v5, intervention margin + attention separation.** At 500 updates it
   separated language and visual attention but retained a shared write mask.
   Target transport relative norm was only `8.5e-8`.
2. **v6, tied read/write + causal factor contrastive loss.** At 500 updates it
   learned the easier source factor (`source margin=0.0937`) but target change
   remained zero. Its factor contrastive loss stayed near the three-way random
   value `log(3)`.

Neither checkpoint is eligible for behavioral claims.

## CABI-v7 amendment

CABI-v7 retains ordinary single-pass inference and adds three training-only
identifiability constraints derived from the tetrad itself:

1. **Tied read/write binding.** A role writes through the same language
   attention distribution used to read it, removing an independent collapsed
   write-mask solution.
2. **Residual role state.** The fused role variable contains a normalized skip
   from its language and visual summaries, so the fusion MLP cannot erase all
   intervention information.
3. **Counterfactual token-change grounding.** The frozen prefix embeddings of
   two paired instructions determine which token positions changed. Source and
   target role queries must place probability mass on the positions changed by
   their respective interventions. No word list, token span annotation,
   withheld action, or outcome label is used.

The existing causal factor contrastive and transport-flow losses are retained.
The new grounding weight is fixed to `0.05`; factor contrastive remains `0.1`,
intervention identifiability `0.5`, and attention separation `0.05`.

## Training-only gate result

After 500 updates, CABI-v7 obtains:

| Geometry metric | v7 |
|---|---:|
| source specificity margin | 0.080944 |
| target specificity margin | 0.092196 |
| source transport relative norm | 0.00003601 |
| target transport relative norm | 0.00003695 |
| language/write attention overlap | 0.00009170 |
| visual attention overlap | 0.005658 |

This passes the identifiability gate only. It is not evidence of action or
closed-loop transfer.

## First long-run result: baseline invalid

The first 9000-update BC run fit supervised action anchors
(`observed-corner MSE=0.000813`) but obtained 0/4 supervised and 0/2 held-out
success at train state 0, fixed `K=3`. It is therefore
`BASELINE_INVALID_AT_9000`; its held-out behavior cannot be used against CABI.

An action-label-free instruction intervention diagnostic exposed the failure:
under one image and shared flow noise, changing the source instruction changed
the predicted action by only `0.0000665` MSE, while changing the target changed
it by `0.5548` MSE. The original 5,611-window view contained 3,595 red-source
windows versus 1,012 white and 1,004 yellow-white windows. Tetrad anchors added
another 4:1 red-source imbalance. This is a factor-exposure confound, not an
eligible compositional-transfer result.

## Factor-exposure amendment

Before inspecting validation or sealed test rollouts, the training view was
reindexed deterministically. An initial v4 view balanced raw supervised-example
counts, but a subsequent reduction audit found that the trainer averages action
loss over the supervised examples inside each microbatch. A regular item thus
has unit loss mass, whereas each of the four supervised examples in an anchor
item has one-quarter loss mass. Raw counts were therefore not the estimand.

The affected v8 runs were stopped before checkpoint eligibility (BC at update
2409 and CABI at update 1948) and carry explicit abort markers. No v8 result is
used for behavioral comparison.

The replacement v5 view changes no image, state, action, split, tetrad, or
held-out-action rule. It changes only deterministic training-window sampling
and is shared by BC and CABI. At anchor period 8, one 5,616-item pass contains:

| Audited quantity | Exact value |
|---|---:|
| source action-loss units: red / white / yellow-white | 7,488 each |
| target action-loss units: left / right | 11,232 each |
| regular edge records: red-left / red-right | 840 each |
| regular edge records: white-left / yellow-white-right | 1,968 each |
| held-out action records loaded | 0 |

Loss units use denominator 4 and explicitly model the trainer's
`mean_over_supervised_examples_per_microbatch` reduction. The four supervised
edges also share one macro-phase distribution over approach, grasp, lift,
transport, and place. The machine-readable audit is stored in
`libero-bind-v0-train-view-v5-loss-balanced/balance_report.json`.

## Frozen next test

Clean BC and CABI-v7 models are retrained as v9 for 9000 optimizer updates from
the same original Pi0.5 checkpoint, seed 41, anchor period 8, gradient
accumulation 2, and the same loss-mass-balanced v5 data order. The primary
behavioral execution horizon remains `K=3`. The earlier v3 CABI run is retained
only as a sampling-bias diagnostic; v4/v8 is retained only as an accounting
failure record.

The method advances beyond calibration only if ordinary BC first reaches 70%
supervised in-distribution success and CABI-v7 then improves held-out transfer
without more than five percentage points of ID degradation. The original
control and paired-state statistical requirements remain unchanged.

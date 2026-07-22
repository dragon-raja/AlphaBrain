# Factor-null-dropout control for LIBERO-Bind

Preregistered: 2026-07-22, before any corrected-view validation or sealed-test
result was available.

## Purpose

This is the mandatory high-risk behavioral control for a positive CABI result.
It adapts [Factored Diffusion Policies](https://arxiv.org/abs/2605.22596) to the
same Pi0.5 backbone and LIBERO-Bind source-target product space. It is not a
CABI ablation and must not use role transport, fourth-corner action labels, or
CABI closure losses.

The source paper independently drops every named factor to a learned null token
with probability `0.1`. With two factors, its composed score is evaluated as

`v_comp = v_empty + (v_source - v_empty) + (v_target - v_empty)`.

This requires three shared-network evaluations at every flow step. The paper's
single-pass jointly conditioned inference is reported separately and cannot be
silently substituted for the composed headline control.

## Pi0.5 adaptation

The control receives two explicit categorical factors:

- source: `red`, `white`, or `yellow_white`;
- target: `left` or `right`.

Each value has a learned embedding. Each factor also has its own learned null
embedding. The two embeddings are inserted as fixed-position conditioning
tokens after a constant neutral task prompt. The original factor-bearing
natural-language instruction is withheld from this control so that it cannot
leak a dropped value through unchanged text.

During training, source and target are independently replaced by their learned
null embedding with fixed probability `0.1`. The only objective is ordinary
Pi0.5 flow matching on the same action-supervised records. No held-out action,
continuation, reward, success, or outcome is loaded. Factor masks are generated
from a dedicated seed stream and recorded, while the DataLoader permutation is
identical to the paired methods.

At evaluation, report both:

1. `factor_composed`: three flow predictions per integration step using the
   formula above and shared noisy action/time state;
2. `factor_joint`: one ordinary prediction with both factor embeddings present.

Both execute at the same fixed action horizon `K=3` as CABI. Wall time, forward
count, and peak memory are reported because CABI retains one ordinary policy
pass per flow step.

## Fairness constraints

- identical Pi0.5 initialization, trainable modules, optimizer, update count,
  batch size, action normalization, and supervised action records;
- `p_drop=0.1` fixed before the first run, with no sweep;
- no fourth-corner action or policy continuation;
- no external detector, segmentation, object slot, world model, or planner;
- source/target IDs may be used only by this explicitly supervised baseline and
  must be disclosed as a stronger conditioning interface than CABI's language;
- fixed seed 41 pilot, followed by seeds 42 and 43 only if the behavioral gate
  is otherwise positive.

## Trigger and interpretation

Do not train this control while the corrected BC baseline is invalid. Train it
only after static CABI or decision-closure CABI clears the seed-41 migration
gate.

CABI's submission-level claim survives only if it either:

- exceeds `factor_composed` on held-out-edge new-state success; or
- matches its success while preserving materially cheaper single-pass
  inference and the action-free binding claim.

If factor-null dropout matches CABI without the action-free fourth corner, the
claimed CABI mechanism is rejected even if both beat ordinary BC.

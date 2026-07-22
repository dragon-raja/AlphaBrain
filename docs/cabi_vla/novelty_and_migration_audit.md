# CABI-VLA novelty and migration audit

## Claim boundary

CABI is not claimed as the first causal representation method, the first activation
interchange method, the first object-centric robot policy, or the first compositional
VLA. Its defensible unit is the combination of:

1. learned language-to-visual role binding inside a continuous generative policy;
2. role-conditioned low-rank transport rather than a hand-selected hidden location;
3. an action-free fourth-corner anchor;
4. cycle and commutator closure;
5. no inference-time intervention.

Removing item 3 reduces the method toward ordinary interchange intervention training.
Removing items 1 and 2 reduces it toward fixed-slot or object-factored architectures.

The present LIBERO-Bind gate is action-free but transductive: CABI may encode the
fourth-corner image-instruction pair at training states, while its action and
continuation remain sealed. It does not establish zero-shot transfer to a combination
that was absent from every training input. The no-unlabelled-anchor ablation is required
to measure that stronger setting.

## Nearest-work audit

| Family | Shared surface | Non-overlap required for CABI |
|---|---|---|
| [Interchange Intervention Training](https://proceedings.mlr.press/v162/geiger22a.html) | hidden interventions align a network with a causal abstraction | CABI must learn multimodal bindings and demonstrate fourth-corner transfer without counterfactual action labels |
| [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) | learns distributed intervention subspaces instead of fixed neurons | CABI must bind visual entities from language and improve policy behavior, not only recover an interpretable subspace |
| [Factored Diffusion Policies](https://arxiv.org/abs/2605.22596) | one diffusion network composes task factors with null-token dropout and additive scores | CABI must beat a factor-dropout control, show that learned visual role binding matters, and retain ordinary single-pass inference |
| [Robust Skills, Brittle Grounding](https://arxiv.org/abs/2602.24143) | diagnoses failure on held-out object-region combinations while grasping skill remains intact | this paper establishes the failure mode, not a remedy; CABI must improve held-out source-target behavior under the same decomposed source-selection and task-success metrics |
| [OA-WAM](https://arxiv.org/abs/2605.06481) | enforces object addressability with externally extracted identity/content slots and causal slot swaps | CABI cannot claim object addressability alone; it must show action-free combinatorial transfer in a pretrained VLA without a world head, external segmentation pipeline, or inference-time slot interface |
| [Action with Visual Primitives](https://arxiv.org/abs/2605.22183) | inserts a spatially grounded VLM-to-action interface and reports strong real-robot spatial-compositional transfer | CABI cannot claim the first explicit solution to VLA spatial grounding; it must achieve held-out source-target transfer without kinematic primitive labels, hand-eye projection, an autoregressive primitive decoder, or two-stage inference |
| [entity-factored control policies](https://arxiv.org/abs/2203.05960) | explicit entities improve compositional control | CABI must work on pretrained VLA tokens and beat an equal-data entity/fixed-slot control |
| programmatic/neurosymbolic grounding | source, target, and skills are explicitly composed | CABI has no program executor at inference and learns a distributed transport algebra |
| [temporal representation alignment](https://openreview.net/forum?id=yaS3JWQRQ6) | auxiliary representation learning improves compositional robot instruction following | CABI must isolate simultaneous source-target recombination rather than long-horizon skill concatenation |
| [ObjectVLA](https://arxiv.org/abs/2502.19250) | vision-language data improves object-level VLA generalization | CABI must transfer a relation between two grounded roles without extra target-object demonstrations |
| sparse-autoencoder VLA analysis | finds interpretable VLA features | CABI trains behaviorally useful bindings and is not a post-hoc probe |
| activation-patching modularity metrics | causal patching diagnoses compositional circuits | CABI is a training objective; diagnostic metrics alone are not the contribution |

## Identifiability-amendment neighbors

The CABI-v7 amendment introduces counterfactual token-change grounding. A targeted
search found related ingredients but no equivalent robot-policy training protocol:

- [Counterfactual Contrastive Learning for Weakly-Supervised Vision-Language
  Grounding](https://papers.nips.cc/paper_files/paper/2020/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html)
  perturbs visual proposals and cross-modal relations to improve weakly supervised
  grounding. It does not derive role targets from paired robot instructions or test
  fourth-corner continuous actions.
- [COMPASS semantic-role circuits](https://aclanthology.org/2026.findings-acl.1964/)
  uses role-cross minimal pairs and causal tracing to localize semantic-role circuits
  in language models. It is a mechanistic analysis method, not an action-policy
  objective or learned visual binding transport.
- [MoDA](https://arxiv.org/abs/2506.01850) uses instruction-conditioned
  cross-attention for fine-grained visual grounding, but has no tetrad intervention
  algebra, action-free fourth corner, or behavioral composition test.
- [Token Steering](https://arxiv.org/abs/2606.15021) intervenes in action-token space
  at inference to guide VLA trajectories. CABI-v7 instead uses training-only role
  interventions and preserves ordinary inference.

Accordingly, token-difference grounding alone is not claimed as the contribution.
The candidate unit remains its use to identify multimodal role transport from
three action-labelled corners and an action-free fourth corner. This claim still
depends on closed-loop transfer and the factor-null-dropout control.

The May 2026 Factored Diffusion Policies paper is the highest-risk behavioral neighbor.
It makes generic claims such as "factorized diffusion improves compositional control"
indefensible for CABI. The remaining candidate claim is narrower: a pretrained
vision-language policy can learn *which visual entity fills each language-defined
role* from three action-labelled corners plus an action-free fourth corner, and can use
that binding under an ordinary inference pass. A fair factor-null-dropout baseline is
mandatory before any submission-level claim.

The May 2026 OA-WAM paper independently makes generic "object-addressable policy"
claims indefensible. It uses foundation-model segmentation and tracking, persistent
identity/content slots, a world-prediction head, and an address-only attention pathway.
CABI remains distinct only if its role binding is learned directly over a pretrained
VLA prefix, needs no explicit object slot at inference, and transfers behavior from an
action-free fourth corner. If those conditions do not survive ablation, the novelty
claim must be withdrawn rather than reframed after the result.

The June 2026 revision of Action with Visual Primitives is an even closer capability
neighbor. AVP predicts source/destination visual primitives between the VLM and a
flow-matching action expert, supervised by projecting end-effector kinematics through
calibrated cameras. It reports large real-robot gains, including unseen direct spatial
transitions. CABI therefore has no defensible claim to inventing a grounded
VLM/action-expert interface or to solving spatial compositionality in general. Its
remaining distinction is supervision and execution: an action-free fourth-corner
constraint, no primitive coordinates or camera calibration, and one normal policy
pass. If AVP code becomes available, an equal-backbone primitive-supervision upper
bound is required before submission; its current "code coming soon" status does not
block the present falsification gate.

The February 2026 Robust Skills, Brittle Grounding study makes the problem statement
stronger but narrows the empirical obligation. It reports that held-out object-region
pairings can destroy instruction-conditioned reach while leaving generic grasp motion
partly intact. CABI must therefore report correct-source reach and source-target task
success separately; smooth motion or grasp-anything success is not evidence of
compositional grounding.

The related-work search must be rerun before submission. A new paper that combines
learned visual role queries, activation transport, and unlabelled tetrad closure would
invalidate or narrow the claim.

## Internal failure evidence

Existing artifacts under `/share/longjunyu/capt-vla` establish two facts:

- explicit or synthetic compiled causal composition can work across multiple seeds;
- prior real-token CIP/ICWM gates did not establish robust scene-conditioned action
  transfer.

Therefore a new synthetic win is insufficient. The unresolved migration question is:

> Can the role variable be identified from real visual-language tokens strongly enough
> that its algebra predicts held-out robot behavior?

The LIBERO-Bind gate is designed around that question. It must not reuse FRESH
attached/slip data as positive evidence because those records vary physical outcome,
not language-to-entity role composition.

## Migration risks and controls

| Risk | Failure signature | Required control |
|---|---|---|
| task-specific initial-state leakage | instruction recoverable from state/image file identity | one canonical state bank shared by every edge |
| extra action supervision | CABI sees withheld actions through generated labels | withheld edge actions are sealed; only image and instruction enter closure |
| factor-exposure confound | policy responds to target but ignores an underrepresented source | source and target action-loss mass is balanced under the trainer's actual per-microbatch reduction, after tetrad insertion, for every method |
| trivial language memorization | correct plate but wrong mug, or vice versa | source and target subgoal metrics plus single-role swaps |
| explicit slot advantage | fixed hand slots equal CABI | `fixed_slot_iit` control |
| generic regularization | equal-data consistency equals CABI | `pi05_equal_data` control |
| generic factorized diffusion | factor null-token dropout and score composition equal CABI | `factor_null_dropout` control with identical action data |
| explicit spatial supervision | kinematic visual primitives equal or exceed CABI | AVP-style primitive upper bound when public code is available; disclose its extra labels and two-stage inference |
| latent metric without behavior | low closure error but unchanged success | behavioral gate overrides representation metrics |
| intervention-only inference | gain disappears on normal forward pass | normal-pass evaluation is mandatory |
| additive toy dynamics | synthetic pass fails on contact-rich policy | no positive claim before LIBERO full-task success |
| benchmark overfitting | gain only in one scene or one held-out edge | both edges, then a preregistered second scene |
| static-factor timing mismatch | role geometry closes but source selection and target placement remain unchanged | supervise source transport at the approach decision and target transport at the pre-transport decision; compare with the original single-anchor method |

## Theory target

A submission-level theorem should be conditional, not universal. Under role-local
mechanism invariance, a Lipschitz flow decoder, and bounded single-swap, anchor, and
commutator errors, the fourth-corner flow error should be bounded by observed-corner
error plus those representation errors. The proof must expose every assumption and
must not imply that arbitrary manipulation dynamics are additive.

Until that proof and the real gate both pass, CABI is a falsifiable research candidate,
not a validated algorithm.

# CCV-VLA Related-Work Audit

Audit date: 2026-07-18.

## Closest families

| Family | Representative work | Overlap | Required distinction |
|---|---|---|---|
| Best-of-N VLA critics | VGAS, arXiv:2602.07399 | Scores sampled chunks with a learned critic | CCV must beat scalar value ranking through coupled labels and ordered viability, not merely rerank |
| Runtime action verification | Pre-VLA, arXiv:2605.22446 | Predicts safety and advantage before execution | CCV predicts policy-conditional downstream reachability from exact sibling interventions |
| Value-guided robot RL | DICE-RL, arXiv:2603.10263 | Uses values for selection and finetuning | Initial CCV freezes the policy and tests causal sibling supervision without online RL |
| Coupled rollout estimation | Yadav et al., arXiv:2605.04732 | CRN reduces relative-utility variance after a shared rollout policy | CRN is prior art and a tool, not the novelty claim |
| Progress estimation | ProgressVLA, arXiv:2603.27670; PALM, arXiv:2601.07060 | Estimates task progress or completion | CCV estimates action-conditioned future milestone survival under a particular base policy |
| Adaptive execution | AAC, arXiv:2604.04161; A3, arXiv:2605.11567 | Changes action chunk length or replanning rate | CCV keeps `K` fixed and changes candidate identity only |
| Counterfactual robot data | PAIR-VLA, arXiv:2605.13105; PACT, arXiv:2606.03949 | Uses paired or counterfactual behavior data | CCV pairs exact physical sibling actions and downstream stochastic continuations for credit assignment |

## Crowded claims that CCV must not make

- "VLA needs a critic."
- "Best-of-N improves action generation."
- "Progress is useful for long-horizon tasks."
- "Common random numbers reduce variance."
- "Counterfactual data improves robustness."
- "A multi-head value network is novel."

## Falsifiable novelty boundary

The credible contribution is the complete combination:

1. exact same-state sibling action interventions for a generative VLA;
2. depth-coupled base-policy continuations producing low-variance pairwise credit;
3. an ordered policy-conditional viability target that identifies where continuation fails;
4. a conservative deployable reranker that recovers the expensive continuation Oracle while
   abstaining when the evidence is weak;
5. source-grouped closed-loop evidence that this improves failure recovery rather than only an
   offline score.

Removing any of points 2-5 risks reducing the work to an existing critic or dataset recipe.

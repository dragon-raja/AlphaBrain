# Active View Search for Frozen Flow-Based Robot Policies

Working manuscript seed, 2026-10-25. Not a submission-ready abstract. Results and scope remain conditional; this file does not authorize experiments. See the [delivery and evidence plan](paper_delivery_plan_20261025_zh.md).

## Abstract scaffold

Robot policies operate on observations produced by a particular camera configuration, yet a reproducible camera pose need not provide the most useful view after the scene changes. We study whether a frozen flow-based manipulation policy can supply a useful signal for selecting observations before acting. Our approach scores candidate observations using the policy's action-generation dynamics and learns a separate, query-limited camera-search policy with a stopping action. The manipulation policy remains fixed throughout search-policy training and evaluation. We compare staying at the perturbed view, returning to a known nominal pose, local score-guided search, a matched random-selection control, and learned search under paired initial states and execution noise. [Insert audited perturbation loss, net recovery, additional harm, and cost on the declared population.] [Insert independently completed real-robot and second-model evidence, or remove these claims.] The study examines both the utility and the failure modes of the internal score, rather than assuming that a lower score implies a higher task-success probability.

Do not submit the bracketed scaffold. Do not describe the base VLA as untrained on LIBERO, the search policy as training-free, or the tested setup as arbitrary physical camera motion.

## 1. Introduction — initial prose

Changing the camera configuration can alter a robot policy's action predictions even when the manipulation goal is unchanged. A practical response is to return the camera to a saved pose. This is a strong solution when the pose is known, reachable, and still suitable for the scene. However, restoring joint angles or camera extrinsics does not restore the arrangement of objects and occluders. A previously useful pose may therefore become less informative, while a new installation may not have a task-specific saved pose at all. These cases motivate choosing observations for the current situation rather than assuming that a single stored configuration is sufficient.

The relevant question is not whether every task benefits from camera motion. Moving a camera consumes time and can expose a view on which the policy performs worse. We ask when a limited amount of active view search can improve manipulation relative to the current observation, and whether that improvement remains useful against direct nominal-pose return when the latter is available. This formulation admits an unchanged view or immediate stopping, and requires measuring both rescued failures and newly introduced failures.

We investigate this question using a frozen flow-based manipulation policy. For each queried observation, a score derived from its action-generation dynamics supplies a candidate-selection signal. A separate PPO policy chooses continuous camera adjustments or stopping under a fixed query budget. It does not update the manipulation policy, and task execution outcomes are not supplied as online search feedback. This separation lets us evaluate whether the internal signal contains useful information and whether learning improves how the query budget is spent.

The central difficulty is that a proxy for action-generation behavior is not a task-success model. A low score may indicate a view that produces consistent but incorrect actions. Furthermore, the scale of the score changes across manipulation models, potentially changing the effective penalty for acquiring another view. We therefore isolate a reward-scale correction while retaining the existing network, stopping action, query penalty, and training budget. We evaluate final task outcomes independently from search reward and report harms as well as gains.

Active perception and camera-motion policies already have substantial precedents, including [ActiveVLA](https://arxiv.org/abs/2601.08325), [LIME](https://arxiv.org/abs/2607.02417v2), [SaPaVe](https://arxiv.org/abs/2603.12193), and [See2Act](https://arxiv.org/abs/2606.23625v2). Our intended contribution must be established through the specific combination of a fixed manipulation policy, its internal flow-derived signal, and bounded search evaluated against strong return-to-pose and nonlearned alternatives. We do not claim that active viewpoint selection itself is new.

## 2. Problem and information contract

Define manipulation policy pi, search policy mu, observation history h, feasible camera actions U(h), current view v0, query cap Q, and task outcome Y. Distinguish a saved pose v_nom from an oracle best view. State whether pi was benchmark-adapted upstream; specify exactly which tasks/initial states are held out from mu. Training success labels, if any, must be documented separately from test-time information.

Three deployments to distinguish: (i) known/reachable nominal pose in a stable scene; (ii) same pose with a changed scene; (iii) no task-specific nominal configuration, with a general home/fixed-view baseline. Current full simulation tests (i); claims about (ii)/(iii) require new evidence.

Primary outcomes: success, rescue, harm, net recovery, queries, camera motion and end-to-end latency. Cost is reported explicitly; search reward is not the performance metric.

## 3. Method — freeze before filling details

Document the implemented Accel definition, action coordinates, flow time grid, noise members and aggregation directly from frozen source. Do not replace it with physical acceleration or invent an uncertainty interpretation.

The isolated reward comparison uses:

\[
r_t = (A^*_{t-1}-A^*_t)/s - 0.01\,\Delta Q_t.
\]

A uses the original scale 0.16535300100378683. B uses the training-only fixed median 0.01214765438717935. Both retain the observation normalization, history encoder, continuous motion, STOP, PPO settings and query limit. Specify how initial and final return/recheck queries are counted using frozen implementation; the equation alone is not the complete episode accounting.

Describe actor/critic training and final-checkpoint export. Independent score evaluations are not used for selecting the returned view. Confirmation-based acceptance and supervised task-value heads are not part of the current A/B method.

### 3.1 Online-scale extension under evaluation, 2026-10-27

This extension is a separate, approved development pilot, not a retroactive change to A/B. Let \(b_t\) be the lowest Accel observed so far. After collecting a complete on-policy batch \(u\), let \(S_u\) be the median of all initial-view mean scores collected during the current adaptation phase, including this batch. Each episode contributes one initial score regardless of its search length. The online global comparison uses

\[
r_t^{G}=\frac{b_{t-1}-b_t}{S_u}-0.01\Delta Q_t.
\]

The local-relative candidate uses the same online reference statistics:

\[
\tau_u=\max\{0.1S_u,\operatorname{quantile}_{0.75}(d_1,\ldots,d_n)\},\qquad
r_t^{L}=\log\frac{b_{t-1}+\tau_u}{b_t+\tau_u}-0.01\Delta Q_t,
\]

where \(d_i\) is the population standard deviation of the two initial scoring members, equivalently their absolute half-difference. The quantile uses linear interpolation. Raw score differences and query increments are retained during collection; rewards and GAE are computed once the batch closes, and the scale stays fixed throughout all PPO epochs. Action sampling in this implementation does not depend on immediate rewards. No separate calibration queries, operation-success labels, nominal target scores, or global view optima are used to estimate these statistics. The cost of the initial query remains an explicitly logged episode constant; return/recheck queries remain charged decision costs. Undefined zero scale stops the update rather than inventing an arbitrary unit floor.

The denominator \(b_t+\tau_u\) makes a given absolute gain more valuable in a low-score region, while bounding its amplification near zero. This is a design motivation, not a guarantee of escaping local optima or improving manipulation. Selecting a minimum over noisy scores remains optimistic; a two-member floor is not a confidence bound. The pilot independently scores both the final candidate and the starting reference with the same held-out noise members, after selection. These four diagnostic model calls are explicitly budgeted and do not train the policy or change its final selection.

The optimizer is standard [PPO](https://arxiv.org/abs/1707.06347). Online target-scale estimation has precedents in [PopArt](https://arxiv.org/abs/1602.07714); robust return scaling and bounded amplification appear in [DreamerV3](https://arxiv.org/html/2301.04104v2). We do not implement their complete normalization systems or claim their results. Learning costly search decisions using RL also precedes this work, including [MetaBO](https://arxiv.org/abs/1904.02642). Our intended method contribution is the information-restricted, query-accounted use of a frozen manipulation policy's internal score, with online scale estimation and a separately tested local-relative utility. Neither PPO, normalization, nor the logarithm itself is claimed as novel. Novelty and practical value must be judged against the related literature and task-level restoration evidence.

The first screen forks three B checkpoints into matched B/G/L branches, 256 episodes and eight updates per branch, and evaluates all branches on the same fixed development panel. B continuation controls for the extra training. L already combines the online global reference with local-relative utility; adding another division by \(S_u\) would be a different, untested objective. This warm-start screen cannot establish calibration-free learning from scratch, arbitrary cross-model transfer, or few-episode physical adaptation. The policy's input normalization and the simulation's installation-coordinate prior remain unchanged.

## 4. Evaluation — prespecified structure

- Main native model: 40 tasks, 400 evaluation initial states, 27 baseline views and two paired execution seeds; dense baseline 21,600 episodes.
- Search evaluation: eight starts per state; methods and seeds differ by development/seen/unseen partition according to the frozen schedule. Total 18,240 searches and 36,480 execution episodes; do not imply every method was run on every split.
- Report task-level paired statistics and intervals, plus all task outcomes. Do not pool dependent views as independent task samples.
- Three project model conditions are checkpoints within one model family, not three architectures. The main campaign uses the native checkpoint; auxiliary model panels must be labeled separately.
- Main simulation currently rotates a virtual camera at a fixed optical center. The LIBERO-plus runtime is not evidence of completing the full LIBERO-plus benchmark.
- Real robot: [hardware, compatible base policy, camera calibration, scene blocks, order balancing, actual motion time, all failures; pending].
- Independent FM family: [checkpoint, interface validation, fixed task panel and information contract; pending].

## 5. Results slots — no invented values

Audited aggregate baseline snapshot, 2026-10-25: nominal 761/800 (95.125%); perturbed stay 13,035/20,800 (62.6683%); task-macro paired loss 32.4567 percentage points, task-cluster bootstrap 95% interval [26.62,38.68] points. All 3,600 result aggregates match their completion hashes and the exact 400-state × 27-view × 2-repeat population. This establishes degradation on the declared panel, not recovery or physical deployment utility. The 800 nominal outcomes are paired references reused across angles, not independent nominal repeats for every perturbed view. Recovery, model transfer and real-robot results remain pending.

Table 1: nominal / perturbed-stay / local / random / A / B on matched declared populations; success, net recovery and cost.

Figure 1: deployment regimes and information available to each method.

Figure 2: full perturbation response and paired losses, including tasks with little or no degradation.

Figure 3: spatial correspondence of independent Accel and success, with representative successes and failures selected by a declared rule.

Figure 4: search quality versus query and wall-clock cost; reward normalization and STOP diagnostics.

Table 2 / Figure 5: physical saved-pose return versus scene-adaptive search, with true camera motion and replicated reset blocks.

Table 3: independent-model panel and same-family checkpoint controls, without implying a full cross-product evaluation.

## 6. Discussion and decision rules

If perturbation loss is small, report limited opportunity for recovery. If local/learned search lowers Accel but harms success, identify proxy failure. If nominal return dominates in stable scenes, retain it as the preferred practical solution in that regime. If B only changes query count, do not claim successful reward calibration at the task level. If external model or real-robot evidence is incomplete, narrow the claim rather than imply generality.

No test-outcome-driven adjustment of reward scale, task selection, checkpoint choice or scene severity within the current evaluation. Any follow-on algorithm revision needs a new development protocol and a clearly distinguished test set.

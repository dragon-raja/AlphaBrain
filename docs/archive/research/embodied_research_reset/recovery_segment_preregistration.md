# 固定预算闭环恢复段预注册

## 研究问题

旧 Gate 0 曾显示单个 `K=3` action chunk 可能影响稳定重抓，但该运行后来因 runtime snapshot 不完整而失效，不能作为已证事实。当前正式重跑不把整个 winner trajectory 都标成好动作，也不做 inference-time best-of-N；只验证：

> 在恢复段的连续 replanning 节点上，重复进行合法的同状态 sibling intervention，能否把局部动作信用累积成持久的 transport 与 full-success 改善？

该方案只是训练数据识别与 policy-improvement 上界，不是最终部署算法，也不预设某个命名方法成立。只有动作效应跨 continuation 可辨、可由部署信息预测，并且 posttrained Pi0.5 的 `sample0` 无搜索推理提高成功率，学习路线才成立。根因与候选方法见 `root_problem_method_review.md`。

## 与整段拒绝采样的区别

不从同一初态采样 N 条完整轨迹后给全段 winner/loser 标签。第一步后轨迹观测已经不同，这种全段配对会重新引入错误的时间信用。

本轮固定执行 4 个 replanning 节点。每个节点都在当前真实状态重新：

1. 采样 N=4 action chunks；
2. 在与 live control 隔离的 branch env 中，从完全相同的 simulator/controller snapshot 执行每个候选；
3. 用 matched frozen-policy lookahead 估计候选的物理过程结果；
4. 只在该节点内形成 action preference；
5. 在从未被搜索 rollout 或 restore 改写的 live env 中重新执行选出的 action prefix；
6. 读取真实新观测后进入下一节点。

因此每个训练 pair 的候选共享当前状态；不同节点之间不构造伪反事实动作对。

## 固定配置

- task：`put_the_cream_cheese_in_the_bowl`
- source：完整 LIBERO v2 val snapshot groups
- model checkpoints：Full-H seeds `41, 42, 43`
- intervention segment：恰好 `4 replans x K3 = 12 actions`
- candidate count：`N=4`
- lookahead：30 actions
- selection continuations：3
- decision-heldout continuations：5，只审计每个节点的局部排序稳定性
- full-heldout continuations：5，从固定 12-action segment endpoint 继续到 120-action 总预算，作为 Gate 主结果
- total live control budget：120 actions
- stage dwell：2 actions
- Oracle selection key：`formal success -> non-regress -> transport -> lift -> next-stage -> stable-regrasp -> quantized progress AUC` 的固定 lexicographic rule
- segment 不因 grasp/lift/transport event 提前结束；只有正式 task success 可终止 episode
- decision-heldout 和 full-heldout continuation 均不参与候选选择、segment 长度或任何阈值调整
- 每个候选的 K3 selection endpoint 只在 branch env 物理执行一次，selection 与 decision-heldout 必须从同一个已保存 endpoint 分叉
- candidate 0 额外做一次同 snapshot、同动作的 K3 branch replay；它只用于核对分支环境可复现性，不参与选择、训练标签或搜索收益，成本单列为 parity audit
- 选出的 K3 action prefix 必须在未被搜索改写的 live env 重新执行，并与 branch endpoint 在像素、robot state、sim state 和物理语义上严格匹配

所有方法从同一 feedback snapshot 开始，执行动作数和观测频率相同。搜索 rollout 只存在于隔离的 branch env，不得改变 live env；它不计入真实控制动作预算，但必须单独报告 simulator transition 成本。

runtime snapshot 必须覆盖 MuJoCo flattened state、model body/friction/site visibility、gripper、OSC goal/orientation/update flag、position/orientation interpolator、robot history buffers、environment timestep、observation cache、每个 Observable 的采样时钟和缓存值，以及接触求解会延续到下一步的 `qacc_warmstart`、active acceleration/constraint caches、外力、control 与 mocap runtime arrays。只验证 restore 当下的 sim state 或像素不充分；必须验证同动作 K3 endpoint。长程接触动力学会把 `1e-14` 级浮点差异放大，因此不把“额外 restore 后 120 步像素 bit-exact”当作有效门槛；真实 live rollout 必须完全隔离，长程效果由相同 branch protocol 的 held-out repeats 估计。旧 Gate 0 因缺少这些运行态而标记为 `INVALIDATED_REQUIRES_RERUN`，不得并入本轮结果。

## 方法与对照

1. `sample0`：自然且不中断的 Full-H seed schedule；
2. `random4`：每个节点从同一 N=4 pool 随机选一个，使用 3 套预注册随机选择 schedule 后聚合；
3. `myopic_stage`：只看候选 K3 后的真实短程 stage/progress，不使用 future continuation；
4. `receding_oracle`：用 3 套 matched lookahead 选择，5 套 held-out 只审计。

后续训练必须在等 simulator-transition 数据预算下比较：

- recovery-success-only SFT；
- VLAC-CUT-style progress/recovery segment curation；
- state-only stage reward / SARM2-style weighting；
- AFIL-style failure-negative guidance；
- unpaired outcome weighting；
- 同状态 sibling winner-only；
- 同状态 pairwise physical advantage。

[PACT](https://arxiv.org/abs/2606.03949) 已在 human intervention state 做反事实动作偏好与 credit reassignment，[SARM2](https://arxiv.org/abs/2606.10305) 已用 stage reward 做 on-policy VLA improvement，[AFIL](https://arxiv.org/abs/2605.08434) 已用失败 rollout 形成负指导，[VLAC-CUT](https://arxiv.org/abs/2607.09776) 已做 progress/failure/recovery segment curation。当前方案不能靠“用了 segment 或 preference”成立，只能靠同状态物理 sibling credit 在等预算下带来额外 sample0 闭环收益成立。

## 数据隔离

训练视图只含：

- 当前 agent/wrist images；
- 当前 robot state；
- N 个实际执行候选的 `K=3` action prefixes；
- selection physical outcome vectors；
- selection Oracle index。

privileged audit 视图单独保存：

- simulator/controller snapshots；
- policy 未执行 action suffix；
- candidate、selection 与 held-out seeds；
- held-out physical outcome vectors 与 held-out Oracle index；
- friction 与 controller runtime state。

接触约束缓存允许随 active contact 数量改变 shape；privileged audit bank 必须零填充到分片最大 shape，并为每条记录保存原始 valid shape，禁止截断或使用 pickle object arrays。

训练 loader 不得读取 audit bank。未来图像、object pose、branch outcome、sim state 和 seeds 不得输入 policy。

## 统计

- candidate、decision、continuation 和 frame 都不是独立样本；
- 先在 `(model_seed, source_initial_state_index)` 内聚合同源 pair groups；
- 再跨 seed 聚合同一 source cluster；
- 最后按 source initial-state cluster 做 paired bootstrap 95% CI；
- 同时报告每 seed 结果和绝对百分点差；
- 所有 feedback states intention-to-treat，不按 regrasp eligibility 筛选。

## 运行有效性锁

正式 decision run 必须硬锁：

- `N=4, replans=4, K=3, total=120, lookahead=30`；
- `selection=3, decision-heldout=5, full-heldout=5, random schedules=3, dwell=2`；
- split 为完整 val：13 groups、9 source clusters；
- seeds 为 `41, 42, 43`，checkpoint SHA256 与 seed 映射固定；
- Git worktree clean，Git SHA 与本文 SHA256 记录在输出；
- sample0 通过完整 12-action intervention segment 的 live-env-vs-branch-env parity：两条 rollout 使用完全相同的 seed schedule，branch env 在 4 个 replan 边界执行 capture/restore；动作、agent/wrist 图像、robot state 与 success 完全一致，sim/controller 数值误差不超过 `1e-8`；
- 每个 Oracle 干预节点的 candidate 0 必须通过同 snapshot、同动作的两次 K3 branch replay parity；
- 每个最终选中的 K3 prefix 必须通过 branch endpoint 与未受搜索影响的 live endpoint parity，像素和语义结果完全一致，sim/robot-state 数值误差不超过 `1e-8`；
- 各方法首个状态的候选池一致；
- 所有 candidate、continuation、policy-call 和 simulator-transition 成本完整记录。

任何自定义参数、单组运行或 dirty worktree 只能标记为 smoke，汇总器必须拒绝其进入裁决。

## Gate S0

进入 policy post-training 必须同时满足：

- receding Oracle 相对 sample0 的 full-heldout transport `>= +15 pp`，或 formal success `>= +10 pp`；
- source-cluster paired CI 不跨 0；
- 三模型 seed 同向；
- 明显优于 random4；
- 不被 myopic stage selector 匹配；
- drop/regress 不增加；
- decision-level selection 与 held-out 排序具有可审计稳定性。

任一核心条件失败，则停止 simulator sibling-search/physical-preference 路线，转向直接 recovery demonstrations、VLAC-CUT-style curation 或 AFIL/SARM2 一类更简单方法。

## Gate S1

S0 通过后，冻结数据和预算，比较后训练目标。最终必须看到：

- posttrained `sample0` slip recovery/full success 相对 Full-H `>= +10 pp`；
- 相对简单 baseline `>= +5 pp` 或 paired CI 排除 0；
- attached/no-intervention 退化不超过 5 pp；
- 固定 K=1/2/3 趋势一致；
- 推理时 N=1、无 critic、无 simulator、无额外观察频率。

Oracle 通过但 posttrained sample0 失败，只能说明轨迹可被昂贵搜索，不能说明信用学习方法有效。

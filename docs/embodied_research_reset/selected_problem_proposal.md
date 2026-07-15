# 入选课题方案（问题优先修订）：同状态物理反事实过程信用分配

工作描述：**Same-State Counterfactual Physical Process Credit Assignment**。这只是便于讨论的描述，不以命名或避开近邻作为研究目标。

## 1. 为什么修订

此前方案把重点放在“候选动作物理后果 verifier + frozen-policy reranking”。它是合理的最小诊断，但不是对根因的完整解法：

1. reranking 不改变策略生成分布，候选池缺少有效恢复动作时无能为力；
2. best-of-N 收益可能只来自额外推理预算，不能证明策略真正学会恢复；
3. 只观察 K=3 后果过于短视，可能奖励短期接触而破坏后续 transport/place；
4. 普通 state-only progress/reward、failure classifier 或任意候选打分器都可能取得相同收益；
5. 若只强调与 A3、SAFE 或 Reflective VLA 的形式差异，会把热门问题缩成局部插件，构成伪创新风险。

因此，本课题不再把 verifier 本身作为最终方法。verifier 仅是测量和部署组件，主问题改为：

> **失败恢复中的物理过程信用分配：当当前观测已经揭示失败、策略也能生成局部合理动作时，如何识别并学习真正导致稳定接触、下一阶段转移和最终完成的 action chunk，而不是只拟合专家动作或当前状态进度？**

## 2. 已有证据指向的真实断点

当前 Pi0.5 / LIBERO `put_the_cream_cheese_in_the_bowl` 结果表明：

- attached/slipped 在 feedback frame 可由当前视觉与状态 100% 区分；
- post-feedback N=8 中 100% 至少含一个前三步局部正确 recovery mode，sample0 也有 97.4%；
- isolated K3 中 100% 触发 recovery-action proxy，61.5% 完成 regrasp；
- 仅 15.4% 到达 transport 和最终 success，regrasp->transport 条件率为 25.0%；
- explicit recovery prompt、动态/Oracle commitment、自一致性均未修复该断点；
- 训练 suffix weighting 改善部分 offline 指标，但没有转化为闭环恢复收益。

这些证据否定了三个简单解释：反馈不可见、随机 mode 没采到、语言路由不足。剩余核心更像是：

```text
当前状态可辨认
  + 局部恢复动作可生成
  + 训练目标只约束动作相似度
  -> 不知道哪个动作会产生可持续的物理阶段推进
```

标准行为克隆监督的是 `a ~= a_expert`；state-only reward 监督的是 `V(o)`。这里真正需要估计的是动作条件化的物理过程价值：

```text
Q_phys(o_t, a_t:t+K; continuation, tau)
```

## 3. 核心假设

从完全相同的反馈状态出发，不同候选 action chunks 会导致可重复且可区分的物理后果。通过同状态干预得到的配对结果，可以消除状态难度、初始布局和失败强度等混杂因素，形成比“成功轨迹 vs 失败轨迹”更干净的 action-level process advantage。

可证伪拆分：

- **H1 候选因果差异**：同一状态的候选在稳定接触、下一阶段转移和 regress 上存在显著差异；
- **H2 持久影响**：差异不仅存在于执行 K=3 后，也能在统一 continuation 下影响后续阶段；
- **H3 可预测性**：仅使用部署时可得的当前观测、proprioception 与候选动作，可在 held-out source states 排序实际后果；
- **H4 可学习性**：同状态 process preference 后训练能提升单样本策略，而不只是 best-of-N reranking；
- **H5 特异性**：收益稳定优于 state-only reward、未配对 outcome weighting、failure-negative guidance 和专家动作相似度。

H1/H2 不成立，不训练 verifier 或策略；H3 不成立但 H1/H2 成立，仅允许一次 history/geometry 诊断；H4 不成立，则不能把 reranking 收益写成策略学习贡献。

## 4. 方法：同状态物理反事实过程信用

### 4.1 同状态候选干预

对 train/val snapshot groups 的关键恢复状态保存完整 simulator/controller state：

- feedback-reveal attached；
- feedback-reveal slipped；
- policy 自己闭环 regrasp 后的 replanning boundary；
- lift 后；
- transport 前。

主分析不得直接使用 scripted expert 的干净 post-regrasp state 代替 policy-induced state。expert state 只作为“状态分布是否导致饱和”的控制，因为真实断点可能正是策略以较差接触姿态完成 regrasp，随后无法持续 transport。

对每个状态：

1. 从冻结 Pi0.5 采样 `N=8` 个 action chunks；
2. 每个候选都恢复到完全相同的 simulator/controller state；
3. 使用相同物理参数、matched random seeds 和相同执行长度 `K=3`；
4. 保存逐步 contact、grasp、EEF/object pose、图像、stage 与正式 success；
5. 同一 source state 的全部候选始终位于同一 split。

正式结果只允许从 clean Git commit 启动，并记录 commit、checkpoint SHA256、完整参数与 runtime identity。dirty-worktree 运行只能作为开发 smoke，不得进入方法裁决。

状态恢复必须先通过数值和像素 parity。当前初步 parity 已通过：重复恢复与相同三步动作的 agent/wrist 图像误差为 0，sim state 最大误差约 `1.62e-15`。

### 4.2 区分短程后果和可持续后果

仅用候选执行后的 K=3 状态会奖励“暂时抓住但马上掉落”的动作。每个候选同时计算两类标签：

1. **direct effect**：执行 K=3 后的稳定接触、对象位移、drop/regress；
2. **bridge value**：执行候选 K=3 后切换到统一 reference continuation `mu`，继续 `tau` 步或直到下一阶段事件。

主 continuation 使用冻结基线策略；scripted teacher 只作为可达性上界。每个候选至少在三套 matched continuation random numbers 下评估并按平均结构化结果排序；入选候选与 sample0 再用至少两套未参与选择的 held-out continuation 复测。候选之间共享同一 repeat 的随机数，避免把后续策略随机性误记为当前动作贡献，也避免 Oracle 只对一条随机未来过拟合。

默认多时间尺度：

```text
tau_short = 3
tau_bridge = 10
tau_event = until next stage or timeout
```

### 4.3 结构化过程结果

不先把所有后果压成任意单标量。保存：

- stable grasp/contact after K；
- first regrasp/lift/transport/place timestamps；
- next-stage reached；
- object-height 与 object-to-bowl progress AUC；
- drop、regress 和已完成 stage 丢失；
- bridge continuation 的正式 task success；
- completion steps 与控制成本。

阶段必须是带 dwell 的有序状态机：

```text
stable regrasp -> lift -> transport -> formal success
```

`post_regrasp` 的直接 next stage 是 lift，不得跳过 lift 直接把 transport 当作 next stage。单帧 grasp/contact 不算稳定阶段完成。

候选优先关系采用预注册的 lexicographic rule：

1. 避免 drop/regress；
2. 达到下一物理阶段；
3. 更高 progress AUC；
4. 更少完成步数。

只在该规则出现大量平局时，才在 val 上固定一个标量化分数。

### 4.4 同状态 process advantage

同一状态内候选 `a_i` 和 `a_j` 的监督为：

```text
DeltaG(s, a_i, a_j) = G(s, a_i; mu) - G(s, a_j; mu)
```

配对只在同一 restored state 内构造。不同状态的绝对 outcome 不直接形成 preference，避免把“容易状态”误当成“好动作”。

### 4.5 Action-conditioned process critic

critic 输入：

```text
agent-view + wrist-view
robot/proprioceptive state
candidate action chunk a[t:t+K]
optional short action-consequence history (only as ablation)
```

critic 输出多时间尺度结果分布，而不是单一 failure 概率：

- `P(stable_grasp)`；
- `P(next_stage_reached | tau)`；
- progress AUC；
- `P(drop_or_regress)`；
- calibration uncertainty。

推理输入不得包含 simulator object pose、branch outcome、未来图像、teacher action 或 oracle stage label。

### 4.6 两种使用方式

**A. 冻结策略 reranking（诊断与可部署基线）**

- 当前观测采样 N=8；
- critic 排序候选；
- 固定执行 K=3；
- 重新观测并重复。

它回答“策略支持集和 critic 是否足够”，但不作为最终学习主张。

**B. Pi0.5 process-preference 后训练（主方法）**

- 使用同状态 `(a+, a-)` 与 `DeltaG` 更新 Pi0.5 action expert；
- 保留原始成功 demonstration flow-matching loss 作为 anchor；
- 第一轮冻结 VLM/backbone，仅训练 action expert LoRA 或与现有训练配置一致的可训练模块；
- 使用 reference-regularized flow preference objective，pair 内共享 flow time 与 noise；
- 同时实现更简单的 winner-only advantage-weighted flow matching 作为强基线。

一个待验证的最小 pairwise 目标为：

```text
d_theta = flow_loss_theta(a+) - flow_loss_theta(a-)
d_ref   = flow_loss_ref(a+)   - flow_loss_ref(a-)
L_pref  = softplus(beta * (d_theta - d_ref))
L_total = L_anchor + lambda * weight(DeltaG) * L_pref
```

该目标借鉴 flow-matching preference optimization，但不能假设图像生成中的结论自动适用于机器人控制；必须与 winner-only、AFIL-style guidance 和普通 SFT 正面对照。

## 5. 不回避强近邻

本课题不靠“没人做过 reward/preference/recovery”成立，这些方向已有强工作：

| 近邻 | 正面关系 | 本课题必须额外证明的价值 |
| --- | --- | --- |
| [SARM2](https://arxiv.org/abs/2606.10305) | stage-aware dense reward + on-policy policy improvement | action-conditioned same-state causal pairing优于 state-only stage value |
| [AFIL](https://arxiv.org/abs/2605.08434) | 在线失败 rollout、成功/失败双生成器、failure-negative flow guidance | 真实执行后果配对优于仅区分成功/失败分布 |
| [HAPO](https://arxiv.org/abs/2506.07127)、[PACT](https://arxiv.org/abs/2606.03949) | 人类纠正动作偏好与 intervention credit reassignment | 无人工、同一物理状态、多候选实际后果能给出更细的 process credit |
| [Reflective VLA](https://arxiv.org/abs/2606.25215) | observation-action-consequence history 改善部署泛化 | 预测 prospective candidate consequence；history 只作必要性消融 |
| [A3](https://arxiv.org/abs/2605.11567) | 通过 group consensus/conditional invariance 动态接受前缀 | 固定 K，比较候选造成的真实物理进展，而非模型内部一致性 |
| [LifeLong-RFT](https://arxiv.org/abs/2602.10503) | action consistency、trajectory alignment、format process reward | process credit 来自实际物理结果而非参考动作对齐 |
| [Linear-DPO](https://arxiv.org/abs/2605.21123) | diffusion/flow matching 的 preference optimization | 把目标适配并验证到连续机器人 action chunk 与闭环物理结果 |

如果 unpaired weighting、state-only stage reward 或 AFIL-style failure guidance 匹配本方法，应采用更简单方法并放弃同状态因果配对主张。

## 6. 必须比较的系统

### 6.1 冻结策略诊断

- base Pi0.5 sample0，fixed K3；
- random candidate；
- A3/self-consistency candidate；
- expert-action RMSE oracle；
- exact physical-consequence oracle；
- learned action-conditioned critic reranking；
- state-only progress/reward reranking。

### 6.2 策略学习

- original Full-H SFT；
- recovery-success-only SFT；
- unpaired outcome-weighted flow matching；
- state-only stage reward weighting（SARM2-style）；
- failure-negative dual guidance（AFIL-style nearest implementation）；
- correction-action preference（HAPO-style）；
- same-state winner-only flow matching；
- same-state pairwise process-preference flow matching；
- pairwise posttrained policy + critic reranking。

最终必须单独报告：

```text
base sample0
base + reranker
posttrained sample0
posttrained + reranker
```

若只有 `+reranker` 提升，结论只能是 inference-time selection，不得宣称策略学会恢复。

## 7. 数据划分与防泄漏

- train/val/test 按 snapshot group 和 source initial state 分组；
- Gate 0 开发只使用 train/val，test 在方法、阈值和权重冻结后开启；
- 同一 restored state 的全部候选、continuation、随机种子不得跨 split；
- 统计独立单位是 `source_initial_state_index`，不是 candidate、frame 或 pair；
- paired bootstrap 先在 `(seed, source_initial_state_index)` 内聚合 pair，再跨 seed 聚合 source cluster，最后按 source cluster 重采样；
- 报告 source-state cluster 敏感性；
- 所有视频 H.264/avc1、yuv420p、faststart。

candidate bank 必须物理拆分为两类文件：

- **training view**：仅含当前图像、robot state 和实际执行的 `candidate[:K]`；
- **privileged audit view**：完整未执行 chunk、simulator state、friction 和 continuation seeds。

后续 critic loader 只能读取 training view，避免未来动作或 simulator 特权字段泄漏。

## 8. 分层证伪门槛

### Gate 0A：干预有效性

已通过 deterministic restore parity。下一步检查：

- 同状态重复相同动作的 stage/outcome 完全一致；
- 候选改变后，物理结果差异明显大于恢复噪声；
- direct effect 与 bridge value 标签一致性可审计；
- 20 组视频人工确认没有状态恢复、控制器或相机泄漏。

### Gate 0B：候选支持与持久因果收益

在 val pilot 上，不训练网络：

- 至少 70% slipped states 的 N=8 中存在 next-stage-positive candidate；
- physical oracle 相对 sample0 的 regrasp->transport 至少 +15 pp，或 isolated recovery 至少 +10 pp；
- 收益在 `tau_bridge`/next-stage 上仍存在，而非仅 K=3 接触 proxy；
- 三 seed 同向或 group-level paired CI 排除 0；
- attached/no-intervention 退化不超过 5 pp。

未通过则判断为 candidate support/generation 问题，停止 reranker，转向恢复动作生成或新数据后训练。

### Gate 1：critic 可预测且有闭环价值

- held-out source-state pairwise ranking accuracy >=75%；
- next-stage AUROC >=0.75，且 group/bootstrap CI 明显高于 chance；
- learned reranker 恢复至少 50% 的 physical-oracle 闭环收益；
- 稳定优于 state-only reward、self-consistency、entropy 和 expert-RMSE proxy；
- success-cost Pareto 不被更简单 baseline 支配。

### Gate 2：策略真正学会 process preference

- `posttrained sample0` 相对 base slip recovery/full success 至少 +10 pp；
- 相对 unpaired/state-only/failure-negative baseline 至少 +5 pp，或 paired CI 排除 0；
- regrasp->transport、drop/regress 至少一项有明确机制改善；
- attached/no-intervention 退化不超过 5 pp；
- 三 seed 同向；
- 固定 K=1/2/3 下趋势一致，不靠给本方法额外观察频率获益。

Gate 2 失败但 Gate 1 通过：保留窄的 inference-time 方法，不把它包装成通用学习算法。

### Gate 3：跨任务/失败类型

论文级主张前至少增加：

- 两个新的 LIBERO manipulation tasks；
- 至少一种非 slip 失败（misalignment、premature release 或 collision）；
- 新 source-state blind split；
- 第二环境在资源就绪后验证，不让 RoboCasa 阻塞当前 Gate 0--2。

## 9. 下一轮执行顺序

1. 固化 simulator/controller restore parity 测试；
2. 增加 first-stage timestamps、matched continuation 和 structured outcome schema；
3. 在 train/val 小规模状态上生成 N=8 candidate intervention bank；
4. 先跑 Gate 0B physical oracle，不训练任何模型；
5. Gate 0B 通过后，扩展 candidate bank 并训练 action-conditioned critic；
6. Gate 1 通过后，比较 winner-only 与 pairwise process-preference 后训练；
7. 冻结方法后开启 test，并完成固定 K 闭环、统计和视频审计。

本轮不实现 dynamic horizon、horizon head、world model、主动视觉或无界 RL。它们不是当前根因验证所必需的。

## 10. 一句话研究主张

本课题要验证的不是“候选 verifier 是否新”，而是：

> **在失败恢复中，从完全相同物理状态执行不同 action chunks 得到的真实过程差异，能否提供比动作模仿、state-only progress 和粗粒度失败负样本更准确的信用分配，并据此让 VLA 本身学会产生可持续的恢复动作。**

若答案是否定，应直接采用更简单的近期方法或转向动作生成，不为保持独特性继续堆叠模块。

SELECT_NEW_RESEARCH_PROBLEM: 同状态物理反事实过程信用分配

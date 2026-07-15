# 入选课题方案：候选动作物理后果驱动的恢复阶段验证

暂定英文描述：**Candidate-Conditioned Physical Consequence Recovery Verification**。这只是便于讨论的工作名，不主张名称或问题“首次提出”。

## 1. 问题定义

当前 Pi0.5 在 LIBERO slip recovery 中呈现一个具体断点：

1. attached/slipped 在 feedback frame 已能由当前视觉与状态完全线性区分；
2. post-feedback N=8 样本 100% 包含前三步局部正确恢复 mode，sample0 也有 97.4% 位于该 mode；
3. isolated K3 中策略 100% 触发 recovery-action proxy，61.5% 能重新抓取；
4. 但只有 15.4% 到达 transport 和最终 success，regrasp->transport 条件率仅 25.0%；
5. Oracle execution commitment、self-consistency 和显式 recovery prompt 均不能修复该断点。

因此入选问题是：

> 当失败已经可见、局部恢复动作已经存在时，能否依据候选短 action chunk 的预期物理阶段转移来选择动作，使 `regrasp -> transport -> place` 连续推进，而不是依据动作相似度、采样一致性或固定执行长度？

## 2. 核心假设

同一反馈状态下，冻结 VLA 的多个候选 K=3 chunks 虽然常处于同一个粗粒度 recovery mode，但会产生不同的接触稳定性、对象位移和下一子目标结果。一个 candidate-conditioned verifier 可以从当前多视角、robot state、候选动作和可选的短 action-consequence 历史中预测这些结果，并选择真正推进下一物理阶段的候选。

可证伪拆分：

- **H1 候选支持集**：N=8 候选中存在比 sample0 更高概率完成下一阶段的动作；
- **H2 可预测性**：不执行候选的情况下，轻量 verifier 能在 held-out source states 上排序实际后果；
- **H3 闭环转化**：固定 K=3 reranking 提高 regrasp->transport 和 final recovery，且不损害 attached/no-intervention；
- **H4 非平凡性**：收益不能被动作熵、自一致性、expert-action RMSE proxy、普通 progress head 或 failure detector 匹配。

H1 不成立就立即停止，不训练 verifier。

## 3. 与近期工作的创新边界

| 近邻 | 近邻的核心 | 本课题必须保持的差异 |
| --- | --- | --- |
| [AAC](https://arxiv.org/abs/2604.04161)、[A3](https://arxiv.org/abs/2605.11567) | 按熵/共识动态决定执行多少步 | 固定 K=3，只选择哪个候选；输出物理阶段后果而非执行长度 |
| [VLA-Corrector](https://arxiv.org/abs/2607.01804) | 运行时动作错误监测与修正 | 聚焦同一状态候选之间的可验证物理阶段转移，并用配对 candidate rollout 训练 |
| [SAFE](https://arxiv.org/abs/2506.09937) | 从 VLA 特征预测任务失败概率 | 不是当前失败二分类；输出 candidate-conditioned next-stage consequence |
| [RoboFAC](https://arxiv.org/abs/2505.12224)、[FailSafe](https://arxiv.org/abs/2510.01642) | 失败理解、恢复数据与语言纠正 | 不依赖外部语言 supervisor；选择低层动作候选并直接评测物理转移 |
| [Reflective VLA](https://arxiv.org/abs/2606.25215) | 用已发生的 observation-action-consequence 历史适应部署偏差 | 历史只是可选输入；核心输出是对多个候选未来 chunk 的后果排序，必须有 current-only/history-only 对照 |
| [ProgVLA](https://arxiv.org/abs/2605.28231) | progress heads 与 offline RL 加权策略训练 | 第一阶段冻结 policy，以 candidate-level consequence reranking 验证机制 |
| [RoboReward](https://arxiv.org/abs/2601.00675) | 通用视觉语言奖励模型和 RL | 使用 1--3 步、动作条件化的接触/阶段标签，不先做任务级通用 reward 或 RL |
| [ProbeAct](https://arxiv.org/abs/2606.09740) | object probe、运动学状态机和 CBF 修正 | 不假设手工状态机足够；学习候选在不同任务阶段的后果，但必须把 ProbeAct 类规则作为 baseline |

若最终方法退化成 dynamic K、failure classifier、普通 reward model、语言恢复 supervisor 或 history-conditioned VLA，就失去本课题的边界。

## 4. 最小方法原理

### 4.1 固定策略与候选生成

- 冻结 Full-H Pi0.5；
- 每次从当前 observation 采样 `N=8` 个 action chunks；
- 统一只考虑前 `K=3` 步；所有方法保持相同 K，不引入动态 horizon；
- branch outcome、future state、expert action 和 oracle labels 不进入 policy 或 verifier 推理输入。

### 4.2 Verifier 输入

最小 current-only 版本：

```text
agent-view image
wrist-view image
robot/proprioceptive state
current stage estimate
candidate normalized actions a[t:t+3]
```

可选 history 版本只在 current-only 上界不足时增加：最近 `L` 个 `(observation, executed action, observed delta)` triplets。必须同时训练 matched frame-history 和 action-history 对照，以隔离 Reflective VLA 风格 action consequence 的贡献。

### 4.3 Verifier 输出

对每个 candidate 输出：

- `P(stable_grasp_after_K)`；
- `P(next_stage_reached)`，stage 为 regrasp/lift/transport/place；
- 归一化 `delta_progress`；
- `P(drop_or_regress)`；
- calibration uncertainty。

选择分数可先固定为：

```text
score = P(next_stage_reached) + 0.25 * delta_progress - 0.5 * P(drop_or_regress)
```

第一轮不扫大量权重；只在 val 固定一次，并以各输出的独立校准和消融解释收益。

### 4.4 推理流程

1. 当前 observation 生成 N=8 候选；
2. verifier 对每个固定 K3 candidate 打分；
3. 执行最高分候选的 3 步；
4. 获取真实新 observation，更新当前 stage/history；
5. 重复直到正式 LIBERO success 或超时。

不在推理时模拟未来像素，不使用 oracle state，不改变 VLA 权重，不给 Oracle 方法额外观察频率。

## 5. 数据构造

### 5.1 Candidate rollout bank

从 train/val snapshot groups 的关键状态 clone simulator state：

- pre-grasp；
- feedback-reveal attached；
- feedback-reveal slipped；
- regrasp 后；
- lift 后；
- transport 前。

每个状态从冻结 Full-H 三 seeds 各采样 N=8 candidates，单独恢复相同 simulator state 并执行固定 K=3，保存：

- 当前 agent/wrist image 与 robot state；
- candidate actions；
- K 步后的 image/state；
- grasp/contact、object pose、EEF pose；
- drop/regress；
- next-stage predicate 与 delta progress；
- source snapshot group、model seed 和 rollout seed。

同一 snapshot group 的所有 candidate 只能位于同一 split。test candidate outcomes 只用于最终评测，不参与阈值、特征或权重选择。

### 5.2 标签

主标签来自执行后的真实 simulator predicates，不来自 expert-action MSE：

- stable regrasp：event 后恢复有效 grasp 且 K 步末仍保持；
- lift progress：对象高度相对当前状态达到固定增量且未 drop；
- transport progress：对象到 bowl 的 XY 距离下降并达到/接近阈值；
- place success：正式 `env.check_success()`；
- regress：drop、离目标更远、丢失已完成 stage。

必须新增 first-regrasp、first-lift、first-transport、first-place 时间戳，修复当前逻辑漏斗无法证明时序的限制。

## 6. 训练流程

1. 冻结 Pi0.5 和视觉 backbone，先用预提取视觉/state/action features 训练小型 verifier；
2. 对每个 stage 使用 calibrated BCE/ordinal progress loss；
3. 同一状态 candidate pairs 使用 pairwise ranking loss；
4. class/group balance 按 snapshot group，不按帧扩增显著性；
5. 超参数只在 val source states 选择；
6. 若 current-only 已过闭环门槛，不增加 history；
7. 只有轻量 verifier 成功后，才考虑有限 LoRA 或 joint feature adaptation；本阶段不使用 RL。

## 7. Baselines

必须包含：

- frozen Full-H sample0，fixed K3；
- random candidate；
- action entropy/min-variance；
- A3/self-consistency 风格 candidate consensus；
- expert-action RMSE oracle，仅作不可部署上界；
- physical-consequence oracle，执行所有 candidate 后选最优，仅作 H1 上界；
- SAFE-style current failure score + sample0；
- current-only progress head；
- matched history-only 与 action-consequence-history；
- 原任务 prompt、显式 recovery prompt；
- Stage A Oracle Commit 与 gripper boundary 结果。

若开源实现可用，应直接复现 AAC/A3/VLA-Corrector/ProbeAct 的最接近配置，而不是只做自定义弱版。

## 8. 主指标与统计

主指标：

- isolated slip recovery；
- end-to-end slip final recovery；
- overall full-task success；
- attached/no-intervention success；
- regrasp->transport transition；
- transport->place/success；
- final progress 与 progress AUC。

安全/效率：drop、regress、failure-continuation、policy forward calls、verifier latency、wall-clock、success-vs-cost Pareto。

统计：snapshot group 独立；同一 group 内先聚合 candidates/seeds；paired group bootstrap 95% CI；source-initial-state cluster 敏感性；报告每 seed 和绝对百分点差异。不得把 candidate rollouts 或帧当独立任务样本扩大显著性。

## 9. 最小验证与硬门槛

### Gate 0：Physical-consequence Oracle

不训练网络。从每个 test state 的 N=8 candidates 中，执行后按真实 next-stage outcome 选最优，再从相同起点正式闭环复跑。

继续条件：

- candidate pool 在至少 70% 的 slipped feedback states 中包含 next-stage-positive candidate；
- 相对 sample0，`regrasp -> transport` 至少 +15 pp，或 isolated recovery 至少 +10 pp；
- 三 seed 同向或 paired group CI 排除 0；
- attached/no-intervention 退化不超过 5 pp。

未通过：立即停止本课题，不训练 verifier。

### Gate 1：Learned verifier

继续条件：

- held-out candidate ranking AUROC >=0.75，且 source-state cluster CI 明显高于 chance；
- 恢复至少 50% 的 Oracle-vs-sample0 闭环收益；
- 相对 self-consistency/entropy 至少 +5 pp，或 paired CI 排除 0；
- 额外 forward-equivalent 成本下仍位于 success-cost Pareto 前沿。

未通过：判定物理后果虽有 Oracle 信号但当前输入不可预测；只允许做一次 current-only vs consequence-history 诊断，不做无界架构搜索。

### Gate 2：泛化

在新的 LIBERO source states/第二任务上复现，并在 RoboCasa 资源实际迁移到 8 卡机后做第二环境验证。当前智能体不主动访问 4 卡机，也不让 RoboCasa 阻塞 Gate 0/1。

## 10. 消融

- 去掉 candidate action，只看当前 observation；
- 去掉 wrist/agent 任一视角；
- 去掉 robot state；
- current-only vs frame history vs action history vs consequence triplets；
- binary next-stage vs continuous delta-progress；
- 去掉 drop/regress risk；
- N=`2/4/8`，但固定 K=3；
- expert-MSE selector vs physical consequence selector；
- train only success trajectories vs 加入 failed/recovery candidates；
- stage-specific verifier vs shared verifier。

## 11. 两周实验计划

### 第 1 周

1. 为 evaluator 增加 first-stage timestamps 与可恢复 simulator state snapshot；
2. 生成 train/val/test candidate rollout bank，不下载新模型或数据；
3. 做数据审计：candidate diversity、positive coverage、stage 标签一致性；
4. 完成 Gate 0 physical-consequence Oracle 与 paired bootstrap；
5. 若 Gate 0 不过，写 No-Go 报告并停止。

### 第 2 周

1. 训练 current-only 轻量 verifier；
2. 跑 fixed K3 isolated/end-to-end 三 seeds；
3. 与 sample0、random、entropy、自一致性和 current progress baseline 比较；
4. 只在 current-only 预测不足时补 consequence-history 对照；
5. 完成 Gate 1 与视频/统计审计。

## 12. 两个月论文计划

- **第 1--2 周**：LIBERO candidate Oracle、轻量 verifier、完整负对照；
- **第 3--4 周**：新增 2--3 个失败类型和新的 source-state blind split，验证阶段标签泛化；
- **第 5--6 周**：在 RoboCasa 已迁移后做第二环境；若未迁移，改用额外 LIBERO tasks，不主动跨机访问；
- **第 7 周**：真机可行性只做小规模 grasp/drop/recovery，不以真机规模代替统计；
- **第 8 周**：统一消融、成本曲线、失败案例、论文与可复现实验包。

论文主张必须保持窄：candidate-conditioned physical consequence verification improves recovery-stage composition。不得扩张成“解决通用机器人失败恢复”。

## 13. 失败后的转向

- **Gate 0 失败**：候选支持集不足。停止 selector/verifier，转向 recovery data generation 或 policy post-training，但重新预注册，不直接上 RL。
- **Gate 0 通过、Gate 1 失败**：后果可选但不可由当前输入预测；只比较 action-consequence history/几何接触特征。若仍失败，停止。
- **只在 LIBERO 单任务有效**：结论限定为任务特定，不进入论文主张；优先增加新 task/source states。
- **普通 self-consistency 或 current progress 匹配**：放弃原创方法，采用简单 baseline。
- **收益来自更高 forward calls**：按 success-cost Pareto 判定失败。

当前选择的依据是“可观测、局部 mode 存在、regrasp 后阶段流失、简单强控制无效”的联合证据；candidate consequence 本身仍需 Gate 0 证明。

SELECT_NEW_RESEARCH_PROBLEM: 候选动作物理后果驱动的恢复阶段验证

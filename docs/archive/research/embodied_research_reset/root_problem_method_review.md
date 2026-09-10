# 从根因出发的方法复审

## 结论先行

研究目标不是避开热门近邻，而是解决这个可操作的控制问题：

> 当失败结果已经可由当前视觉辨认，基础 VLA 也能生成局部合理的恢复动作时，怎样让策略选择并学会那些能够跨越后续重规划、持续保留可恢复性并最终完成任务的动作？

现有结果已经排除了“反馈不可见”“恢复模式完全采不到”和“只需缩短执行 horizon”这三个简单解释。真正断点出现在：

```text
反馈可见 -> 局部重抓 -> 稳定抬升 -> 搬运 -> 放置成功
                         ^
                         当前主要断裂
```

因此，单步 failure classifier、state-only progress、固定或动态 horizon、局部 action reranker 都不是完整解法。但目前也没有证据足以把某个新名称定为最终方法。下一主线先回答一个更严格的问题：

> 在部署可用的信息历史 `h_t` 下，基础策略支持集中是否存在跨 continuation 稳定为正、可由 `(h_t, a_t)` 预测、且经一次策略更新后仍能提高完整成功率的 recovery advantage？

**Counterfactual Recovery Advantage Distillation（反事实恢复优势蒸馏）** 仅作为待证候选，不是预设结论。贡献是否成立，只由无搜索 Pi0.5 的闭环恢复和完整任务成功率决定。

## 1. 根因假设

行为克隆给出的监督是“专家在这个状态做了什么”，但没有回答：

1. 同一视觉和机器人状态下，多个看似合理动作中哪个会形成稳定物理接触；
2. 哪个局部动作在后续策略随机性下仍能保留 lift、transport 和 place 的可达性；
3. 失败恢复过程中，哪个重规划节点真正造成了后续成功或退化；
4. 策略进入自己的恢复状态分布后，原始 expert 数据是否仍覆盖正确动作。

这不是单纯的 horizon 问题，而是部分可观测、接触敏感控制中的 action-level credit 和 on-policy state coverage 问题。

## 2. 问题的必要条件

### 2.1 支持集与动作可控性

从同一 feedback state 改变当前 action chunk 后，必须在共享 continuation 随机数下稳定改变后续结果。`K3` 和 12 步接管用于测局部纠错，但当前 teacher 在 slip 后约 12 步仍主要执行开夹爪，不能单独作为动作充分性的反证。因此还必须测量 teacher 接管到稳定重抓、抬升和 transport 后再交回 Pi0.5 的分阶段上界；若这些有物理意义的 handoff 仍不能明显提高最终成功率，局部动作选择就不是当前瓶颈，应先修复控制接口、任务数据或长程规划，而不是训练动作级 critic。

### 2.2 部署可观测性

优势必须能由部署时真实可用的当前视觉、proprioception，或一个短的 observation-action-consequence history 预测。sim state、object pose、branch outcome、未来图像和 continuation seed 只能用于审计。若当前信息不能区分 latent contact mode，应明确增加历史或反馈探测，而不是让训练标签偷偷补足不可观测信息。

### 2.3 信用可辨识性

动作效应必须大于 continuation noise。候选按预注册 outcome key 形成 top set；只有唯一或具有正稳健优势下界的候选才能提供方向性监督。并列时必须弃权，禁止用候选数组中的第一个元素制造稳定标签。

### 2.4 策略可学习性与闭环充分性

即使昂贵的 sibling search 能选出更好动作，也必须证明 `N=1` 策略更新能吸收该信号，并在 fixed-K 闭环中提高 recovery 和 formal success。否则它至多是 simulator planning 上界，不是可部署学习方法。

### 2.5 诊断结果决定方法，而不是反过来

| 观测结果 | 根因判断 | 下一步 |
|---|---|---|
| sibling action effect 不超过 continuation noise | 当前支持集没有稳定的动作杠杆 | 停止精确 Oracle 标签，补 recovery demonstrations 或做 on-policy residual improvement |
| full expert 有效，但 milestone handoff 后 Pi0.5 仍失败 | 下游 transport/place 策略或状态覆盖不足 | 先稳定 Full-H 和分阶段恢复数据，不训练局部 preference |
| handoff 有效，但 `(h_t,a_t)` 无法预测优势 | 信息不足或标签不可辨识 | 加最短 O-A-C history；仍无效则转向 feedback-aware replanning、probe action 或 belief model |
| 优势可预测，但一次 `N=1` 更新无闭环收益 | 蒸馏目标或离线分布不充分 | 优先比较 direct recovery SFT、DICE-RL/AFIL/SARM2 类简单强基线 |
| `N=1` 在 paired closed loop 稳定获益 | 动作级信用路线获得支持 | 再做第二环境、预算匹配和 on-policy data aggregation |

这张表允许最终答案落在热门近邻上。若 DICE-RL 风格 residual improvement 或 direct recovery SFT 解决了实质问题，采用它比维护一个名义更远、效果更差的新模块更正确。

## 3. 待证候选方法

### 3.1 同状态 sibling intervention

在策略实际到达的重规划状态 `s_t`，从 Pi0.5 采样多个候选 action chunks。每个候选都从完全相同的 simulator、controller 和相机状态执行，得到配对物理结果。

这样比较的是 `a_i` 与 `a_j` 在同一状态下的边际影响，而不是把容易 episode 的成功误归因给动作。

搜索 rollout 必须运行在独立 branch env，不能反复 restore 正在执行任务的 live env。候选选定后，action prefix 在 untouched live env 重新执行，并与 branch K-step endpoint 做严格 parity。这里解决的是测量干预污染：接触临界点会把 `1e-14` 级求解差异在长程放大，因此不能用“搜索后恢复 live env”制造看似更好的恢复行为，也不能用不对应真实算法操作的 120 步 bit-exact 压力测试误杀合法的 K-step 分支。

### 3.2 matched continuation survival

候选执行固定 `K` 步后，从各自真实 endpoint 出发，使用共享随机数的相同 frozen-policy continuations。标签不只看 K 步后的瞬时 contact，而看一条有序恢复 survival chain：

```text
no regress -> stable regrasp -> lift -> transport -> formal success
```

同时保存 stage 到达时间、progress AUC 和动作成本。不同候选必须共享 continuation seeds，每个候选只执行一次 K 步并从该唯一 endpoint 分叉，避免重复执行引入伪差异。

### 3.3 receding causal teacher

Oracle 不从同一初态挑一整条 winner trajectory。它只在当前真实状态内选择一个 chunk，执行后读取新的真实观测，再在新状态重新构造 sibling comparison。

这避免把第一步以后已经不同的状态错误地当作同一反事实，也直接测试局部优势能否跨多次重规划累积成最终收益。

### 3.4 可弃权的稳健 advantage distillation

只有 receding Oracle 在独立 full continuation 上显著提高 transport 或 formal success，才允许训练。

训练时先从同状态 action pairs 和 matched continuations 估计动作效应分布。学习标签必须同时满足：

- 相对 state baseline 的优势为正；
- 跨 continuation 的稳健下界为正；
- top set 不是不可分的全并列；
- 优势可由部署输入预测。

不满足时对该 decision abstain。满足时再更新 Pi0.5 action expert：

- winner-only advantage-weighted flow matching 作为第一候选；
- 原始成功 demonstration flow-matching anchor；
- reference-anchored pairwise flow objective 只在简单候选有效后测试；
- 当前图像、proprioception 和候选动作是唯一可用输入；
- simulator state、object pose、未来图像、stage label 和 continuation seed 只存在于 audit 数据。

最终部署必须是 `N=1`、无 simulator、无 critic、无搜索、固定 K。若只有 Oracle 或 reranker 提升，方法主张不成立。

### 3.5 on-policy data aggregation

一轮蒸馏后策略会进入新的恢复状态分布。若第一轮 sample0 有正向信号，下一轮必须由更新后的策略重新采集 sibling interventions，而不是无限复用 base-policy bank。

这是对 compounding error 的直接处理，也区分本方法与纯离线轨迹筛选。第一轮没有闭环增益时不启动迭代数据飞轮。

## 4. 与强近邻的关系

- [SARM2](https://arxiv.org/abs/2606.10305) 的核心是 stage-aware state reward 与 on-policy improvement；本方法只有在 action-conditioned same-state advantage 超过 state-only stage reward 时才有额外价值。
- [AFIL](https://arxiv.org/abs/2605.08434) 用成功/失败生成器形成负指导；本方法必须证明真实 sibling consequence 比 unpaired failure distribution 提供更准确的动作信用。
- [PACT](https://arxiv.org/abs/2606.03949) 已在人工 intervention state 构造 counterfactual preference；本方法必须证明无需人工纠正动作、由真实物理 sibling 结果产生的细粒度信用能改善 VLA。
- [VLAC-CUT](https://arxiv.org/abs/2607.09776) 已做 progress/failure/recovery segment curation；本方法必须超过普通 segment 保留和 recovery-success-only SFT。
- [A3](https://arxiv.org/abs/2605.11567) 解决 action-prefix 接受长度；本方法固定 K，检验的是候选动作的真实后果，不以改变观察频率获益。
- [Reflective VLA](https://arxiv.org/abs/2606.25215) 使用 action-consequence history 适应执行差异；history 可以作为后续消融，但不能替代当前 action 的因果比较。
- [DICE-RL](https://arxiv.org/abs/2603.10263) 已把 frozen flow prior、residual actor、chunk critic、best-of-N 和在线回报过滤组合成高效 VLA 后训练；如果它或其简化版本在相同交互预算下达到同样收益，应直接采用，不应把 sibling 数据来源本身包装成独立贡献。

若这些更简单方法达到同样闭环结果，应采用更简单方法。热门近邻不是需要规避的威胁，而是必须正面比较的基线。

## 5. 当前 receding Gate 能证明什么

本轮只验证第一个必要条件：base Pi0.5 的支持集中，是否存在能够通过连续重规划累积为最终收益的 action choices。

它必须同时满足：

1. `sample0` 是真正自然、不中断且不受搜索 restore 影响的 Full-H rollout；
2. candidate K-step endpoint 只在独立 branch env 生成一次，selection 与 decision-heldout 从同一 endpoint 分叉；
3. 选中 action 在 live env 重新执行，K-step sim、pixels、robot state 和物理语义必须与 branch endpoint 严格一致；
4. 4 次干预结束后，从每种方法 endpoint 启动 5 条共享 seed 的 full-heldout continuations；
5. Gate 指标使用 full-heldout transport、formal success、regress 和 progress，而不是 selection lookahead；
6. random selector 使用多个预注册 schedule，不能让一次随机抽样充当基线；
7. 三个 checkpoint seed 使用完全相同的 13 个 val groups、9 个 source clusters；
8. 运行配置、checkpoint hash、Git commit 和预注册 hash 必须硬锁。

若 receding Oracle 仍不能提高 full-heldout transport/success，结论是 base policy candidate support 或 continuation dynamics 不足。此时不训练 critic，也不把更多标签技巧包装成解法，直接转向高质量 recovery demonstrations、AFIL/SARM2 类 on-policy improvement 或显式反馈重规划。

Gate 后还必须做固定总预算 expert-handoff ladder：`sample0`、`expert-K3`、`expert-12`、`expert-to-regrasp`、`expert-to-lift`、`expert-to-transport` 从同一 feedback state 出发并使用 matched continuation seed schedule，另以 full expert 检查控制器本身。若到达稳定重抓或后续阶段的 expert handoff 仍无明显收益，说明当前完整任务失败不是局部候选选择问题；若 handoff 有效而 base-policy candidates 无效，说明是支持集或数据覆盖问题，应优先收集 recovery demonstrations。

## 6. 禁止提前声称的内容

以下结果都不构成最终方法成功：

- K3 contact、regrasp 或 offline MSE 改善；
- Oracle selection 优于 sample0，但 posttrained sample0 不提升；
- 只在一个 seed 或重复帧统计上显著；
- 使用额外 simulator rollout、更多观察频率或动态 K 才提升；
- Oracle 不稳定优于 random、state-only、unpaired 或 failure-negative baseline；
- attached/no-intervention 明显退化。

## 7. 执行顺序

1. 先修复并锁定 receding causal-teacher Gate；
2. dirty 单组 smoke 只验证状态恢复、endpoint 复用、数据隔离和成本计数；
3. clean commit 上运行三 seed val Gate；
4. 用 tie-aware 汇总、action-vs-continuation 方差分解和 method-order parity 审计 Gate；
5. 执行从 `expert-K3` 到 `expert-to-transport` 的固定总预算 handoff ladder，确定哪一个恢复阶段足以改变完整成功；
6. 比较 current-only 与一个 observation-action-consequence triplet 的优势可预测性；
7. 任一必要条件失败即停止该数据教师，转向恢复数据、稳定 baseline 或反馈重规划；
8. 全部通过才生成训练 bank，并先比较简单 winner-only、state-only、unpaired、direct recovery SFT 与 DICE/AFIL-style baseline；
9. 最后只用无搜索 sample0 闭环指标裁决反事实优势蒸馏。

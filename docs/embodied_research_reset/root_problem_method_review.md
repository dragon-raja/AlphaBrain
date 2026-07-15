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

因此，单步 failure classifier、state-only progress、固定或动态 horizon、局部 action reranker 都不是完整解法。下一主线定为：

**Counterfactual Recovery Advantage Distillation（反事实恢复优势蒸馏）**。

它不以名称作为贡献。贡献是否成立，只由无搜索 Pi0.5 的闭环恢复和完整任务成功率决定。

## 1. 根因假设

行为克隆给出的监督是“专家在这个状态做了什么”，但没有回答：

1. 同一视觉和机器人状态下，多个看似合理动作中哪个会形成稳定物理接触；
2. 哪个局部动作在后续策略随机性下仍能保留 lift、transport 和 place 的可达性；
3. 失败恢复过程中，哪个重规划节点真正造成了后续成功或退化；
4. 策略进入自己的恢复状态分布后，原始 expert 数据是否仍覆盖正确动作。

这不是单纯的 horizon 问题，而是部分可观测、接触敏感控制中的 action-level credit 和 on-policy state coverage 问题。

## 2. 最终方法构成

### 2.1 同状态 sibling intervention

在策略实际到达的重规划状态 `s_t`，从 Pi0.5 采样多个候选 action chunks。每个候选都从完全相同的 simulator、controller 和相机状态执行，得到配对物理结果。

这样比较的是 `a_i` 与 `a_j` 在同一状态下的边际影响，而不是把容易 episode 的成功误归因给动作。

搜索 rollout 必须运行在独立 branch env，不能反复 restore 正在执行任务的 live env。候选选定后，action prefix 在 untouched live env 重新执行，并与 branch K-step endpoint 做严格 parity。这里解决的是测量干预污染：接触临界点会把 `1e-14` 级求解差异在长程放大，因此不能用“搜索后恢复 live env”制造看似更好的恢复行为，也不能用不对应真实算法操作的 120 步 bit-exact 压力测试误杀合法的 K-step 分支。

### 2.2 matched continuation survival

候选执行固定 `K` 步后，从各自真实 endpoint 出发，使用共享随机数的相同 frozen-policy continuations。标签不只看 K 步后的瞬时 contact，而看一条有序恢复 survival chain：

```text
no regress -> stable regrasp -> lift -> transport -> formal success
```

同时保存 stage 到达时间、progress AUC 和动作成本。不同候选必须共享 continuation seeds，每个候选只执行一次 K 步并从该唯一 endpoint 分叉，避免重复执行引入伪差异。

### 2.3 receding causal teacher

Oracle 不从同一初态挑一整条 winner trajectory。它只在当前真实状态内选择一个 chunk，执行后读取新的真实观测，再在新状态重新构造 sibling comparison。

这避免把第一步以后已经不同的状态错误地当作同一反事实，也直接测试局部优势能否跨多次重规划累积成最终收益。

### 2.4 advantage distillation

只有 receding Oracle 在独立 full continuation 上显著提高 transport 或 formal success，才允许训练。

训练时使用同状态 action pairs 和 matched-continuation advantage，更新 Pi0.5 action expert：

- reference-anchored pairwise flow objective；
- 原始成功 demonstration flow-matching anchor；
- winner-only advantage-weighted flow matching 作为更简单强基线；
- 当前图像、proprioception 和候选动作是唯一可用输入；
- simulator state、object pose、未来图像、stage label 和 continuation seed 只存在于 audit 数据。

最终部署必须是 `N=1`、无 simulator、无 critic、无搜索、固定 K。若只有 Oracle 或 reranker 提升，方法主张不成立。

### 2.5 on-policy data aggregation

一轮蒸馏后策略会进入新的恢复状态分布。若第一轮 sample0 有正向信号，下一轮必须由更新后的策略重新采集 sibling interventions，而不是无限复用 base-policy bank。

这是对 compounding error 的直接处理，也区分本方法与纯离线轨迹筛选。第一轮没有闭环增益时不启动迭代数据飞轮。

## 3. 为什么不是近邻的形式改名

- [SARM2](https://arxiv.org/abs/2606.10305) 的核心是 stage-aware state reward 与 on-policy improvement；本方法只有在 action-conditioned same-state advantage 超过 state-only stage reward 时才有额外价值。
- [AFIL](https://arxiv.org/abs/2605.08434) 用成功/失败生成器形成负指导；本方法必须证明真实 sibling consequence 比 unpaired failure distribution 提供更准确的动作信用。
- [PACT](https://arxiv.org/abs/2606.03949) 已在人工 intervention state 构造 counterfactual preference；本方法必须证明无需人工纠正动作、由真实物理 sibling 结果产生的细粒度信用能改善 VLA。
- [VLAC-CUT](https://arxiv.org/abs/2607.09776) 已做 progress/failure/recovery segment curation；本方法必须超过普通 segment 保留和 recovery-success-only SFT。
- [A3](https://arxiv.org/abs/2605.11567) 解决 action-prefix 接受长度；本方法固定 K，检验的是候选动作的真实后果，不以改变观察频率获益。
- [Reflective VLA](https://arxiv.org/abs/2606.25215) 使用 action-consequence history 适应执行差异；history 可以作为后续消融，但不能替代当前 action 的因果比较。

若这些更简单方法达到同样闭环结果，应采用更简单方法。热门近邻不是需要规避的威胁，而是必须正面比较的基线。

## 4. 当前 receding Gate 能证明什么

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

## 5. 禁止提前声称的内容

以下结果都不构成最终方法成功：

- K3 contact、regrasp 或 offline MSE 改善；
- Oracle selection 优于 sample0，但 posttrained sample0 不提升；
- 只在一个 seed 或重复帧统计上显著；
- 使用额外 simulator rollout、更多观察频率或动态 K 才提升；
- Oracle 不稳定优于 random、state-only、unpaired 或 failure-negative baseline；
- attached/no-intervention 明显退化。

## 6. 执行顺序

1. 先修复并锁定 receding causal-teacher Gate；
2. dirty 单组 smoke 只验证状态恢复、endpoint 复用、数据隔离和成本计数；
3. clean commit 上运行三 seed val Gate；
4. Gate 不通过即停止该数据教师；
5. Gate 通过才生成训练 bank，并先比较简单 winner-only、state-only、unpaired 与 AFIL-style baseline；
6. 最后只用无搜索 sample0 闭环指标裁决反事实优势蒸馏。

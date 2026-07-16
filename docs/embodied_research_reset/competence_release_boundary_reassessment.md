# 反馈因果纠正与恢复支持：问题复审与可证伪方案

状态：概念修订；在 baseline validity repair 的 validation 结果、恢复强基线结果和
test split 之前冻结。本文不把候选方案声明为新方法，也不改变当前 baseline gate。

## 1. 从问题本质重新定义

当前失败不是一个需要靠“远离近邻”包装的新颖性问题。正式 handoff 实验已经说明：

- slip 反馈后的 policy-only 完整成功率为 21.85%；
- teacher 接管 3 或 12 个动作没有帮助；
- teacher 接管到稳定重抓后，原策略可达到 85.19% 完整成功率；
- 接管到 lift、transport 后继续提高到 88.52% 和 98.89%。

因此，已经被证据直接支持的根问题是一个策略相对的可达性问题：

> 如何用尽量少的纠正监督，把失败后的策略诱导状态带回一个区域，使当前闭环策略在
> 扰动和重规划下仍有足够高的任务完成概率？

这一区域不是固定语义阶段。`stable regrasp` 只是当前任务中观察到的一个有效上界，
不应被当作通用方法定义。对于不同初始状态、策略 checkpoint 或任务，正确释放点可能
早于或晚于重抓。

## 2. 不再混淆两个不同失败机制

原始 FRESH 叙事把两个问题压进了同一个 suffix loss：

1. **反馈前/反馈揭示时的旧动作承诺**：chunk 是根据旧观测生成的，物理结果变化后仍可能
   延续错误 tail；
2. **反馈后的恢复支持缺口**：即使每步重规划，policy 也没有学会从自己诱导出的 slip
   状态回到任务主轨迹。

固定 `K=1/2/3` 和 corrected handoff 结果已经证明第二项真实存在，但尚未量化第一项在当前
任务中的独立贡献。一个只调整训练 suffix 权重的方法既没有增加失败状态支持，也没有让
执行器读取 chunk 生成后的新观测，因此不应继续作为主方案。

### 2.1 三种子 train-only 机制诊断结果

在 102 个 train snapshot group、三个独立 Pi0.5 checkpoint 上，比较反馈前生成的旧 chunk
tail 与反馈时从最新图像重新生成的 fresh chunk。每个 group 先跨 policy seed 平均，再以
snapshot group bootstrap；validation/test 均未打开。`stale age=1, horizon=1` 的结果为：

- slip teacher-action MSE 从 stale 改成 fresh 后平均降低 0.4582，95% CI
  `[0.4285, 0.4859]`，相对降低 83.5%；
- 首步 recovery-action rate 提高 81.0 个百分点，95% CI `[75.8, 85.9]`；
- fresh attached/slip twin assignment 正确率为 96.7%，95% CI `[94.4, 98.7]`；
- attached 分支 fresh-minus-stale MSE 为 -0.0023，95% CI `[-0.0110, 0.0067]`，未见正常
  分支损害；
- 三个 seed 的 fresh slip recovery-action rate 分别为 97.1%、78.4% 和 80.4%，方向一致。

这确认旧 chunk tail 在反馈后会过时，但也**否定了“基策略在反馈后的第一步普遍不会恢复”**：
从最新观测重规划时，基策略大多能立即选择正确恢复方向。它不否定后续 support gap，因为
正式 multi-step handoff 的完整成功率仍只有 21.85%。因此，任何新方法都必须把“首步 revision”
和“后续恢复轨迹覆盖”分开测量；不能拿首步 MSE 改善代替闭环成功。

## 3. 修订后的主假设：Counterfactual Feedback Revision

暂用工作名 **CFR**，不作为原创性声明。冻结基策略在时刻 `t` 从观测 `o_t` 生成 chunk
`A_t`。执行到 chunk 内位置 `k` 时，一个轻量纠正器读取最新可部署观测 `o_{t+k}`、原始
动作 `A_t[k]`、chunk 位置、语言/robot state 和冻结策略特征，输出选择性 residual：

```text
a_exec[t+k] = A_t[k] + g_phi(o_latest, A_t[k], k, z_t) * delta_phi(...)
```

其中 `g_phi` 不是 horizon head，不改变固定 K，也不接收 branch outcome、contact oracle、
simulator state 或未来信息。它只回答：最新视觉是否提供了足以推翻旧动作的证据。

当前 attached/slip outcome twins 提供了比普通 replay 更干净的监督：

- feedback reveal 前，两支 conditioning 相同，纠正器不得提前泄漏或猜测结果；
- reveal 后，attached twin 是“保持/小修正”的匹配负例，slip twin 是“停止错误抬升并
  进入恢复”的匹配正例；
- gate 标签来自同一状态下 teacher action 与 base action 的可执行差异，而不是 branch
  名称；
- clean、attached 和 no-intervention 状态使用零 residual anchor，约束正常能力退化；
- slip 纠正只持续到当前策略重新具备稳定闭环完成能力的位置。

这个对象直接针对“新物理反馈何时应覆盖旧 chunk”，同时用 policy-induced recovery 数据
补齐覆盖后的动作能力。若普通 A2C2-style correction 或 RaC/VLA-OPD-style 数据已经解决
问题，就采用近邻方案并停止 CFR 主张。

新诊断也进一步收缩了 CFR 的必要性：fresh `K=1` 重规划是首步 revision 的最简单上界。
若完整闭环中 `K=1` 已解决问题且计算成本可接受，就直接采用高频重规划，不训练 residual；
若 `K=1` 只能修正第一步而无法完成恢复，则先补 policy-state recovery support；只有在
高频重规划成本不可接受、固定 `K>1` 的 stale tail 又造成显著失败时，A2C2-style residual
或 paired CFR 才有独立工程价值。paired CFR 仍必须正面胜过非成对 A2C2-style 控制，不能
靠组合两个已知组件成立。

## 4. 辅助对象：policy-relative competence set

对冻结策略 `pi` 和状态 `s`，定义闭环能力：

```text
C_pi(s) = P(task success | start from s, execute pi with the deployed fixed K)
```

概率必须由相同 episode 预算、扰动分布、策略 seed schedule 和正式成功条件下的重复
闭环 continuation 估计。训练组上的 simulator state 可用于准确恢复候选状态，但不得
成为部署时 policy 输入。

沿一条从失败状态出发的 teacher recovery trajectory，候选释放点 `t*` 是最早同时满足
以下条件的状态：

1. `C_pi(s_t)` 的预注册置信下界超过能力阈值；
2. 后续若干候选状态的下界保持在阈值之上，避免一次幸运 rollout；
3. 到达该状态前没有违反接触、掉落或安全约束；
4. 相同结论在多个 source snapshot group 上成立，而不是由重复帧放大。

能力集合只决定 CFR 或恢复纠正数据何时停止，不再作为独立主方法。训练数据只蒸馏从失败
状态到 `t*` 的最短 teacher bridge，并用原始成功数据作为 anchor。策略进入能力域后继续
使用原策略，不额外部署 simulator、teacher、reranker、动态 K 或 privileged state。

## 5. 为什么能力边界仍可能有用

固定 `feedback -> stable regrasp` 解决的是当前单任务的已知断点，但会产生两类浪费：

- 原策略已经能从更早状态完成时，继续提供 teacher 动作会增加纠正数据并覆盖原有能力；
- 原策略从语义上“已重抓”但仍处于脆弱状态时，固定释放会过早，导致再次掉落或漂移。

能力释放边界直接用下游闭环成功定义“何时可以交回策略”。它要解决的是监督长度与当前
策略能力不匹配，而不是重新命名 recovery replay。

## 6. 不得回避的强控制组

后续恢复实验必须包含以下控制；它们是答案候选，不只是陪衬：

1. frozen Full-H / Base continuation；
2. clean feedback-to-regrasp full-policy replay；
3. RaC/DAgger/VLA-OPD-style policy-state full-policy correction；
4. A2C2-style latest-observation residual correction，使用非成对但等量数据；
5. CFR paired correction，不使用 competence truncation；
6. CFR paired correction 加固定 stable-regrasp 截断；
7. CFR paired correction 加 policy-relative competence 截断；
8. full-recovery correction 与 teacher-action-count-matched random truncation。

如果 direct replay、固定重抓、普通 DAgger 或 residual/on-policy improvement 已经达到同等
成功率和成本，就采用更简单方法。热门近邻是答案候选和基线，不是需要规避的对象。

### 6.1 近邻审计后的主张收缩

- [RaC](https://arxiv.org/abs/2509.07953) 已经把 human intervention 明确定义为
  recovery-to-familiar-state 再 correction，并证明数据组成可以比普通 HG-DAgger 更有效；
- [VLA-OPD](https://arxiv.org/abs/2603.26666) 已经在 student-induced trajectories 的每个
  timestep 查询强 teacher，并用 reverse-KL 做 on-policy dense distillation；
- [A2C2](https://arxiv.org/abs/2509.23224) 已经用最新观测、base action、chunk position 和
  base-policy feature 做逐步 residual correction；
- [REMAC](https://arxiv.org/abs/2601.20130) 已经用 masked action chunking 学习感知与执行
  不一致下的纠正；
- [Dream2Fix](https://arxiv.org/abs/2603.13528) 已经生成成对反事实 failure-correction 数据
  并学习可执行恢复轨迹；
- [FailSafe](https://arxiv.org/abs/2510.01642) 已经自动生成 failure-action pairs，并用最终
  任务成功系统验证 corrective action；
- [AIM](https://openreview.net/forum?id=TC1sQg5z0T) 已经用 proxy Q 自适应请求干预并核算
  expert cost；
- [Selective Imitation Learning](https://openreview.net/pdf/ddb3065506b0d54af340c0c1e7abff5c7b7fe203.pdf)
  和 [ViSkill](https://arxiv.org/abs/2307.16503) 说明能力、价值或可恢复性驱动的 handoff/
  chaining 也不是新的抽象。

因此，“回到熟悉状态”“在 policy states 上纠正”“用最新观测修正 chunk”“生成成对反事实
恢复”或“用成功概率定义 handoff”均不能单独成为贡献。当前保留的可证伪问题是：在真实
outcome twins 上，匹配正负反馈的选择性 residual 是否比非成对 A2C2-style correction 和
RaC/OPD-style full-policy correction 更准确地保留正常 chunk、推翻失败 chunk；能力截断是否
在此基础上进一步减少 teacher 成本。只有等成本闭环结果和第二任务都成立，才讨论这是否
构成方法贡献，而不是若干已知组件的组合。

## 7. 成本必须完整记账

“更少纠正数据”不能掩盖昂贵的 simulator 搜索。每个方法分别报告：

- teacher-labeled action 数；
- teacher intervention episode 数；
- policy/environment rollout 数和总环境步数；
- GPU 训练与推理时间；
- slip recovery、overall 和 attached/no-intervention 成功率；
- 每提升一个成功率百分点所需的 teacher 动作和环境步数。

第一轮先验证普通恢复监督和 fixed-boundary correction，不训练 competence head。只有纠正
本身有效后才测量 policy-relative boundary；若测量版没有数据效率优势，不用预测器、
world model 或更多模块继续包装。

## 8. 严格的阶段 Gate

### Gate 0：baseline 有效

当前 baseline validity repair 必须先通过预注册 validation gate。否则保持
`BASELINE_INVALID_OR_DATA_INSUFFICIENT`，不开 test，不比较恢复方法。

### Gate 1：恢复监督本身有效

在同一冻结训练预算下，clean replay 或 policy-state correction 至少有一项必须相对 Base
稳定提高 slip recovery；否则能力边界没有可压缩的有效监督，应停止该离线 SFT 路线。

### Gate 2：paired feedback revision 有独立价值

在相同 teacher action 数、训练更新、冻结模块和 fixed K 下，CFR 必须同时满足：

- slip recovery 相对非成对 A2C2-style correction 或 RaC/OPD-style 强对照至少提高 10 个
  百分点，或 paired group-level 95% CI 下界大于 0；
- failure-continuation 与 feedback-to-revision latency 明显下降；
- attached/no-intervention 退化不超过 5 个百分点；
- reveal 前 gate 假阳性率不高于 5%，证明没有把 branch timing 偷渡成输入。

若只有所有 recovery 数据方法都提升、paired CFR 没有额外优势，结论应采用强近邻方法，
不得将组件组合包装成新算法。

### Gate 3：能力边界优于固定边界

在三个 seed 和相同 fixed K 下，能力边界必须满足至少一个条件：

- 在成功率不低于固定重抓对照 5 个百分点以上的前提下，teacher 动作减少至少 25%；或
- 在 teacher 动作严格匹配时，slip recovery 提高至少 10 个百分点且 paired group-level
  95% CI 下界大于 0。

同时 attached/no-intervention 退化不得超过 5 个百分点。若不满足，不声称方法价值。

### Gate 4：近邻和跨任务验证

Gate 1 通过后，选择 validation 上成功率最高且成本最低的简单方法。若 CFR 通过 Gate 2，
选择 CFR；若 CFR 没有独立优势但 RaC/OPD 或 A2C2-style 控制解决了问题，选择该近邻方案并
明确不作 CFR 贡献主张。只有选定方案在预注册 validation gate 上通过，才允许打开 test，
随后在第二个具有不同恢复几何的 LIBERO 任务上验证。Gate 3 只决定是否保留能力截断，
不是解决恢复问题或打开 test 的必要条件。单个 cream-cheese 任务只能证明机制，不能证明
一般性。

## 9. 当前执行决定

- 继续四 checkpoint 的 Full-H validation，因为它验证所有后续比较的共同地基；
- 在 baseline gate 完成前，不启动 B/C，不打开 test；
- 旧 B/C runner 绑定旧 checkpoint，不能用于 repaired baseline；
- 第一轮恢复实验先回答 clean-state exposure 与 policy-state coverage 哪个是根因；
- 把最终 `K=1/2/3` 闭环差异作为 stale-tail 的行为验证；train-only action 诊断不单独裁决；
- 只有普通恢复监督有效，才实现最小 CFR residual prototype；
- 只有 paired CFR 胜过非成对强对照，才评估 policy-relative competence truncation；
- 任一更简单强基线解决问题时，停止增加模块并采用该基线。

### 9.1 Baseline v2 已释放 Gate 1

10,353-step Full-H 三 seed validation gate 已通过：K=3 attached 跨 seed
均值为 53.85%，三个 seed 均达到单 seed 门槛，正式状态为
`BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS`。test 仍未打开。与此同时，
failure-continuation 从 K=1 的三个 seed 全 0% 上升到 K=3 的
62.5%/55.6%/75.0%，说明 stale-tail 现象在具备基本任务能力的策略上仍存在。

因此下一步不再修 baseline，也不直接实现 CFR。按结果前冻结的 Gate 1，先比较
Base continuation、clean feedback-to-regrasp replay 和 policy-state recovery；三者
从相同 10,353-step seed checkpoint 出发，统一增加 6,902 updates。具体协议见
`recovery_support_v2_preregistration.md`。只有 policy-state correction 在等预算下
胜过两个简单强对照，才有理由继续考察 paired CFR 的独立价值。

这条路线允许最终答案是一个热门近邻，也允许答案是负结果。唯一不能接受的是为了形式差异
而牺牲闭环成功、数据效率或实验可解释性。

## 10. 范围限制

CFR 与能力释放边界只针对当前实验证据指向的 feedback revision 和 post-feedback recovery
support 缺口。它们不自动解决一般的 pre-feedback 多模态承诺问题，也不证明原始 FRESH
suffix weighting 有效。
如果在一个 recovery policy 已充分有效的任务上，主要失败仍来自反馈到来前执行了过长的
branch-specific action prefix，正确解法可能是显式 plan-commit/replan、action-prefix
acceptance 或 feedback probing。届时应与 A3、SEAM 等方向正面比较，而不是把本候选强行
扩展成万能模块。

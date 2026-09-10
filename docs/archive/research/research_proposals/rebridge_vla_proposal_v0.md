# ReBridge-VLA：基于策略相对最优停止的 VLA 闭环恢复

状态：`SUPERSEDED_TECHNICAL_NOTE`。正式研究计划见
[`rebridge_vla_formal_research_proposal_v1.md`](rebridge_vla_formal_research_proposal_v1.md)。
本文保留为方法构思与 Gate 设计记录，不作方法有效性或原创性结论。

日期：2026-07-19

## 0. 摘要

当前实验已经把问题收缩到一个很具体的闭环控制断点：失败结果在视觉上可辨认，Pi0.5
也常能生成局部合理的恢复动作，但它无法稳定穿过从 slip feedback 到 stable regrasp 的
长桥接区间。完整 teacher 接管到 stable regrasp 后，冻结 Pi0.5 的任务完成率从 21.85%
上升到 85.19%；相反，clean recovery replay、policy-state recovery SFT、局部候选标签、
一次 policy-response surrogate 和静态 continuation winner distillation 都没有解决这个问题。

本 proposal 暂用工作名 **ReBridge-VLA**。它不再修改基础 VLA，也不把完整任务成功作为
恢复策略唯一的稀疏奖励，而把恢复形式化为一个以冻结基础策略能力为停止收益的最优停止问题：

> 恢复 option 应在自己的 on-policy 状态上做最小必要纠正，直到把系统带到一个冻结基础
> 策略能够可靠接管并完成任务的状态；何时交回控制权由基础策略的闭环 continuation value
> 决定，而不是由固定阶段、固定时长或外部 VLM 语义标签决定。

核心对象是冻结策略 `pi_0` 的闭环成功概率 `V_0(h)`，以及允许“继续恢复”或“立即交回
pi_0”的组合价值：

```text
U*(h) = max {
    V_0(h),
    max_delta E[ U*(h') - c(delta, h) ]
}
```

其中 `h` 只含部署可用的短 observation-action history；`delta` 是叠加在 Pi0.5 动作上的
有界 residual；`c` 同时惩罚干预幅度、恢复时间和物理退化。`V_0(h)` 是停止动作的收益，
不是候选 action ranker。该 Bellman 最优停止结构同时学习 bridge control 和 handback，
并保持基础 VLA 冻结。

这是一个初步算法 proposal，不是把“恢复 option”“residual RL”或“policy-relative value”
本身声明为新贡献。能否形成论文贡献，取决于它是否在等预算下稳定胜过 full-task residual RL、
A2C2-style correction、ReCoVLA-style stage reward、固定 stable-regrasp handoff 和 on-policy
distillation，并在多个任务上保持基础策略正常能力。

## 1. 已有实验告诉了我们什么

### 1.1 已经成立的事实

| 证据 | 结果 | 对方法设计的约束 |
| --- | --- | --- |
| Feedback observability | feedback frame 的 vision+state 分支识别为 100% | 当前任务不需要主动视觉或 belief model 才能看见 slip |
| Local recovery mode | post-feedback sample0 correct-mode 为 97.4% | 不是简单的 mode 缺失或采样不足 |
| Full-H fixed K | K=1 仍不能解决完整恢复；K=3 更好 | 不是只需更频繁重规划 |
| Expert handoff | policy-only 21.85%，teacher-to-regrasp 85.19% | 主要能力缺口在 feedback 到 stable regrasp |
| Bridge length | 到 stable regrasp 平均约 80.5 teacher actions | 三步、十二步局部纠正不足 |
| Offline recovery SFT | clean slip -23.1 pp；policy-state slip -17.9 pp，相对 Base | 整网离线更新产生干扰和 distribution drift |
| Exact receding continuation Oracle | slip 87.2%，比 single 高 15.4 pp | 闭环逐步反馈中存在可利用 headroom |
| Local/static surrogates | teacher distance、short physical、one-response、static winner 均失败 | 不应再压缩成单步局部评分或固定标签 |
| Stable continuation labels | 主要集中在 post-regrasp | 静态监督恰好缺失于真正的 bridge 区间 |

这些事实共同排除“再调 loss”“再选一个局部 action”“再加一个 horizon head”作为主线。
它们支持的不是某个现成算法，而是如下机制判断：

```text
真实反馈 -> option 自己诱导的新状态 -> 新反馈 -> 再纠正 -> ...
          -> 到达 base policy 能可靠完成的状态 -> handback
```

### 1.2 前几轮失败的共同结构

1. **目标错位**：动作相似、局部接触和短期 physical score 都不等于冻结 Pi0.5 的长期可接续性。
2. **状态分布错位**：离线 expert 或一次性 policy-state bank 没有覆盖更新后策略实际到达的状态。
3. **更新范围过大**：恢复数据直接更新完整 Pi0.5，破坏 attached 和原有长程能力。
4. **监督时间尺度过短**：真正 bridge 平均约 80 个动作，3 或 12 步 correction 无法改变结果。
5. **静态标签假设错误**：pre-regrasp 区域中，正确动作依赖后续真实物理反馈，固定 winner 不稳定。

ReBridge-VLA 的每个设计选择都必须对应其中至少一个断点，否则不应加入。

## 2. 研究问题与假设

### 2.1 研究问题

对于一个在正常任务上已有能力、但在可观测失败状态上恢复不足的冻结 VLA，能否学习一个
小型闭环 recovery option，使其：

1. 只在失败后进入；
2. 在 option 自己诱导的状态分布上学习；
3. 以“冻结 VLA 已恢复完成能力”为目标，而非模仿固定 teacher 轨迹；
4. 自动决定何时停止纠正并交回控制；
5. 在提升恢复率的同时，基本不改变正常任务行为？

### 2.2 核心假设

**H1：策略相对 bridge 比完整任务 residual 更容易学习。**

从 slip 状态直接优化完整 task success，需要同时学习 regrasp、lift、transport 和 place；但
handoff 结果表明，冻结 Pi0.5 在 stable regrasp 之后已经有 85.19% 的完成能力。只学习到
“可接管集合”的 first-passage bridge，学习时域更短、探索空间更小。

**H2：基础策略价值适合作为 handback 的停止收益。**

固定 `stable_regrasp` 是当前任务的诊断里程碑，不是通用定义。不同任务、checkpoint 和物理
状态的最早可靠接管点不同。`V_0(h)` 直接回答当前 frozen policy 能否从这里继续成功。

**H3：冻结基础策略加小 residual option 能减少灾难性干扰。**

现有 recovery SFT 的 attached 与 slip 性能同时下降。冻结 Pi0.5，并让 option 仅在失败状态
激活，可以把学习容量和梯度限制在 recovery bridge。

**H4：on-policy option data 是必要条件。**

静态 bank 在 bridge 区间缺少稳定 winner。option 必须根据自己执行后的真实观测继续收集和
更新，而不是无限复用 base-policy 或 teacher 状态。

## 3. 方法

### 3.1 POMDP 历史与冻结基础策略

部署输入为短历史：

```text
h_t = E(o_{t-L:t}, s_{t-L:t}, a_{t-L:t-1}, language)
```

其中 `o` 包含 agent-view 和 wrist-view，`s` 是部署已有 robot state。branch outcome、object
pose、contact oracle、sim state、未来图像和 continuation seed 不进入 policy 或 handback
网络。训练期可以使用它们生成审计标签和正式环境成功判据。

冻结 Pi0.5 生成基础动作：

```text
a_t^0 ~ pi_0(. | h_t)
```

第一版固定 `K=3`，不训练 dynamic K 或 horizon head。

### 3.2 基础策略 continuation value

定义：

```text
V_0(h) = P(task success before timeout | h, execute frozen pi_0 with fixed K)
```

训练标签来自对恢复轨迹上状态的 matched frozen-policy continuations。学习一个分布式、可校准
的 `V_psi(h)`，输出 success probability 或 success-return quantiles。正式 handback 使用保守
下界：

```text
L_psi(h) = calibrated lower confidence bound of V_0(h)
```

关键限制：`V_psi` 不给候选 action 排序，也不参与修改 Pi0.5；它只给“现在停止恢复并交回
base”这个动作定价。这与已经停止的 BASIN policy-conditioned candidate committor 不同：

- BASIN 的问题是同状态候选排序是否随 checkpoint 改变；该现象不够广泛，已停止；
- ReBridge 固定一个 base policy，学习 state-level stopping payoff；
- 即使不同 checkpoint 的候选排序一致，每个 checkpoint 仍可能有不同的接管区域；
- 不使用 policy fingerprint，不做 cross-policy action ranking。

### 3.3 Residual recovery option

失败 guard 激活后，小型 option 输出有界 residual：

```text
delta_t ~ rho_phi(. | h_t, a_t^0)
a_t = clip(a_t^0 + m_t * delta_t)
```

`m_t` 是固定 action-dimension mask，只开放当前 recovery 所需的 EEF 与 gripper 维度。第一版
网络读取冻结 Pi0.5 的视觉特征、最新 proprioception、base action 和最近三次 action-result
history，不读取 privileged state。

option 激活前 `delta=0`。激活后，Pi0.5 仍每 K 步提供动作 prior，residual 只做最小必要偏移。

### 3.4 最优停止 Bellman 目标

在每个 option 状态有两个高层选择：

1. `STOP`：立即交回 Pi0.5，收益为 `V_0(h)`；
2. `CONTINUE(delta)`：执行 residual action，读取新真实观测后再次选择。

组合价值为：

```text
U*(h) = max(
    V_0(h),
    max_delta E[ U*(h') - lambda_a ||delta||^2
                            - lambda_t
                            - lambda_f I(regress/drop) ]
)
```

实现时训练 distributional critic `Q_theta(h, delta)` 和 residual actor `rho_phi`。`STOP` 不是
另一个神经动作，而是由保守的两值比较决定：

```text
handback if L_psi(h) >= U_continue_upper(h) - epsilon
```

为了避免 value overestimation 造成过早交回，必须同时满足：

- `L_psi(h) >= tau_min`；
- 条件连续满足 `q` 个 replan；
- 最近窗口无 drop/regress；
- calibration set 上的 handback false-positive 不超过预注册阈值。

`tau_min` 和 `q` 只在 development split 选择，不能在 test 扫描。

### 3.5 基础价值势能与最小干预

可选但仍属于同一目标的 shaping 为：

```text
r_shape = beta * (stopgrad(L_psi(h_{t+1})) - stopgrad(L_psi(h_t)))
```

它奖励把系统推向 base competence，而不是手写 `regrasp/lift/transport` 阶段奖励。终端 task
success、drop 和正式 constraint 仍由环境真值判定；value potential 不能单独宣告成功。

最小干预项 `||delta||^2` 与 option 时长成本使算法优先保留 base 动作，并尽早 handback。
这不是纯安全约束，而是防止 recovery policy 重新学习整个任务。

### 3.6 On-policy bridge expansion

训练从已有 feedback states 启动，但每轮数据由当前 option 自己执行：

```text
for iteration j = 1 ... J:
    sample a train failure snapshot
    activate current recovery option
    collect h, base action, residual, physical outcome
    when handback fires, execute frozen pi_0 to task end
    store true composed success and constraint outcomes
    update V_psi only with frozen-pi_0 continuation targets
    update rho_phi and Q_theta on option-induced transitions
    retain a fixed normal/attached zero-residual audit set
```

`V_psi` 与 bridge critic 的数据职责分离：前者估计 frozen base 的停止收益，后者估计 option
继续控制的收益，禁止用 option rollout 假装成 base continuation label。

### 3.7 训练期与部署期边界

训练期允许：simulator state restore、object/contact audit、重复 continuation、正式 success
predicate 和 privileged curriculum initialization。

部署期只保留：冻结 Pi0.5、失败 guard、短视觉动作历史、residual option 和 handback value。
不允许 simulator rollout、teacher、future image、object pose、external VLM、candidate search、
dynamic K 或 world model。

## 4. 为什么它不是前几轮方法的延续

| 已失败路线 | 失败假设 | ReBridge 的变化 |
| --- | --- | --- |
| FRESH suffix weighting | 通过 loss allocation 改善公共动作 | 不改 base loss，直接学习闭环 recovery control |
| Oracle/plan commitment | 调整 chunk 执行时长 | 固定 K，学习真实动作与 handback |
| CORA/BASIN reranking | 从 Pi0.5 candidate 中选局部 winner | 不做 candidate ranking，residual 可走出原支持局部 |
| CCV coupling | 用低成本 continuation label 近似 action value | 不压缩一次 action 标签，直接在 option on-policy 分布学习 |
| Policy-response surrogate | 一次 policy response 足以预测长期结果 | 保留多步交互，不要求单次响应充分 |
| Continuation self-distill | 固定 winner 可蒸馏进 base | 冻结 base，bridge 中每个新状态重新闭环决策 |
| Recovery SFT | expert/policy-state replay 可补支持 | 不更新整网；使用 RL/optimal stopping 而非固定轨迹模仿 |

## 5. 近邻工作与原创性边界

### 5.1 必须正面比较的工作

- [A2C2](https://arxiv.org/abs/2509.23224) 已经提出读取最新观测、base action、chunk position
  和 base feature 的逐步 residual correction。ReBridge 不能把“小 correction head”当贡献；
  必须证明最优停止 bridge 对长恢复区间有独立价值。
- [ReCoVLA](https://arxiv.org/abs/2606.09630) 已经冻结 VLA，以外部 VLM 编译 failure/stage reward，
  再训练 residual recovery policy。ReBridge 的差异候选是不用外部语义 stage，把 frozen-base
  continuation value 作为可组合停止收益。
- [Object-Centric Residual RL](https://arxiv.org/abs/2606.18953) 已经证明 frozen VLA 加 residual
  RL 可以显著提高真实机器人成功率。ReBridge 不能把 residual RL 或 frozen base 当贡献。
- [DICE-RL](https://arxiv.org/abs/2603.10263) 已经研究基于 diffusion/flow prior 的高效 residual
  off-policy RL。它是 full-task residual 强基线。
- [VLA-OPD](https://arxiv.org/abs/2603.26666) 已经在 student-induced states 上做 dense teacher
  distillation并缓解 SFT distribution shift。ReBridge 必须说明无需逐状态强 teacher 的价值。
- [RaC](https://arxiv.org/abs/2509.07953) 已经用 intervention 把机器人 rewind 到熟悉状态，再做
  correction。ReBridge 不能把“回到熟悉区域”当新概念。
- [Recovery RL](https://arxiv.org/abs/2010.15920) 已经分离 task policy 与 recovery policy，并学习
  recovery zones。ReBridge 必须以任务完成能力而非安全约束区分问题和目标。
- [Reach-Avoid RL](https://arxiv.org/abs/2112.12288) 已有正式 reach-avoid Bellman 理论。本文只能
  把它作为组合恢复的理论工具，不能声明发明 reach-avoid objective。

### 5.2 暂定的贡献候选

只有实验支持后，才可能形成以下贡献：

1. **Base-value optimal stopping for VLA recovery**：把冻结 VLA 的 continuation success
   probability 作为 recovery option 的停止收益，联合定义继续纠正和 handback，而非手写阶段。
2. **Policy-compatible bridge learning**：训练目标不是重新完成整项任务，而是以最小干预首次
   到达当前 VLA 的 competence region。
3. **Conservative composition**：用校准 lower bound 控制 handback，并报告组合成功下界与实际
   calibration，而不是用语义 milestone 代替能力。
4. **Evidence-backed failure benchmark**：利用成对物理分支和完整 recovery episodes，隔离
   failure-to-competence bridge，而不把帧级 MSE 当最终结果。

这四点仍可能与 option-critic、skill chaining 或近期 residual VLA 工作高度重叠。Gate 0 前还需
做一次更系统的论文级 novelty search；检索发现直接等价方法时，应改为复现强基线或停止命名，
不能为了投稿制造形式差异。

## 6. 初步理论主张

令 `B_tau = {h: V_0(h) >= tau}`，失败集合为 `F`。若 recovery option 从失败分布 `D_f`
以概率 `p_B` 在进入 `F` 前到达 `B_tau`，且 `V_0` 的 handback lower bound 在覆盖事件上以
概率至少 `1-alpha` 有效，则组合策略的成功率满足直观下界：

```text
P(success | D_f) >= p_B * tau - alpha
```

更准确的版本应直接积分 handback state 的真实 `V_0(h_T)`，而非统一阈值：

```text
P(success | D_f) = E[ I(T_B < T_F) * V_0(h_T) ]
```

这只是由条件概率分解得到的组合性质，不是假装获得全局最优保证。可发表的理论部分需要证明：

- stopping-payoff Bellman operator 在选定折扣/episodic 条件下的性质；
- lower-confidence handback 对过早停止风险的界；
- potential shaping 不改变理想化最优停止策略的条件；
- 近似 `V_psi` 和部分可观测 history 对组合下界造成的误差项。

如果理论只能重述标准 options/reach-avoid 结果，就将其降为方法解释，不包装成理论贡献。

## 7. 实验计划

### Gate 0：proposal 与 novelty 审计

不训练。完成：

- option-critic、skill chaining、Recovery RL、reach-avoid RL、residual VLA、VLA post-training 的
  系统 related-work matrix；
- 明确每个组件的最早近邻和 ReBridge 的必要差异；
- 固结主指标、计算预算和 stop conditions；
- 新建 train/development/confirmation split，旧 test 不再作为盲测。

**停止条件**：发现已有方法同时使用 frozen-policy continuation value 作为 recovery stopping
payoff，并联合训练 VLA residual bridge 与 handback，且本 proposal 无实质新机制。

### Gate 1：停止收益是否有效

只做测量，不训练 option。

1. 从 train recovery trajectories 的多个阶段恢复状态；
2. 每个状态运行 matched frozen Pi0.5 continuations；
3. 训练视觉历史 `V_psi`，在 source-disjoint development states 校准；
4. 与固定 milestone、stage classifier、object-distance 和 state progress 比较。

门槛：

- source-level Brier/ECE 明显优于固定阶段基线；
- `LCB >= tau` 的 handback precision 至少 90%；
- coverage 至少 30%，不能靠只接受极少状态得到高 precision；
- value 排序在三个 Pi0.5 seeds 上方向稳定，但不要求 BASIN 式 candidate preference flips。

**失败裁决**：`STOP_POLICY_RELATIVE_STOPPING_VALUE`。不训练 bridge。

### Gate 2：privileged optimal-stopping upper bound

先用 simulator 真值和重复 continuations 构造高成本 stopping oracle，比较：

- policy-only；
- teacher-to-fixed-regrasp；
- full teacher；
- full-task residual RL；
- fixed-duration residual option；
- ReBridge oracle stopping。

所有方法使用相同 failure states、环境交互预算、K=3 和 episode timeout。

门槛：ReBridge oracle stopping 相对 fixed-regrasp 或 full-task residual 至少满足一项：

- slip final success `+10 pp` 且 paired source CI 下界大于 0；
- 成功率不低于 5 pp 的前提下，option actions 减少至少 25%。

**失败裁决**：`STOP_REBRIDGE_OBJECTIVE`。说明 stopping formulation 本身没有优势。

### Gate 3：最小 learned pilot

仅 seed 41、development split。Pi0.5 完全冻结，训练小 residual actor、distributional critic
和 calibrated stop value。比较：

1. frozen Pi0.5 K=1/K=2/K=3；
2. A2C2-style supervised residual correction；
3. direct recovery SFT 已有结果；
4. DICE-style full-task residual RL；
5. ReCoVLA-style fixed stage reward residual；
6. fixed stable-regrasp option；
7. ReBridge，无 value potential；
8. ReBridge，完整 objective。

门槛：

- slip recovery 相对 frozen base 至少 `+15 pp`；
- 相对最佳 residual 强基线至少 `+5 pp` 或 paired CI 下界大于 0；
- attached 与 no-intervention 退化不超过 `5 pp`；
- 至少 80% handback 后 episode 不重新进入 recovery；
- option 平均时长显著短于完整任务长度；
- 三次独立训练初始化方向一致后才进入 Gate 4。

### Gate 4：三 seed 与多任务验证

在全新 source-disjoint confirmation groups 上运行 seeds `[41,42,43]`，然后增加至少：

- 一个恢复几何不同的 LIBERO task；
- 一个无需 forced slip、具有自然执行失败的 task；
- 条件允许时增加 RoboCasa 或真实机器人，不让其阻塞首轮 LIBERO gate。

主指标：overall success、slip recovery、normal success、recovery re-entry、handback precision、
completion steps、residual energy、environment interactions 和 wall-clock cost。统计单位始终是 source
snapshot group，使用 paired group bootstrap 和每 seed 结果。

## 8. 必须做的消融

1. `V_0` stopping payoff vs fixed stable-regrasp termination；
2. learned optimal stopping vs fixed option length；
3. on-policy collection vs frozen offline bank；
4. frozen base vs LoRA/full-policy update；
5. residual action vs standalone recovery action；
6. conservative LCB vs mean value handback；
7. short history vs current frame；
8. base-value potential vs hand-designed stage reward；
9. intervention cost `lambda_a/lambda_t` 的少量预注册设置，不做无止境 sweep；
10. policy-relative value vs task-agnostic progress value。

## 9. 风险与反证

### 风险 A：`V_0` 估计本身比恢复更昂贵

continuation rollout 只在训练期使用，但成本必须完整记账。若达到可用 calibration 需要与 exact
Oracle 相近的无限 rollout，方法不具实用性。可接受方向是 active state selection 或 offline
reuse，不是偷偷降低置信要求。

### 风险 B：value hacking 与过早 handback

residual 可能把观测推向 `V_psi` 高估区域。用独立 frozen-policy rollout 校验 handback states、
LCB、persistence 和 option-induced calibration drift。若仍频繁 false handback，停止 learned
stopping，不加 world model 掩盖问题。

### 风险 C：residual action 不足以跨 bridge

若 privileged residual upper bound 明显低于 standalone option 或 teacher，则 residual parameterization
限制了控制。此时可以切换 standalone option，但必须作为方法变更重新 preregister，不能事后混报。

### 风险 D：方法只是已有 residual RL 的小变体

如果 DICE/ReCoVLA/A2C2-style baseline 在等预算下达到相同成功率、正常能力和干预成本，结论应
采用更简单基线，不声称 ReBridge 贡献。

### 风险 E：单任务过拟合

cream-cheese forced slip 只适合机制验证。没有不同 failure geometry、自然 failure 和第二环境，
最多形成内部技术报告，不足以支持顶会主张。

## 10. 资源与工程边界

第一阶段不需要下载新模型或大数据集。复用：

- `/share/longjunyu/alphabrain/pretrained_models/pi05_base`；
- `/share/longjunyu/alphabrain/pretrained_models/paligemma-3b-pt-224`；
- 现有 LIBERO counterfactual/full-episode 数据；
- 已有 frozen Pi0.5 checkpoints 与 continuation infrastructure。

预估最小 Gate 1 主要消耗 simulator rollout 和少量 value-head 训练，不应重新训练 Pi0.5。Gate 2
先用现有多 GPU 并行跑 continuation oracle。只有 Gate 1 和 Gate 2 均通过，才启动 residual RL。

所有新 artifact 写入 `/share/longjunyu`，代码和小型文档留在仓库。不得读取旧 test、confirmation
或 sealed 数据；由于历史 test 已多次使用，论文级验证必须新建并冻结 confirmation source states。

## 11. 预期论文形态与投稿门槛

若 Gate 4 成功，论文主线应是：

1. 诊断 VLA recovery 中“基础能力存在但中间 bridge 不可达”的组合失败；
2. 提出以 frozen-base continuation value 为 stopping payoff 的 recovery option；
3. 给出保守 handback 的组合分析；
4. 在多任务、多 seed、至少两个环境或仿真加实机上胜过最新 residual/correction/post-training 基线；
5. 报告成功率、正常能力保持、交互成本和 handback calibration。

只有单任务 simulator 增益时，目标应是机制报告或后续扩展，不应直接以 ICML/NeurIPS/CVPR
级主张包装。要达到这些 venue 的标准，必须证明算法超出某个机器人脚本，具有跨任务的学习
原则、强基线优势和足够统计功效。ICRA/CoRL 类机器人 venue 也仍需要多任务和可信行为证据。

## 12. 当前建议

当前不应继续大规模训练。推荐顺序是：

1. 接受或修改本 proposal 的问题定义与方法边界；
2. 完成 Gate 0 系统 novelty audit；
3. 只实现 Gate 1 的 frozen-base stopping-value 数据与 calibration；
4. Gate 1 通过后再实现 privileged optimal-stopping upper bound；
5. 两个必要条件均成立，才投入 residual RL 工程。

本 proposal 的优点是每一步都能被低成本否定，而且失败后不会再回到 suffix weighting、局部
reranking 或离线 recovery replay。它的最大不确定性不是“能否写出代码”，而是策略相对停止
收益是否在视觉历史上足够可校准，以及它是否比已有 residual RL 的完整任务目标带来实质优势。

**当前裁决：`PROCEED_TO_PROPOSAL_REVIEW`，不自动启动实验。**

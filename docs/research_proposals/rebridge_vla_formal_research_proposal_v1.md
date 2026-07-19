# 博士研究计划书

## Policy-Relative Recovery Bridges for Robust Vision-Language-Action Control

中文题目：面向鲁棒视觉-语言-动作控制的策略相对恢复桥

暂定方法名：**ReBridge-VLA**

研究领域：机器人学习、具身智能、视觉-语言-动作模型、强化学习

文档状态：`FORMAL_PROPOSAL_V1_FOR_REVIEW`

日期：2026-07-19

计划周期：36 个月

> 本文是一份可供导师或独立评审审阅的初步博士研究计划。方法、贡献与实验结论均为待验证
> 假设。此前 AlphaBrain 实验仅作为 preliminary evidence，不被表述为论文最终证据。

## 摘要

视觉-语言-动作模型（Vision-Language-Action model, VLA）已经表现出跨任务、跨场景的通用
机器人控制能力，但其训练数据仍以无失败的专家轨迹为主。当抓取滑落、物体被意外移动、接触
执行偏差或局部子任务失败时，VLA 往往能够观察到异常并产生局部合理动作，却不能稳定恢复到
任务主轨迹。现有工作主要通过高频动作纠正、人工或 VLM 生成的恢复阶段、失败数据增强、
on-policy teacher distillation 或 full-task residual reinforcement learning 解决这一问题。
这些方法尚未显式回答一个组合控制问题：恢复策略应当纠正多久，以及何时把控制权交回一个
在正常任务上已经具备能力的冻结基础 VLA。

本研究提出 ReBridge-VLA，将失败恢复建模为**以冻结基础策略 continuation value 为停止收益的
部分可观测最优停止问题**。基础 VLA `pi_0` 保持冻结；一个小型 residual recovery option 在
检测到失败后，根据最新视觉、proprioception、基础动作和短 observation-action history 进行
闭环纠正。对于任一历史状态 `h`，`V_0(h)` 表示立即交回 `pi_0` 后的任务成功概率。恢复 option
在“立即停止并获得 `V_0(h)`”与“继续纠正并承担时间、动作和物理风险成本”之间作保守选择。
因此，恢复目标不是固定的 stable-grasp 语义阶段，也不是从失败状态重新学习完整任务，而是以
最小干预到达当前基础策略可以可靠完成任务的 competence region。

研究将分四个层次验证该主张。首先，在现有 LIBERO 反事实失败数据上验证基础策略停止价值能否
由部署可用历史可靠估计。其次，使用高成本 continuation oracle 检查最优停止目标本身是否优于
固定恢复阶段和 full-task residual。再次，在 LIBERO、LIBERO-PRO 扰动和 RoboCasa365 上与
A2C2-style correction、RaC/VLA-OPD-style on-policy correction、ReCoVLA-style stage reward、
DICE-style residual RL 及固定 handback option 进行等预算比较。最后，在第二 VLA backbone 和
条件允许的真实机器人平台上验证跨策略与 sim-to-real 外部有效性。主要结果为完整任务成功率、
失败恢复率、正常任务保持、handback precision、复发率、恢复时长和交互成本；统计单位为独立
初始状态或场景，而不是视频帧。

预期贡献包括：VLA 恢复中的 policy-relative bridge 问题定义；以 base continuation value 为
stopping payoff 的恢复算法；保守 handback 的组合分析；以及带事件触发扰动、自然失败、正常
能力保持和成本核算的恢复评测协议。若任一必要假设失败，研究将按照预注册标准停止对应方法
主张，不以增加动态 horizon、world model 或额外模块挽救结果。

**关键词**：Vision-Language-Action；robot failure recovery；residual reinforcement learning；
optimal stopping；policy composition；robust manipulation

## 1. 研究背景与意义

### 1.1 VLA 从任务执行走向可靠执行

VLA 将图像、自然语言和机器人状态映射为连续动作或动作块，为通用机器人策略提供了统一接口。
`pi_0.5` 通过异构机器人数据、语义预测和 web data co-training 展示了开放环境中的长程操作能力
[1]。OpenVLA-OFT、SmolVLA 等开源模型进一步降低了 VLA 训练和部署门槛 [2,3]。然而，标准
imitation learning 主要拟合成功专家轨迹；一旦实际执行进入训练分布外的接触状态，逐步误差会
改变后续观测，导致策略沿错误状态分布继续运行。

可靠性不能仅由 clean benchmark success 衡量。LIBERO-PRO 表明，标准 LIBERO 上的高成功率在
对象、位置、指令和环境变化下可能大幅下降 [4]。对真实部署而言，机器人不仅要避免失败，还要
能识别已经发生的失败、采取纠正动作，并在恢复后继续完成原任务。恢复能力因此同时涉及：

1. 失败是否可观测；
2. 当前动作是否需要修正；
3. 恢复控制能否跨越接触敏感的状态分布；
4. 何时可以安全地交回原任务策略；
5. 新增恢复能力是否破坏正常任务能力。

### 1.2 为什么“交回控制”是独立研究问题

已有恢复方法通常预先规定恢复段的结束：人工干预结束、固定动作数、达到语义阶段、完成整个
任务，或由统一策略自行隐式决定。这些定义没有显式考虑基础策略的能力边界。同一个物理状态
对于两个能力不同的基础策略可能具有不同含义：较强策略可以立即接管，较弱策略仍需纠正。
反之，固定的 `stable grasp` 可能对某些状态过晚，浪费 recovery supervision；对另一些状态又
过早，导致再次掉落。

因此，本研究关注的不是一般的 failure detection，也不是重新训练一个完整任务策略，而是：

> 如何学习一个与当前冻结 VLA 能力相匹配的恢复桥，使恢复控制以最小干预到达该 VLA 能可靠
> 接管的状态，并以可校准的规则交回控制？

这一问题连接 VLA post-training、options、policy composition、reach-avoid control 和 calibrated
policy evaluation，具有明确的算法和机器人系统意义。

## 2. 前期研究与问题形成

### 2.1 已有实验资产

AlphaBrain 当前建立了 LIBERO `put_the_cream_cheese_in_the_bowl` 的 128 个 snapshot groups。
每组包含相同前缀下的 attached 与 forced-slip 完整物理分支，共 256 条 episode 和 34,551 个
滑动训练窗口。expert 在两分支均达到 100% 正式任务成功。数据按初始 snapshot group 划分，
同组分支不跨 split。

该资产的价值是机制诊断，不是 benchmark 广度。历史 test 已被多轮机制实验使用，后续论文级
结果必须建立新的 source-disjoint confirmation split。

### 2.2 关键经验结果

| 研究问题 | 经验结果 | 推论 |
| --- | ---: | --- |
| slip 是否可见 | feedback frame 的 vision+state branch probe 为 100% | 当前任务不以主动视觉为必要前提 |
| base 是否完全没有恢复 mode | post-feedback sample0 局部 correct-mode 为 97.4% | 局部动作可用不等于长程恢复可用 |
| 高频重规划是否足够 | K=1 未解决完整恢复，K=3 更好 | 单纯缩短执行 horizon 不充分 |
| 瓶颈位于何处 | policy-only 21.85%；teacher-to-stable-regrasp 85.19% | 主要缺口位于 feedback 到 stable regrasp |
| 短纠正是否足够 | 3/12 teacher actions 无改善；到重抓平均约 80.5 actions | 需要长闭环 bridge 而非局部 patch |
| 离线恢复 SFT 是否足够 | clean 与 policy-state recovery 均显著伤害 slip/overall | 整网离线更新存在 drift 与 interference |
| base action support 是否有 headroom | exact receding continuation Oracle 的 slip 为 87.2%，高 15.4 pp | 真实反馈后的连续重决策具有上界 |
| 是否可压缩为局部标签 | local physical、teacher-distance、one-response、static winner 均失败 | bridge 不能由单步静态 surrogate 充分表示 |

### 2.3 前期工作的局限

这些结果来自一个仿真任务、有限初始状态和反复使用的分析 split，不能证明普遍性。它们仅提供
三个 proposal-level 依据：

1. 存在可测量的 failure-to-competence bridge；
2. 恢复终点应与基础策略的下游能力相关；
3. 静态离线动作监督不是当前最有希望的机制。

## 3. 研究问题、目标与假设

### 3.1 总体目标

建立一种能够与冻结 VLA 组合的闭环恢复方法，使机器人在可观测执行失败后恢复任务，同时保持
正常任务能力，并给出可复现的失败恢复 benchmark protocol。

### 3.2 研究问题

**RQ1：基础策略能力边界是否可测且可预测？**

从失败恢复轨迹上的不同状态交回冻结 VLA，其完整任务成功概率是否呈现可复现结构？该概率能否
仅由部署可用的视觉、proprioception 与短 action-consequence history 校准估计？

**RQ2：policy-relative stopping 是否比固定恢复目标更有效？**

把基础策略 continuation value 用作恢复 option 的停止收益，是否能在相同环境交互和训练预算下，
比固定时长、固定语义阶段和 full-task residual RL 获得更高恢复率或更低干预成本？

**RQ3：on-policy bridge learning 是否能解决恢复分布漂移？**

仅在 option 自己诱导的状态上更新小型 residual policy，是否优于离线 recovery SFT、on-policy
teacher distillation 和 supervised real-time correction？

**RQ4：方法是否跨任务、场景与基础策略成立？**

在不同失败类型、LIBERO/LIBERO-PRO、RoboCasa365 和第二 VLA backbone 上，policy-relative
handback 是否仍具有稳定优势？

### 3.3 可证伪假设

**H1（可校准性）**：在 source-disjoint states 上，部署历史预测的 `V_0` handback set 可达到
至少 90% precision 和至少 30% coverage，且明显优于固定阶段与几何 progress baseline。

**H2（恢复效果）**：ReBridge 相对 frozen base 的失败恢复率提高至少 15 个百分点，相对最佳
等预算 recovery baseline 提高至少 5 个百分点或 paired 95% CI 排除 0。

**H3（能力保持）**：clean/no-intervention success 的绝对退化不超过 5 个百分点。

**H4（组合效率）**：在成功率不低于最佳固定 handback 5 个百分点以上的前提下，ReBridge 的
recovery-controlled actions 或 teacher-equivalent supervision 至少减少 25%。

**H5（外部有效性）**：主要恢复增益在至少两个 benchmark、三随机种子和两个基础策略中方向一致。

若 H1 或 objective upper bound 失败，不进入 full learned method；若 H2/H3 失败，不作算法成功
主张；若仅 H5 失败，结论限定为单 backbone 或单 benchmark 机制。

## 4. 文献综述与研究缺口

本版采用 scoping review 而非声称已经完成系统综述。检索范围覆盖 arXiv、OpenReview、CVF Open
Access、机器人会议 proceedings 与 benchmark 官方仓库，时间截至 2026-07-19；关键词组合包括
`VLA failure recovery`、`robot corrective learning`、`residual VLA reinforcement learning`、
`action chunk correction`、`policy handoff`、`recovery option`、`reach-avoid`、`robust manipulation
benchmark`。纳入标准为：直接处理机器人/VLA 失败恢复、闭环纠正、policy composition，或提供
相关标准 benchmark；优先引用论文与官方项目页。Gate 0 将补做 backward/forward citation tracing，
记录检索式、日期、去重和排除理由，并在每次论文投稿前更新近 12 个月工作。

### 4.1 动作块实时纠正

A2C2 使用最新观测、基础动作、动作块位置和基础模型特征，每个控制步输出 residual correction，
在不重训基础 VLA 的情况下提高长 horizon 和延迟下的反应性 [5]。REMAC 通过 masked action
chunking 处理感知与执行不同步 [6]。这类方法解决 stale action 或短时偏差，但没有专门定义跨越
数十步恢复状态后何时交回基础策略。

### 4.2 恢复数据与 on-policy correction

RaC 在 policy rollout 中由人先把机器人带回熟悉状态，再提供 correction segment [7]。FLARE
通过 Retry/Reset 和 bridge segments 训练自主恢复 [8]。Dream2Fix 合成 counterfactual failures 与
可执行恢复轨迹 [9]，FailSafe 构造 failure-action pairs 并以任务成功验证纠正 [10]。VLA-OPD
在 student-induced states 上由强 teacher 提供 dense token supervision，并用 bounded reverse-KL
缓解 SFT drift [11]。这些研究证明恢复数据与 on-policy states 的重要性，但通常由人工、teacher
或数据过程规定纠正何时结束。

### 4.3 Residual RL 与 VLA post-training

ReCoVLA 冻结 VLA，使用外部 VLM 识别 failure mode/recovery stage 并编译 residual-policy reward
[12]。Object-Centric Residual RL 用 object pose residual 在模拟中训练并进行 zero-shot sim-to-real
[13]。DICE-RL 通过 distribution-contractive residual off-policy RL 利用 diffusion/flow prior [14]。
RobustVLA 则针对观测和动作扰动进行 robustness-aware RL post-training [15]。这些工作表明 frozen
VLA 加 residual control 是强基线，因此 residual architecture 本身不是本研究贡献。现有方法多以
完整任务回报、对象状态或语义 stage 为目标，没有把基础策略 continuation competence 作为 option
的显式停止收益。

### 4.4 Options、恢复区与 reach-avoid

Option-Critic 能联合学习 intra-option policy 和 termination [16]；Recovery RL 分离 task policy
与 safety recovery policy，并以 learned recovery zones 控制切换 [17]；reach-avoid RL 为“到达目标
同时避开失败集”提供 Bellman 形式与分析工具 [18]。因此，learned termination、独立 recovery
policy 与 reach-avoid objective 都是成熟概念。本研究的候选新意只能来自其具体组合对象：
**冻结 VLA 的任务 continuation value 作为 recovery stopping payoff，以及由此产生的 base-compatible
bridge objective 与保守 handback**。

### 4.5 Benchmark 与评测缺口

LIBERO 提供 130 个任务和四个知识迁移 suite [19]，但原始目标是 lifelong transfer，不是 failure
recovery。LIBERO-PRO 增加对象、位置、指令和环境扰动，主要考察 generalized robustness [4]。
CALVIN 用连续五任务序列评估长程语言条件控制 [20]。RoboCasa365 提供 365 个家庭任务、2,500 个
厨房环境及大规模 demonstrations，并区分 pretrain 与 target scenes/objects [21]。这些 benchmark
可以承载恢复研究，但目前缺少统一的：

- 物理事件触发而非绝对 step 触发的失败注入；
- 同一初始状态的 clean/failure 配对评测；
- isolated recovery 与 end-to-end recovery 分离；
- handback precision、复发和正常能力保持；
- 环境交互、teacher、推理与训练成本核算。

### 4.6 Gap analysis

| 方法 | 使用最新反馈 | 处理 option 自身状态 | 冻结 base | 显式 handback | handback 由 base task competence 决定 | 保持正常能力为主指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2C2 [5] | 是 | 否/监督分布 | 是 | 否 | 否 | 是 |
| RaC [7] | 是 | 是 | 否 | 人工 intervention 结束 | 熟悉状态但非 calibrated base value | 部分 |
| FLARE [8] | 是 | 由数据覆盖 | 否 | Retry/Reset 结构 | 否 | 部分 |
| VLA-OPD [11] | 是 | 是 | 否 | 否 | 否 | 是 |
| ReCoVLA [12] | 是 | RL rollout | 是 | recovery stage | 外部 VLM stage | 是 |
| DICE-RL [14] | 是 | RL rollout | prior/base 受约束 | 否 | 否 | 是 |
| Recovery RL [17] | 是 | 是 | task policy 可分离 | safety switching | 安全而非 task competence | 是 |
| ReBridge（拟议） | 是 | 是 | 是 | 是 | **是** | **是** |

表格只说明待验证的结构差异，不证明原创性或性能。正式实现前需继续追踪 2026 年以后新增工作；
若发现直接等价方法，应取消新名称并将研究改为复现、扩展或 benchmark study。

## 5. 理论与算法框架

### 5.1 问题定义

考虑部分可观测控制过程。部署历史为：

```text
h_t = (o_{t-L:t}, s_{t-L:t}, a_{t-L:t-1}, language)
```

`o` 包含外部和 wrist RGB，`s` 为部署可用 proprioception。冻结基础策略 `pi_0` 每次产生固定
长度动作块，正式主实验固定 `K=3`：

```text
a_t^0 ~ pi_0(. | h_t)
```

环境失败集合为 `F`，正式任务成功集合为 `G`。失败 detector 不属于主要贡献；isolated recovery
使用 evaluator 的 event trigger 隔离恢复能力，end-to-end 使用所有方法共享的 deployable detector。

### 5.2 基础策略停止价值

定义冻结 `pi_0` 的有限预算 continuation value：

```text
V_0(h) = P(hit G before timeout or F | h, thereafter execute pi_0)
```

对 train states 运行 matched repeated continuations，学习分布式视觉历史估计器 `V_psi(h)`。
部署使用经 source-disjoint calibration 得到的 lower confidence bound `L_psi(h)`。sim state、object
pose、branch label 和未来轨迹只用于训练标签和审计，不进入模型输入。

该对象不同于先前已停止的 BASIN candidate committor：BASIN 试图对 action candidates 做
cross-policy ranking；这里固定单个 base policy，`V_0` 只给“现在停止 recovery”定价。

### 5.3 Residual recovery option

失败后激活小型 policy：

```text
delta_t ~ rho_phi(. | h_t, a_t^0)
a_t = clip(a_t^0 + M delta_t)
```

`M` 为预注册 action mask；residual 范围满足控制安全界。Pi0.5 backbone 和 action expert 保持冻结。
第一实现使用冻结视觉 features 加轻量 temporal encoder、actor 和 distributional critic。若 residual
parameterization 的 privileged upper bound 不足，再把 standalone option 作为独立、重新预注册的
替代方法，而不是事后混合。

### 5.4 最优停止目标

每个 recovery state 有两个选择：立即 handback，或继续 residual control。

```text
U*(h) = max {
    V_0(h),
    max_delta E[ U*(h')
                 - lambda_a ||delta||^2
                 - lambda_t
                 - lambda_f I(drop or regress) ]
}
```

`V_0(h)` 是 stopping payoff。continue value 由 distributional critic 学习。保守 handback 规则为：

```text
STOP if L_psi(h) >= U_continue_upper(h) - epsilon
        and L_psi(h) >= tau_min
        and condition persists for q replans
        and no recent drop/regress
```

为改善稀疏学习，可以使用 potential shaping：

```text
r_shape = beta [ stopgrad(L_psi(h_{t+1})) - stopgrad(L_psi(h_t)) ]
```

但真实 task success 和 failure 仍由环境判据决定；`V_psi` 高分不能替代真实成功。

### 5.5 On-policy bridge learning

每轮从 train failure states 开始，由当前 option 执行并收集其诱导状态。handback 后冻结 Pi0.5
执行到 episode 结束，得到 composed success。训练数据严格区分：

- `D_base`：冻结 Pi0.5 continuation，用于 `V_psi`；
- `D_bridge`：option transitions，用于 actor/critic；
- `D_clean`：attached/no-intervention audit，只检查 residual false activation 与能力保持。

不得把 option success 当作 base continuation label，也不得用 privileged future 输入 policy。

### 5.6 初步组合性质

若 recovery option 以概率 `p_B` 在进入失败集合前到达 handback states，且这些状态的真实
`V_0(h_T) >= tau`，则组合成功率满足：

```text
P(success) = E[I(T_B < T_F) V_0(h_T)] >= p_B tau.
```

使用 approximate calibrated lower bound 时需加入 miscoverage 与 estimation error。论文理论部分
将研究 stopping Bellman operator、LCB handback 的过早停止风险及 potential shaping 的策略不变
条件。若只能直接复用标准 options/reach-avoid 结论，则将其作为解释而非理论贡献。

## 6. 研究方法与实验设计

### 6.1 总体设计原则

研究采用 staged, falsification-first design。每阶段只验证进入下一阶段的必要条件。所有主比较
在运行前冻结配置、Git commit、checkpoint hash、数据 manifest、primary endpoint 和停止门槛。
开发集可用于调参，confirmation set 只在方法和预算冻结后打开一次。

### 6.2 Benchmark 组合与作用

| 层级 | Benchmark/数据 | 研究作用 | 是否支持最终主张 |
| --- | --- | --- | --- |
| B0 | 现有 LIBERO-CF cream-cheese 128 groups | 机制诊断、代码 smoke、Gate 1 calibration | 否，单任务且历史 split 已使用 |
| B1 | LIBERO 四 suites + 新 source-disjoint states | 标准单臂 VLA、多任务恢复和 clean retention | 是，核心 benchmark 1 |
| B1-R | LIBERO-PRO position/object/environment variations | 检查 handback 与恢复对分布变化的鲁棒性 | 是，B1 外部分布 |
| B2 | RoboCasa365 target tasks/scenes/objects | 厨房多任务、articulated/contact、长程 composition | 是，核心 benchmark 2 |
| B3 | CALVIN ABC->D 五任务序列 | 检查跨子任务组合和长序列 recovery | 扩展，不替代 B1/B2 |
| B4 | SO-101 或 Franka 真实机器人 [22] | 检查真实接触、延迟和 detector-handback 联动 | 强烈建议，取决于硬件合作 |

选择依据如下。

**LIBERO** 与当前 Pi0.5/AlphaBrain 管线兼容，支持 RGB、proprioception、语言和正式 task predicates，
可低成本建立事件触发 counterfactual pairs。正式任务集不只使用 cream-cheese：从 Spatial、Object、
Goal 和 Long 中选择至少 12 个任务，覆盖 pick-place、容器放置、articulated interaction 和多阶段
组合。任务必须满足 frozen base clean success 位于可比较区间，且 scripted/teacher success 至少
90%，避免所有方法均失败的无效 benchmark。

**LIBERO-PRO** 用对象、初始位置、指令和环境变化检查方法是否只记住训练场景。恢复算法的核心
primary perturbations 仍是物理执行失败；LIBERO-PRO variations 用于外部分布，不与物理 failure
intervention 混为同一变量。

**RoboCasa365** 提供大规模厨房场景、原子与 composite tasks、disjoint target scenes/objects，适合
检验 policy-relative handback 是否跨任务阶段和环境成立。第一批选择至少 12 个 target tasks，
覆盖 pick/place、door/drawer、knob/lever 及至少四个 composite tasks。RoboCasa 迁移在资源可用后
进行，不阻塞 LIBERO Gate 1/2。

**CALVIN** 的五任务序列指标可以测试一次恢复是否影响后续任务 composition，但 embodiment 与
policy adapter 成本较高，列为扩展验证。**真实机器人**不作为早期算法 gate；SO-101 已有包含
failure taxonomy 与 recovery-aware metrics 的低成本 VLA 评测工作，可作为协议参考 [22]。若无硬件，
论文必须明确限定为仿真方法，不使用 sim-to-real 语言。

### 6.3 失败类型与 intervention protocol

建立 task-compatible failure taxonomy：

1. **grasp miss/slip**：在真实 gripper/contact event 附近降低接触或施加小位移；
2. **post-grasp drop**：在 lift/transport event 后触发受控滑落；
3. **target displacement**：在 policy 已观察目标后，将对象移动到可达但不同位置；
4. **articulation rebound/jam**：仅用于 door/drawer/knob 任务，施加可恢复阻力或回弹；
5. **execution lag/action attenuation**：连续有限控制步改变 action execution，不污染观测；
6. **natural failures**：不施加 intervention，按预注册 taxonomy 记录策略自然失败。

intervention 必须由物理事件触发，禁止固定 absolute step。每种 intervention 的强度在 train/development
冻结；confirmation 包含已见强度和一个未见但可恢复强度。每个 failure episode 有同 snapshot 的
clean pair，并使用相同随机种子与 episode budget。

所有 intervention 需通过 teacher recoverability gate：teacher 在该条件下最终成功至少 90%；若
teacher 无法恢复，该样本属于不可恢复扰动，不进入主要恢复成功率分母，而单独报告 robustness limit。

### 6.4 数据构成与划分

数据分四类：

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| 成功 demonstrations | LIBERO/RoboCasa 官方数据与现有 expert trajectories | base adapter、anchor、normal competence |
| Paired failure episodes | 同 snapshot clean/failure 物理分支 | detector audit、isolated recovery、行为分析 |
| Frozen-base continuations | recovery states 上重复执行 `pi_0` | `V_0` label 与 calibration |
| On-policy bridge rollouts | 当前 residual option 自己产生 | actor/critic learning |

划分遵循四级隔离：

1. 同一 simulator snapshot 的所有分支只属于一个 split；
2. 近重复 source scene/object layout 不跨 split；
3. confirmation 包含 disjoint source configurations；
4. cross-task generalization 另留完全未用于 method selection 的 tasks。

禁止以视频帧扩充统计样本量。旧 test、confirmation、sealed artifact 不进入新开发流程。最终 manifest
记录每个 source、task、scene、object、failure type、severity 和 split。

### 6.5 基础策略与模型范围

主 backbone 为当前可用的 Pi0.5 PyTorch/LeRobot checkpoint。第二 backbone 优先选择 OpenVLA-OFT
或 SmolVLA，依据 benchmark adapter 成熟度和资源决定。第二 backbone 的作用是验证方法不依赖
Pi0.5 flow head；不要求从零预训练 VLA。

同一 benchmark 内所有方法使用相同基础 checkpoint、observation、fixed K、control frequency、
flow/decoding steps 和 episode horizon。若不同 backbone 的 action spaces 不同，分别归一化后只做
backbone 内 paired comparison，不直接比较原始 action MSE。

### 6.6 对照方法

主要比较必须包括：

1. frozen base，`K=1/2/3`；
2. ordinary continuation 或等更新 base adapter；
3. recovery SFT 与 policy-state correction；
4. A2C2-style supervised residual correction；
5. RaC/VLA-OPD-style on-policy teacher correction；
6. ReCoVLA-style failure/stage reward residual；
7. DICE-style full-task residual RL；
8. fixed-duration residual option；
9. fixed stable-grasp/semantic-stage handback；
10. privileged continuation/stopping oracle；
11. ReBridge learned stopping。

失败 detector 对所有 runtime recovery 方法共享。isolated recovery 使用 oracle event trigger，防止
detector 差异掩盖 recovery control；end-to-end 再使用 deployable detector，并分别报告 detection、
recovery 和 composed system 结果。

### 6.7 公平预算

至少报告并匹配：

- environment interactions；
- successful/failure demonstration actions；
- teacher queries 或 VLM calls；
- frozen-base continuation rollouts；
- trainable parameter 数；
- GPU hours、wall-clock time 和 peak memory；
- deployment forward calls、control latency 和 residual-active actions。

主比较同时提供 interaction-matched 与 performance-matched cost。不能把昂贵 simulator search 隐藏为
“训练免费信息”。

### 6.8 评价指标

**Primary outcomes**：

- end-to-end full-task success；
- isolated recovery success；
- failure-branch final success；
- clean/no-intervention success retention。

**Handback outcomes**：

- handback precision：交回后 base 最终成功的比例；
- false handback rate；
- recovery re-entry/relapse rate；
- time/actions to handback；
- base continuation Brier score、ECE 与 coverage-risk curve。

**行为与效率 outcomes**：

- failure continuation、premature commitment；
- regrasp、lift、transport、place 子目标；
- recovery latency、completion steps、progress AUC；
- residual energy、constraint violations、drops；
- interactions、GPU hours 和推理延迟。

动作 MSE、critic loss 与 mode coverage 仅为机制指标，不参与最终 Go/No-Go。

### 6.9 统计分析与样本量

独立单位为 source initial-state group；同组 clean/failure、不同 method 和 policy seed 形成 paired block。
主分析采用 hierarchical paired bootstrap：先在 group 内聚合 policy seeds，再在 task 和 source 层重采样。
同时报告每 seed、每 task、绝对百分点差异和 95% CI。低成功率条件补充 Wilson interval；多 primary
baseline comparisons 使用 Holm correction。

Gate pilot 完成后，基于 observed paired discordance 做正式 power analysis。confirmation 的最低规划
为每 task/failure condition 30 个独立 source states、至少 8 个核心任务、3 个 policy seeds；若该规模
不能对 10 pp 恢复差异达到 80% power，则增加 source groups，而不是把帧当样本。benchmark aggregate
使用 task-stratified estimate，防止少数重复任务主导结论。

### 6.10 阶段 Gate

**Gate 0：novelty 与 protocol audit**

完成系统 literature matrix、benchmark adapters、new split 和 preregistration。若发现直接等价方法，
取消 ReBridge 原创性主张。

**Gate 1：`V_0` 可校准性**

仅 train/development。要求 handback precision >=90%、coverage >=30%，并优于 fixed-stage baseline。
失败则裁决 `STOP_POLICY_RELATIVE_STOPPING_VALUE`。

**Gate 2：objective upper bound**

用 privileged repeated continuation 测量 ReBridge stopping objective 是否胜过 fixed-stage 与 full-task
residual upper controls。要求恢复 `+10 pp` 且 paired CI 排除 0，或等成功下干预减少 25%。失败则
裁决 `STOP_REBRIDGE_OBJECTIVE`。

**Gate 3：learned LIBERO pilot**

一个 policy seed、development tasks，比较全部强基线。要求 H2/H3 均通过才扩大。

**Gate 4：三 seed、多任务 confirmation**

冻结方法和预算后打开新的 LIBERO confirmation，并运行 RoboCasa365。最终算法结论以 Gate 4 为准。

**Gate 5：第二 backbone 与真实系统**

验证跨 policy 与真实接触。Gate 5 不反向改变 Gate 4 方法；失败时收缩外部有效性主张。

## 7. 预期贡献

若主要假设成立，预期形成以下贡献：

1. **问题贡献**：提出 failure-to-competence bridge，区分 failure detection、local correction、完整任务
   relearning 与 policy-compatible handback。
2. **算法贡献**：以 frozen VLA continuation value 为 stopping payoff 的 on-policy residual option，
   联合学习 bridge control 与 conservative handback。
3. **理论贡献候选**：给出 approximate stopping value 下组合成功和过早 handback 风险分析。
4. **评测贡献**：在标准 benchmark 上建立 event-triggered paired failure protocol、isolated/end-to-end
   分解、clean retention 和完整成本核算。
5. **经验贡献**：系统比较 supervised correction、recovery data、on-policy distillation、stage-reward
   residual 与 full-task residual RL 在相同 VLA 和 failure distribution 下的适用边界。

若 ReBridge 不优于强基线，贡献应降为严谨负结果或 benchmark analysis；不得仅凭新术语声称算法创新。

## 8. 可行性、资源与研究训练

### 8.1 现有资源

- 8 x RTX 5090 32GB 开发机；
- AlphaBrain PyTorch 框架与 Pi0.5 adapter；
- Pi0.5 base 与 PaliGemma 本地权重；
- LIBERO 环境、完整 counterfactual episodes、闭环 evaluator 和 H.264 视频审计工具；
- 多 seed fixed-K evaluation、group-level bootstrap 与 artifact manifests。

第一阶段不需下载大模型或数据集。RoboCasa365、第二 backbone 和真实机器人资源在 Gate 2 后申请或
迁移，避免在必要假设未通过时消耗大量存储和带宽。

### 8.2 所需研究训练

- distributional RL、off-policy actor-critic 与 option learning；
- calibration、conformal/risk-control 方法；
- MuJoCo/robosuite/RoboCasa 仿真与接触控制；
- hierarchical statistics、power analysis 和 reproducible systems evaluation；
- 真实机器人安全协议与实验设计（若进入 Gate 5）。

## 9. 风险、伦理与替代方案

### 9.1 技术风险与 Plan B

| 风险 | 诊断 | 预注册处理 |
| --- | --- | --- |
| `V_0` 无法由短历史校准 | Gate 1 失败 | 停止 learned stopping；报告 partial observability，不自动加 world model |
| continuation label 成本过高 | cost curve 无法收敛 | active state selection/复用；仍昂贵则停止实用性主张 |
| residual 无法跨 bridge | privileged residual upper bound 低 | 重新预注册 standalone option；不与 residual 结果混报 |
| fixed-stage baseline 同样好 | Gate 2/3 无优势 | 采用更简单 baseline，停止 ReBridge 方法主张 |
| value hacking/过早 handback | option-induced calibration drift | 独立 rollout audit、LCB、persistence；无法控制则停止 |
| clean capability 退化 | H3 失败 | 收紧 activation/mask；若仍失败则 No-Go |
| 单任务或单 backbone 成立 | Gate 4/5 不复现 | 限定结论，不宣称通用 VLA recovery |
| teacher 本身恢复失败 | recoverability <90% | 修复 benchmark/controller，不用低基线裁决算法 |

### 9.2 伦理与安全

仿真实验风险较低，但真实机器人实验需要速度、力矩、工作空间和急停限制；先在 simulation replay
验证，再逐级开放动作。若收集 human teleoperation/intervention，需获得参与者同意、记录劳动时间并
避免保存不必要的身份信息。使用公开数据与模型时遵守 license 和 attribution；视频发布需移除人员
身份信息。计算实验报告 GPU hours 与失败运行，减少无门槛 sweep 带来的资源浪费。

### 9.3 复现与研究诚信

- 配置、代码、checkpoint hash、split manifest 和统计脚本版本化；
- confirmation 访问 fail-closed，并记录文件访问审计；
- 所有成功与失败 episode 保存为 H.264/AVC `avc1`, `yuv420p`, fast-start；
- 报告负结果、协议偏差和 quarantine artifacts；
- 不选择性汇报 seed、task 或 K；
- 论文图表由 machine-readable result 自动生成。

## 10. 工作计划与里程碑

| 月份 | 工作包 | 主要交付物 | 决策点 |
| --- | --- | --- | --- |
| M1-M3 | 系统综述、proposal 修订、novelty audit | related-work taxonomy、gap matrix、最终 protocol | Gate 0 |
| M4-M6 | LIBERO recovery benchmark 工程 | 12+ tasks、interventions、new splits、teacher audit | benchmark validity |
| M7-M9 | Frozen-base continuation value | value dataset、calibration、cost curves | Gate 1 |
| M10-M12 | Privileged stopping upper bound | fixed-stage/full-task/oracle comparison | Gate 2 |
| M13-M17 | Residual option 与 on-policy learner | seed-41 pilot、ablation、video audits | Gate 3 |
| M18-M21 | 三 seed LIBERO confirmation | preregistered final LIBERO results | Gate 4A |
| M22-M25 | RoboCasa365 transfer | multi-scene/multi-task results | Gate 4B |
| M26-M28 | 第二 backbone | Pi0.5 vs OpenVLA-OFT/SmolVLA | Gate 5A |
| M29-M31 | 真实机器人或 CALVIN 扩展 | real recovery 或长序列 external validity | Gate 5B |
| M32-M34 | 理论、系统整理、论文撰写 | main paper、benchmark release | submission |
| M35-M36 | thesis integration 与答辩准备 | thesis chapters、reproducibility package | completion |

潜在论文/论文章节划分：

1. VLA recovery failure taxonomy 与 paired benchmark；
2. policy-relative stopping value 与 recovery bridge algorithm；
3. cross-benchmark/backbone/real-system study。

## 11. 成功标准与投稿定位

单个 LIBERO task 的正结果只构成 pilot。机器人顶会或综合 AI/ML/CV 顶会所需最低证据为：

- 两个 benchmark、多个任务和三 seeds；
- 与最新 correction、data-centric recovery、on-policy distillation 和 residual RL 强基线比较；
- 至少一个第二 backbone；
- clean capability、恢复效果、calibration 和成本同时成立；
- source-level 统计与 untouched confirmation；
- 对最接近工作的清晰机制差异，而非组件堆叠。

若方法贡献偏控制与机器人系统，优先考虑 ICRA/CoRL/RSS；若形成一般 optimal-stopping/policy-
composition 学习原则并跨环境成立，可考虑 ICML/NeurIPS/AAAI；若视觉反馈表示与跨视觉扰动成为
关键贡献，才适合 CVPR/ICCV/ECCV。投稿 venue 应由最终贡献决定，不反向驱动伪创新。

## 12. 参考文献

[1] Physical Intelligence et al. [pi_0.5: a Vision-Language-Action Model with Open-World
Generalization](https://arxiv.org/abs/2504.16054). 2025.

[2] Kim, M. J., Finn, C., and Liang, P. [Fine-Tuning Vision-Language-Action Models: Optimizing
Speed and Success](https://arxiv.org/abs/2502.19645). 2025.

[3] Tu, C.-H. et al. [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient
Robotics](https://arxiv.org/abs/2506.01844). 2025.

[4] Zhou, X. et al. [LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action
Models Beyond Memorization](https://arxiv.org/abs/2510.03827). 2025/2026.

[5] Sendai, K. et al. [Leave No Observation Behind: Real-time Correction for VLA Action
Chunks](https://arxiv.org/abs/2509.23224). 2025.

[6] Wang, H., Zhang, G., Yan, Y., Shang, Y., Kompella, R. R., and Liu, G.
[Real-Time Robot Execution with Masked Action Chunking](https://arxiv.org/abs/2601.20130).
ICLR 2026.

[7] Hu, Z. et al. [RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and
Correction](https://arxiv.org/abs/2509.07953). 2025.

[8] Zhao, G. et al. [FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in
Visual-Language Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html).
CVPR 2026.

[9] Li, D. et al. [Learning Actionable Manipulation Recovery via Counterfactual Failure
Synthesis](https://arxiv.org/abs/2603.13528). 2026.

[10] Lin, Z. et al. [FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action
Models](https://arxiv.org/abs/2510.01642). 2025.

[11] Zhong, Z. et al. [VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action
Models via On-Policy Distillation](https://arxiv.org/abs/2603.26666). 2026.

[12] Hu, H. et al. [ReCoVLA: VLM-Guided Reward Compilation for Failure Recovery in
Vision-Language-Action Policies](https://arxiv.org/abs/2606.09630). 2026.

[13] Kim, K. et al. [Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA
Enhancement](https://arxiv.org/abs/2606.18953). 2026.

[14] Sun, Z. and Song, S. [From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL
Finetuning](https://arxiv.org/abs/2603.10263). 2026.

[15] Zhang, H. et al. [Robustness-Aware Reinforcement Post-Training for Vision-Language-Action
Models](https://arxiv.org/abs/2511.01331). 2025.

[16] Bacon, P.-L., Harb, J., and Precup, D. [The Option-Critic
Architecture](https://arxiv.org/abs/1609.05140). AAAI 2017.

[17] Thananjeyan, B. et al. [Recovery RL: Safe Reinforcement Learning with Learned Recovery
Zones](https://arxiv.org/abs/2010.15920). 2020.

[18] Hsu, K.-C. et al. [Safety and Liveness Guarantees through Reach-Avoid Reinforcement
Learning](https://arxiv.org/abs/2112.12288). RSS 2021.

[19] Liu, B. et al. [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
Learning](https://arxiv.org/abs/2306.03310). NeurIPS Datasets and Benchmarks 2023.

[20] Mees, O. et al. [CALVIN: A Benchmark for Language-Conditioned Policy Learning for
Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227). 2021.

[21] Nasiriany, S. et al. [RoboCasa365: A Large-Scale Simulation Framework for Training and
Benchmarking Generalist Robots](https://arxiv.org/abs/2603.04356). ICLR 2026.

[22] Yu, Y. and Qiu, X. [Benchmarking Vision-Language-Action Models on SO-101: Failure and
Recovery Analysis](https://arxiv.org/abs/2606.08881). 2026.

## 附录 A：正式 proposal 结构合规说明

本文结构参照高校博士研究计划的共同要求，而不是套用某一学校的页数限制：

- [University of Cambridge, Department of Computer Science and Technology](https://www.cst.cam.ac.uk/local/phd/year1-report)
  要求问题定义、批判性文献综述、最接近工作及其优缺点、gap analysis、风险与 Plan B；
- [University of Cambridge postgraduate proposal guidance](https://www.postgraduate.study.cam.ac.uk/apply/how/research-proposal)
  要求研究问题、领域定位、重要性、方法、范围、资源和时间表；
- [University of Edinburgh proposal guidance](https://geosciences.ed.ac.uk/study/degrees/research-degrees/application-information/research-proposal)
  明确列出 introduction、theory/research context、research question、methodology、ethics、timetable
  和 references；
- [Imperial College Engineering initial research plan](https://www.imperial.ac.uk/engineering/departments/bioengineering/admin/research/doctoral/initial-research-plan/)
  要求 methodological approach、potential difficulties、back-up strategies 和 timeline。

正式提交到具体院校时，仍需按该院系的页数、字体、导师匹配、个人背景和签字模板压缩或调整。

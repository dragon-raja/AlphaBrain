# 具身智能 Research Reset：近期工作地图

更新时间：2026-07-15。

## 调研范围与证据标准

本地图围绕本项目已经观测到的 `slip -> recovery -> completion` 断点，检查最近两年的 VLA 执行调度、失败检测与恢复、动作后果表征、进度/奖励建模、主动感知和 post-training 工作。优先引用论文原页、作者项目页和官方代码仓库。仅有预印本的工作按预印本处理；“没有在本轮检索中发现”不等价于“此前没有人做过”。

对 AlphaBrain 当前资源的可复现性分为：

- **高**：冻结 Pi0.5、现有 LIBERO 数据和 8 卡 5090 即可做最小复现；
- **中**：需要新增轻量模块、rollout 或标签，但不需要重新预训练大模型；
- **低**：需要真机、大规模跨任务数据、专用传感器或昂贵 online RL。

## 1. Action chunk 与执行调度

| 工作 | 解决的问题与方法 | 没有直接解决的部分 | 代码/复现 | 对本项目边界的影响 |
| --- | --- | --- | --- | --- |
| [Real-Time Chunking](https://arxiv.org/abs/2506.07339) | 以异步生成、冻结已承诺动作和 inpainting 消除 VLA 推理延迟与 chunk 边界停顿。 | 不判断抓取失败后的候选动作能否带来正确物理后果。 | 作者项目可用；复现中。 | “更平滑/低延迟地执行 chunk”不是新的恢复问题。 |
| [Adaptive Action Chunking (AAC)](https://arxiv.org/abs/2604.04161) | 用动作熵在推理时自适应决定 chunk 长度，平衡反应性和 mode jumping。 | 熵不能直接验证候选 chunk 的任务进度或接触后果。 | 有项目页与代码；高。 | 泛化的动态 K 已被直接覆盖。 |
| [A3: Dynamic Execution Commitment](https://arxiv.org/abs/2605.11567) | 通过多样本共识、条件不变性与 prefix-closed 验证选择可接受动作前缀。 | 主要验证动作内部一致性，不验证执行后对象是否完成目标状态转移。 | 预印本；中。 | 与 FRESH Oracle Commit/self-consistency 高度相邻，不能再以“学习执行长度”为主贡献。 |
| [VLA-Corrector](https://arxiv.org/abs/2607.01804) | 运行时监测与修正 VLA 动作，目标是阻止错误继续累积。 | 与候选动作的多阶段物理进度排序仍有区别，但边界需要逐项实证。 | 最新预印本；中。 | “外挂修正器”本身不构成足够原创性。 |
| [Adaptive Action Chunking via Multi-Chunk Q](https://arxiv.org/abs/2605.10044) | 以多 chunk 长度 Q 值在 offline-to-online RL 中动态选择执行长度。 | 依赖 RL/Q 学习，且核心仍是长度选择。 | 预印本；低到中。 | 动态 K + value 的组合也已拥挤。 |

**当前判断**：Stage A 的 privileged event-aligned Oracle Commit 与 `fixed_k3` 成功率完全相同，且不优于 random/self-consistency 控制。结合上述近邻，执行长度不是值得继续扩展的主课题。

## 2. 失败检测、解释与恢复监督

| 工作 | 输入、输出与训练方式 | 尚未覆盖的缺口 | 代码/复现 | 最可能的审稿质疑 |
| --- | --- | --- | --- | --- |
| [SAFE](https://arxiv.org/abs/2506.09937) / [项目页](https://vla-safe.github.io/) | 从 VLA 内部特征预测通用失败概率，并用 conformal prediction 平衡准确率与提前量；训练使用成功/失败 rollout。 | 输出失败标量，不负责给多个候选动作排序，也不保证恢复完成。 | 项目公开；高。 | 新 failure detector 若只用图像二分类，会被认为弱于内部特征基线。 |
| [RoboFAC](https://arxiv.org/abs/2505.12224) | 9,440 条错误轨迹与 QA，训练任务理解、失败分析和纠正模型；作为外部监督产生修正指令。 | 语言纠正是否落实为接触稳定、运输、放置的连续后果仍依赖底层 VLA。 | 数据/代码状态需复核；中。 | 简单 recovery prompt 或 VLM supervisor 很难与其区分。 |
| [FailSafe](https://arxiv.org/abs/2510.01642) | 自动生成失败与可执行恢复动作，训练 VLM 检测并帮助多种 VLA 恢复。 | 偏高层 reasoning/supervision；未专门验证候选低层 chunk 的物理后果。 | 宣称计划开源；中。 | “生成失败数据再给恢复指令”已被直接覆盖。 |
| [TCoT](https://ojs.aaai.org/index.php/AAAI/article/view/37577) | 以时间链式推理组织失败理解和恢复。 | 高层推理与低层连续控制之间仍可能断裂。 | 论文可得；中。 | 语言化恢复流程不新，必须证明低层控制独特机制。 |
| [RACER](https://arxiv.org/abs/2409.14674) / [代码](https://github.com/sled-group/RACER) | 从失败中生成恢复策略并进行闭环纠正。 | 对 Pi0.5 多样本动作后果的细粒度选择不是核心。 | 有代码；中。 | 通用 recovery framework 已有强基线。 |
| [FPC-VLA](https://arxiv.org/abs/2509.04018) | 面向失败预测与修正的 VLA 框架。 | 与候选 chunk 物理进度的区别需靠输入输出和评测明确。 | 预印本；中。 | “预测失败后修正”命名层面高度重叠。 |
| [ViFailback](https://arxiv.org/abs/2512.02787) | 利用视觉反馈进行失败回退与恢复。 | 不必然解决局部正确动作无法组合成长程成功。 | 预印本；中。 | 仅增加 visual feedback/replan 会被认为是直接近邻。 |
| [ProbeAct](https://arxiv.org/abs/2606.09740) | 训练外 hidden-state object probe、运动学状态机与 CBF 修正，处理抓取/放置失败。 | 依赖可定义的对象状态和安全约束；不是通用候选后果学习。 | 预印本；中。 | 主动 probe/状态机式恢复已经有直接实现，不能泛泛提出“探测动作”。 |

**当前判断**：通用失败检测、语言错误解释、恢复指令和训练外纠正都已有密集近邻。若 AlphaBrain 只加入失败分类器或固定 recovery prompt，原创空间和机制证据均不足。

## 3. 动作后果、历史与进度/奖励

| 工作 | 解决的问题与监督 | 尚未解决的部分 | 复现性 | 与候选课题的边界 |
| --- | --- | --- | --- | --- |
| [Reflective VLA](https://arxiv.org/abs/2606.25215) | 把历史 observation-action-consequence triplet 输入策略，帮助识别相机、标定和执行偏差；以 block-causal 方式训练。 | 重点是利用**已发生**后果适应部署偏差，不是对同一当前状态的多个**候选未来 chunk**做恢复进度排序。 | 中。 | 任何 action-consequence history 方法都必须与其做 matched history-only 和 no-selector 对照。 |
| [ProgVLA](https://arxiv.org/abs/2605.28231) | 用 progress heads、remaining-horizon critic 和 offline RL 目标增强长程技能学习。 | 进度作为策略内部训练信号，不等同于运行时验证候选动作是否实现下一物理子目标。 | 中。 | 泛化“加进度 head”不新；候选级、反事实执行结果是可能边界。 |
| [RoboReward](https://arxiv.org/abs/2601.00675) | 用 OXE/RoboArena、counterfactual negative 和 temporal clipping 训练通用视觉语言奖励模型，并用于真机 RL。 | 以视频/任务结果给奖励，时空粒度可能不足以区分 1--3 步接触后果。 | 模型较大；低到中。 | 新方法必须证明细粒度 candidate-action 条件化，而不是另一个通用 reward model。 |
| [Reflective VLA 项目页](https://lianqing11.github.io/reflective-vla-page/) | 提供 action consequence 与普通 history 的对照。 | 尚未回答 recovery candidate ranking。 | 中。 | 是“动作后果表征”最强的近期重叠风险。 |
| [pi0.5](https://arxiv.org/abs/2504.16054) / [OpenPI](https://github.com/Physical-Intelligence/openpi) | 开放世界泛化 VLA 与官方训练/部署代码。 | 基础模型并不提供显式失败后果验证。 | 当前环境已具备；高。 | 继续使用冻结 Pi0.5 可隔离新增机制。 |

**关键空隙（需要进一步证伪，不是新颖性声明）**：现有工作分别覆盖历史后果条件化、任务级进度/奖励和失败检测，但“在反馈已经可见且局部恢复 mode 已存在时，按预期物理阶段转移验证并选择候选短 chunk”仍可能形成更窄的研究问题。该边界最接近 Reflective VLA、ProgVLA、SAFE 和 VLA-Corrector，必须共同比较。

## 4. 主动感知、probe 与 belief

| 工作 | 方法 | 对当前证据的含义 |
| --- | --- | --- |
| [ActiveVLA](https://arxiv.org/abs/2601.08325) | 主动选择 3D 关键区域、视角和 zoom，降低遮挡并提高精细操作。 | 若 slip 在当前双视角中不可识别，主动视角有价值；若单帧 probe 已高准确，则不应把它作为当前主因。 |
| [ProbeAct](https://arxiv.org/abs/2606.09740) | 从内部特征定位对象，以运动学状态机检测失败并用 CBF 修正。 | 已覆盖大量抓取/放置 probe 和训练外恢复空间。 |
| [Back to a Familiar Future](https://arxiv.org/abs/2606.09258) | 以恢复到熟悉未来状态为目标处理偏离。 | 与“回到训练分布”类恢复相关，候选方法必须证明自己不是简单回退。 |

当前 LIBERO 任务同时有 agent-view、wrist-view 和 proprioception。主动感知是否必要由反馈可观测性 probe 裁决，不凭直觉升级为主课题。

## 5. Post-training 与 RL

| 工作 | 方法范围 | 资源与边界 |
| --- | --- | --- |
| [pi0.6 technical report](https://www.pi.website/download/pistar06.pdf) | 以更强训练与后训练提升开放世界机器人能力。 | 大规模数据与后训练资源显著高于当前最小验证预算。 |
| [RL Token](https://www.pi.website/download/rlt.pdf) | 将 RL 信号引入 VLA 训练。 | 可作为长期 baseline；不能在尚未定位失败断点时先用 RL 掩盖机制。 |
| [Action-chunk PPO](https://arxiv.org/abs/2509.25718) | 对 action chunk 直接做 PPO 式优化。 | online rollout 成本高，且难区分收益来自表示、探索还是 reward。 |
| [RoboReward](https://arxiv.org/abs/2601.00675) | 通用 reward model + 真机 RL。 | 构成 reward/post-training 强近邻。 |

本轮不选择 RL 为第一步：现有 Full-H 已表现出“局部恢复 mode 存在但阶段推进中断”，应先用冻结策略和候选级诊断验证可选性，再决定是否需要 policy update。

## 6. 近期工作后的排除与保留

### 已排除为主贡献

1. **精确 suffix loss weighting**：完整三 seed 闭环已失败。
2. **Oracle/dynamic execution horizon**：Oracle Commit 无成功率收益，且 AAC/A3/RTC 已覆盖邻域。
3. **普通失败二分类器**：SAFE 是更强、跨任务且使用内部特征的直接基线。
4. **通用 recovery prompt/VLM supervisor**：RoboFAC、FailSafe、TCoT、RACER 等近邻密集。
5. **泛化的 progress head、reward model 或 RL**：ProgVLA、RoboReward 与多种 VLA post-training 已覆盖宽问题。
6. **默认主动视角/probe action**：须先证明当前观测不可辨；ProbeAct/ActiveVLA 已是直接近邻。

### 仍保留、但必须通过最小实验的窄问题

**反馈后候选动作的物理后果与阶段推进验证**：不是预测执行多长，不是仅检测失败，也不是把完整 world model 输入策略；而是从冻结 VLA 的多个短候选 chunk 中，依据当前多视角、状态和短 observation-action-consequence 历史，判断哪个候选最可能完成下一可验证物理转移（重新抓稳、抬升、运输、放置），并在后果不符时立即重新选择。

保留该问题的必要条件：

- 反馈结果在当前输入中可辨；
- Full-H 能生成局部正确恢复 mode；
- 简单 prompt、固定 K、Oracle K 和 self-consistency 不能解决完整恢复；
- 行为漏斗显示主要损失发生在 regrasp 之后的阶段推进，而不是纯 failure detection；
- 一个轻量 candidate-level 后果分数能在冻结策略上提高短程阶段转移，之后才值得训练完整方法。

若这些条件任一不成立，应输出 `NO_VALID_RESEARCH_PROBLEM_YET`，而不是扩大方法范围。

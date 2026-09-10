# VERIFY-VLA 研究地图：从动作生成到决策性物理验证

审计日期：2026-07-18 UTC

## 1. 现有证据给出的起点

本项目不从“再加一个模块”出发，而从 AlphaBrain 已经完成的闭环证据出发：

- FRESH 的 suffix loss weighting 改善过离线前缀误差，但没有改善闭环成功；
- 当前 LIBERO 抓取滑落任务在 feedback frame 已经可由图像和 state 线性区分，现有任务不是单纯的“看不见”；
- Full-H 能生成局部恢复动作，普通 sample availability 不是主瓶颈；
- teacher action distance 和 K=2 short-physical heuristic 都不能预测最终闭环表现；
- 只有昂贵的 policy-in-the-loop continuation oracle 稳定提高 slip recovery：71.8% -> 87.2%，overall 83.3% -> 93.6%。

因此，已经被证据排除的直接延伸包括：重新调 suffix 权重、只缩短执行 horizon、普通
self-consistency、prompt-only recovery，以及把局部动作相似度包装成长期价值。

## 2. 近邻路线与拥挤程度

### 2.1 VLA post-training 与失败数据

- [FlowPRO](https://arxiv.org/abs/2606.05468) 用 intervention-and-rollback 构造偏好对，并以
  proximalized flow preference objective 更新 VLA。
- [AFIL](https://arxiv.org/abs/2605.08434) 同时学习成功和失败 action generator，以失败分布
  在采样时提供负 guidance。
- [HAPO](https://arxiv.org/abs/2506.07127) 从人类干预轨迹构造 action preference optimization。
- [Z-1](https://arxiv.org/abs/2606.31846) 在 Pi0.5 上用 task-wise GRPO、共享前缀分支和完成度
  reward 做大规模 post-training。

判断：这是重要且有效的热门主线，但“使用失败/偏好/RL”本身已经不是独特贡献。新方法若最终
进入 post-training，必须先证明它提供了普通成功标签、未配对失败或通用 reward 没有提供的监督。

### 2.2 World model、test-time dreaming 与 policy-in-the-loop

- [DreamAvoid](https://arxiv.org/abs/2605.11750) 在关键阶段采样多个 action chunks，预测短期未来并
  排序以避开失败。
- [Feedback World Model](https://arxiv.org/abs/2605.15705) 用真实执行反馈在线修正 latent world model，
  再指导 diffusion policy。
- [WMPO](https://openreview.net/forum?id=qE2FyvRvuF) 在像素 world model 内进行 VLA policy optimization。
- [PiL-World](https://arxiv.org/abs/2606.05773) 明确交替执行 VLA 和视频 world model，支持闭环
  policy-in-the-loop 评测。
- [RISE](https://arxiv.org/abs/2602.11075) 将 controllable dynamics 与 progress value 组合，用想象
  rollout 产生 policy advantage。

判断：把 world model 用于动作候选排序、闭环想象或 policy improvement 已非常拥挤。AlphaBrain
的 continuation oracle 可以作为问题证据和上界，但不能直接改名成新 world-model 方法。

### 2.3 动态 action chunk 与 failure detection

- [AutoHorizon](https://arxiv.org/abs/2602.21445) 从 action self-attention 估计模型的预测极限。
- [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161) 用 action entropy 动态选择执行长度。
- [ActProbe](https://arxiv.org/abs/2606.08508) 用连续 action chunks 的一致性与幅度提前检测失败。
- [SAVE/VFD](https://arxiv.org/abs/2606.18043) 从 flow velocity-field disagreement 估计 epistemic
  uncertainty，用于失败检测和主动微调。

判断：何时重规划、何时报警和如何量化 flow uncertainty 已有强近邻。此前 Oracle Commit 也没有
改善当前任务成功率，不能再以 dynamic K 为主张。

### 2.4 Memory 与 partial observability

- [MemoryVLA](https://arxiv.org/abs/2508.19236) 使用 working memory 与 perceptual-cognitive memory bank。
- [ReMem-VLA](https://arxiv.org/abs/2603.12942) 使用 frame/chunk 两级 recurrent queries。
- [muVLA](https://arxiv.org/abs/2606.12497) 在控制实验中隔离最小 recurrent memory 的作用。

判断：给 VLA 加 history、recurrent token 或 memory bank 同样拥挤。只有明确证明历史中存在当前帧
不可恢复的任务变量，才值得进入这条路线。

### 2.5 Active perception 与 information-seeking control

- [DISaM](https://arxiv.org/abs/2410.18964) 把 information-seeking policy 与 information-receiving
  manipulation policy 分开，并根据后者对不同 context 的 action uncertainty 切换。
- [AAWR](https://arxiv.org/abs/2512.01188) 用训练期 privileged sensors 学 advantage，已在 NeurIPS 2025
  证明可从 generalist policy 初始化出主动感知行为。
- [Vision in Action](https://arxiv.org/abs/2506.15666) 从共享视角的人类示范学习主动头部相机策略。
- [Neural Graspness Field](https://openreview.net/forum?id=6FYh6gxzPf) 通过 next-best-view 降低抓取几何
  不确定性。
- [The Value of Sensory Information to a Robot](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1126028d9f1f69c13571ec462084d31-Abstract-Conference.html)
  表明许多标准任务中传感信息并不总是关键，任务随机性与信息出现时机必须被显式控制。

判断：通用“主动看”、最大化状态信息、训练独立探索策略都已有成熟近邻。仍未被这些工作完整覆盖的
窄问题是：**接触操作中，同一个机械臂动作同时改变任务状态并产生关于执行结果的证据，信息动作与
任务动作无法因子化；怎样让 VLA action chunk 以低风险尾动作主动验证刚发生的物理结果？**

## 3. 候选问题筛选

| 候选方向 | 真实缺口 | 当前可证伪性 | 近邻压力 | 结论 |
|---|---|---:|---:|---|
| 再做 VLA preference/RL | 闭环目标优于 BC | 中 | 极高 | 暂不立项 |
| 闭环 world model distillation | continuation oracle 昂贵 | 高 | 极高 | 作为强 baseline |
| recurrent memory | 隐状态可能影响接触 | 中 | 高 | 需要先证明当前帧不足 |
| action-effect latent | motor label 与物理结果脱钩 | 中 | 高 | 保留为后续 representation baseline |
| 通用 active vision | 遮挡下需主动取景 | 高 | 高 | 不足以形成主张 |
| **decision-valued physical verification** | 接触结果决定后续动作，但可通过低风险物理动作主动揭示 | **高** | **中** | **进入 Gate 0** |

## 4. 暂定研究命题

> Chunked VLAs should not only predict task-progressing controls. At decision-critical interaction
> boundaries, they should allocate a short verification tail whose physical consequence makes the
> next optimal continuation identifiable, but only when the expected reduction in decision regret
> exceeds its task risk.

中文表述：动作块不是在未知结果上继续“猜后缀”，而是在关键接触后用一个兼具控制和感知作用的
短尾动作，把不可判定的物理结果变成下一次重规划可用的证据。

这不是新颖性结论，只是 **novelty hypothesis**。若 Gate 0 不存在安全且有信息的 probe，或普通
hold/reobserve 已完全匹配，则立即停止，不进入模型设计。

# BASIN-VLA：Policy-Relative Committor Fields for Action-Chunk Routing

状态：Gate 0 已停止，`STOP_POLICY_RELATIVE_COMMITTOR`

## 1. 从哪个真实断点出发

当前 AlphaBrain / Pi0.5 证据不是“基础策略没有恢复动作”：on-policy correct-mode recall@16 为 96.2%。
真正断点是候选动作的局部合理性不等于它对部署策略的长期可接续性：

- teacher-distance selector 的即时正确率为 99.3%，slip success 却为 0%；
- short-physical Oracle 的即时正确率为 96.3%，slip success 为 61.5%，不优于 random；
- frozen-policy continuation Oracle 的 slip success 为 87.2%，相对 single 提升 15.4 pp，
  paired group 95% CI `[+2.6,+28.2]`，但 wall time 是 single 的 14.3 倍。

这说明候选价值不只由环境几何决定，还可能取决于执行候选后 **实际部署策略是否会继续成功**。

## 2. 核心对象

对冻结策略 `pi`、当前观测历史 `h`、候选 action chunk `a`、先执行的长度 `K` 和预算 `B`，定义
有限预算 policy-relative committor：

```text
q_pi^B(h, a) = P_pi(hit target event before regress within B | do(a[:K]))
```

目标事件不是任意手工 reward，而是有序物理事件的 hitting probability / hazard：stable grasp、lift、
transport、formal success；同时预测 regress/drop。若两个冻结 VLA 在同一个候选 endpoint 上具有不同的
`q_pi`，则“好动作”是 policy-relative 的，单个与 policy 身份无关的 Q critic 会系统性混合不同
competence basins。

工作方法设想为：

1. 用同状态、同候选、不同冻结 policy continuation 得到干净的 policy-relative labels；
2. 从少量 canonical policy responses 构造 policy fingerprint；
3. 学习 `q_phi(h, a, fingerprint(pi))` 的多事件、分布式 committor；
4. 在当前 VLA 的候选池内 listwise 排序，固定 K 执行并重规划；
5. 用 leave-one-policy-out 评估是否能泛化到未参与训练的 checkpoint，而不是记住 seed id。

policy fingerprint、best-of-N、critic 和 committor 各自都不是新颖性主张。潜在贡献必须来自
**VLA action chunk 的 policy-relative causal ranking、跨 policy 泛化，以及相对 generic Q 的闭环收益**。

## 3. 与强近邻的关系

| 近邻 | 已覆盖内容 | BASIN 必须额外证明 |
|---|---|---|
| [VGAS](https://arxiv.org/abs/2602.07399) | 单个适配 VLA 的 Q-Chunk-Former 与 best-of-N | 同一候选的价值随 downstream policy 改变；policy-conditioned critic 优于 generic critic |
| [DreamAvoid](https://arxiv.org/abs/2605.11750) | 预测候选短未来并选高价值动作 | 不生成像素未来；建模实际 frozen-policy competence，而非 policy-agnostic dream value |
| [N2M](https://openreview.net/forum?id=yW9uNbRgOh) | 从 rollout 学习一个 manipulation policy 偏好的 base pose | action-chunk 级、同干预跨 policy committor及未见 policy 泛化 |
| [General Policy Evaluation](https://openreview.net/pdf/ff664abd75ad3bbf4a9d51b9bc4ce4f04e94ad10.pdf) | policy fingerprint 与跨 policy value evaluation | VLA 局部 action intervention、事件 committor和闭环 chunk routing |
| [CORA-VLA](../cora_vla/sequential_oracle_gate.md) | 同策略候选与昂贵 policy continuation Oracle | 不再使用已失败的 teacher/local physical target；先证明 policy relativity 是否存在 |

若 Gate 0 显示三个 Full-H checkpoint 对候选的排序近似一致，则 policy conditioning 没有必要；此时应
采用 VGAS/DreamAvoid 类更简单路线，BASIN-VLA 立即停止。

## 4. Gate 0 结果

21 个 validation on-policy states、七个阶段和三个冻结 Full-H checkpoints 的 same-state、
same-candidate 实验已完成。两组 policy pairs 的 source-mean preference flip 略高于 15%，且
target-policy Oracle 相对 leave-one-policy-out selector 的 percentile gain 为 10.6 pp，95% CI
`[2.9,19.2]`。但所有 policy pairs 的 state-median comparable fraction 均为 0；差异只集中在
`preclose/reapproach` 两阶段和一个 source。它不是广泛的 policy-relative candidate ordering。

因此按结果前冻结门槛停止 policy-conditioned committor，不训练 fingerprint predictor，也不以增加
策略家族或挑选高差异状态挽救假设。完整结果见 `gate0_results.md`。

## 5. 顶会级证据要求

单任务 LIBERO pilot 只能建立机制，不足以投稿。完整工作至少需要：

- 多任务、多个策略家族/训练阶段，而不只是三个 seed；
- same-state same-candidate cross-policy interventions；
- generic Q、policy one-hot、per-policy Q、fingerprint committor、world-model evaluator强基线；
- leave-one-policy、leave-one-task 和分布外 failure state；
- 固定计算预算下的闭环成功率、恢复率、校准、推理成本；
- 真实机器人或第二个高可信仿真环境；
- 明确报告 rollout 数据成本，不能只把昂贵搜索移到训练期后隐去。

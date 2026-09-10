# Branch-VLA：Observation-Contingent Action Chunks

状态：当前 cream-cheese Gate 0 已停止，`STOP_BRANCH_ACTION_CHUNK`

## 1. 问题不是“chunk 太长”，而是输出对象不闭环

标准 VLA 在时刻 `t` 输出一条线性 action chunk：

```text
o_t -> [a_t, a_{t+1}, ..., a_{t+H-1}]
```

这个表示默认未来只有一条需要立即提交的控制路径。接触操作并不满足该假设：在抓取闭合、插入、
释放或受力接触后，环境会产生多个合理物理结果；结果在动作开始前不可知，却可能在 chunk 执行中
由新图像或本体反馈辨认。线性 chunk 只能提前猜一个结果，或缩短执行长度并重新调用昂贵 VLA。

Branch-VLA 暂时把输出对象改成一个短时、观测条件的 policy fragment：

```text
o_t -> shared_prefix + {(guard_m, branch_actions_m)}_{m=1..M}
```

共享前缀只执行不同结果都同意的动作；结果揭示后，一个只读最新部署观测的轻量 guard 选择对应
branch。重型 VLA 不必在每个低层控制步重新运行，branch 也不在结果揭示前被随机选定。

## 2. 最小形式化

深度一的 contingent chunk 写作：

```text
C_t = (A_shared, {(g_m, A_m)}_{m=1..M})
```

其中 `g_m(h_{t+k})` 只能使用截至执行时刻的图像、proprioception 和已执行动作历史。执行器在
`k` 前执行 `A_shared`，在 guard 可辨后执行 `A_m`。训练标签可以由 paired physical outcomes
构造，但 policy 输入不得包含 outcome id、contact oracle、simulator state 或未来图像。

这不是一个普通 mixture policy。mixture 在 `t` 时采样一个 latent mode，仍会在未知结果揭示前
提交；contingent chunk 输出的是“看到哪种结果后做什么”的小型闭环策略。

## 3. 当前证据为什么支持检验这个表示

现有 AlphaBrain 结果给出四个约束：

- attached/slipped 在 feedback 前动作与 conditioning 完全相同，feedback 时动作显著分叉；
- feedback frame 的 vision+state 可线性区分结果，而前一帧为 chance；
- `K=1` 能消除 failure-continuation，但完整恢复仍不足，说明只提高重规划频率不是完整算法；
- FRESH loss weighting、recovery replay、局部 reranking 与 policy-relative critic 均未形成可靠闭环解。

因此 Branch-VLA 的第一问题不是宣称成功，而是检验：同一 pre-feedback 输入能否可学习地输出两条
真实后续，以及 learned post-feedback guard 能否把它们路由得显著优于单条回归 chunk。

## 4. 与强近邻的边界

| 近邻 | 已覆盖内容 | Branch-VLA 必须额外证明 |
|---|---|---|
| [Bidirectional Decoding](https://bid-robot.github.io/) | 每步采样并以 backward/forward coherence 选一条 chunk | 不在结果前选择单一路径；一次生成可执行的 observation-contingent branches |
| [DREAM-Chunk](https://arxiv.org/abs/2606.18589) | world model 预测多个候选未来，并以真实 rollout 匹配候选 | 无测试时 world rollout；branches 在一次 policy generation 中被显式摊销 |
| [DCDP](https://arxiv.org/abs/2603.01953) / [Tube Diffusion Policy](https://arxiv.org/abs/2604.23609) | 在单条 nominal chunk 周围做高速连续修正 | 对离散物理 outcome 使用模式级分支，而不只做局部 residual |
| [AAC](https://arxiv.org/abs/2604.04161) / [VLA-Corrector](https://arxiv.org/abs/2607.01804) | 自适应截断或事件触发重新规划 | guard 路由已生成的 continuation，不靠再次调用重型 VLA 获得反应性 |
| [B2FF](https://arxiv.org/abs/2606.09258) | 从预想未来 milestone 中选择恢复目标 | 输出低层 contingent policy fragment，不在失败后选择视觉 milestone |
| 经典 POMDP policy tree | 用 observation-labeled edges 表示有限时域策略 | 在连续 VLA action chunk、图像 guard 与 counterfactual paired demos 上实现并验证可扩展参数化 |

policy tree、branch classifier、多头网络各自都不是新颖性主张。潜在贡献只能来自：

1. 把 action chunk 从 open-loop trajectory 改写为可摊销的有限时域闭环策略；
2. 从同一前史、不同真实物理结果的数据中学习 branch set 与 observation guards；
3. 在相同 VLA 调用预算下，同时改善随机接触鲁棒性与执行效率；
4. 相对 K=1、动态 K、fast residual、world-model matching 和普通 mixture policy 的闭环优势。

## 5. 证据路线

1. **Gate 0：表示可学性。** 只用 train/val outcome twins，检验 branch-set prediction 与视觉路由。
2. **Gate 1：真实物理上界。** 在完全相同 LIBERO snapshot 上比较 fixed branch、event replan 和
   oracle/learned contingent execution；固定重型 policy-call 预算。
3. **Gate 2：Pi0.5 branch head。** 冻结 VLM，训练最小 set-valued action head 与 guard；不得先加
   world model、RL 或动态 horizon。
4. **Gate 3：跨事件与跨任务。** 至少包含 grasp、release/contact 两类 outcome；单一 cream-cheese
   任务只能证明机制。

任何 Gate 失败都停止该层主张。尤其是，若 learned routed branches 不优于普通 K=1 或 DCDP-style
fast correction，就采用更简单方法。

## 6. Gate 0 结果与边界

真实 paired train/val 结果证明 outcome guard 与多分支风险差异存在：post-feedback guard 为 100%，
pre-feedback 为 50%；learned route 相对单条 linear chunk 的 normalized suffix MSE 降低 58.0%，
source bootstrap 95% CI `[51.4,64.5]`。但是每个 `(branch, lead)` 直接使用 train 均值的常量模板
MSE 为 0.134，而 state-conditioned two-branch predictor 为 0.208，后者反而差 49.6%。

所以当前任务只支持“可观察结果路由两个高度固定脚本”，不支持需要 VLA 表征的 state-conditioned
contingent action generation。该数据不能作为 Branch-VLA 方法验证；不得通过删除常量基线或放宽门槛
继续。若未来重启，必须先有跨 source、跨任务、分支内动作确实随状态变化的数据，并在结果前保留
per-event template baseline。

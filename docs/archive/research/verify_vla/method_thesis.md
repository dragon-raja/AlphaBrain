# VERIFY-VLA：Decision-Value Verification Tails for Chunked VLAs

## 1. 问题定义

令 `h` 为部署时可见历史，`z` 为刚发生但尚未可靠观测的物理结果，例如：

- 物体是否真正稳定夹持；
- 插头是否真正进入接口；
- 抽屉是否解锁；
- 软物体是否被两侧同时抓住；
- 放置是否稳定而非临界滑落。

后续 continuation `c` 的优劣依赖 `z`。如果当前 observation 对不同 `z` 发生 aliasing，直接生成
长 action suffix 会面对不可消除的 Bayes decision error。仅加大模型或重加权 loss 不能预测输入中
不存在的信息。

机器人可以执行一个短验证动作 `u`，得到新观测 `y`。关键是 `u` 不是独立相机动作：微抬、轻退、
小幅 back-drive 或撤手同时改变动力学状态并揭示接触结果，属于 dual control。

## 2. Decision Value of Verification

给定 belief `b(z | h)`、continuation loss `L(c, z)` 和 probe cost/risk `k(u)`，无 probe 的 Bayes risk 为：

```text
R0(h) = min_c E_{z ~ b(z|h)} L(c, z)
```

执行 `u` 并观察 `y` 后：

```text
Ru(h) = E_{y ~ p(y|h,u)} [ min_c E_{z ~ b(z|h,u,y)} L(c,z) ] + k(u)
DVoV(h,u) = R0(h) - Ru(h)
```

它与普通 information gain 不同：

- 只奖励会改变后续最优 continuation 的信息；
- 将 probe 对任务状态的风险和时间成本显式计入；
- 如果所有 latent outcomes 都对应同一 continuation，信息再多也没有决策价值；
- 如果当前帧已经足够辨认结果，额外 probe 的价值应接近零。

在 `k(u)=0` 时，DVoV 非负，因为观察后决策者始终可以忽略 `y` 并复用原 continuation。若 `y` 与
`z` 条件独立，或所有 posterior 下同一 `c` 最优，则 DVoV 为零。加入成本后，只有严格降低预期决策
损失的 probe 才应执行。

## 3. 方法结构假设

当前只预注册结构，不在 Gate 0 前实现大模型：

1. **critical interaction detector**：识别 close/contact/insert/release 等可能产生 latent outcome 的边界；
2. **outcome belief**：由当前多视角历史估计 continuation-relevant latent outcome，而不是重建完整状态；
3. **verification-tail proposer**：生成 2--6 步低幅度动作，可与 task motion 共用机械臂自由度；
4. **decision-value predictor**：预测每个 tail 后的 posterior Bayes risk 与物理风险；
5. **replan**：执行正 DVoV tail，摄取真实反馈，再由原 VLA 生成 continuation；若 DVoV 不为正则直接执行原策略。

训练监督优先来自 matched counterfactual outcome twins：相同前史、相同 probe、不同物理 outcome。
simulator privileged state 只用于构造 `z` 和训练 label，不输入部署 policy。

## 4. 与最近邻的实质边界

- 不同于 DISaM：不假设信息动作维度和 manipulation 动作维度可因子化；同一接触动作同时 act/sense。
- 不同于 AAWR：第一阶段不用通用 privileged critic 学整条 active-perception policy，而先识别短验证动作的
  决策价值；后续必须以 AAWR 为强 baseline。
- 不同于 ViA/next-best-view：不依赖额外可动相机，也不优化几何重建或通用视觉覆盖。
- 不同于 DreamAvoid/PiL-World：不以生成完整像素未来或长闭环 rollout 作为必要运行时组件；world-model
  版本只能作为强 baseline。
- 不同于 AutoHorizon/AAC：改变的不只是“执行几步”，而是动作块末端的功能，从任务 suffix 变为受风险
  约束的结果验证。
- 不同于 failure detector：目标不是报警，而是主动产生能改变下一决策的观测。

## 5. 论文成立所需的完整证据

1. 存在自然或受控接触 outcome aliasing，当前 observation 确实不足；
2. 存在短、低风险、非平凡的 verification tail，优于 hold/reobserve 和 random motion；
3. tail 提高的是 continuation decision accuracy，而不只是 branch classification；
4. 学习模型在 held-out state/task/object 上预测 DVoV，并在没有 privileged input 时工作；
5. 闭环成功、恢复率或不可逆失败显著改善，正常 fully-observed 任务退化不超过 5 pp；
6. 击败 fixed chunk、adaptive chunk、uncertainty trigger、AAWR、world-model candidate planning 和额外视觉帧；
7. 至少第二类接触事件和第二环境复现，不以单一 forced-slip task 支撑论文结论。

## 6. 明确的停止条件

- `hold_closed` 通过自然动力学已经提供同等信息；
- 有信息的 probe 都显著破坏 attached 成功或 detached recovery；
- 当前帧在严谨 source-state-disjoint 测试中本来就能稳定区分 latent outcome；
- branch classification 提升，但 continuation 决策和闭环成功不提升；
- 收益只存在于按 simulator outcome 手调的单任务动作模板；
- 普通短 world model 或 AAWR 在相同交互预算下完全匹配。

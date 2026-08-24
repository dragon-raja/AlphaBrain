# Accel 复现与跨视角选择审计

日期：2026-08-24

## 1. 审计结论

当前实现可以准确称为：

> **原文 Accel 核心几何分数的公式级实现，以及固定状态跨视角选择的扩展实验。**

不能称为：

> **原论文完整实验协议或论文数值的复现。**

原因是原论文验证的是“单个动作 chunk 的 flow-matching 不确定性与在线失败检测”，而本地实验
验证的是“同一物理状态下，能否用最低 Accel 从候选相机中选出行为更好的视角”。后者是新的下游
假设，不在原文主张范围内。

## 2. 原理

对一个 observation 和初始高斯噪声 `x0`，Pi0.5 用 10 次 Euler 更新产生动作。第 `t` 次去噪
输出动作速度 `v_t`。若条件动作后验高度确定，理想 velocity field 接近各向同性仿射收缩，去噪轨迹
近似直线且速度变化小；多模态或 OOD 条件会使轨迹弯曲。

前 `p` 个去噪步的离散分数为：

\[
\mathrm{accel}_p
=
\frac{p\sum_{t=1}^{p-1}\lVert v_t-v_{t-1}\rVert_2}
{\sum_{t=0}^{p-1}\lVert v_t\rVert_2}.
\]

分数越低表示该次生成轨迹越直、模型内部越确定；它并不直接度量任务相关信息、可见性或该视角的
闭环价值。

## 3. 实现一致性

| 项目 | 本地状态 | 判断 |
|---|---|---|
| 10-step Pi0.5 Euler velocity trace | 记录 `[candidate,10,10,7]` | 一致 |
| Accel 离散公式 | 与原文公式逐项一致 | 一致 |
| 单次生成、无需额外训练或重采样 | 从已有 velocity trace 读取 | 一致 |
| 去噪前缀 | 保存 `accel_2...accel_10` | 一致 |
| 候选间共享 `x0` | 同一状态所有视角逐元素相同 | 本地更严格的公平控制 |
| 实际执行动作窗口 | trace 为 10 个执行动作 | 一致 |
| 动作坐标尺度 | 本地 v1 使用归一化动作坐标；官方参考代码还做路径坐标 z-score | 近似但不完全一致 |
| `K=32` 后验重采样相关性 | 未运行 | 未复现 |
| CUSUM + conformal 在线失败检测 | 未运行 | 未复现 |
| 从视角池取最小 Accel | 本地新增 | 原文外扩展 |

因此，代码没有把 Accel 公式写错，但实验任务和完整数值协议与原文不同。特别是 `accel_3` 来自
早期 RoboCasa 对应设置；原文官方 Pi0.5 × LIBERO 结果的最佳相关前缀是 `p=5/10`。本地仍保留
预注册 `p=3` 作为主结果，并把 `p=2...10` 全部作为事后敏感性分析，禁止按闭环结果挑选最优前缀。

## 4. 本地扩展协议

每个状态保持物理状态、语言、机器人状态和推理预算不变，只改变相机观测：

```text
同一状态 s
  -> Canonical / Strong-info / Matched-control / Blind
  -> 所有候选共享同一 x0
  -> 各自运行一次 10-step flow matching
  -> 计算 accel_p
  -> 选择分数最低的候选
  -> 与四视角真实闭环结果比较
```

固定状态 dense bank 有 102 个候选，用于分析训练视角熟悉度和角色排名；闭环结果只覆盖
`Canonical / Strong-info / Matched-control / Blind` 四个角色，因此 selector 的行为裁决严格限制在
这四个视角中。

## 5. 预注册 `accel_3` 结果

下表为 21 个 selected frame states 的描述性结果；正式差值另按 6 条 source demonstrations 等权。

| 模型 | 选择 Canonical | Accel 所选成功率 | Canonical | 任一候选可成功 | 状态级差值 |
|---|---:|---:|---:|---:|---:|
| Broad practical | 18/21 | 57.1% | 57.1% | 85.7% | 0.0pp |
| Broad state-matched | 16/21 | 57.1% | 52.4% | 76.2% | +4.8pp |
| Broad paired FM | 15/21 | 47.6% | 52.4% | 71.4% | -4.8pp |
| Broad paired consistency | 18/21 | 57.1% | 57.1% | 76.2% | 0.0pp |

Source-demonstration 等权差值范围为 `-2.5pp` 到 `+3.3pp`。Broad practical 中，四视角至少一个
能够成功的比例为 85.7%，比 Canonical 的 57.1% 高 28.6pp；但 Accel 所选仍为 57.1%。这说明
候选池存在真实行为上限，而 `accel_3` 没有把该上限转化为选择收益。

## 6. 前缀与坐标尺度敏感性

![Accel 前缀敏感性](figures/accel_prefix_selector_audit.png)

敏感性审计包括：

1. 本地冻结版本：直接在归一化动作 velocity 上计算；
2. 官方参考式版本：重建 denoising path，并在每个固定状态候选 bank 内按路径坐标标准差缩放。

个别模型在部分事后前缀出现正值，例如未缩放的 Broad paired FM 在 `p=6/7` 上为 +14.3pp；
该峰值在官方参考式尺度下缩小为 0/+4.8pp，并且其他模型没有同向复现。由此可见 selector 结果对
前缀和尺度敏感，不能把任何单个事后峰值当作算法收益。

## 7. 最终解释

1. Accel 能有效排斥明显异常输入：blackout/look-away 的平均排名靠后，说明其“不确定性/兼容性”
   含义在本地仍有可见信号。
2. Accel 经常选择 Canonical，因为 Canonical 最接近训练分布，通常形成更直、更熟悉的去噪轨迹。
3. “低不确定性”不等于“高任务信息价值”。新视角可能揭示更多任务实体，但只要它较陌生，Accel
   仍可能高于 Canonical。
4. 当前负结果否定的是 `argmin_v accel(v)` 作为独立 view-value selector，不是否定原论文的
   uncertainty proxy 或 failure detector。
5. Accel 可保留为安全拒绝项或兼容性项；若以后设计主动视角价值，应与任务可见性、信息增量或
   预期行为收益组合，而不能单独最小化 Accel。

## 8. 可复现入口

- 原始 trace：`/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2/`
- M1 join：`/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2/m1-joins/`
- 审计脚本：`scripts/dsol_paper1/audit_accel_view_selector.py`
- 机器可读结果：`docs/dsol_paper1/accel_view_selector_audit.json`
- 敏感性图：`docs/dsol_paper1/figures/accel_prefix_selector_audit.png`

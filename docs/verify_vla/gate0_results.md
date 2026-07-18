# VERIFY-VLA Gate 0 结果

正式裁决：**STOP_VERIFY_VLA**

本轮只使用 `/share/longjunyu/fresh-vla/libero-full-episode-v2-128` 的 train/val groups，未打开
sealed test 或 confirmation episode。完整机器结果位于：

`/share/longjunyu/verify-vla/gate0-probe-v1/gate0_results.json`

## 数据有效性

| Split | 预期组数 | 有效 twin | 无效 twin | 有效率 |
|---|---:|---:|---:|---:|
| train | 102 | 88 | 14 | 86.3% |
| val | 13 | 12 | 1 | 92.3% |

15 个无效组全部因为沿冻结方向在 3 mm 上限内仍无法令 grasp predicate 变为 false。没有放宽网格、
视觉阈值或位移上限。有效样本量通过预注册的 twin-construction 门槛。

## 核心结果

固定 16x16 双视角 block-average ridge（正则 1.0）在 val 上得到：

| 条件 | Vision accuracy | Vision+state accuracy | Pair ranking |
|---|---:|---:|---:|
| pre-probe | 91.7% | 91.7% | 100.0% |
| hold closed | 66.7% | 66.7% | 91.7% |
| micro lift | 62.5% | 58.3% | 100.0% |
| full lift | 66.7% | 70.8% | 100.0% |
| micro lateral | 66.7% | 70.8% | 100.0% |
| release | 75.0% | 70.8% | 91.7% |

pre-probe vision group-bootstrap 95% CI 为 `[79.2%, 100.0%]`，远高于冻结的 65% 上限。也就是说，
最小 1--3 mm pose intervention 虽然双视角平均像素 MAE 不超过 raw RGB 2.0，却产生了稳定、可泛化的
视觉标签泄漏。这个 benchmark 没有构成需要主动物理验证的 latent-contact ambiguity。

所有有效 val endpoint 的 privileged teacher attached completion 和 detached recovery 均为 100%。
`micro_lift` 的合并中位物体位移为 1.10 cm 且没有新增不可逆失败；`full_lift` 为 4.20 cm，超过冻结的
3 cm 安全阈值。probe 可恢复性不是失败原因。

## 科学解释

本结果否定的是当前 **pose-offset latent twin + fixed four-step probe** 构造，不是广义主动感知定理。
由于 outcome 在 probe 前已经可由 RGB 识别，继续调 probe 幅度、时长或网络会把任务设计成支持方法，
而不是验证方法解决真实问题。

随后只在已披露 train calibration group 上检查了同像素 hidden-friction 与 hidden-mass interventions：

- 将 object friction 缩放至 0，四步 hold/micro-lift/full-lift 中仍保持几何 pinching；
- 将质量放大至 100 倍仍保持 grasp，只出现连续的小幅跟踪误差，没有形成必须改变 continuation 的
  成功/失败分支。

因此不继续为 VERIFY-VLA 人为雕刻 LIBERO 奶油奶酪任务。下一研究方向必须直接解释既有最强事实：
局部物理/teacher 标签不能预测闭环结果，而 policy-continuation rollout 可以。

最终裁决：**STOP_VERIFY_VLA**

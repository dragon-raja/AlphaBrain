# 视角价值反向发现 Pilot

## 研究判断

当前应并行推进两条证据链：

1. 构造具有明确 Blind-Reveal 差异、且 Reveal 证据会影响后续任务行为的评测状态；
2. 在大候选池中穷举闭环行为，先测量行为最优视角上限，再反向研究什么观测属性能够预测该上限。

可见像素分数继续作为冻结的主基线。新的可见性定义只能在 discovery source 上提出，并在独立 calibration source 上冻结；最终测试不得继续调定义。

## 已有 A97 反向审计

审计对象为 Broad practical 模型的 15 个 canonical-failure/difficult states，共 7 个任务、11 条 source demonstrations。每个状态穷举 canonical、64 个训练支持视角和 32 个留出视角，共 1,455 个完整闭环 episode。

| 指标 | 结果 |
|---|---:|
| Canonical 闭环成功率 | 6.7% |
| 97 视角中至少一个成功的状态比例 | 80.0% |
| 每个状态平均成功候选占比 | 31.3% |
| 当前可见性 Top-1 成功率 | 33.3% |
| Accel Top-1 成功率 | 33.3% |
| 当前可见性同状态排序 AUC | 0.541 |
| Accel 同状态排序 AUC | 0.478 |
| 留一 source 的全局固定视角成功率 | 46.7% |

该结果说明候选池中存在显著行为上限，但当前可见性和 Accel 几乎不能识别该上限。训练支持视角和留出视角的平均成功候选比例分别为 31.5% 和 31.9%，当前缺口也不能简单解释为 exact training-pose support。

## 不能直接得出的结论

- 单个 state-view 目前只有一次 rollout，成功标签包含策略采样方差；
- 这些 test outcomes 已被查看，只能作为 post-hoc discovery 数据；
- `Oracle@97` 证明“存在可用视角”，不证明该视角因新增任务信息而成功；
- 不能把一次成功的 view 当作唯一最优视角，也不能在这批状态上训练 selector 后汇报最终效果。

## 下一轮执行

1. 优先扩展 top-drawer、wine-rack、bottom-drawer 等已有行为 headroom 的任务族；
2. 同时完成 cream-cheese、drawer-bowl-plate 等构造任务的 Blind、Reveal 和 matched-control；
3. 新建 source-disjoint discovery/calibration/final-test 三组；
4. discovery 目标为 4 个以上任务、48 个状态、每状态 97 个候选的一次初筛；
5. 对 canonical、可见性 Top-1、Accel Top-1、全局固定视角及行为候选组成的 8-view shortlist 使用 3 个共享 policy-noise seeds 重跑；
6. 使用同状态 pairwise ranking 学习 view value，并与可见性、Accel、随机和固定视角比较；
7. 仅在 untouched calibration 上确认稳定后，冻结 selector 并进入一次性 final test。

正式协议见 `configs/dsol_paper1/view_value_reverse_discovery_v1.json`。

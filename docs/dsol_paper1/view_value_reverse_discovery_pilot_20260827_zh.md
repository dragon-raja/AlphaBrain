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

## 构造场景 Dense-View Pilot

在 2 个构造任务、4 条 validation source demonstrations 上各选择两个阶段状态。每个状态使用相同物理快照穷举 97 个 operational views，共完成 776 个完整闭环 episode；无错误，所有同状态记录的物理哈希一致。

| 指标 | 结果 |
|---|---:|
| 状态 / source demonstrations | 8 / 4 |
| Canonical 成功率 | 75.0% |
| Oracle@97 成功率 | 100.0% |
| Oracle 相对 Canonical 上限 | +25.0pp |
| 当前可见性 Top-1 成功率 | 75.0% |
| 当前可见性同状态排序 AUC | 0.489 |
| 最小实体可见性排序 AUC | 0.458 |
| 留一 source 全局固定视角成功率 | 75.0% |

任务分解：

| 任务 | 状态数 | Canonical | Oracle@97 | 平均成功视角占比 |
|---|---:|---:|---:|---:|
| Cream cheese -> bowl | 4 | 100.0% | 100.0% | 96.4% |
| Drawer bowl -> plate | 4 | 50.0% | 100.0% | 48.2% |

Cream-cheese 构造状态没有行为 headroom，不应继续作为信息价值主任务。Drawer-bowl-plate 出现两个 canonical-failure states：一个仅 5/97 个视角成功，另一个 77/97 个视角成功，说明同一任务内同时存在稀疏救援与宽救援状态。

当前可见性没有兑现 Oracle 上限。在 5/97 的稀疏救援状态中，成功视角的实体可见性普遍低于 canonical，可见性排序 AUC 仅 0.284。这表明当前构造仍混合了任务证据、策略视角兼容性与 rollout 随机性；必须先对行为候选做多 seed 重复，再从稳定正例中学习信息定义。

## 三噪声重复性审计

从上述 8 个状态中预先选取四类状态：单次稀疏救援、单次宽救援、换视角伤害和简单视角不敏感。每个状态保留 8 个具有明确角色的候选，并使用 3 个新的独立 policy-flow noise 重跑完整闭环，共 96 episodes。

| 指标 | 结果 |
|---|---:|
| 状态 / 候选 / 新噪声 | 4 / 8 / 3 |
| Canonical 三噪声平均成功率 | 50.0% |
| 可见性 Top-1 三噪声平均成功率 | 50.0% |
| Post-hoc Best-of-8 平均成功率 | 58.3% |
| Canonical 多数失败但 Best-of-8 多数成功 | 0 / 4 |
| 单次成功候选 | 20 / 32 |
| 单次成功且重复后多数成功 | 9 / 20 |
| 单次失败但重复后多数成功 | 3 / 12 |
| 单次成功对稳定成功的阳性预测率 | 45.0% |

![单次行为发现的三噪声重复性](/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-repeatability-v1/analysis/view_repeatability.png)

两个 canonical-failure 状态中没有任何候选达到 2/3 稳定成功。原先 5/97 和 77/97 的一次性救援不能继续作为稳定视角价值证据；单次成功标签受 flow noise 明显影响。后续 97-view 搜索只承担候选发现，所有候选价值结论必须经过独立噪声复验。

## 八任务构造扩展

新的构造扫描覆盖 8 个任务、16 条 validation source demonstrations 和独立 test demonstrations。共扫描 260 个前 65% 轨迹状态；每个状态加入同一静态、无碰撞遮挡构造，并渲染 8 组左右镜像 task-centric pairs。筛选不使用策略输出。

Validation 冻结门要求任务级 Strong-info 中位增量至少 `+0.5pp`，同一镜像 pair 的 Matched-control 绝对变化不超过 `0.5pp`。

| 任务 | Strong 增量 | Control 增量 | Gate |
|---|---:|---:|---|
| Cream cheese -> bowl | +0.56pp | -0.04pp | PASS |
| Bowl -> top drawer | +7.39pp | +0.41pp | PASS |
| Wine bottle -> rack | +1.24pp | -0.12pp | PASS |
| Book -> caddy | +3.38pp | +0.45pp | PASS |
| Bowl -> bottom drawer | +7.32pp | +1.17pp | Control FAIL |
| Mug -> microwave | +10.09pp | +2.33pp | Control FAIL |
| Cream cheese -> basket | +2.76pp | -0.13pp | PASS |
| Drawer bowl -> plate | +0.91pp | -0.30pp | PASS |

6/8 个任务通过严格方向分离门。底层抽屉和微波炉不是没有可见性变化，而是镜像两侧都显著改善，无法用作 Strong-vs-Control 的严格因果对照；它们仍可进入 97-view 行为发现，但不能进入 matched-control 主结论。

下一阶段使用 32 个 validation 状态、16 条独立 source demonstrations 和每状态 97 个 operational views 做一次候选发现。随后每个状态仅保留 8 个代表候选，并以 3 个新 noise 重跑。Test source 保持未触碰，待 view-value 规则在独立 calibration 上冻结后再使用。

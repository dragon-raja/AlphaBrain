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

## 八任务 97 视角闭环发现

完整运行包含 8 个任务、32 个 validation 物理状态、16 条 source demonstrations 和每状态 97 个 operational views，共 `3,104/3,104` 条完整闭环 episode。所有候选共享各自状态的 MuJoCo 初态；候选池由 canonical、64 个训练支持位姿和 32 个留出位姿组成。

| 指标 | 单噪声发现结果 |
|---|---:|
| Canonical 成功率 | 68.75% |
| Oracle@97 成功率 | 93.75% |
| 平均成功候选比例 | 66.59% |
| 可见性 Top-1 成功率 | 71.88% |
| 可见性状态条件 AUC | 0.538 |
| 训练支持位姿平均成功率 | 66.65% |
| 留出位姿平均成功率 | 66.41% |

Oracle@97 相对 canonical 的单次上限为 +25.0pp，但它是检查了闭环结果后的 post-hoc 上限。可见性 Top-1 只比 canonical 高 3.13pp，且排序 AUC 接近随机；单次发现不能据此声明视角选择有效。

## 三噪声全任务复验

每个状态在查看单次发现结果后冻结 8 个具有预定义角色的候选，再使用 3 个新 policy-flow noise 重跑，共 `768/768` 条闭环 episode。稳定成功定义为 3 次中至少成功 2 次。

| 指标 | 三噪声结果 |
|---|---:|
| Canonical 平均成功率 | 69.79% |
| 可见性 Top-1 平均成功率 | 65.63% |
| Post-hoc Best-of-8 平均成功率 | 82.29% |
| 稳定 Rescue 状态 | 5/32，15.63% |
| 可见性直接 Rescue | 2/32，6.25% |
| 可见性 Harm | 4/32，12.50% |
| 单次成功候选的稳定阳性预测率 | 79.67% |

以 16 条 source demonstration 为独立单位进行 10,000 次 paired bootstrap：

| 配对差值 | 点估计 | 95% CI |
|---|---:|---:|
| 可见性 Top-1 - Canonical | -4.17pp | [-10.42, +2.08] |
| Post-hoc Best-of-8 - Canonical | +12.50pp | [+6.25, +18.75] |

![八任务三噪声视角复验](/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/dense-repeatability/analysis/view_repeatability.png)

任务级结果显示，稳定 Rescue 分布在 wine-rack、book-caddy、bottom-drawer、microwave 和 drawer-bowl-plate 五个任务中；当前可见性只在 wine-rack 和 book-caddy 各命中一个，同时在 wine-rack、book-caddy 与 bottom-drawer 产生四个稳定 Harm。Cream-cheese-bowl 与 cream-cheese-basket 均为 100% ceiling 对照，没有 Rescue 空间。

因此，本轮扩展支持两个不同结论：

1. 候选视角包含可重复的行为价值，Best-of-8 的稳定上限并非完全由单次 flow noise 造成；
2. 现有等权实体可见像素分数不是可部署的 view-value selector，平均上反而低于 canonical，且 CI 未排除零。

Best-of-8 使用了复验结果选择候选，只能证明 headroom，不能作为算法结果。下一步应在 validation 的稳定正负候选上审计几何支持、任务阶段、实体关系与 Accel 特征；冻结新打分规则后，只在未触碰 test source 上做一次选择与三噪声闭环验证。

## 冻结选择器独立测试

选择规则在 validation 上冻结后，第一次独立测试使用 8 个任务、24 条未参与规则设计的 source demonstrations 和 48 个状态。每个状态只执行 6 个预先确定的选择器结果，而不是再次用闭环结果挑视角；每个结果使用 5 个独立 policy-flow noise，共完成 `1,440/1,440` 条闭环 episode。

| 选择器 | 五噪声成功率 | 相对 Canonical | Source-level 95% CI | 稳定 Rescue / Harm |
|---|---:|---:|---:|---:|
| Canonical | 66.67% | 0.00pp | [0.00, 0.00] | - |
| Validation 全局固定视角 | 62.08% | -4.58pp | [-9.17, +0.42] | 1 / 4 |
| 可见性增量门控 | 67.92% | +1.25pp | [-2.92, +5.83] | 3 / 2 |
| 实体可见性调和平均 | 66.25% | -0.42pp | [-4.58, +4.17] | 1 / 2 |
| 实体可见性算术平均 | 67.50% | +0.83pp | [-4.17, +5.83] | 3 / 1 |
| 最小实体可见性 | 65.00% | -1.67pp | [-6.25, +3.33] | 1 / 3 |

该批数据中，可见性门控的点估计为正，但区间跨零，尚不能确认收益。五个噪声下相对 Canonical 的差值依次为 `0.00pp`、`+8.33pp`、`+4.17pp`、`-4.17pp`、`-2.08pp`，已经显示明显的噪声敏感性。

## Source-disjoint 来源扩展

为排除第一批 source demonstrations 的偶然性，继续使用同一冻结规则纳入 split 中剩余的 18 条 test source demonstrations。新批次含 7 个任务、36 个状态和 5 个相同独立噪声；选择前仍先对每个状态扫描 97 个视角的静态可见性特征，但未使用任何 test policy outcome。最终完成 `1,080/1,080` 条闭环 episode。

| 选择器 | 五噪声成功率 | 相对 Canonical | Source-level 95% CI | 任务等权差值 | 稳定 Rescue / Harm |
|---|---:|---:|---:|---:|---:|
| Canonical | 83.89% | 0.00pp | [0.00, 0.00] | 0.00pp | - |
| Validation 全局固定视角 | 75.00% | -8.89pp | [-15.56, -2.22] | -9.05pp | 1 / 4 |
| 可见性增量门控 | 72.22% | -11.67pp | [-18.89, -4.44] | -12.14pp | 0 / 5 |
| 实体可见性调和平均 | 74.44% | -9.44pp | [-16.67, -2.78] | -10.05pp | 0 / 4 |
| 实体可见性算术平均 | 71.11% | -12.78pp | [-19.44, -6.11] | -13.86pp | 0 / 5 |
| 最小实体可见性 | 72.22% | -11.67pp | [-18.89, -4.44] | -14.33pp | 1 / 4 |

![新来源冻结选择器复验](/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/independent-source-extension/dense-test-selector-eval/analysis/dense_test_selectors.png)

可见性门控在五个噪声上的差值依次为 `-5.56pp`、`-16.67pp`、`-5.56pp`、`-19.44pp`、`-11.11pp`，五次方向全部为负。任务等权差值同样显著为负，因此该结果不能由新批次任务数量不均单独解释。

## 42 来源合并审计

两批测试合并后共覆盖 8 个任务、42 条独立 source demonstrations、84 个状态、5 个噪声和 `2,520` 条闭环 episode。统计单位是 source demonstration；另行报告每个任务等权、任务内重采样的 bootstrap 结果。

| 选择器 | 五噪声成功率 | 相对 Canonical | Source-level 95% CI | 任务等权差值 | 稳定 Rescue / Harm |
|---|---:|---:|---:|---:|---:|
| Canonical | 74.05% | 0.00pp | [0.00, 0.00] | 0.00pp | - |
| Validation 全局固定视角 | 67.62% | -6.43pp | [-10.48, -2.38] | -5.73pp | 2 / 8 |
| 可见性增量门控 | 69.76% | -4.29pp | [-8.57, 0.00] | -3.60pp | 3 / 7 |
| 实体可见性调和平均 | 69.76% | -4.29pp | [-8.33, -0.24] | -3.76pp | 1 / 6 |
| 实体可见性算术平均 | 69.05% | -5.00pp | [-9.52, -0.48] | -4.45pp | 3 / 6 |
| 最小实体可见性 | 68.10% | -5.95pp | [-10.24, -1.67] | -5.95pp | 2 / 7 |

![42 来源合并选择器审计](/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/independent-source-extension/combined-analysis/dense_test_selectors.png)

合并后，可见性门控在五个噪声上的差值为 `-2.38pp`、`-2.38pp`、`0.00pp`、`-10.71pp`、`-5.95pp`，没有一次为正。原 24 条来源上的弱正点估计没有跨来源复现。

## 任务异质性

下表给出可见性增量门控相对 Canonical 的任务级成功率差值。第一批与新增来源中，同一任务的构造、候选池和规则均未变化。

| 任务 | 原 24 来源子集 | 新 18 来源子集 | 合并 |
|---|---:|---:|---:|
| Cream cheese -> bowl | 0.0pp | 0.0pp | 0.0pp |
| Bowl -> top drawer | -6.7pp | -6.7pp | -6.7pp |
| Wine bottle -> rack | -10.0pp | -5.0pp | -8.0pp |
| Book -> caddy | +10.0pp | -26.7pp | -8.3pp |
| Bowl -> bottom drawer | +6.7pp | -26.7pp | -10.0pp |
| Mug -> microwave | +3.3pp | -20.0pp | -2.5pp |
| Cream cheese -> basket | 0.0pp | 0.0pp | 0.0pp |
| Drawer bowl -> plate | +6.7pp | 未含新来源 | +6.7pp |

Book、bottom-drawer 和 microwave 在相同任务定义下由正转负，说明选择收益依赖具体 source state，而不是稳定的任务级规律。两个 cream-cheese 任务接近 ceiling，只能作为无收益/不应伤害的控制，不能证明主动视角价值。

当前门控选择也高度集中：84 个状态中 29 个保持 canonical，其余状态只使用 7 个不同的非 canonical 位姿，远少于 97-view 候选池。等权实体可见性在实践中接近“按任务选择少数固定大视野位姿”，没有学到状态、阶段和关系条件下的 view value。

## 当前裁决与样本充分性

当前构造规模足以支持以下裁决：

1. 97-view 候选池确实包含可重复的行为 headroom；validation 上 Post-hoc Best-of-8 相对 Canonical 为 `+12.50pp`，95% CI `[+6.25,+18.75]`。
2. 现有等权实体可见像素分数不能可靠识别该 headroom；在来源扩展中它产生 `0` 个稳定 Rescue 和 `5` 个稳定 Harm。
3. 第一批 test 的 `+1.25pp` 是来源和噪声敏感的弱信号，不能作为方法效果。
4. “更多可见像素”与“策略能够利用该观测”是两个独立条件；单纯扩大对象像素可能同时增加相机分布偏移。

当前规模不足以支持以下结论：

- 不能据此否定所有可学习的 view-value selector；
- 不能声称所有任务都没有主动视角空间；
- 不能在已经查看结果的 42 条 test sources 上继续调分数后仍称为独立测试；
- 不能把 Post-hoc Best-of-8 报告为可部署算法。

下一阶段应把现有等权可见性冻结为弱基线，在 discovery/validation 的稳定候选标签上研究 `任务阶段 × 实体关系 × 相机支持距离 × 遮挡/可见性 × Accel` 的组合预测。正例必须在多个 policy-noise 下稳定优于 canonical，负例应包含 matched-control 与稳定 Harm；冻结后需要新增任务或新的 source split 做一次性测试，而不是继续消耗当前 42 条已查看来源。

原始结果入口：

- 第一次独立测试：`/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/dense-test-selector-eval/analysis/analysis.json`
- 新来源复验：`/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/independent-source-extension/dense-test-selector-eval/analysis/analysis.json`
- 42 来源合并统计：`/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/independent-source-extension/combined-analysis/analysis.json`
- 完成回执：`/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1/independent-source-extension/completion.json`

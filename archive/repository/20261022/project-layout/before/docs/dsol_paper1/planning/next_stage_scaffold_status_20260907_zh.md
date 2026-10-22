# 下一阶段：资产归位、脚手架与实验推进清单

日期：2026-09-07。状态：`SCAFFOLD_AND_ENGINEERING_SMOKE_COMPLETE / SCIENTIFIC_RUN_NOT_RELEASED`。

后续执行补记：用户继续授权后，已完成训练身份统一并启动单个匹配canonical补训，见[当前训练执行状态](../execution/training_match_execution_status_20260907_zh.md)。下文记录的是上一批脚手架交付时点；当前主对照选择以Broad64 M-B为锚补同初始化canonical，不再将quick匹配对作为拟主对照。主动科学评测仍未放行。

这是[中心研究计划 v2](paper_master_plan_v2_active_bridge_20260907_zh.md)的实施进展，不替换历史预注册或结果。代码根目录：`/workspace/projects/alphabrain-dsol-paper1`。

本轮已经完成资产审计、三个执行/规划组件及真实 LIBERO 工程检查。没有新增 VLA 训练或任务成功率评测，没有启动 10,752 条开发矩阵，没有改动旧 runner、模型、noise bank、数据集或归档结果。

## 1. 中心不变，但把主张落实为三项检验

> 多视角训练之后，还剩下哪些值得获取的观察差异？能否根据当前合法输入，判断何时保持、何时获取、获取什么，并在新场景中兑现收益？

不是继续证明“多视角一般有用”，也不是把一个有限候选最大值叫作已经实现的主动感知。论文的贡献成立与否，要看下面三项是否得到新的、独立确认的结论。

| 拟贡献 | 需要的主要对比 | 现有证据能支持到哪里 | 下一项缺口 |
|---|---|---|---|
| 训练怎样改变剩余观察价值 | 同资源 Narrow/Broad，各自与自己的冻结固定规则比较，再比较二者差值 | 已有训练改进与强 Broad 的静态搜索结果；尚无完整匹配的训练×获取交互 | 先确定匹配模型身份；不能混用旧 canonical 和 v2 Broad |
| 什么条件使新视角值得获取 | 预定义的等价投影、任务证据揭示、匹配负控制，在相同获取预算下比较 | 可见性代理和旧构造能作资产/失败证据；不能直接当信息性定义 | 审计两路相机、语言和历史中的任务证据；验证任务可解且同一 VLA 能用该证据 |
| 能否形成可执行的使用规则 | 合法输入预测的查询/保持规则，对直接继续、冻结固定查询、随机/几何规则以及门控匹配 hold | 旧 ridge 尚未证明部署收益；新主动规则没有结果 | 独立来源、独立噪声确认成功—成本取舍；不能用拟合数据自验 |

这些是拟贡献，不是已证明的新颖性。近邻及重叠边界沿用中心计划第4节与 v1 的近邻矩阵；本轮没有新增“首次”判断。一次运动学外部获取不等于实际头部动力学或腕部耦合验证；需要那些主张时再做对应桥接。静态搜索收益小，不作为主动分支的否决条件。

## 2. 资产审计的关键修订

完整路径、来源、训练参数及可复用工具见[资产与证据清单](next_stage_asset_evidence_inventory_20260907_zh.md)。

发现不能忽略的训练差异：旧 canonical quick 和 v2 Broad64 M-B 虽然均实际训练 2,000 updates、64,000 model examples，但 LR schedule 分别为 3,000 / 2,000 steps，最终记录 LR 为 `1.6284082688567607e-5` / `5e-6`。GPU 拓扑也分别为8卡/2卡。因此这对模型只能先作为 recipe 比较，不能直接归因于视角覆盖。

当前默认推进方式：

- **工程及初步能力开发优先复用 canonical-unique quick 与 Broad32-practical quick。**保存配置除训练臂/输出身份外一致；源容器、seed、预算、学习率计划一致。不因本轮审计就立即重训全套模型。它们仍需权重内容身份和实际接口哨兵。
- **v2 Broad64 M-B 和原始 O/P/Q 完整保留**，作为历史强模型证据。其行为记录不能贴到 Broad32 quick 或 Broad64 quick 身上。
- 正式训练交互若坚持以 v2 强 Broad 为锚，须审计并补一个必要的匹配 canonical 处理，再单独冻结训练预算；本轮没有启动。若正式改用 quick 匹配对，则旧 v2 O 不能抵扣这两个新模型的静态评测。

预算修订：v2 中 S 新增约28,928条的草案依赖“原 Broad O 可复用且有匹配 Narrow”。现在不能直接采用这个数。若使用 quick 匹配对且两侧 O 都须新测，原32来源、97候选、8噪声的草案变为49,664条 O，再加最多2,560条 P 与1,536条 Q，共53,760条开发评测。**这个完整矩阵本轮不启动**；先完成匹配身份和小规模执行/能力审计，比较必要补训与新评测的真实开销后再冻结路线。

旧 v2 的242,816条、22份 PASS audit 保留。主模型 canonical/task-fixed/state-search 为59.08/60.45/62.99%；state-search 相对 task-fixed 的事后区间仍跨0。不能宣布“没有任何好视角”，也不能宣布“状态选择已经有效”。所有看过的旧 test 此后均属于历史材料。

## 3. 已搭好的脚手架

| 新入口（均在当前代码根） | 已实现 | 尚不承担 |
|---|---|---|
| `scripts/dsol_paper1/view_acquisition_protocol.py` | 严格 JSON、未解决项登记、匹配训练参数、来源分组/泄漏检查、独立噪声声明、动作/时间预算、开发量核算；正式预检失败即拒绝 | 不验证声明的文件哈希是否真实，不自动授权或启动任务 |
| `configs/dsol_paper1/view_acquisition_a1_draft_v1.json` | 继续、匹配 hold、4次候选移动；2训练处理×3证据块；来源与噪声分组草案 | 具体 checkpoint、任务证据、路径、阈值和来源尚未冻结 |
| `scripts/dsol_paper1/view_acquisition_motion.py` | 五次平滑位置参考轨迹、最短弧旋转插值、速度/加速度与世界坐标 AABB 检查；环境步/观察/策略调用账本 | 不认证相机实体碰撞、IK、颈部/腕部机构或连续执行器动力学 |
| `scripts/dsol_paper1/view_acquisition_executor.py` | 实际推进仿真时间，保持 OSC/夹爪执行器目标，逐步验证实际相机位姿与新标定；只向调用方交付端点观察；失败留账 | 不是完整 rollout runner；尚未连接 VLA/selector、全历史恢复、输入白名单及整个任务的总账本 |
| `scripts/dsol_paper1/run_view_acquisition_engineering_smoke.py` | 一个历史开发状态的 continue/hold/hold/move 对照；私有 LIBERO 配置；新目录落盘、拒绝覆盖；零 VLA 调用 | 不是科学成功率评测，也不能解除所有正式执行门 |

`queries` 已统一定义为**交付一次 external+wrist 观察包**，不是单张图像；成本项使用 `observation_queries`。仿真内部沿途渲染不提供给模型，但它仍可能消耗计算。本账本不声称测量全部渲染耗时、传感器能耗或单张图像数；正式性能记录须分别记账。

no-acquisition 在执行层保持零环境步、零新增观察、零策略调用，并返回原观察对象。这个局部等价性不等于已经验证与旧 VLA 全 rollout 行为等价。

## 4. 已实际完成的验证

### 单元与回归

Python 3.12：新增组件与相关旧评测/审计回归合计 **65 tests、52 subtests 通过**。

```bash
cd /workspace/projects/alphabrain-dsol-paper1
/alphabrain/.venv/bin/python -m pytest -q \
  tests/dsol_paper1/test_view_acquisition_protocol.py \
  tests/dsol_paper1/test_view_acquisition_motion.py \
  tests/dsol_paper1/test_view_acquisition_executor.py \
  tests/dsol_paper1/test_libero_eval_sensor_controls.py \
  tests/dsol_paper1/test_statewise_view_oracle_v2_stage.py \
  tests/dsol_paper1/test_statewise_view_oracle_v2_audit.py \
  tests/dsol_paper1/test_statewise_view_oracle_v2_analysis.py
```

实际仿真环境 Python 3.8：三个新增测试文件共 **53 tests 通过**。

```bash
/workspace/envs/fresh-libero/bin/python -m unittest discover \
  -s tests/dsol_paper1 -p 'test_view_acquisition_*.py'
```

### 真实 LIBERO 工程检查

首次输出保留在：

`/share/longjunyu/alphabrain/experiments/dsol-view-acquisition-scaffold-20260907-LZuOl3/report.json`

首次状态为 `FAIL_ENGINEERING_ONLY`。夹爪零输入使 `current_action` 从初始化的一维广播成二维，导致严格相等检查失败；更重要的是，零归一化指令不保证维持进入时的 actuator ctrl。没有靠放松判断把它改成 PASS。

修复后改为在每个物理 substep 显式保持进入时夹爪 actuator ctrl，并保持原 `current_action` 不变，退出恢复临时局部接口。新增记录保留初末控制值、手指关节位置和位移。

复跑输出：

`/share/longjunyu/alphabrain/experiments/dsol-view-acquisition-scaffold-20260907-fixed-pxff3O/report.json`

结果为 **`PASS_ENGINEERING_ONLY`，16项检查全部通过**：

- 一次实际相机平移5厘米；hold/move 各推进20个控制步，即1秒仿真时间，continue为0。
- 两次 hold 与 move 的最终物理状态 hash 完全相同；移动确实改变外部图像和标定。
- continue 保持物理状态和图像；所有条件 VLA 调用为0；相机获取结束只交付一个观察包。
- 夹爪指令状态及 actuator ctrl 均保持不变，局部控制接口正常恢复。
- 持位条件的 EEF 末端位移约0.420毫米，手指最大末态位移约0.385毫米。**这些是一个来源上的观测，不是已经满足正式漂移门槛。**

限制：HDF5 没有完整恢复原示范的 controller/夹爪控制历史。本次机械臂参考为恢复后的当前 EEF 姿态，夹爪参考为进入时 sim.ctrl；后者不是已认证的原示范控制目标。相机为按控制步采样的运动学外部相机，不是真实头部动力学。没有测对象漂移阈值、任务能力、视角净收益或完整模型随机流等价性。旧 v2 没有这段新增持位阶段，本次修复不追溯改变它。

## 5. 当前协议状态与实验量

草案机器检查：`draft_schema_valid=true`、`release_valid=false`，84项发布检查信息；实际来源0，确认/校准来源尚未配置。

草案规范化 SHA256：`4f870c1e0309674b4210e203b6b5237e6f07cae70e2a2fb08f6c011ab6cf0121`。

可运行下列命令查看详细计划；第二条应退出2，表明目前禁止按正式协议放行。两个命令都不启动实验。

```bash
/workspace/envs/fresh-libero/bin/python scripts/dsol_paper1/view_acquisition_protocol.py \
  configs/dsol_paper1/view_acquisition_a1_draft_v1.json
/workspace/envs/fresh-libero/bin/python scripts/dsol_paper1/view_acquisition_protocol.py \
  configs/dsol_paper1/view_acquisition_a1_draft_v1.json --require-release
```

A1 开发预算：16基础来源×2分支×3证据块×2模型×(6动作+1直接揭示控制)×8噪声 = **10,752条**，其中动作银行9,216、正控制1,536。独立基础来源仍只有16，不是10,752，也不是32或96。当前只是算术草案，未实际构造16来源；只配置了 O 开发分组，声明 Q 银行不等于已经设计并启动独立确认。

## 6. 接下来的有界执行顺序

| 顺序 | 具体工作与产物 | 进入下一步的条件 |
|---|---|---|
| P0a 输入/恢复 | 显式 policy 与 selector 白名单；完整 snapshot 包含控制器/夹爪/缓存、action queue、模型/selector 历史与 RNG；整段共享账本；源文件哈希回执 | 重复恢复、无获取旧行为等价、非请求图像不泄漏、控制频率/耗时和基策略噪声索引审计通过 |
| P0b 训练身份 | 先完成 quick 匹配候选的权重内容和接口核验；审定是否为 v2 Broad 补必要匹配 canonical | 明确训练对比是 coverage 还是 recipe；模型/数据/预算实际匹配，不仅是布尔声明 |
| E0 任务证据与可解性 | 先选少量开发模板，构造三类证据关系；审计所有合法输入和后续 wrist 补偿；验证物理可解、直接揭示时同一 VLA 能完成 | 条件由任务/观察干预预先定义，不由最终收益挑选；若基策略不会新规则，先解决共同能力支持 |
| A1 小型执行/能力试验 | 在明确的少量开发来源上检查 continue、hold、4移动、正控制；记录对象/EEF/夹爪漂移、失败分母、实际吞吐 | 有效性与成本口径通过后，才冻结10,752条开发银行；不因静态 S 为零而跳过 A1 |
| 规则开发与新来源确认 | 开发内按基础来源分组拟合；冻结固定、随机/几何、预测查询/不查询及门控hold；新来源与新噪声检验 | 正式主张看来源外收益及区间，不看开发 Oracle 最大值；确认预算依来源方差和预定实际效应冻结 |

P0a 的三个硬边界尚未解决，不能将本次工程 PASS 填成所有 gate=true：

1. executor 还不绑定完整冻结协议，也不管理触发前与恢复后 VLA 的全部步数和随机流。
2. 当前 backend 的原始观察字典可能包含对象低维信息；连接模型时必须构造白名单，不能直接透传。
3. 完整恢复和持位/对象漂移尚未通过多来源、带操作历史的哨兵验证。

成本先报成功、实际环境时间、观察包、策略调用与失败，净值权重仅在开发/校准期固定。不能用这次无 VLA 的渲染速度估计正式8卡吞吐或承诺完成日期。正式 runner 接好后再做少量真实并发吞吐验证，给出吞吐与规模依据的时间估算。

本阶段不扩大背景研究、不新增复杂时序记忆算法、不铺满全部训练臂×所有相机本体。若条件审计失败，先修构造/接口；若有效协议下结果仍不确定，就按预定预算报告不确定性。只有新的可确认条件关系与可兑现规则成立，才把对应项提升为论文贡献。

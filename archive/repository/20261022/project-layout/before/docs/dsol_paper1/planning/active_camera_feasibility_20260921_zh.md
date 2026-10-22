# 主动相机本体可行性：核验结果与下一步决策

2026-09-21。用户授权的并行工程核验；不改变P0任务、相机、模型、噪声、渲染和分片。未调用VLA、未启动图像渲染、未新增任务闭环。CPU加载MuJoCo资产，包含一次独立机器人1500步/3秒原生电机检查。不是正式主动感知实验。

## 结论

1. 当前PandaOmron的400个冻结初态，相机祖先关节均为躯干升降与底盘平移/偏航，没有独立头部关节。通过移动底盘/躯干可以改变世界观察位置，但同时改变操作本体状态，不是只调相机。
2. 原候选并非全部物理可安装：10个预选seen初态中，CoffeeSetupMug有6组、OpenCabinet有20组候选的相机中心进入碰撞盒内部；规范安装在这10例中未标记。不能把这一小样本比例推广到400初态，也不能说其余候选已通过完整物理验收。
3. 本机已有robosuite TIAGo资产，原生头部关节、相机、执行器和头部控制配置均存在。49点正运动学和3秒原生电机测试通过，无需靠改cam_pos来伪造可动相机。
4. TIAGo不是当前RoboCasa π0.5的直接替换件：相机输入、本体状态、双臂动作及任务初始化需要适配，操作策略未验证。当前证据支持将其列为优先原生本体候选，不支持立即迁移整个benchmark或开始重训。

![工程核验图](/workspace/research-reviews/active-camera-feasibility-20260921/verified/feasibility.png)

## 已完成的检查

### P0相机运动链

对正式release关联的400份compiled XML进行哈希核对，逐一提取左外部相机的祖先关节；左右相机共用同一支撑体已由前次97候选几何核查确认。

```text
底盘：前后平移 / 左右平移 / 偏航
  └─ 躯干升降（0–0.34m）
      ├─ 左外部相机（安装固定）
      ├─ 右外部相机（安装固定）
      └─ 机械臂安装
```

当前VLA动作转换输出12维语义：末端平移3、旋转3、夹爪1、base_motion4、control_mode1；没有独立相机动作。不能把base_motion当作只动相机的等价操作。

### 相机安装初筛：不是完整碰撞认证

预先取每任务最低seed的seen初态，共10例；不读成功率或Accel筛选案例。校验原始快照与候选缓存，恢复模型运行时几何、qpos/qvel和mocap，CPU forward求当前几何世界位姿。相机是零体积中心点，检查是否严格进入启用碰撞的box/sphere/capsule/cylinder/ellipsoid或mesh凸包内部；边界余量1e-5m。没有仿真推进、没有移动物体、没有改变相机。

| 预选任务初态 | 标记候选组 / 97 | 训练目录 / 留出目录 |
| --- | ---: | ---: |
| CoffeeSetupMug | 6 | 5 / 1 |
| OpenCabinet | 20 | 15 / 5 |
| 其余8个任务各1初态 | 0 | 0 / 0 |

两个独立可核对的box例子：

- CoffeeSetupMug `train_048`右相机：在柜门碰撞盒局部坐标为`[0.091183,-0.009850,-0.040436]m`，盒半尺寸`[0.132027,0.021822,0.346592]m`，三轴绝对值均严格小于半尺寸。
- OpenCabinet `train_015`左相机：柜体后板局部坐标`[0.091583,0.009288,-0.263506]m`，半尺寸`[0.371942,0.015000,0.430000]m`，同样位于内部。

因此至少这些是明确的碰撞模型冲突，而非仅凭视觉怀疑。完整flags包括几何体名称、相机世界/局部坐标和尺寸；没有删除候选、改变标签或重跑正式实验。检查不包含传感器实体尺寸、支架、布线、扫掠路径、整个rollout动态碰撞。即使所有中心点未进入实体，也仍不能签发安装/运动可行证书。

## TIAGo原生相机验证

使用本机原始`robots/tiago/robot.xml`，无资产编辑。相机`robotview`安装在`head_2_link`；两个头部关节分别有原生motor。注意电机名称中的pitch/yaw与实际轴顺序不宜望文生义，应依据关节轴和父变换。

| 检查 | 结果 |
| --- | --- |
| 机械关节范围 | 第一关节约±75°；第二关节约−60°至45° |
| 当前头部控制配置 | 配置含output_min/max ±0.5，但absolute分支直接使用目标关节角；不能把此字段当作有效机械限幅，适配器须显式验证目标与路径 |
| 49组关节采样 | 两关节各7点，均在±0.5rad内 |
| 相机最大位移 / 光轴变化（相对零位） | 0.04919m / 39.63° |
| 非头部body位置与旋转变化（FK） | 0 |
| cam_pos/cam_quat安装数组 | 完全不改 |
| 原生电机测试 | 1500步×0.002s=3s，目标[0.3,−0.25]rad |
| 最终头部关节最大误差 | 0.0001201rad以内 |
| 其他关节最大漂移 | 3.08×10⁻⁷以内（各关节自身单位） |

电机检查使用诊断PD（kp100、kd20）及bias补偿，对原生执行器限幅；不是在每步直接改qpos。该独立模型检查未使用厨房场景、图像与VLA；模型零位有1个接触，终点仍为1个，所以绝不标记为“无碰撞任务认证”。厨房原生控制器的后续检查见下节。

### 2026-09-22：RoboCasa厨房原生控制接口检查

仅CPU、无渲染、无VLA；没有修改P0或安装的RoboCasa/robosuite源码。使用原生Tiago控制配置，在layout 23、style 54进行有界检查。

| 任务 / seed | 原生设置结果 | 解释 |
| --- | --- | --- |
| OpenDrawer / 0 | 两次独立运行均完成reset与40个控制步 | 2秒物理时间，头部目标[0.4,−0.3]rad，终点[0.39988,−0.29991]rad，光轴改变28.38°，相机安装数组不变 |
| OpenDrawer / 1 | reset失败 | `FactorizeHessian: rank-deficient sparse Hessian` |
| TurnOnMicrowave / 0 | reset失败 | 同类求解错误；qpos/qvel有限不代表约束系统有效 |

另做TurnOnMicrowave实例内dense Jacobian诊断，未在总计180秒运行预算内完成；保留超时记录，不视为修复成功。当前证据不够区分碰撞/约束配置、初始放置与求解器的具体责任，需要隔离诊断，不能靠换seed避开。

同日继续对上述三例进行原生设置的只读接触诊断：OpenDrawer seed 0再次通过，reset后24个接触，最深约6.4毫米（夹爪内部），无超过1厘米穿透；OpenDrawer seed 1失败时477个接触，其中245个超过1厘米，最深约21.75厘米，为柜门与机器人底盘；TurnOnMicrowave seed 0失败时516个接触，其中264个超过1厘米，最深约21.57厘米，同为柜门与机器人底盘。计数是接触点数，不是独立碰撞物体数。

后续有界预筛试验：只在独立诊断进程内替换出生位置函数，保持原生采样/扩张规则，用运动学与几何碰撞查询先于完整约束求解；每例最多500个候选。三例均避免了原先的Hessian报错，但均因500次未找到合格位置而失败，包含此前能原生重置成功的seed 0，因此此实现不合格、不可推广。末例最终位置无超过1厘米穿透，仍有轮子与地板约1e−13m的接触，被原生“机器人与外部物体任意接触”规则拒绝。合法支撑接触与禁止碰撞分类需要进一步验证，不能通过任意加穿透容差来放行；也不能说所有500个拒绝均由地板造成。旧失败与成功记录不覆盖。[预筛失败记录](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-placement-prescreen-20260922/results.json)。未修改已安装依赖或P0。

源码显示出生位置采样后先调用完整`sim.forward()`、再用`detect_robot_collision()`拒绝碰撞位置；原生失败发生在前者，未进入拒绝该位置的检查。这提供了具体的适配方向：在完整动力学约束求解前做几何碰撞预筛，保留正常碰撞物理与原生求解器。上述有界原型尚未通过，不宣称已修复；也不能据单次接触证据排除其他约束问题。[接触诊断完整记录](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-contact-diagnostic-20260922/results.json)。接触摘要与预筛预算共三项纯CPU单测通过，三项仓库静态检查通过。

原生厨房控制动作20维，head占[13:15]；`robot0_robotview`确实挂在`robot0_head_2_link`。左右agentview仍挂在`mobilebase0_support`，不会因为打开head控制就变成头部图像。因此后续必须显式接入头部观测通道，当前P0输入与12维动作适配器不可直接复用。

#### 限定支撑接触后的诊断结果

新增显式支撑分类：只允许原生轮/脚轮几何与命名room floor之间、法线近竖直（abs(n_z)≥0.999）、穿透不超过1微米的接触。此阈值仅针对出生几何查询中约1e−13m的轮地数值接触，不用于豁免柜体碰撞，也不用于修改仿真求解。机器人与外部的其余接触均拒绝；规则单测覆盖深穿透、柜体、底盘、倾斜法线和NaN不放行。

在相同三个预定案例上，OpenDrawer seed 0、seed 1及TurnOnMicrowave seed 0均通过重置和40步/2秒原生头部控制，光轴变化分别28.38°、28.50°、28.37°，相机安装数组未变。完整forward后再次核查禁止接触，原生求解器/接触物理保持不变。只在独立进程内替换出生函数并在结束时恢复；未修改P0或安装依赖。

这验证了已复现的出生问题有可行适配方向，不是全面修复验收：接受位置经过重新筛选，不能称为与原协议初态逐值相同；允许规则目前依赖TIAGo及厨房命名，尚未扩展多布局，也未验证任务目标是否在臂可达范围、整条运动的碰撞、图像或操作策略。历史失败仍保留。[支撑预筛原始结果](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-support-prescreen-20260922/results.json)、[输入与实现哈希](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-support-prescreen-20260922/provenance.json)。本轮4项相关纯CPU测试与三项仓库静态检查通过。

#### 5秒本体保持诊断

上述三例继续以100控制步/5秒运行：明确给torso绝对控制器传入当前关节位置，双臂保持原生零增量、底盘零速度，不锁定物理世界。三例均正常完成，左右末端最大平移漂移均小于0.264毫米，躯干最大漂移小于0.117毫米，头部到达目标，安装数组不变。这是定量控制诊断，不是任务成功或安全通过证书。

原出生支撑分类直接应用于动态阶段时，三例分别有4、0、97个控制步被标记为存在未豁免外部接触。保存的首批例子是脚轮与地板约1e−14m的接触，说明还需要核对动态接触法线与分类；不能把全部标记解释为危险碰撞，也不能忽略它们直接报告全程无碰撞。当前日志只留首批接触例子，后续应保存每类接触及法线后判定。[原始保持诊断](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-hold-diagnostic-20260922/results.json)。本轮未启动渲染、VLA或抓取执行，任务可达性仍待检查。

这只是环境和头部动作接口烟雾测试，不代表抽屉任务成功、机械臂严格保持、图像质量、路径无碰撞或策略可用。未驱动的其他动作分量由原生create_action_vector填零，不能把填零等同于已验证的保持控制。

下一步先隔离重置失败、实现显式非头部保持与限位检查；通过后再做小规模头部图像检查。仍不启动TIAGo训练，不占P0 GPU，不把成功案例升级为稳定benchmark支持。

- [首次检查（含失败）](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-integration-20260922/results.json)
- [超时说明](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-integration-20260922/execution_status.json)
- [独立复核与相机父节点](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-integration-confirmation-20260922/results.json)
- [复核配置和源码溯源](/workspace/research-reviews/active-camera-feasibility-20260921/kitchen-integration-confirmation-20260922/provenance.json)

入口仍为`active_camera_feasibility.py`，使用`--kitchen-integration --execute`及独立配置；新增实现位于`benchmarks/Robocasa365/tiago_integration.py`。所有检查结果独立归档，未改变正式矩阵。

这也显示空间不等价：该头部在本次采样中相机中心只移动约5cm，主要改变朝向，而旧RoboCasa候选非规范点平移0.11–0.72m。不能原样套用97点结果，也不能把机械头部强行移动到旧候选。

## 平台选择

| 方案 | 已核实的可用性 | 适配代价与研究边界 | 当前建议 |
| --- | --- | --- | --- |
| 原PandaOmron直接“主动调相机” | 没有独立相机关节 | 改安装并不是动作；底盘运动又耦合操作 | 不作为已实现主动本体 |
| 保留PandaOmron，增加有实体/关节的相机机构 | 可设计，尚未装配验收 | 操作模型与场景复用潜力最高；必须公开为自建扩展，不能叫原生配置；0.7m双目架不能未经检查直接移植到头部 | 原生本体适配过重时的备选 |
| 现有robosuite TIAGo | 原生头部＋相机＋电机，CPU核验通过 | 仍需任务、观测与操作策略适配；本机RoboCasa相机配置默认指向mobilebase0_support，未见TIAGo专用覆盖 | 优先做小型集成门槛验证，不大迁移 |
| ManiSkill Fetch | 官方实现有head_pan/head_tilt、头部相机与body控制器 | 新仿真/任务/动作与模型适配；尚未本机运行 | 第二候选 |
| SaPaVe ActiveManip-Bench | 项目页明确有Isaac Sim主动头部任务 | 这次检索项目页仅见论文/视频，未核实可安装代码、资产与预训练权重；不能把论文存在当成可复现环境已就绪 | 保留，获取入口核实后再决定 |
| ActPerMoMa TIAGo++ | 公开gym式环境与代码 | 以主动移动抓取为主，README测试Isaac Sim 2023.1.0，依赖/任务差异较大 | 方法与可达性参考，暂不优先迁移 |

一手来源（2026-09-21访问）：[robosuite TIAGo模型](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/robots/tiago/robot.xml)、[头部控制配置](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/robosuite/controllers/config/robots/default_tiago.json)、[ManiSkill Fetch实现](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/robots/fetch/fetch.py)、[SaPaVe项目](https://lmzpai.github.io/SaPaVe/)、[ActPerMoMa代码与依赖](https://github.com/pearl-robot-lab/ActPerMoMa)。当前推荐是基于这些工程条件的判断，不是性能排名。

## 明确的下一阶段门槛

### ActPerMoMa配套任务作为补充验证（待决策，未启动）

年份与适用性：ActPerMoMa为ICRA 2024工作，使用早期YCB物体模型构造随机抓取场景；YCB物体集（2015）不是一份现成的VLA示范轨迹数据集，不能等同于我们需要的TIAGo后训练数据。TIAGo平台历史较长并不让可达性、碰撞与主动观察实验失效；具体仿真模型/控制器版本仍须冻结。主要局限是桌面移动抓取任务范围、现有抓取执行器与当前VLA不同，以及旧Isaac Sim依赖迁移成本，不是年份本身。可作为真实获取视角的机制验证，不能替代LIBERO/RoboCasa多任务、语言条件与训练覆盖结论，成功率也不能跨任务体系直接横比。

ActPerMoMa是方法论文，其开源Isaac Sim TIAGo++移动抓取环境可作为复用的评测任务；是否是大型通用benchmark不是科学有效性的必要条件。可以测试同一任务初态/观察扰动下，真实相机运动后是否改善闭环抓取。需保留作者任务与成功判据，明确我们新增的扰动、预算和方法接口，不能把改动后的设置冒充作者原始成绩。

两条实施路径必须区分：复用作者VGN/规划执行系统，只替换观察策略，可验证视角搜索的跨系统可迁移性，但原生VGN不提供π0.5流匹配Accel，不能宣称直接验证VLA-Accel恢复；在该环境接入一个已具备TIAGo抓取能力的VLA，再使用Accel与当前搜索模块，才能直接验证VLA视角恢复。后者的控制与策略适配成本尚未估计，不预先安排新训练。搜索框架可替换评分器不代表指标可不经验证任意替换。

预设比较建议为：正常观察基线、扰动后不恢复、同移动/查询预算随机、我们的恢复、原生ActPerMoMa观察规划。明确最后一组可能具有底盘与头部联合行动权限；若我们的首版仅动头部，需单独列出不同权限，不能作为等权限优劣排名。仅比较原生方法与新方法，不足以证明“恢复”：需先确认扰动导致下降，并测试是否缩小与正常基线的差距。

观察历史/TSDF初始化规则、RGBD与标定权限、时间/距离/碰撞、执行器与随机性必须一致；搜索中不倒回物理时间，不额外给方法提供未查询视角真值。相机位姿变化与错误外参标定是不同扰动，首版选择前者并给所有方法正确标定。新增场景与物体留出用于方法冻结后的确认。实验优先作为补充，不替代LIBERO/RoboCasa训练覆盖与视角恢复主线。[论文与任务说明](https://arxiv.org/html/2310.00433v2)、[官方环境](https://github.com/pearl-robot-lab/ActPerMoMa)。

不是继续把所有事情无限并行。建议依次过以下门槛，每次通过才扩大。

1. **原生本体集成门槛**：TIAGo接入1–2个静态任务场景；确认实际head action驱动图像变化、机械臂保持控制、时间推进、限位与碰撞记录。需要少量渲染但不用VLA，可等P0空出资源后做；不擅自占当前GPU。
2. **操作策略门槛**：确认可用示范/预训练操作策略、相机输入通道、本体状态和动作语义。先做标准头位的任务基线；不能将PandaOmron checkpoint直接包装成TIAGo已可用策略。若必须另建大量示范与训练，先停止扩大，比较Panda相机扩展的工程成本，由用户决定。
3. **主动获取门槛**：在关节空间生成候选，从实际到达后的图像评分；搜索期间保持机械臂但不冻结世界，计入仿真时长、路径长度、查询次数和墙钟延迟。不得搜索完恢复物理初态再把终点成绩称为真实获取收益。
4. **小型方法比较**：先看任务开始前的相机调整，完成后再操作；不做中途频繁切换。基线为原位、同预算随机头部移动、指标搜索、已知home标定时的回零，并加入匹配等待时长的控制。关节范围、任务/初态、评分及执行噪声在跑之前冻结。

框架接口复用现有`ObservationOption → AcquisitionReceipt → AcquiredObservation`与`EpisodeCursor`，不再复制第二套框架。适配器需提供：本体ID/版本、当前关节及标定、目标关节、限位与路径检查、实际执行动作和起止tick、到达误差/碰撞状态、真实新观测、评分模型和噪声地址。预测器只能提议可达区域/目标；真实执行由本体控制器完成，首版不改VLA动作头。

## 当前P0结束后的分析口径准备

- 原400×97×2×4矩阵保持不变，执行身份与完整性验收先于成功率汇总；8次评分与4次执行明确分离。
- 两模型、seen/heldout、10任务分别汇总；保持每任务每初态等权，使用配对差值与分层区间，不能将候选×噪声全当独立场景。
- 原位扰动、预先固定的Accel/组合规则、同预算随机、规范几何参考分开报告；完整查询与有限查询权限分开。
- 本次发现的物理问题不能用成功率决定取舍。下一步先将同一定义的几何审查扩到全部初态，形成不使用成败的标记；全候选作为视觉敏感性分析保留，物理标记后的结果作为另一个明确总体，未经用户确认不替换主报告数字。
- “保持原位的起点是否可行”与“选出的终点是否可行”分别约束，搜索路径仍需另验证。若终点改了或有初态无合格候选，明确报告，不能默默删除。
- 每初态噪声均值后取最大值只称经验候选上限；4次执行不支持精细稳定排名，方法最终确认另用独立执行噪声，不能宣称本次几何初筛解决了Oracle偏差。

## 文件与复现

验收：新增3项纯CPU单元测试通过，完整CPU测试790项通过、182个subtests通过；仓库、架构与目录静态检查通过。MuJoCo检查为显式独立诊断，不包含在默认测试内；未修改P0的冻结执行源码。TIAGo通过的是本体级检查，厨房图像、标准任务操作基线和主动搜索闭环仍未验收。

- [完整核验记录及碰撞点坐标](/workspace/research-reviews/active-camera-feasibility-20260921/verified/audit.json)
- [输入资产、快照及源码哈希](/workspace/research-reviews/active-camera-feasibility-20260921/verified/provenance.json)
- [CPU检查实现](../../../benchmarks/Robocasa365/active_camera_audit.py)
- [纯几何与路径原语](../../../AlphaBrain/research/dsol/acquisition/kinematic_audit.py)

首次初筛保留在同一审查目录根部；详细坐标与电机验证保留在`verified`，不是两个竞争的论文结果。当前PDF未修改。执行命令：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /workspace/projects/openpi-robocasa/.venv/bin/python scripts/dsol_paper1/diagnostics/active_camera_feasibility.py --config configs/dsol_paper1/analysis/active_camera_feasibility.json --execute
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/dsol-review-mpl /alphabrain/.venv/bin/python -m reports.paper1.active_camera_feasibility --config configs/dsol_paper1/analysis/active_camera_feasibility.json
```

诊断输出要求新目录；复跑须在配置中显式指定新目录，不覆盖已有核验记录。模型编译需要能只读访问原场景网格，沙箱权限不足不意味着资产缺失。所有新接口默认导入不启动仿真。

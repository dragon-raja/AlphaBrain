# Paper 1：VLA 的视角利用

2026-10-21 执行更新：[原生 SigLIP 真实准入及批量诊断](execution/rl_search_attention_20261020_zh.md)。
批量原因诊断通过后，用户批准采用batch1新参考及6,208条重算。新参考准入66条已通过，
累计准入102/114；两噪声评分与原生SigLIP重算进行中。混合预算训练和真实仿真接口已准备，
尚无本轮新训练或完整任务成绩，不混用旧 PPO 结果。

2026-10-20 后续：[查询成本、策略原生 SigLIP 与共享 attention 设计](planning/rl_search_attention_20261020_zh.md)。
已准备无额外视觉前向的捕获组件、候选共享 actor 和隔离特权 critic，完成 CPU 组件测试；
旧数据噪声/几何审计不产生新模型调用。混合预算训练、实际策略准入与头部查询尚未运行。

2026-10-20：已完成[RL 搜索接口准入](planning/rl_search_interface_20261020_zh.md)及
[独立 CPU 环境下 PPO 第一轮训练与比较](execution/rl_search_20261020_zh.md)。
复用 INITIAL32 缓存，累计 100,352 个搜索训练决策，新增 VLA 推理/仿真执行为 0。
主预算 8：PPO 37.82%、局部 Accel 36.90%，差值区间跨零；少查询但几何移动量更大。
外部相机与头部适配分别验证，旧队列、历史结果及汇报 PDF 未修改。

本仓库当前主线：**比较规范视角与多视角后训练后，模型在固定候选相机空间中的闭环表现，以及指标能否利用其中的选择收益。**

2026-10-12 新授权补测：[LIBERO五窗口、官方Plus相机子集及反馈结构审计](execution/libero_plus_followup_20261012_zh.md)。保持原8任务×4初态、两份模型；Plus每格4次新闭环须先通过测速准入。渲染在Fresh8槽运行，LIBERO完整初态短探测准入在GNU运行；不表示完整闭环矩阵已经启动，不替换下方已完成的RoboCasa结论。

2026-10-12 最终独立确认：两模型各80个新初态、97候选短探测与8次独立执行已验收，合计44,376次完整任务；620,800个Accel评分从原速度序列重算逐值一致，窗口1与独立初态评分一致。主窗口3预先冻结：π0.5总体40.98%→42.34%（+1.36 pp，95%区间[-3.89,+6.56]），XR-1为51.64%→50.63%（−1.02 pp，[-9.59,+6.48]），没有确认跨模型稳定恢复。全部五窗口、seen/heldout、逐任务和冻结开发分组均保留，不按结果改选主窗口、删任务或自动加预算。唯一[研究进展PDF](reports/current/vla_view_research_progress_zh.pdf)更新为20页：前18页正文与图表逐页像素保持，末页展望替换为独立确认的两页结果。详见[最终验收记录](execution/robocasa_recovery_validation_20260929_zh.md#2026-10-12-新初态独立确认收尾)。这条进度替代下方10月10日“运行中”的历史截点。

当前实验从官方任务初态执行完整任务；LIBERO外部相机每次执行中世界位姿固定，RoboCasa外部相机安装偏置固定、随支撑体自然运动。不是中间状态恢复实验，也不是任务途中主动调整相机安装。候选图可查询的指标分析不能写成单实体相机无需移动就能获得所有视角。

2026-09-23：RoboCasa P0 的32个分片、310,400次完整任务已完成，逐文件身份、覆盖、噪声地址和全部Accel前缀重算验收通过。两模型均为300任务匹配后训练的8250步最终权重，评价取10任务、400初态、97安装、每格4次执行，另8次初态评分。当前报告合并LIBERO关键证据与RoboCasa结果，仍只维护下面一个PDF入口。原生渲染微差、每格4次的经验Oracle偏差与未验证实际相机获取均明确保留，不用负结果反向改变协议。

## 从这里开始

2026-10-10 16:08 完成时间均衡：两模型短探测均已完成；本轮独立完整任务确认现为 Fresh 的π0.5/XR-1各4卡、Video π0.5 8卡、GNU XR-1 8卡，每卡2 worker，48个槽实际运行。只迁移未启动块：174个XR-1块转Fresh、100个π0.5块转Video；全量1,451块对照确认科学任务、地址、输出不变，总预算仍44,376条。Fresh XR-1先通过双worker固定输入逐值重放，再以原预算首块32条完整任务验收，未新增重复科学试验。当前控制目录和后续估时见共享`CURRENT_EXECUTION.json`；下面较早的整机模型分工已被本条替代。

2026-10-10 调度工具更新：标准队列现支持三机所有权/闲置巡检、按物理GPU排空/恢复、只读结果抽查与独立1/2/3 worker容量测试。π0.5纯推理1/2/3并发的固定输入动作与velocity逐值一致；真实短探测双worker吞吐为单worker约1.94倍，三worker仅剩463MiB显存，触发安全停止，未获生产准入。Video测试卡已自动恢复原双worker队列。操作、原始证据及边界统一见[执行管理协议](protocols/execution_management_zh.md)，实时状态仍以共享`CURRENT_EXECUTION.json`为准。

2026-10-10 12时提速更新：Video不再空等，已分担约半数剩余official短探测；Fresh与Video各16槽（每卡2 worker），GNU完整评测也扩为16槽。只替换协调器并接管原运行句柄，未向GPU worker发停止信号；未启动任务互斥划分，原科学地址与输出位置保持不变。现行控制目录统一查共享实验根`CURRENT_EXECUTION.json`，旧控制目录有retired标记，不可重启。后续official完整评测仍自动衔接，改为两机各16槽；下方“Video等待”是更早截点。见[提速与实际验收](execution/short_view_probe_20261008_zh.md#2026-10-10-短探测分流与双worker提速)。

2026-10-10 Video接入：用户授权停止Video原Qwen服务和网页应用，保留保活、文件和恢复命令。新环境已复制并通过8评分噪声的固定输入推理逐值核验；官方π0.5下一阶段改为Fresh/Video互斥初态分区，GNU的XR-1正式队列与Fresh当前短探测保持不动。待全部80初态选位冻结后按原生时限×视角数均衡分配，两台各自16条准入计入原预算。Video目前是已接入的等待执行节点，不能写成三机正式闭环已同时开跑。见[三机接入记录](execution/short_view_probe_20261008_zh.md#2026-10-10-video接入与三机分工)。

2026-10-10后续：GNU完整任务准入16条通过，已进入722块、8卡正式确认；本机官方π0.5短探测继续，独立official-only自动衔接已启动等待。15日双机本地盘将清空，共享盘保留；现已开始共享盘首轮备份，目标14日完成本轮确认和收尾。不能把此目标写成已经保证的工期，也不包含尚未批准全量GPU规模的LIBERO/Plus。见[运行记录](execution/short_view_probe_20261008_zh.md#2026-10-10-gnu独立进入完整任务阶段)及[关机迁移与恢复规范](protocols/execution_management_zh.md#2026-10-10-部分闭环恢复与15日清盘准备)。

2026-10-10北京时间10:28：按用户要求，GNU XR-1不再等待本机π0.5。全80初态选位审计完成，去重后21,960条独立噪声完整任务确认；已启动持久tmux托管的16条完整轨迹准入，通过后自动接8卡标准动态队列，准入结果计入总量。此时仍在实际准入，不能写成全量已经铺开。原双模型CPU钩子只准备计划，后续不得重复执行XR-1分区。见[GNU独立进入完整任务阶段](execution/short_view_probe_20261008_zh.md#2026-10-10-gnu独立进入完整任务阶段)。

2026-10-10北京时间10:17中断核查：GNU XR-1短探测319块全部完成（62,080条）；本机官方π0.5控制器已不在，已启动的57块全部正常完成，但262块未分发，实际只完成11,376条（18.32%）。旧队列账本仍写RUNNING，CPU衔接也停止刷新；不能把历史RUNNING当作仍在运行。完整闭环尚未启动。已保留中断账本，按标准队列显式核验恢复，并改为独立tmux会话托管；详见[中断与恢复记录](execution/short_view_probe_20261008_zh.md#2026-10-10-控制器中断与恢复)。

2026-10-09 CPU准备继续推进：LIBERO原32初态×97候选共3,104组图像/内外参、两模型49,664个逐噪声评分、原执行噪声库核验通过，重建256份首次评分噪声。Plus原8任务的341条相机定义已生成1,364条明确的wrapper/物理初态请求。输出统一位于规划根`libero-plus-preflight`，不是新增GPU评测结果。窗口1实际推理复现、Plus原生渲染和运行准入仍待GPU验证，见[CPU输入准备](execution/short_view_probe_20261008_zh.md#liberoplus-cpu输入准备2026-10-09)。

2026-10-09北京时间19:13推进：双机各8卡短探测健康运行，官方π0.5约11.2%、XR-1约13.2%，尚无完整任务确认成绩。已有7/8个初态分别完成全97候选选位校验与冻结；CPU阶段衔接已启动，待全体评分完成后自动生成去重闭环计划。完整执行worker及标准队列适配已通过CPU检查，仍待真实GPU准入，不能宣称闭环已自动启动。见[选位冻结与确认衔接](execution/short_view_probe_20261008_zh.md#选位冻结与确认衔接2026-10-09)。

2026-10-09最新执行截点：新80初态及80×97候选缓存全部通过；本机官方π0.5与GNU XR-1的新初态两候选准入均通过。两台机器各8卡的正式五窗口短探测队列已运行，并已产生分支进度；完整任务8噪声确认阶段尚未启动。LIBERO／Plus已完成旧8任务的分层统计与相机—初态映射，未据此删任务或新增GPU实验。见[统一执行记录](execution/short_view_probe_20261008_zh.md#后续推进候选缓存与新总体探测接口)。

2026-10-09：用户批准有界独立确认，保留10任务、增加80个新初态，两模型、97候选短探测、8套新执行噪声；主窗口固定为3，完整任务上限48,640条，不因结果不显著自动扩量。新初态采集已启动，正式策略队列仍须通过新总体的渲染、评分锚点与双机准入，不能将准备阶段写成闭环已开跑。LIBERO／Plus同时进行CPU身份与相机定义审计，不自动新增其GPU实验。见[五窗口记录中的新确认阶段](execution/short_view_probe_20261008_zh.md#2026-10-09-有界独立确认启动准备)。

2026-10-08后续授权：改为与原8评分严格匹配的五窗口短探测，纳入XR-1与官方π0.5，双机分别采集。窗口1须精确复现旧方法；新增短轨迹、不重跑完整任务矩阵。双机小验收通过，正式协议与调度入口见[五窗口执行记录](execution/short_view_probe_20261008_zh.md)。下条“不新增闭环”是本日较早决定，现已由本次有限短探测授权更新；旧1,072条完整任务确认草案仍未启动。

2026-10-08：已完成XR-1前1/3/5次实际重规划Accel的跨执行噪声离线诊断。Heldout对应55.42%/59.58%/55.42%，不调整54.22%；没有稳定的时序增益证据，不能把候选执行分支当作初态免费输入。80单元/320条原始执行抽查通过。用户明确本轮**不新增闭环**；1,072条去重补测清单仅为未启动草案，LIBERO-Plus Camera也未启动。旧数据及PDF未改，详见[最新恢复验证记录](execution/robocasa_recovery_validation_20260929_zh.md)。

2026-09-30静态规则诊断：评分稳定性、均值/中位数/标准误/几何邻域以及seen监督校准已完成，只读复用原数据，新增闭环0次、PDF未改。XR-1“几何+Accel”heldout为70%，但seen固定先验已有67.5%，两者差异尚无可靠证据；π0.5上未显示相同组合收益。详见[恢复验证记录的静态决策诊断](execution/robocasa_recovery_validation_20260929_zh.md#初态静态决策规则诊断2026-09-30)，不可把该探索线索当作已确认方法。

2026-09-30：XR-1同40初态×97候选全矩阵已完成，绘图尾项已在独立CPU环境补齐；唯一PDF更新为19页，保留原1–17页。规范66.88%、扰动不调整53.52%、最低Accel48.13%，未显示恢复。seen预选固定位置的探索性heldout对照为67.50%，不能把指标失败等同于没有恢复空间。下一步及边界见恢复验证执行记录。

2026-09-29报告更新：统一PDF为18页，保留LIBERO及RoboCasa完整矩阵主体，补入官方模型40初态全候选诊断、seen/heldout恢复差异、低值区域与任务归因线索。完整heldout200初态的规范模型恢复仍为+3.24pp；不与诊断子集混报。本次仅更新报告，未启动实验，也未将新决策规则写成已验证方法。

| 要找什么 | 唯一入口 |
| --- | --- |
| 并行任务调度、恢复与后端接入标准 | [执行管理协议](protocols/execution_management_zh.md)：通用队列、安全CLI与CPU CI；活跃冻结评测不热切换，旧后端接入另行验收 |
| 当前推进：评分范围核验与更强RoboCasa模型 | [2026-09-29恢复验证执行记录](execution/robocasa_recovery_validation_20260929_zh.md)：官方40初态离线评分核验已完成；更强模型资源/工程准入，不替换旧矩阵与PDF |
| 全项目真实路径、OpenPI后端、官方RoboTwin版本差异与开源边界 | [完整资源地图](protocols/project_inventory_zh.md)；包含LIBERO/Plus/RoboCasa原数据、派生数据、权重、外部框架及本机路径绑定 |
| 跨数据集、权重、实验与源码的版本对应 | [统一资源管理](protocols/resource_management_zh.md) · [RoboTwin工程试点](execution/robotwin_resource_pilot_20260929_zh.md) · [前置资源审计](audits/robotwin_resource_admission_20260928_zh.md)；原始、派生、实验和代码分离，不移动旧证据 |
| 当前完整实验定义与执行协议 | [标准初态 AA0 记录](execution/standard_initial_aa0_execution_20260908_zh.md) |
| 当前汇报材料 | [研究进展 PDF](reports/current/vla_view_research_progress_zh.pdf) · [图表说明](reports/current/vla_view_research_progress_zh.md) |
| RoboCasa P0验收、选择器与结果分析 | [统一报告的详细说明](reports/current/vla_view_research_progress_zh.md)；原始审计、逐任务表、空间切片和查询预算图均由此进入 |
| LIBERO / RoboCasa 差异与随机性审查 | [噪声降采样、协议差异、几何分层及80次固定噪声小验证](results/libero_robocasa_attribution_20260923_zh.md)（不修改原矩阵，不自动扩大评测） |
| 待审查：真实相机几何与可见性分层 | [LIBERO与RoboCasa几何、可见性和覆盖范围统一审查](results/geometry_visibility_review_20260920_zh.md)（2026-09-20；未改PDF） |
| 主动相机本体选择 | [候选几何冲突、TIAGo原生关节/电机核验与平台选择门槛](planning/active_camera_feasibility_20260921_zh.md)（2026-09-21；CPU工程验证，无新增VLA闭环） |
| Accel 空间结构与模板恢复核查 | [模板闭环结果、覆盖曲线、形状分解与三维切片](results/accel_field_20260915_zh.md)（2026-09-15，复用归档，未替换 PDF） |
| 局部几何梯度恢复原型 | [五张过程／方向／噪声图、真实 Pi0.5 验证与指标改良分析](protocols/local_view_derivative_20260916_zh.md)（2026-09-16，机制诊断；尚非闭环恢复算法） |
| 当前推进：条件指标场与 RoboCasa | [第一轮学习结果、现有权重、显存验证与两阶段计划](protocols/conditional_field_robocasa_20260916_zh.md)（2026-09-16，已获授权推进；不重训全量Human300） |
| 下一阶段方法：结果驱动的模型辅助主动观察学习 | [方法归类、预测/价值/策略融合、CPU 原型与后续验收](planning/outcome_active_observation_20260918_zh.md)（2026-09-18，CPU 契约原型已实现；尚无学习模型/动态实测，不替换当前实验） |
| 双周汇报交付 | [2026-09-09 双周报源文件](../../reports/paper1/historical/biweekly_20260909.py) |
| 代码用途与入口 | [脚本导航](../../scripts/dsol_paper1/README.md) |
| 开发规则与模块边界 | [CONTRIBUTING](../../CONTRIBUTING.md) · [可复用研究包](../../AlphaBrain/research/dsol/README.md) |
| 仓库迁移后的真实闭环验收 | [运行一致性记录](audits/repository_runtime_equivalence_20260911_zh.md) |
| ACD / basin / branch 等旁支 | [代码归档与恢复映射](../../archive/research/README.md) |
| 共享服务与相机工具 | [VLA 共享模块](../../scripts/vla_shared/README.md) |
| 训练匹配与证据边界 | [训练匹配审计](audits/training_match_anchor_audit_20260907_zh.md) |
| 过去的计划、结果、版本 | [历史项目索引](HISTORY.md) |
| 不属于当前论文的旁支文档 | [历史研究档案](../archive/research/README.md) |
| 已退休的操作脚本与路径映射 | [操作归档](../../archive/paper1/README.md) |

历史文档中的“当前”“运行中”和预计用时只代表文档当时的截点。最新实际进度读取运行目录的状态和完成收据；已有 PDF 不因代码整理自动成为新矩阵结果。

2026-09-11 更新：198,656 条正式记录已完成并完成本轮全量身份／覆盖核对；当前唯一 PDF 已加入新初态统计、空间图、预定规则与两类探索性离线选择器。选择器只用初态 0、1 开发，在 2、3 评价；部分测试汇总此前已查看，不声称完全盲测确认。本次新增闭环 0 次、VLA 训练 0 步。统计与选择器产物入口为 `/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908/analysis/research-v1/latest.json`。

## 当前实验边界

2026-09-14 更新：第六页四图保留独立自适应色条及原始数值；第九页以受扰动后保持原位为主基线，第十一页加入去掉规范图特权的新选择器与有限查询图搜索，第十二页区分已完成离线重放和未完成实际获取。仍为同一份12页 PDF，原版按哈希归档，冻结闭环矩阵不变。Accel 是同一初态8评分噪声均值，不是执行全程均值。新分析的目录导航与静态缓存假设不等于真机验证，详见[视角恢复协议](protocols/view_recovery_v1_zh.md)。

- 两个匹配后训练模型；8 个任务、每任务 4 个官方初态。
- 每初态 97 个固定候选位姿、32 套显式 Flow 噪声序列，共 198,656 次整任务闭环。
- 按重规划索引配对噪声，去噪网格、时限和 AA0 渲染协议固定。
- 同一矩阵比较规范视角、全局/每任务经验最佳固定视角、每初态经验 Oracle；先对噪声平均，再选视角。
- Oracle 是有限样本事后参照，不等于可部署选择器。选择规则使用初态留出，且要说明候选图查询权限。
- 冻结矩阵不扩写、不混入历史恢复快照。2026-09-16已另行授权条件场核验与RoboCasa资源/pilot准备；新任务范围和执行链独立登记，不隐式改变本矩阵。

具体定义、划分和限制以当前执行协议及冻结 release 为准；本 README 不另立实验口径。

## 运行与源码分离

正式运行目录：

```text
/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908/
  release.json                         冻结科学协议
  repo/                                冻结执行代码，不修改
  scheduling/dynamic-v1/               单独验收过的调度扩展
  hosts/fresh/status.json               fresh 当前状态
  hosts/gnu/status.json                 gnu 当前状态
```

当前工作仓库 `/workspace/projects/alphabrain-dsol-paper1` 用于后续开发；整理它不会热更新上面的冻结副本。不要在本仓库直接重跑建库、迁移或调度命令来“检查状态”。

2026-09-11 运行迁移验收：修复 Python 3.8 提前导入及共享工具源码绑定问题后，192 次独立诊断闭环、96 对新旧轨迹全部精确一致并匹配历史验收结果。它们不计入上面的 198,656 条科学结果，PDF 保持不变。覆盖范围及完整来源见 [验收记录](audits/repository_runtime_equivalence_20260911_zh.md)。

## 开发检查

2026-09-23 P0 后新增 [归因验证协议](protocols/robocasa_attribution_v1.md)。用户随后将范围收缩为简单验证：大规模补充调度已取消，不得自动恢复；此前已启动的304条工程pilot保留。当前只做[离线跨benchmark分析与80次固定噪声复跑](results/libero_robocasa_attribution_20260923_zh.md)。旧矩阵、当前 PDF 和已发布结论不在诊断启动时改写。

在具备本项目依赖的 Python 环境中，从仓库根目录运行：

```bash
python tools/paper1/check_layout.py
python tools/paper1/check_architecture.py
python tools/paper1/check_layout.py --list
python tools/paper1/test.py
```

本机可使用 `/alphabrain/.venv/bin/python` 替代 `python`。测试入口会关闭 GPU 可见性并限制 CPU 数值库线程；部分测试只读已有冻结 release，不代替真实闭环验收。默认不会安装依赖、训练模型或启动正式评测。

## 保存约定

- 代码、测试、配置、文字说明和来源收据进入 Git。
- 新生成的报告 PDF、DOCX、预览图和字体副本留在本地/共享盘，由限定目录的忽略规则管理；原文件不删除，已跟踪历史证据不取消跟踪。
- 报告源文件仍在 Git，但 PDF 重建还依赖外部图表与数据，不声称仅凭克隆仓库即可完整重建。
- 冻结协议、权重、原始结果和判废记录保持原路径及来源链，不在本轮清理。
- 95 份旁支文档已转入 `docs/archive/research/`；论文引用的两份 KYC 证据和原图/历史配置保留原位。
- 当前共享服务、观测和相机实现位于 `scripts/vla_shared/`。旧 CABI 四个同名文件仅作兼容入口；绑定旧冻结 release 的驱动只在哈希核验后使用其中的原副本。
- 2026-09-11：新初态数据核查、指标规则、选择器与统计已抽入 `AlphaBrain/research/dsol/`；报告布局移到 `reports/paper1/`。旧入口保留兼容，科学结果不因目录调整重写。
- 2026-09-11 整理：原 186 个平铺脚本已分两轮全部按职责迁出（149 + 37），85 个平铺测试也已模块化。第一轮见 [迁移清单](../../archive/paper1/maintenance/script-layout-20260911/manifest.json)，最终路径见 [运行模块迁移](../../archive/repository/20260911/runtime-modules/manifest.json) 与 [测试导航](../../tests/dsol_paper1/README.md)。旧控制进程已获准结束，原源码和冻结实验身份保留；没有因整理重新评测或替换 PDF。
- 8 个退出主线的旁支共 208 个文件已原字节归档到 `archive/research/code/scripts/`；数据、权重、冻结运行目录不动。FRESH 中仍被历史 CABI 引用的两项工具保留共享实现。
- 21 个混放测试已统一到 `tests/dsol_paper1`；12 个已核对的一次性操作脚本转为历史文本归档。其余脚本不能据此视为已完成模块化重构。

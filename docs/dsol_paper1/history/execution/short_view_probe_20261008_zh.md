# 五窗口短探测：与原初态评分严格匹配

用户于2026-10-08授权开始五窗口实验，并加入π0.5、使用双机。不是执行途中切换相机；候选探测后恢复原初态选位。此次加入官方π0.5，不自动扩展两份后训练模型。

## 正式启动截点

### 2026-10-09 修复与补跑

XR-1已完成40/40初态块、3,880候选单元、31,040条短探测。官方π0.5首轮保留3,353个完整单元，剩527个单元。旧队列退出后已核实39个旧进程组均无存活成员，不接触保活进程。

停机原因不是Flow噪声不一致：`TurnOnMicrowave`初态`67ad4142fb3c0383556c7d01`、`train_001`、评分重复2，在动作步99出现`success=true`，但`terminated=false`且`truncated=false`。旧运行器把任务成功也作为短探测提前停止条件，导致缺少步100的第5次评分。

经用户授权修复，本次明确将短探测定义为固定五次重规划：成功标志只记录，不决定提前停止、评分或选位；继续真实仿真以取得评分，不补零、不外推。实际`terminated/truncated`仍保留partial并停止。后续评分可能包括任务完成后的观察，分析须记录并披露，不能解读为全部发生于任务完成前。

修复验收使用同一失败初态的规范和`train_001`，共16条分支；再次复现重复2在第99步成功，随后正常取得步100评分。16条首次velocity、动作、Accel及两格窗口1均值均与旧记录精确一致。11项短探测测试、缺项拆分/进程/队列等共45项通过，portable99项通过。

全部3,353格旧完整结果已验证payload哈希、8×5完整评分、记录覆盖和原窗口1均值；旧失败文件与原源码不变。确定性缺项清单包含527格、4,216条分支，分成27个最多24格的块，本机8卡标准队列动态领取。修复不改变原40初态、97候选、8评分噪声、4独立执行噪声或评分公式。GNU上已完成的XR-1不重跑。

- 新运行源码：共享根下`repair-source-20261009`，哈希清单`repair-source-20261009-manifest.json`。
- 修复验收：`admission/pi05-repair-20261009`。
- 补跑结果：`official-repair-20261009`；`reuse.json`固定可复用证据，`ownership.json`为每一原始格指定唯一来源，不能按结果挑选来源。
- 补跑队列：`/workspace/research-reviews/short-view-probe-20261008/queue-official-repair-20261009`。
- 运行配置：`configs/dsol_paper1/short_view_probe_repair_fresh_20261009.json`。
- 统一统计：`analysis/short_view_windows.py`，配置`configs/dsol_paper1/analysis/short_view_windows_20261009.json`。按原独立完整任务矩阵统计窗口1–5，seen/heldout分开，窗口1逐格分数和最终选位必须精确复现。

准备队列时曾因配置以相对路径调用，产物清单校验拒绝准入；尚未启动GPU任务。已统一解析为绝对路径并保留`manifest.preparation-failed.json`。正式准入清单通过后才启动，不影响科学协议或已有结果。

以下为旧启动记录；其中“早成功即停止”已由上面的显式修订替代，不回写旧协议。

2026-10-08：两份正式标准队列已启动。本机官方π0.5与GNU XR-1均为8个RUNNING任务块、32个PENDING任务块，worker已产生短探测进度，不只是队列准备。此处是启动截点，不代表全量完成或窗口1全总体统计验收通过。后续读取上述本机/远端队列与共享worker状态；不存在自动重跑失败项或放宽门槛。

冻结正式源码371个文件，短探测运行器与通过v2验收的字节相同。新进程/数值与原队列测试35项通过，仓库portable99项通过，架构/布局/仓库检查通过。默认沙箱无法读真实/proc时两项进程测试显式跳过，主机权限下已实测通过；不以跳过冒充验收。

## 冻结科学条件

- 原10任务、40初态（seen20/heldout20）、97候选，两模型分别处理，全部初态保留。
- 每候选8条短探测，首次使用原评分噪声、输入、模型和指标；原始速度轨迹逐值核对，π0.5还核对原动作。每格窗口1均值须精确等于原均值。
- 后续重规划复用原显式评分噪声地址规则的replan1–4，从未使用最终execution噪声流；跨视角配对，不依赖线程或调用顺序。
- 同一组轨迹计算窗口1/2/3/4/5，分数为八次重复及窗口的算术均值。全部窗口并列报告，禁止按heldout挑一个赢家作主结果。
- XR-1采用原生16动作块/4帧历史/5步Flow，Accel仍取16×12与去噪前缀3；π0.5采用25动作块/原生状态和图像/10步Flow，Accel仍取50×32与前缀3。窗口长度指**重规划数**，不能与Flow前缀混淆。
- XR-1评分时刻0/16/32/48/64动作步，π0.5为0/25/50/75/100；不执行最后一块动作。跨模型不是等物理时长因果比较。
- 每模型31,040条短分支，最多155,200次带记录推理；两模型62,080条，不新增完整任务成功率评测。
- 选择终点后复用旧97候选的4次独立完整执行。旧成败不进入评分、窗口选择或超参数调整。它是已有矩阵上的方法比较，不是新噪声/新场景盲测。
- 早成功、terminated、truncated、退化分数：保留partial证据并停止准入/队列；不填零、不丢初态、不利用成功标签给候选打分。若出现，需要另行确定科学处理规则，不能为完成矩阵静默修改。

## 小验收与调度

验收为原目录第一初态`702bbff034638941f7c7e07d`，规范与预定`extra_002`两候选，各8条，不按结果挑样本。新仿真实测与CPU进程管理测试均为运行准入，不混入最终研究总体。

XR-1本机/GNU两处各16条短分支通过；官方π0.5本机16条通过。每条第一次velocity/Accel精确等于旧数据。首版在均值聚合时发现NumPy二维axis=0求和与旧连续一维求和的末位差异，已恢复旧求和顺序，新版通过；旧失败目录及源码保留。

热身后每候选八分支：XR-1本机35.45秒/GNU36.22秒；π0.5本机44.70秒。仅一个任务的小测速，推算双机各8卡、两模型并行约6–10小时，正式并发吞吐与任务间差异可能改变估计。不能直接用短探测步数比例承诺完整工期。

生产调度使用既有`execution`持久队列，新增`benchmarks/Robocasa365/probe_queue.py`后端。每初态97候选为一个块，每模型40块；卡完成块后领取下一个，不固定每卡五块。GNU只跑XR-1，本机只跑官方π0.5。资源分开，但任务/噪声地址不含主机信息。

进程记录包含PID、启动时钟、boot ID、独立进程组；仅清理本轮拥有的进程组。存在未核实的launch gap时禁止自动重发。GPU准入只允许空闲卡或已识别的保活进程，发现其他GPU进程则等待，不结束其他任务。默认沙箱不暴露真实/proc，因此真实进程测试在主机权限下单独执行：新增7项与原队列28项，共35项通过。

小验收为有限、明确的调度兼容例外：每host/model最多一个直接启动worker、两候选，不作为正式矩阵分片。正式工作必须由标准队列启动。

## 资产与复现入口

- 代码：`benchmarks/Robocasa365/short_view_probe.py`；队列准备：`AlphaBrain/research/dsol/data/short_view_program.py`。
- 配置：`configs/dsol_paper1/short_view_probe_20261008.json`及fresh/gnu运行配置。
- 共享实验根：`/share/personal/longjunyu/robocasa/experiments/short-view-probe-20261008`。
- 冻结源码：`formal-source`；首轮验收：`admission-source`，修复后验收：`admission-source-v2`。
- 正式结果：根下`official`和`xr1`，包括协议、manifest、逐初态/候选分数、原始velocity及首次一致性回执。
- 本机队列：`/workspace/research-reviews/short-view-probe-20261008/queue-official`；GNU本地队列：`/workspace/dsol-short-view-probe-20261008/queue-xr1`。队列控制文件使用各主机本地磁盘，不依赖共享盘flock。

正式统计必须等待覆盖审计与窗口1全体选位复现通过：XR-1 heldout48.75%，官方π0.5 heldout33.75%。不能将两模型的基线相混。随后统一生成成功率—窗口曲线、相对窗口1配对增益、逐任务结果、排名关系、选位变化与实测成本；当前PDF及旧结果不覆盖。

## 2026-10-09 完成及空间分析

两模型均已完成各3,880格、31,040条短探测，窗口1全体分数与选位精确复现。π0.5补跑27块全部完成；统一统计在北京时间11:59:20完成。统计根为`/workspace/research-reviews/short-view-probe-20261008/matched-analysis-20261009`。其后仅做CPU空间分析，没有新增闭环或训练。

## 2026-10-09 有界独立确认启动准备

上述旧40初态的探测已结束。用户现批准新一轮独立确认，不能把下述准备状态混入旧结果。

- 原10任务×2官方环境划分×4新初态，共80个物理起点，由官方π0.5和XR-1共享；不按成功率替换任务或初态。
- 97视角×8评分噪声×5次重规划，共124,160条短分支。窗口3为冻结主比较，窗口1/2/4/5均保留但不在确认结果上挑赢家。
- 每模型/初态完整评测规范位、全部原32个扰动位及最多5个窗口赢家，使用8套新执行噪声；相同视角合并，最多48,640条完整任务。不是重跑全97视角结局，也不生成新Oracle上限。
- 固定预算后报告正、负或不确定结果，不因区间跨零继续加量。只允许有明确工程故障且未得到有效结果的原定试验补跑；失败任务结局不重试。
- 这是已知任务与官方环境划分中的新初态/新噪声，不是新任务或全新厨房泛化。旧40初态用于开发分组；不修改历史seen/heldout标签。

批准与冻结入口：`/share/personal/longjunyu/robocasa/experiments/recovery-confirmation-20261009/release/authorization.json`。同目录保留原提案字节、80个seed、开发分组和旧400物理快照指纹。批准不等于运行准入。

新采集实现为`benchmarks/Robocasa365/confirmation_population.py`，快照写入能力由同源已审计历史实现提升到`snapshot_bundle.py`，不从归档import。保留第一次生成的XML、完整运行状态与RGB，另建环境还原并检查4次；物理哈希严格一致、像素门槛沿用既有值，不改材质或抗锯齿。重复物理起点或失败门槛立即留证停止，不换seed。

准备源码冻结于新实验根的`preparation-source`，首轮采集输出`population-attempt-1`。这是一个串行、无策略推理的准备worker，不是新的多worker调度器。正式双机推理仍使用标准持久队列；当前尚待新总体缓存与评分锚点适配、小规模实机准入，不得直接把旧40初态的探测程序指向新seed。

并行CPU结果：`/workspace/research-reviews/recovery-confirmation-planning-20261009/libero-plus-cpu/summary.json`及`camera-entries.json`。旧LIBERO初态来源哈希核验通过。Plus本地分类表含1,599条相机定义、40个基础任务，BDDL/初态资源无缺项；其中341条属于旧8任务。资源存在不等于完整示范数据下载或闭环验收。Plus的五参数定义与自建97视角目录分开记录，不把前者当作已评测。没有自动启动LIBERO/Plus训练、推理或下载。

当前PDF保持不变；本轮有结果并验收后才合并。计划工期仍以双机正式吞吐校准，之前3–4天是估计，不从准备worker启动时刻承诺固定完成时间。

### 后续推进：候选缓存与新总体探测接口

80/80新物理初态已通过，未替换seed、未筛任务。新GPU渲染阶段为80×97个候选，保留三路图像，外部联动与腕部安装定义不变。首初态完整97候选作为准入并直接复用，其余79初态进入本机8卡标准持久队列，不使用新的静态分片控制器。

- 新共享计划：实验根下`render-plan.json`；渲染源码`render-source`；首样本`render-admission`；其余结果`render-queue`。
- 本机队列：`/workspace/research-reviews/recovery-confirmation-planning-20261009/queue-render`；运行配置`configs/dsol_paper1/confirmation_render_fresh_20261009.json`。
- 运行适配：`benchmarks/Robocasa365/confirmation_render.py`与`render_queue.py`。复用`probe_queue.py`的进程身份、排他所有权、取消和缺项恢复契约，仅新增worker命令及科学校验。
- 所有候选通过后，生成`render-release.json`，旧P0的候选目录原样保留。发布仍需逐文件、物理状态和像素验收，不能用完成数量替代。
- 新初态探测接口在`short_view_probe.py`增加显式`fresh_confirmation`模式。先独立做8次初态评分并保存输入、velocity与动作，再逐值核对窗口1；之后才计算窗口2–5。历史40初态入口保持原逻辑，冻结副本不变。
- 新短探测源码`probe-source`，本机官方π0.5、GNU XR-1的配置分别为`configs/dsol_paper1/confirmation_probe_fresh_20261009.json`和`confirmation_probe_gnu_20261009.json`。实际推理准入限定首个预定新初态的规范位与`extra_002`；准入结果复用到正式矩阵，不按效果选择。

LIBERO／Plus的CPU工作已从资源清点推进到协议映射。输出为`/workspace/research-reviews/recovery-confirmation-planning-20261009/libero-plus-protocol`，含逐任务规范能力、扰动损失、旧初态Accel效果及独立的Plus初态映射。描述性分组只用旧开发初态0/1，原8任务全部保留，不以选择器收益决定纳入。341条Plus相机定义映射到原32初态，共1,364个可定位组合；这是地址清单，不是已批准或已完成的GPU工作量。Plus的`initstate_0`是机器人构型变体标识，不应误当作只允许第0个物理初态。

CPU准备明确保留缺项：LIBERO多步短探测、Plus原生位姿/图像/运行准入、Plus评分与结局，以及需要独立泛化结论时的新初态/新噪声。旧LIBERO两份后训练模型的完整矩阵不能直接冒充RoboCasa官方模型条件；扩充模型条件仍须另立匹配方案。

本次实际启动验收：80初态与97候选缓存全部通过，`render-release.json`已发布。本机官方π0.5、GNU XR-1分别完成预定2候选×8条分支的小验收，窗口1的8份初态分数、完整velocity与动作逐值复现；独立评分payload及哈希保留在各格`initial-anchor.npz/json`。不是从新结果选择更容易通过的初态或视角。

两份正式短探测队列已经启动，每台8卡、每模型319个动态任务块，最多每块32个候选；复用2个已完成准入格，其余7,758格待采集。每模型总计80×97×8=62,080条短分支，不扩预算。启动后已核实两边worker产生`completed_probes`进度，非仅PREPARED清单。

- 官方π0.5本机队列：`/workspace/research-reviews/recovery-confirmation-planning-20261009/queue-official`，结果位于共享实验根`probes-official`。
- GNU XR-1本地队列：`/workspace/dsol-recovery-confirmation-20261009/queue-xr1`，结果位于共享实验根`probes-xr1`。
- 共享运行配置：`probe-runtime-fresh.json`、`probe-runtime-gnu.json`；冻结模型/观测定义：`probe-inputs/plan.json`、`probe-inputs/config.json`；唯一格来源：各模型结果根`ownership.json`。
- 短探测不是完整任务成功率评测。后续须先核验全覆盖、冻结窗口1–5选位，再运行已批准的8独立执行噪声确认；截至此启动截点，该完整任务阶段尚未运行。

新增与相关主机CPU回归45项通过，渲染后端与短探测测试15项通过（两组有重叠，不相加），portable99项通过；仓库、架构与布局检查通过。新源码冻结和旧实验副本分离，未修改旧结果、权重或当前PDF。

### 选位冻结与确认衔接（2026-10-09）

北京时间19:13:24检查：官方π0.5已完成6,958/62,080条短分支（11.21%），XR-1完成8,184/62,080（13.18%），均包含16条已复用准入分支。两机各8个RUNNING worker，无FAILED，状态持续刷新。此前累计实际吞吐线性估计剩余约11.2/9.4小时，仍受后续任务耗时影响；完整闭环2–3天仍是待新阶段实测速率校准的规划值。

新增可复用选位冻结实现`AlphaBrain/research/dsol/analysis/recovery_confirmation.py`。每个初态必须有完整97候选×8评分×5窗口，核验来源所有权、payload哈希、worker身份和独立初态锚点；按原目录顺序解决并列，不读完整任务成败。窗口1沿用原算术，主窗口仍为3，不用确认结果挑窗口。规范位、全部32扰动位及五个赢家去重，一视角只对应一组8执行噪声。

真实记录首次核验已冻结官方7个、XR-1 8个完整初态，分别对应1,976/2,232条待执行闭环（不是已执行结果）。冻结文件位于`/workspace/research-reviews/recovery-confirmation-planning-20261009/selected-views/{official,xr1}/initials`，待处理初态不会被当作缺失候选删除。再次读取已冻结选择会核对来源哈希，不改写选择结果。

CPU衔接配置为`configs/dsol_paper1/confirmation_followthrough_20261009.json`，源码独立冻结在上述规划根的`followthrough-source`，状态为`followthrough-status.json`。该进程已启动：等待两模型原队列全部完成，再核验全部选择，生成`confirmation-execution-plan.json`。遇到工程失败或输入变更留证停止，不自动重试；它不启动GPU、不消费确认结局。不能将其`READY_FOR_ENDPOINT_RUNTIME_ADMISSION`误读为完整闭环已经运行。

完整任务适配实现`benchmarks/Robocasa365/confirmation_execution.py`与`confirmation_queue.py`。复用原生π0.5/XR-1 rollout，而非另造成功判据；严格保持任务时限、25/16步重规划节奏、execution噪声地址、初态缓存输入、相机安装不变与复位检查。标准队列适配核验试验覆盖、噪声与轨迹来源；success=false同样是有效结果，不因失败重跑。当前仅通过CPU验收，尚待实际完整轨迹准入及正式任务分块发布；未在运行中的短探测GPU上争抢资源。

本轮相关21项CPU测试及portable99项通过，仓库/架构/布局检查通过。静态检查曾指出新核心函数写死路径映射，已改为显式配置后重测通过。LIBERO/Plus仍为已完成CPU资源/协议准备、GPU补测未启动；未修改PDF、旧结果或冻结探测源码。

### LIBERO／Plus CPU输入准备（2026-10-09）

用户授权继续利用CPU准备，未改变任务、模型、初态或GPU预算。新增核心准备器为`AlphaBrain/research/dsol/data/libero_probe_preparation.py`，配置`configs/dsol_paper1/libero_plus_cpu_preflight_20261009.json`。实际产物统一在`/workspace/research-reviews/recovery-confirmation-planning-20261009/libero-plus-preflight`：

- `libero-inputs.json`：32个原物理初态、相机缓存、两模型评分锚点、原任务时限与原生接口。外部和腕部图像、相机内外参逐数组比对历史哈希，共3,104组；不只是检查文件存在。
- `libero-anchors.npz`：原49,664个member分数（轴为initial、canonical/broad模型、candidate、8评分重复），以及256份首次评分噪声（initial、8评分重复、10动作块、7动作维度）。两模型原评分噪声一致；按原numpy生成器重建，不借用执行噪声。原32重复执行库文件哈希也已核验。
- `plus-requests.json`：341条官方Plus相机定义在原8任务上的1,364条请求。分别记录含扰动编码的wrapper BDDL参数、实际基础BDDL路径、物理初态数组、相机参数和唯一地址。编码中的`initstate_0`仍是机器人变体，不能代替物理初态0/1/2/3。
- `summary.json`：CPU完成与剩余GPU门槛。请求清单没有分配GPU，也不构成新增实验预算；Plus结果不能从自建97候选按任务名直接复用。

准备过程确认了两项重要约束：一是旧LIBERO只保存逐噪声Accel分数，未保存完整初次velocity/action，因此首次推理还需独立复现原8分数与选位，再建立完整trace锚点；不能假称已通过GPU逐值验收。二是LIBERO原生每5动作步重规划，五窗口对应0/5/10/15/20动作步；RoboCasa π0.5为0/25/50/75/100，XR-1为0/16/32/48/64。比较按原生重规划窗口展开，不冒充相同物理时间长度，不擅自修改步长。

仍需处理：多步评分后续噪声地址协议冻结、LIBERO实际推理/复位准入、Plus实际相机位姿/RGB准入、运行后端接标准队列及最终GPU规模。CPU准备没有训练、物理仿真、策略推理或新闭环。相关10项单元测试及portable99项通过，仓库/架构/布局检查通过。

当前RoboCasa已使用标准动态队列，每机8槽，每模型319块（239块32候选、1块31候选、79块1候选），空闲槽继续领取下一块；复用已完成准入格、地址和所有权冻结。两机当前按模型分工，不是跨机自动工作窃取；模型在块内驻留、跨块重新加载。不能把“动态调度已优化”解释成“跨机均衡与模型加载开销全部消失”。本轮CPU准备未热改任何运行副本或调度清单。

### 2026-10-10 控制器中断与恢复

北京时间10:17只读核查发现，本机协调器及CPU衔接进程已不在；无短探测worker存活。官方π0.5队列账本停在49 COMPLETE、8 RUNNING、262 PENDING，然而原8个RUNNING所属worker也已正常完成，实际57块全部有COMPLETE记录、无FAILED，合计11,376/62,080条（含16条准入复用）。最后worker完成于10月9日20:21:29。其余任务没有分发，不能解释为计算变慢或科学负结果。

GNU XR-1的319块均已被标准队列核验为COMPLETE；最后worker完成于10月10日05:12:00，队列05:12:16完成。CPU自动衔接旧状态仍为WAITING但已停止刷新，所以全体选位和完整闭环均未自动开始。

本机控制器此前依附工具执行会话，没有独立持久进程托管，这是保活设计的缺口。现有证据不足以确定操作系统层面的退出触发原因；没有把它写成OOM、GPU故障或模型数值错误。旧stdout缺少独立持久日志也限制了归因。

恢复不改协议、不热改冻结源码：在`/workspace/research-reviews/recovery-confirmation-planning-20261009/recovery-20261010`保存原`queue-state-before.json`与`followthrough-before.json`；独立tmux服务器`dsol-confirmation`中，`official-20261010`会话按原冻结源码先执行标准`recover`，核查旧所属进程均退出和已完成证据，再执行原队列`run`。不重跑已成功保存的57块，不重新分配科学地址。协调器stdout持久记录到`official-controller.log`。

CPU衔接另开`followthrough-20261010`会话，仍用原`followthrough-source`冻结实现。新尝试的配置/状态/日志在恢复目录，旧状态原样保留。该程序依然只自动完成评分验收、选位冻结和完整任务计划准备，不能宣称自动启动GPU闭环。恢复与真正恢复出新分支须分别核实，队列账本RUNNING本身不作为健康证明。

恢复后核验：57块全部通过原适配器重新校验，原8条陈旧RUNNING账目转为已完成；未丢失、重算已有结果。北京时间10:21原队列重新分发8块，剩余254块PENDING；8个新worker已实际启动加载，不能把STARTING计作已完成新分支。XR-1另行开始全80初态选位证据审计，未启动完整闭环。

### 2026-10-10 GNU独立进入完整任务阶段

用户授权GNU先进入下一阶段，不等待本机官方π0.5。全80个XR-1初态均完成97候选、8评分噪声、5窗口核验，首次评分与独立锚点完全对应；五窗口选位已冻结。规范位、全部32扰动位、五个窗口赢家去重后为21,960条完整任务，8个execution噪声，主窗口仍固定为3。仅调整两模型的启动先后，不改变研究范围、选择规则、任务时限或48,640条双模型总上限。

模型分区计划生成器默认仍要求两模型，单模型发布必须显式传入`model_partition`并限定24,320条上限。XR-1选择文件逐字节发布到共享盘，只有索引中的路径更新；来源哈希保留。核心发布/分块在`AlphaBrain/research/dsol/data/confirmation_execution.py`，实际程序在`benchmarks/Robocasa365/confirmation_queue.py`，配置为`configs/dsol_paper1/confirmation_execution_gnu_20261010.json`。

共享实验根`/share/personal/longjunyu/robocasa/experiments/recovery-confirmation-20261009`下的唯一XR-1执行资产：

- `execution-selections/xr1`：完整冻结选位和原输入锚点引用。
- `execution-xr1-plan.json`、`execution-runtime-gnu.json`：科学分区与资源配置。
- `execution-source-20261010`：新执行源码；不改活跃短探测或旧CPU钩子副本。
- `execution-xr1`：实际准入、正式结果与唯一所有权；准入的16条从正式队列中扣除复用。
- `execution-partitions-20261010.json`：分区授权、总预算和避免重复执行的记录。

GNU控制目录为`/workspace/dsol-recovery-confirmation-20261009/execution-xr1-20261010`；协调器由独立tmux服务器`dsol-confirmation`的`xr1-execution-20261010`会话托管，日志为同父目录的`execution-xr1-20261010.log`。北京时间10:28启动，先用一个GPU完成预定首初态的规范位和extra_002、每位8执行噪声，共16条。实际准入检查初始输入、固定安装、原生16步重规划、任务时限和每次执行噪声地址；成功或失败都保留，成功率不作为准入门槛。

准入通过后程序自动写验收收据、建立每块最多4视角×8执行的标准8槽动态队列并启动；失败则留证停止，不自动重试。当前截点仍在准入，尚不宣称全8卡正式评测已经运行。CPU16项针对性测试、portable99项与仓库/架构/布局检查通过，这些不替代实际轨迹准入。

本机π0.5仍按原短探测队列运行；10:29账本为60 COMPLETE、8 RUNNING、251 PENDING。旧CPU衔接仍只负责准备，不启动GPU。其后生成的双模型计划不能再次用于启动XR-1；π0.5完成后应发布独立official分区并引用此处XR-1唯一结果所有权，避免重复预算。

后续截点：GNU16条完整任务准入已全部通过，约10:47自动进入722块正式队列（21,944条新增＋16条复用＝21,960条）。验收单卡每条平均约67.6秒，仅一个任务初态，不能等同于8卡、10任务总体吞吐。本机10:47完成13,333/62,080条短分支，恢复后约4,493条/小时，线性估计剩余10.85小时；这不是跨任务尾部工期保证。

官方模型GPU自动衔接已独立启动：`confirmation_pipeline.py`等待原CPU准备成功，再逐字节发布official选择、建立official-only计划并调用实际准入→8卡正式队列。配置为`confirmation_handoff_fresh_20261010.json`及`confirmation_execution_fresh_20261010.json`；状态在共享实验根`official-handoff-status.json`，源码`execution-source-official-20261010`。等待阶段不占GPU，CPU准备失败则停止，不重复启动XR-1。

用户确认15日双机本地盘清空、共享盘保留。当前目标为14日前完成本轮RoboCasa确认与收尾，不自动包含LIBERO/Plus的新GPU实验。备份已与评测并行启动，详见[清盘准备与恢复边界](../../protocols/execution_management_zh.md#2026-10-10-部分闭环恢复与15日清盘准备)。旧导入/路径修复、独立缺失条目恢复工具与备份工具在当前开发源码验证，未热改活跃冻结评测副本，也未更新当前PDF。

10月10日10:55截点：官方短探测14,018/62,080（22.58%），70块COMPLETE、8块RUNNING，无FAILED；GNU正式8卡已产生新完整任务，首16条正式记录平均69.1秒（范围67.5–70.7秒），仍主要来自首批任务，不能替代全任务工期实测。当前规划XR-1于12日晚至13日、官方π0.5于13–14日结束，14日做覆盖验收、汇总和最终备份；工程异常可能推迟，不能为赶期限减少预先冻结的样本。

本轮修复后完整CPU测试：1,062项通过、7项按环境/资产条件跳过、182个subtest通过，无失败或收集错误；唯一warning是CPU测试禁用CUDA autocast。仓库/架构/布局检查通过。短探测恢复保留完整候选格、闭环恢复保留完整执行条目，均已有故障/重复/损坏/跨重启CPU测试，未停止活跃GPU试验做人为故障演练。

空间分析实现为`AlphaBrain/research/dsol/analysis/short_view_spatial.py`，绘图为同研究包`plotting/short_view_spatial.py`；配置为`configs/dsol_paper1/analysis/short_view_spatial_20261009.json`。结果根为`/workspace/research-reviews/short-view-probe-20261008/spatial-analysis-20261009`，包含两模型×seen/heldout的四张三维图及逐任务统计。

图使用96个实测扰动候选，排除没有同一参数化的规范位，不插值；实际选位和成功率仍沿用全部97候选。空间相关的候选口径因此与前一份包含97点的窗口总表略有区别，不应把两个系数误认为计算漂移。每个Accel图使用独立全范围色标，绝对数值须读色标。

三维坐标标记初始相机组的方位、俯仰和半径参数，不是世界XYZ。窗口2–5的Accel来自各候选独立短探测后不同的机器人状态；它是“初始候选→后续评分”的场，不是同一中间状态换视角的价值场。完整任务成功率场在五窗口之间保持不变。

heldout的空间平均图排名相关（低Accel对应高成功率为正），XR-1窗口1/3/5为0.701/0.837/0.776，官方π0.5为0.471/0.718/0.750；对应初态内汇总排名相关分别为0.166/0.263/0.123及0.050/0.134/0.096。平均图改善不等于每个初态最小点可靠，窗口也不是越长越好。新增统计6项测试及portable99项通过，仓库、架构、布局检查通过。尚未发布PDF更新，也未启动独立确认实验。

### 2026-10-10 Video接入与三机分工

用户追加Video主机并授权停止原大模型和应用。11时核验为8×RTX5090 32GB，驱动580.159.04，与Fresh相同；共享盘挂载一致。已向明确识别的Qwen控制器和网页uvicorn发送SIGTERM，确认原模型进程全部退出，各卡只保留约0.7GB的GPU保活；SSH、其他服务、原应用文件及模型权重未删除。原启动身份和使用迁移后路径的恢复命令保存在共享实验根`video-onboarding-20261010.json`。

Video的`openpi-robocasa`环境、RoboCasa/robosuite源码和资源及Python解释器从Fresh复制到相同绝对路径，复制后进行rsync内容校验；没有重新安装升级包。CPU导入通过。使用已完成的CloseFridge规范位初态锚点进行独立工程回放，8套评分噪声的动作及velocity trace全部逐值一致，最大误差0；记录`video-anchor-admission.json`，不计为新增科学试验。另在同初态恢复仿真，检查规范位和extra_002三路图像，原生MSAA=4不变；最大像素误差1级、平均不超过0.00009156，原门槛全部通过，未执行机器人动作。该小验收不代替全部任务或完整轨迹准入。

分工与唯一入口：

| 主机 | 当前/下一阶段 | 控制与结果 |
| --- | --- | --- |
| GNU | 原XR-1完整任务队列不变，21,960条 | 原`execution-xr1`及GNU控制目录 |
| Fresh | 原官方短探测继续；完成后承担official一半左右的原生步数负载 | `official-handoff-threehost-status.json`；`execution-official-fresh` |
| Video | 已完成固定输入/渲染检查，等待全80初态选位发布；随后承担另一半official负载 | `video-handoff-status.json`；`execution-official-video` |

已确认旧official自动衔接处于纯CPU等待、尚无计划及端点评测控制目录，才停止该会话；原源码/配置/状态保留，新冻结源码为`execution-source-threehost-20261010`。主发布者在Fresh的`official-threehost-handoff-20261010`会话，Video消费者在`official-video-20261010`会话，均属于独立tmux服务器`dsol-confirmation`且持久记录日志。Video当前状态`WAITING_FOR_FROZEN_PARTITION`，不是完整闭环已经开跑。

全80初态选位冻结后，单一发布者生成`official-host-partition.json`，按候选数×8执行×原生时限、两机各8槽均衡分配，某个初态只属于一台。每台先跑本机首初态规范位及extra_002的16条完整任务准入，通过后8槽动态执行；这32条均从原总量扣除，没有新增预算。主窗口3、五窗口规则、输入、噪声地址和原生时限不变；不重新启动GNU或重复短探测。最终分析须合并三个ownership文件，并确认两份official所有权恰好覆盖原80初态，不能只读Fresh一份。

CPU全套1,069项通过、7项按条件跳过、182子测试通过。新增覆盖分区完整/互斥、错误计划拒绝、双pilot复用和固定输入回放失败关闭。源码入口及配置索引为`configs/dsol_paper1/confirmation_threehost_20261010.json`。当前PDF未改。

工期规划：10日夜间完成official短探测后，Fresh/Video并行完整任务，GNU持续运行；以当前GNU每条约68秒、8槽及既有official时长估计，目标12日晚至13日完成本轮，14日验收与最终备份。Video尚无完整任务吞吐，以上不是已实测三机承诺；异构任务长尾及块间载模开销会影响时间。Video控制目录每10分钟备份到共享盘，新增运行环境可由Fresh已备份同版本恢复，原应用不进入研究备份范围。

### 2026-10-10 短探测分流与双worker提速

用户指出Video空闲及单卡显存利用不足，授权按优先级提速。约12:08在确认旧协调器身份及无LAUNCHING间隙后，只停止Fresh协调器，不停止任何GPU worker；新队列保留全部正在运行句柄，并把未启动任务划给Video。剩余短分支分配Fresh 21,206、Video 21,200。新队列任务及输出记录的互斥并集与原319块清单逐项相等；原ownership/分析入口不变，已完成结果不重跑。

先在Video GPU0测试双worker，其余卡单worker。同一初态57fc029ce5ccb15b3e2547d5、不同候选块，约273秒内双worker合计81条分支，单worker卡41条；两个worker各40/41条，没有明显互相减速。每个worker已提交5个完整候选格，首次评分与初态锚点的严格检查通过。初期双worker显存约19,876MiB，扩到16槽后的已采样最高22,636MiB。这是工程吞吐比较，不是新的科学对照或所有任务的并发认证。

GNU GPU0另做XR-1双worker完整任务试跑。单worker卡近期900步任务约45–46秒；双worker卡两路各约46–48秒，未明显增加单条时长。已完成的主/副worker 16/5条完整任务通过原输入身份、原生重规划、噪声地址、任务时限及trace哈希检查，尚未完成的条目没有算入验收。采样峰值26,071MiB。随后批准工程布局扩为每卡两个worker，保持原试验预算。

现行运行控制（共享索引`CURRENT_EXECUTION.json`）：

- Fresh短探测：`/workspace/research-reviews/recovery-confirmation-planning-20261009/queue-official-dual-20261010`，16槽。
- Video短探测：`/workspace/dsol-recovery-confirmation-20261009/queue-official-dual-20261010`，16槽。
- GNU完整任务：`/workspace/dsol-recovery-confirmation-20261009/execution-xr1-all-dual-20261010`，16槽。

12:22截点，三份新队列均RUNNING、各16个已启动工作块；部分worker在载模，不能据此直接宣称48路稳态吞吐。官方短探测22,259/62,080；XR-1完整任务783/21,960，无FAILED worker。旧控制根、状态及源码保留并登记退休，不再读取其陈旧状态作进度。所有新增控制目录仍被原每10分钟备份覆盖。

下一阶段official自动钩子也已换为独立加速源码`execution-source-accelerated-20261010`，对应`official-handoff-accelerated-config.json`和`video-handoff-accelerated-config.json`；两机各16槽，仍待全80选择冻结，各机16条完整任务准入计入原总数，失败停止而非缩减预算。当前短探测worker及GNU完整任务worker继续原冻结源码，调度修改没有进入模型、噪声、指标或仿真逻辑。

CPU全套1,080项通过、7项按条件跳过、182子测试通过；包括互斥完整分区、活worker不被信号终止、旧队列禁止重启、双worker只接受已登记同卡进程及显存余量检查。窗口、任务、初态、评分/执行噪声及当前PDF均未变。

12:24实测截点：official短探测22,960/62,080，最近180/300秒按已提交完整候选格统计分别1,000/1,632条分支，即20,000/19,584条每小时；此前Fresh单机约4,700条每小时。XR-1完成821/21,960，最近180/300秒44/67条，即880/804条每小时，仍含扩并发载模阶段，不能直接当作全部10任务稳态速度。三机均16槽运行，无FAILED worker。暂估短探测再2–3小时，整轮目标前移至11日晚–12日，14日收尾目标不变；以共享实验根`acceleration-validation-20261010.json`保存此次工程快照。后续不为达到预估时间删减试验。

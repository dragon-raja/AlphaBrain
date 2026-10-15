# 并行执行、恢复与后端接入标准

适用于后续评测、渲染和有明确独立工作单元的训练编排。分布式训练的一个协同进程组视为一个工作单元，不把FSDP/DDP的rank随意拆成独立任务。本标准不修改研究问题、任务范围、初始化、噪声预算或成功判据。

## 实现与职责

| 内容 | 唯一实现／位置 |
| --- | --- |
| 科学协议与任务身份 | `AlphaBrain/research/dsol/execution/manifest.py` |
| 动态分配、排他控制、持久状态、恢复 | `AlphaBrain/research/dsol/execution/queue.py` |
| 安全参数与显式执行门槛 | `AlphaBrain/research/dsol/execution/cli.py` |
| 标准命令入口 | `scripts/dsol_paper1/operations/manage_work_queue.py` |
| 模型、仿真、SSH、进程与结果校验 | 相应`benchmarks/`或运行适配层的`Backend`实现 |
| CPU故障及恢复验证 | `tests/dsol_paper1/execution/test_work_queue.py`，纳入portable CI |

队列通过接口接收后端对象，不反向import `benchmarks`、`scripts`或报告。CLI仅在明确`--execute`且冻结资产核验后加载配置指定的受信任适配器，工厂函数源码必须在文件指纹清单中。工厂只构建对象，不启动进程；真实资源启动发生在取得协调器锁后的`start`。不得使用陌生清单执行任意后端模块。

## 冻结清单

`format=dsol_work_queue_v1`，必需字段：

- `experiment_id`、`protocol_sha256`：实验身份与科学协议文件的原始SHA256。
- `artifacts`：协议、源清单、权重收据、噪声库、后端源码等文件的绝对路径和SHA256。大权重可以引用既有逐文件校验收据，但适配器必须确认权重本体符合该收据。
- `slots`：显式`host/id`与适配器资源配置；不默认两台机器、8张卡或一GPU一进程。
- `tasks`：每个块包含`addresses`、`output`、`payload`和由`task_key()`计算的`key`。
- `backend`：受审查的`module/factory`与适配器配置；路径、Python环境、SSH参数、显存资源等在这里配置，不硬编码到队列核心。

每个试验地址包含`model/task/initial/view/stream/repeat/noise`。`stream`明确区分`score`和`execution`，`noise`指向冻结噪声地址／序列身份，不是临时调用随机数后猜测的seed。块key由实验、协议哈希和原试验地址生成，不包含host、GPU或执行结果。任务可包含多个试验；重分块应发布新的调度清单并保留旧试验地址及复用映射，不冒用旧块key。

重复试验地址、重复资源槽、重叠输出目录和缺失来源哈希在启动前拒绝。计划覆盖是显式声明的工作集合；适配器还须对照科学release验证其确为完整预期集合，队列不能从清单猜出被遗漏的任务。

## 动态分配和状态

一个协调器管理全部资源槽，空闲且通过准入的槽按冻结任务顺序领取下一个工作块。分配不使用成功率、Accel或其他实验结果；长任务不会阻止其他空闲槽继续领取。

任务状态为 `PENDING → LAUNCHING → RUNNING → COMPLETE`。启动前先保存意图，使用文件及父目录fsync加原子替换；启动后保存可恢复的进程句柄；正常退出且结果身份核查通过才标记完成。**科学任务失败、success=false，同样是有效完成的试验**，不能因此重试直到成功。

运行目录保存：

```text
manifest.json        冻结清单
state.json           清单指纹、任务状态、尝试次数、句柄、验收收据、事件历史
coordinator.lock     单协调器排他锁
```

控制目录应使用支持原子替换和可靠`flock`的本地文件系统；不假设任何共享盘/NFS都支持安全锁。只有协调器写状态，远端worker只写自己预先分配的结果目录。结果和权重可放共享盘。原始worker stdout、轨迹、失败记录和环境收据由适配器写入任务输出目录，不能只保留总成功率。

工作单元应是有界批块，而非直接将数十万条rollout全部作为独立队列项。v1状态账本每次状态转换重写，不声称完成超大任务表的存储性能认证；扩大到大量块前须测试账本IO与调度开销，再决定是否更换本地事务存储。它不影响原科学总体或每条试验地址。

## 后端契约

`Backend`接口见`queue.py`：

| 方法 | 要求 |
| --- | --- |
| `admit(slot)` | 环境、共享资产与模型版本准入；不按试验成功率放行主机 |
| `start(task, slot, attempt)` | 启动精确工作单元；返回可JSON持久化的所属进程身份 |
| `locate(task, slot, attempt)` | 处理启动与记账间隙：找回进程身份或严格证明没有启动；无法证明则抛错，不能猜测为None |
| `poll(handle)` | 未完成返回None；进程组完整退出返回退出码，不能只看SSH客户端已退出 |
| `alive(handle)` | 核验主机、进程启动身份及所属子进程，避免PID复用误判 |
| `cancel(handle)` | 仅停止该工作单元所属进程，确认子进程退出；不得影响其他实验或保活 |
| `validate(task)` | 核验完整地址覆盖、噪声、来源、退出及轨迹／结果哈希，返回稳定非空验收收据；缺失用`MissingResult`，损坏用其他异常 |

适配器必须在返回启动句柄前保留持久启动收据，并解决SSH断联和PID复用；一个SSH客户端PID不够。输入、输出、成功判据和统计口径仍由科学协议定义。后台训练、下载、GPU初始化不能发生在模块import、`prepare/status`或默认测试里。

## 失败与恢复

任何worker非零退出、结果损坏、超时或用户中断，停止后续分发并清理当前所属worker，保留失败与清理错误。SIGTERM/SIGHUP通过CLI转换为受控中断；SIGKILL、主机断电可能留下RUNNING状态，必须先核验原worker。

不自动重试失败实验。显式`recover`首先取得协调器锁、核查冻结资产，证明所有旧worker已退出；然后校验既有结果。有效完成结果保留原收据，仅缺失工作重新进入PENDING，损坏或已完成结果被改写则拒绝恢复。恢复事件及尝试次数保留。同一块部分完成时，由适配器只重跑其缺失地址并保留原结果；当前通用状态机不自行解释某benchmark的单条轨迹文件。

状态不明的旧进程不能通过删除状态文件“解决”。需要人工诊断时保留证据并另立恢复记录，不取消验收门槛。

## 命令

从仓库根目录调用唯一入口：

```bash
python scripts/dsol_paper1/operations/manage_work_queue.py prepare --manifest PLAN.json --root NEW_CONTROL_DIR
python scripts/dsol_paper1/operations/manage_work_queue.py status --root CONTROL_DIR
python scripts/dsol_paper1/operations/manage_work_queue.py run --root CONTROL_DIR --timeout-seconds 86400 --execute
python scripts/dsol_paper1/operations/manage_work_queue.py recover --root CONTROL_DIR --execute
```

`prepare`新建控制目录但不导入后端；`status`只读。`run/recover`可启动或管理进程，必须明确授权且显式执行。计划由相应实验准备适配器生成，不能将本文示例命令直接套到活跃实验目录。最终聚合仅消费全部已核验的完成收据；PDF发布属于独立报告步骤，不能因调度维护自动覆盖当前报告。

## 2026-09-30 落实范围与迁移门槛

通用身份／队列／恢复与CLI已实现，CPU测试覆盖双主机模拟动态领取、覆盖不变、重复拒绝、故障清理、排他锁、未准入超时、只读入口和显式恢复。**这是调度基础的CPU验证，不是新双机运行后端已经通过正式仿真认证。**

当前XR-1冻结双机队列`data/xr1_rebalance.py`和历史LIBERO控制器继续使用原版本。它们不因本次维护热切换；历史结果与源快照不变。XR-1现实现仍有资源槽写死、无通用恢复接口及失败收尾不完整等技术债，不以“文件已分类”宣称解决。

后续接入顺序：

1. 当前评测完成后，提取XR-1工作启动、结果校验和双机准入为运行适配器，保留原噪声地址与结果映射。
2. 加入真实本地／SSH进程生命周期、部分结果恢复、启动间隙及断联的有界CPU工程测试。
3. 用独立输出做原实现／新适配器配对推理与完整轨迹验收；保留原生渲染非逐位确定性的已知限制，不借迁移放宽门槛。
4. 验收通过后，下一份新release从标准入口启动。历史LIBERO在实际需要运行时接入，不为“统一”重跑旧矩阵。

新并行工作默认遵守此接口。暂时不能接入时，启动前在对应执行记录中登记兼容例外：release、原因、科学身份不变依据、尾部调度风险、停止／恢复方法与退出条件。代码评审检查标准入口和适配器是否已有实现，禁止再次复制实验专属调度器替代复用。

本轮验收：通用队列28项CPU测试通过；portable CI共99项通过；原XR-1调度／阶段划分／续接／全视角分析18项回归通过；仓库、架构、布局和diff空白检查通过。未启动GPU任务、未重跑正式矩阵、未修改冻结副本、权重或当前PDF。这些结果不替代后端实际运行准入。

## 2026-10-10 部分闭环恢复与15日清盘准备

用户确认：10月15日本机与GNU都会强制关机，本地盘重新初始化，只有`/share`保留。因此按清盘迁移处理，不假定本地tmux、容器、虚拟环境、Git未提交文件或队列账本能够留存。

### 部分闭环恢复

新增`benchmarks/Robocasa365/confirmation_recovery.py`，只在原协调器已退出、取得原本地控制锁且所属worker均已停止后使用。它逐条验证已有完整任务（包括失败）的原始身份、噪声、时限和轨迹哈希，发布新控制目录与新输出目录，只分配缺失的`initial/view/repeat`。已有输出不覆盖、不删除；损坏记录不被伪装为缺失后重跑。跨重启时旧boot ID证明旧进程不可能存活，不向复用PID发信号。

恢复清单继续调用原已准入的冻结worker及其原生rollout；新工具只改变缺失任务的分块和结果所有权。原队列、旧状态与保留结果的哈希一起纳入新清单。合并分析必须同时读取`recovery-ownership.json`中的retained条目和新队列条目，且当前试验的16条准入所有权仍单独保留。不得仅统计新目录或把缺失算失败。

本轮通过CPU故障测试；未人为中断正在运行的正式矩阵来演示恢复。当前标准短探测队列也可复用已提交的完整候选格（每格8次×5窗口），未形成完整格收据的在途计算按缺格重新执行；不宣称能够恢复只在内存中的半条轨迹。其他历史后端不自动获得此能力。

### 备份与收尾边界

统一根：`/share/personal/longjunyu/recovery/shutdown-20261015`。

- `fresh`、`gnu`：明确列出的项目源码（含Git和未提交文件）、虚拟环境、上游代码/资源、分析产物的非删除备份；原绝对目录编码在`files/`下。
- `controls-fresh`、`controls-gnu`：活跃本地控制目录的周期增量快照，保留旧版本。活跃文件可能在复制时改变，必须在最终静止状态再做一轮；不能把运行时快照称为最终一致备份。
- `*-environment.json`：Python包、系统包、驱动/系统版本；不是包含内核与驱动的可启动整机镜像。恢复后仍需重建兼容系统、挂载共享盘，并核验虚拟环境绝对路径和外部软链接。
- `manifest.json`记录文件内容哈希、原路径、模式及软链接；替换备份前保存旧版本，不删除源文件。备份进程不申请GPU，限速并降低CPU优先级。
- SSH私钥、API凭据、Codex登录信息不进入普通研究备份；清盘后须另行恢复/重新配置认证。不会重建已删除的`/share/longjunyu`。

首轮备份10日已启动，不等到实验结束。13日按实测剩余工作评估工期；14日优先完成结果覆盖/来源验收、最终增量备份、校验与恢复清单。若仍未跑完，保留已完成结果和明确缺项，不减少噪声、不删除任务、不因成功率选择性补跑，也不把未完成结果发布成完整结论。未经用户另外确认，不新增LIBERO/Plus的GPU预算或自动启动其全量实验来挤占窗口。

### 增加主机而不重复试验

`confirmation_execution.publish_host_partition`在完整单模型计划冻结后，以候选数×8次执行×原生任务时限估算负载，用LPT贪心规则按显式主机容量分配初态。不读取成功率或窗口效果决定归属。分区记录绑定计划哈希；每台建队前验证全部主机的并集完整、交集为空。每主机使用自己的本地锁/控制目录与共享结果子目录；本机8槽动态领取，暂不支持跨主机运行中偷取任务。

每主机在自己首个初态的规范位和extra_002执行16条正式准入，后续本机完整队列扣除这些地址。准入绑定主机、分区、模型和源码。CPU测试验证两份pilot与两份正式队列的四路并集恰好覆盖原计划，重叠、缺失及错计划分区均拒绝。扩大主机不扩大研究预算。

单一发布者冻结全体选择和分区后写发布收据；其他主机的`confirmation_pipeline.after_publication`只消费该收据，不另算选择或另发计划。旧等待钩子先验证尚未建队后退出，再另立冻结源码和状态。不得热改正在运行的队列来加入机器。汇总必须读取所有主机的ownership，不能只统计一个子目录。

### 运行中增加资源：只替换协调器，接管存活worker

通用实现`execution/repartition.py`发布互斥的新清单，不改原清单或结果目录；运行层`benchmarks/Robocasa365/queue_handoff.py`核验协调器的PID/启动身份/命令，短暂暂停单一协调器，避开LAUNCHING记账间隙，随后只结束协调器PID，不向其GPU worker进程组发信号。原正在运行的handle、slot、attempt完整带入同主机的新控制器；只有PENDING、attempt=0、输出目录不存在的任务可移动到其他主机。接管poll使用原持久进程身份，不因父进程退出重复启动。

旧控制状态/清单按原字节复制到共享证据目录，旧控制根添加retired.json。新队列的任务key、address、output逐项等于原任务；各分区必须互斥且并集完整。新调度源码独立冻结，worker仍使用原已准入源码和环境。故障时保持证据并停止交接，不通过删除输出或重置attempt处理。旧源码尚不识别retired标记，操作人员必须以CURRENT_EXECUTION.json为准，不再运行旧命令。

每卡双worker使用两个不同slot id（例如0与0:1），但显式gpu_id均为0。资源准入仅允许本控制账本已登记worker占用同一卡；陌生GPU进程拒绝，启动新worker前至少保留13,500MiB空闲显存。该门槛不等价于通用OOM保证，扩并发前仍需模型对应的实测峰值、实际吞吐和输出验证；不支持未经准入的三worker。仍是一worker一个独立模型/仿真进程，噪声身份不随并发改变。

### 标准容量测试、三机巡检与排空（2026-10-10）

复用现有入口，不新增实验专属调度脚本：

```bash
python scripts/dsol_paper1/operations/manage_work_queue.py cluster-status --config CLUSTER.json
python scripts/dsol_paper1/operations/manage_work_queue.py resources --root CONTROL_DIR
python scripts/dsol_paper1/operations/manage_work_queue.py drain --root CONTROL_DIR --host video --gpu 0 --reason 'isolated capacity benchmark' --execute
python scripts/dsol_paper1/operations/manage_work_queue.py resume --root CONTROL_DIR --host video --gpu 0 --reason 'benchmark finished' --execute
python scripts/dsol_paper1/operations/manage_work_queue.py audit-results --root CONTROL_DIR --per-status 2 --include-running --output NEW_AUDIT.json
python -m benchmarks.Robocasa365.capacity_benchmark --config CAPACITY_TEST.json --execute
```

- `cluster-status`并行读取明确配置的主机；审计跨机科学地址/输出所有权，报告待办、在途、GPU显存/利用率，以及非主动排空情况下的闲置槽。它是只读快照，不能凭过时快照直接重发任务。连接失败不伪装为空闲。`CLUSTER.json`的`hosts`列表提供`name`、`control`、可选`python`及SSH参数数组`ssh`；不在通用源码硬编码个人地址。
- `drain`按**物理GPU**同时阻止其全部slot接新任务，不取消现有worker。控制文件绑定manifest哈希，所有修改有原因/时间记录。`resources`只有在该卡已无LAUNCHING/RUNNING记录时才报告quiescent。排空不是显存独占证明，测试前还必须检查GPU实际进程。具有活动排空/预留的队列禁止交接，避免新协调器忽略预留。
- `audit-results`仅检查已提交结果，按task key确定抽样，不看成功标签挑样。STARTING尚未提交的worker标为NOT_READY，不误报为科学失败。导入的适配器必须来自清单记录的冻结来源；旧release上的检查需在原来源环境执行。完整任务检查原初态/选择/执行噪声/重规划时刻/时限/trace哈希；短探测还重算前缀并核对窗口1。
- `capacity_benchmark`当前真实适配器支持RoboCasa官方π0.5。先等待已排空卡，再按1/2/3 worker测试固定输入8组噪声的动作与velocity逐值重放，以及原冻结worker的真实五窗口短探测。每2秒记录显存、利用率；错误、超时或剩余显存低于512MiB时只停止本测试所属进程组。结束后恢复原队列派工，不自动将新容量推广至生产。
- 生产推荐至少保留2GiB实测余量、质量检查通过，并比更小并发提高至少5%吞吐；这是显式工程准则，不是普适最优值。测试记录包含模型/配置/来源/硬件与阶段。短探测准入不等于完整rollout准入；Video准入不自动覆盖Fresh；XR-1不沿用π0.5上限。一次小样本不能证明所有任务峰值，持续运行仍需观察。
- 并发容量不是训练microbatch。`execution.resources.training_allocation`将DDP/FSDP训练表示为独占物理GPU的整体作业，一卡一rank、world_size一致；禁止按评测的多slot方式把训练rank拆成独立任务。目前没有迁移所有历史训练launcher，也未验证跨三机训练，不能宣称它们已自动获得负载均衡。后续训练入口接入独占资源契约后才能与评测混排。

标准执行顺序是：冻结科学计划 → 核验主机/模型 → 实测对应阶段容量 → 按容量与原生时限分区 → 每机动态队列 → 巡检剩余负载 → 必要时通过`queue_handoff`迁移从未启动的任务。没有完成显存/正确性准入，不以填满显存为目标强加worker；有空闲卡且另机有待办时，必须先检查模型环境和所有权，再调配未启动任务。正式产物仍由全部主机ownership聚合。

本轮实际入口：共享实验根`recovery-confirmation-20261009`下的`cluster-current-20261010.json`、`CURRENT_EXECUTION.json`、`capacity-pi05-video-config-20261010.json`。容量测试与审计产物单独保存，不进入论文统计。Video切至带排空控制的新冻结协调器，原GPU worker、输入、输出及科学预算未修改。

#### 本轮π0.5实测结果

Video的一张RTX5090，原权重、数值flags、8评分噪声与5次重规划均未改变；每个worker测试同一初态的canonical和extra_002，两候选共16条短探测。工程测试独立输出，不覆盖正式结果。

| 每卡worker | 固定输入动作/velocity | 真实短探测冷启动批耗时 | 实测峰值显存 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 8组逐值一致 | 16条 / 136.03秒 | 11,027MiB | 通过 |
| 2 | 每worker 8组逐值一致 | 32条 / 140.21秒 | 22,131MiB | 通过；总吞吐约1.94倍 |
| 3 | 每worker 8组逐值一致 | 31.72秒时触发保护，未完成 | 31,647MiB，剩余463MiB | 不准入真实探测生产 |

三worker纯策略重放峰值26,901MiB，说明额外的仿真、渲染及运行时内存不能忽略。此次不是“观察到数值错误”，也不是已经发生OOM，而是触及预设显存安全底线，只停止独立测试进程。原生产worker早已自然完成，测试后GPU 0自动恢复派工。当前完整评测仍每卡2个；XR-1已有双worker约26GB的实测，未宣称实测过三worker。

原始证据位于共享实验根的`capacity-pi05-video-20261010/status.json`及各阶段日志、遥测、重放收据。其三worker失败阶段旧runner的`throughput`字段由计划数量计算，**不得作为实际吞吐**；该阶段完全未准入，正文不报告该值，新版工具失败阶段固定记0。冷启动小面板不是长期吞吐上限，不能据此保证每个任务、每种环境或每份权重都适合双worker。

并发结果抽查：Fresh/Video短探测各抽查2个已完成块与2个在途块；GNU完整任务亦核对4个固定块。GNU首次抽样有一个worker处于STARTING且尚未发布identity，旧抽查器报HOLD；保留该记录，并在同一4个key上复核，不换样本。标准工具现明确区分NOT_READY与已提交结果损坏。它不是修改数值容差或删去失败轨迹。抽样通过不能替代最后全量覆盖/身份验收。

### 跨模型按结束时间均衡

`execution.balance.plan_allocation`按剩余GPU时间枚举整数卡数，目标是最小化最后一个模型的完成时间，而非永久绑定“某机器只跑某模型”，也不是各机任务条数相等。需要同时考虑当前已完成量、原生任务时限、不同模型实际耗时及已准入的每卡并发。

`benchmarks.Robocasa365.cluster_balance`提供CPU观测器：每10分钟读取正式完整任务的计时/覆盖，不读取success用于调度。已有足够样本的任务用该任务平均时长，未访问任务按已测每执行步耗时乘原生时限投影；启动开销单独显式配置。当前小样本门槛为每模型128条、至少2任务各16条，预计最晚结束时间改善至少15%才提出调卡建议。投影不是工期保证，也不能直接触发重跑。

本轮配置和输出在共享实验根`finish-time-balance-config.json`与`finish-time-balance.json`。最初π0.5仅有单卡pilot，必须标为WAITING_FOR_REPRESENTATIVE_TIMING，不能冒充双worker全任务吞吐。观测器只发布建议，**不是已完成自动跨模型迁移**。迁移前还需目标主机对应模型运行环境、数值/完整任务准入，之后通过排空和互斥交接迁移从未启动的任务；在途不打断，历史所有权和输出不改。

三机巡检配置可按优先级列`controls`，例如先full/再pilot/；只有manifest与state均已存在的目录才被选中。这避免阶段自动衔接后仍把旧短探测COMPLETE当成当前评测状态。

跨主机迁移只分配从未启动的块；在途句柄、旧槽位字典、输出目录和评分/执行地址原样保留。新控制器可携带物理卡 drain 状态：在途自然结束后不补派，之后才能把这张卡交给其他队列。迁移前核验槽位的完整字典，而不是只比较 GPU 编号；旧主槽位未必含 `gpu_id` 字段。排空本身不等于取消评测。

资源准入与科学运行时分开：目标主机已有同模型、同执行计划和同冻结 worker 的完整任务准入时，可通过指纹绑定的 `resource_admission` 引用目标收据，保留原运行时记录。新主机没有该模型准入时，先复制并校验运行环境、同卡双worker逐值重放，再以转入的首个原预算任务块单worker完成实际闭环验收；通过后才扩展并发。该首块不是额外评测，不得再在原机执行；失败时禁止自动扩大或重跑。

2026-10-10 16:08实际完成12/12配卡：Fresh两模型各4卡，Video 8卡π0.5，GNU 8卡XR-1，全部每卡2 worker。Fresh复制运行环境的201,358个文件、41,867,360,887字节通过GNU备份哈希核对；双worker共32次固定输入推理的动作/velocity逐值相同；首个转入块32条完整任务通过准入。迁移后139条已提交轨迹抽查通过，全量1,451块/44,328正式队列地址逐项相同，加原48条pilot总44,376。最终三机巡检无重复地址、GPU重叠或闲置槽告警。

此次保留了两项控制流失败记录：旧主槽位字段兼容导致一次接管被拒绝（未向worker发信号，原句柄随后接管）；等待器把新控制器的短暂PREPARED误判为不健康，后续从已完成的迁移步骤继续，未重新执行科学任务。槽位检查已前置到停止旧协调器之前并有回归测试；PREPARED必须与FAILED分开。旧收据不改成PASS，新的恢复收据和当前索引给出实际完成状态。

当前工程证据统一位于共享实验根的`rebalance-exact-ownership-audit.json`、`rebalance-audit-*.json`、`rebalance-execution-resumed-status.json`及`CURRENT_EXECUTION.json`。估时新入口为`finish-time-balanced.json`，按当前12/12资源更新；旧16/8观测器已停止。估时仍含未覆盖任务的投影，不是期限保证；最后必须全量验收并做静止后的增量备份。

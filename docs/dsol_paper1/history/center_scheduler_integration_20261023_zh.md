# Center CPU调度接入：仅模拟后端开放

2026-10-23，接续此前接管准备。本轮不运行训练、推理、GPU准入、评测或线上W&B。
持久代码、测试及恢复证据位于项目产物 `runs/center-scheduler-integration-20261023/`。
现有7个暂存文件先与上一轮补丁逐字节核对，原 `.env.example` 删除未暂存。
不配置Git作者或写入认证，不提交、不推送；持续保存补丁及源码快照。

## 新入口实际接入范围

薄CLI为 `scripts/dsol_paper1/operations/manage_controlled_queue.py`，核心组合在
`execution/controller_cli.py` 和 `execution/controlled_queue.py`。
它复用原 `queue.run`，以 `LedgerBackend` 包装后端，未复制一个新的任务调度循环。
本版本固定使用 `SimulationBackend`，不会按清单动态导入研究后端；程序API与CLI都拒绝生产执行。
模拟器只写独立的合成进程/结果记录，不创建worker进程、发SSH、访问GPU或写科学结果目录。
模拟账本须以 `prepare(..., simulation_only=True)` 单独建立；无该标志的原暂停保管账本在run/recover入口即拒绝，不能向其中写模拟消耗。

整个run持有独立的 `launcher.lock`，排除第二个新版控制器及并发恢复；每次预留/启动
另持账本事务锁，直到启动回执持久化，避免预留与启动之间被另一epoch接管。
启动前校验科学地址、queue/ledger指纹、owner、epoch、独立阶段许可及全部peer；
先持久化完整块预算，再重新读取账本、许可和连通性，之后才调用真正的 `backend.start` 接口。
permit类型必须为 `CPU_SIMULATION_ONLY`，绑定owner/epoch、queue和ledger指纹、阶段及模拟时间窗。
暂停保管记录、旧campaign许可或 `--simulate` 本身都不能替代这份独立许可。

入口将断联、账本失效、回执不明、任务失败锁定为HALTED；重连不会自动续派。
预留保留完整上限，未确定的消耗不退款、不换目录重发；完成的负结果正常保留，不择优重跑。
后端报告的完成消耗必须逐维非负且不超过原预留，预留上限不会被挪给别的任务。

## 显式恢复

恢复与run使用同一launcher锁，并持有账本事务锁。先检查全部peer，核对每个已预留尝试的
`locate/alive/validate`，要求所属worker停止且有完整、身份一致的结果与消耗。
不能仅凭一个不透明摘要解除新启动预留；标准队列的恢复也不能把部分块重新置为可重发。

- 预留后、启动前进程崩溃：保留预留，缺完整结果即拒绝恢复/重发；即使模拟器可证明未启动也不自动退款。
- 后端已完整完成、启动确认或控制器丢失：显式验证后纳入COMPLETE，增加epoch，回到PAUSED；不再调用该块启动。
- 启动结果未知、worker仍活着、部分块失败或账目不完整：保持阻塞，需要独立人工核账和未来经批准的恢复方案。
- 队列曾被旧入口直接运行、出现无账本的attempt：拒绝自动收编，不能靠换控制目录消除消耗。
- 恢复后仅尚未预留的任务可在新epoch对应的新模拟许可下继续；每阶段前置完整性检查仍生效。

CLI首次run只能claim尚无owner的测试账本。新进程不得自动attach旧会话；需使用显式recover或
`run --explicit-recovery`，提供previous-owner/previous-epoch及新owner，并在run时提供新epoch许可。
单独recover始终返回PAUSED，不启动任务；恢复中途存储失效也不能绕过旧预留继续启动。

## 旧入口与仍未满足的条件

以下入口仍不检查新ledger，不能声称已全局隔离：

1. `manage_work_queue.py run/recover` 和直接调用 `execution.queue.run/recover`；它们接受原后端。
2. 冻结68f2cd8及当前未改的 `benchmarks.LIBERO.continuous_evaluation_fleet` 的run-phase/worker路径，
   包括原 `BlockBackend.start`；原CampaignGuard只处理旧campaign许可和心跳。
3. 直接调用连续配对/规范评测或其他历史训练/驻留worker入口；另建账本目录也不是跨目录全局排他。

本轮未改这些冻结副本、科学清单、结果或预算，也没有给旧campaign发permit。
新入口的进程锁/epoch属于协作控制契约，不是对恶意修改文件或绕过入口的安全隔离。
共享存储灾难/分区语义、实际SSH生命周期与worker端fencing、受信任资源身份和GPU数值/轨迹准入仍未完成。
模拟连通性测试验证的是后端探测契约，不是对恢复后真实节点的许可。

原72块/672任务、48/240/384阶段、38,592轨迹上限不变。全矩阵仅复制科学身份到独立CPU模拟队列，
不创建生产主机/期限绑定，原暂停保管账本不写入模拟消耗。
等待Mac完成历史目录增量备份、私密恢复及基础服务工作，用户重启并核验环境后，再讨论新的运行绑定和GPU准入。

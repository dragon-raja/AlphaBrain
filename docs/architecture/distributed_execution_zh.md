# 执行与恢复维护手册

## 一条授权链

科学配置 → 不可变运行绑定 → center 单一账本/owner/epoch → 实时 GO → worker → 每次查询许可 → 原始结果 → 验收/结算。

共享代码：

- `execution/controller_ledger.py`、`controlled_queue.py`：预算预留、重复任务拒绝、显式恢复。
- `execution/model_authority.py`：center 连接与资源所有权。
- `execution/model_agent.py`、`model_plan.py`：worker 生命周期、同连接 CHECK/CONTINUE、启动和结果回执。
- `execution/process_backend.py`：本地进程身份、启动间隙、GPU 所有权及所属进程回收。科学准入和结果校验由外层提供。
- `execution/source_release.py`：目录清单哈希、隔离导入、本地不可变镜像。

生产调度不信任共享 NFS 的即时可见性来替代授权。断联停止继续派发；未知消耗保留预留，不自动退款或重放。

## 两种并行不能混淆

PPO：中央学习器发布一个策略版本，采集器返回完整同版本批次，核对后更新。不按最快完成者截取样本，也不混入过期策略。
独立操作评测：预先冻结每条 trial 的初态、噪声和所有权，可以按资源动态分配短块。

当前这条 PPO 链不是通用 DDP/FSDP，也不等于批推理后端。若以后接入，须单独实现进程组资源、同步和失败合同，不能对外标成已支持。

`PrefixCampaign` 可通过冻结资源配置中的 `training_slots` 和 `evaluation_slots_max` 限制每阶段租用的物理卡数，默认保持历史 4/8；这只改变执行并行度，不改变完整 32 回合 PPO 屏障。早期已保存检查点评测和新尾段评测使用同一预算、租约与异步开发评测路径。某阶段失败时保留不明消耗并隔离所属卡，不自动重发。

## benchmark 适配

LIBERO：`continuous_authority.py` 提供科学适配，`prefix_campaign/continuation/evaluation.py` 组合实验；原生模型实现通过明确合同绑定。
GR1：复用同一 model_agent/model_plan；`trial_runtime.py` 连接搜索与最终操作；`block_runtime.py` 组合短小可恢复块。CPU 模拟会经过实际块入口，真实生产模式仍关闭，尚需冻结配置及 GPU 准入。

模型原生特征接入与搜索训练接口：`data/native_features.py`、`native_observation.py`、`task_condition.py` 与 `selectors/search_policy_factory.py`。读取层、相机身份、时序及 preprocessing 来源均为发布合同的一部分，不能只比维度。

## 恢复顺序

1. 核对 owner、epoch、节点实例/GPU UUID、解释器与源码/输入身份。
2. 以 PID + start ticks + 启动归属核查进程；不凭旧文件或名称判断，不盲目杀进程。
3. 完成结果校验后结算；未知任务先审计，禁止自动退款/重发。
4. 用新绑定显式恢复；不沿用旧 PID、旧截止时间或旧许可。
5. 关闭时持久化预算、结果、进度及代码回执。基础保活和认证属于独立运维职责。

## 运行发布清单

至少绑定源码 manifest、模型/processor/norm、任务与输入、评分/执行前缀、噪声、核心/初始化谱系、环境版本、资源与预算。动态路径是部署配置，不写入算法默认值。

CPU 测试不启动模型/渲染或科学任务。实际新后端还需模型输出/特征数值、完整轨迹和容量验收。旧冻结目录不受开发树入口防护控制，不能声称全局隔离。

完整科学约束见 [执行管理协议](../dsol_paper1/protocols/execution_management_zh.md)；原有详细沿革在维护归档，不再作为当前能力说明。

## 显式恢复的已结算额度与隔离卡

`PhaseBudget.delegate_settled_surplus` 只在人工审计后的恢复中调用：完整结算、实际消耗已知的阶段，其未用预留可以为一个独立恢复 owner 提供额度。原预留记录不修改，旧 owner 仍按原预留上限运行；每个阶段的剩余额度只能分配一次。子 owner 另用标准 PhaseBudget 硬限额，父上限减去已证明未用额度再加子额度不得超过原授权。未知/失败阶段的预留不可作为来源。分配记录在父账本的 `surplus-allocations/`，使用独立锁，避免阻塞旧 owner 的非阻塞派发锁。

隔离卡交接不是“看到空闲就使用”：必须确认原阶段授权已 HALTED、dispatch=false、无完整结果可重复纳入、原 owner 的失败分支不会重试或重新释放该隔离租约，并实时核查节点、GPU UUID、可信保活和旧 worker 已退出。新 owner 只绑定明确移交的卡，用原有 PrefixCampaign/实时授权链执行；不得把隔离卡同时恢复到旧池。原失败证据与保守预留保留，新结果另目录并建立恢复索引。

长查询准入必须覆盖长历史边界。`prefix_admission.CapacityProbeActor` 仅用于工程准入：每步真实计算冻结 actor，然后用明确标记的合法小步替代 STOP，保证超过第八次查询。不会修改权重，也不作为策略效果、科学轨迹或训练样本。科学评测继续使用原冻结 actor 的 STOP 与运动输出。

# Paper 1：主动视角搜索

当前主线：冻结 π0.5、GR00T N1.5、RLDX-1，训练共享 PPO 搜索器；benchmark 为 LIBERO 与 GR1 tabletop。FastWAM 与 RoboCasa365 等既有实现保留为历史研究/兼容用途，不是当前新增实验矩阵。

## 从这里开始

| 需要了解 | 唯一入口 |
|---|---|
| 目录职责与依赖方向 | [架构](../architecture/README.md) |
| 输入、Actor/Critic、U/T 配置 | [共享搜索核心合同](../architecture/shared_search_core_zh.md) |
| 调度、预算、故障与恢复 | [执行维护手册](../architecture/distributed_execution_zh.md) |
| 安装、CPU 验证、冻结与复现 | [复现指南](../architecture/reproduction_zh.md) |
| 当前能力与准入边界 | [就绪状态](../architecture/framework_readiness_zh.md) |
| LIBERO 后端 | [LIBERO](../../benchmarks/LIBERO/README.md) |
| GR1 后端 | [GR1](../../benchmarks/Robocasa_tabletop/README.md) |
| 科学协议与数据划分 | [protocols](protocols/) |
| 结果报告 | [reports/paper1](../../reports/paper1/README.md) |

## 当前实验与持久证据

所有运行目录相对于部署时显式配置的持久项目根，不是源码中的机器默认路径。

- U12/U3 三种子长续训与开发评测：`runs/unified-core-long-confirmation-20261104/`。
- 查询预算/成本、评分反馈与 STOP 筛查：`runs/unified-core-mechanism-screen-20261104/`，按已批准资源交接执行。
- 长训尾段、早期曲线与部署预算补证：`runs/unified-core-final-checks-20261105/`；U12/U3 从 3,072 续至 4,096，Q16 仅部署诊断，使用独立 300,000 调用上限。实际准入/启动状态以该目录 `CURRENT.json` 为准。
- GR1 CPU 准备：`runs/gr1-cpu-preparation-20261104/r15/`（整块输入预检、authority 接入与实际任务 CPU 组合）；早期重放和模型加载证据在同级 `r14/`。
- 框架整理前审查：`runs/framework-maintainability-audit-20261104/`。
- 共同核心对外协作索引：`runs/parallel-sim-real-plan-20261104/core-release/`。

原 600,000 调用包已完成结算；新三项补证独立记账，不复用旧余额。GR1 真实模型调用不包含在内。
实验状态必须同时核对账本、结果和所属活进程；旧 RUNNING 文件不能证明仍在运行。
当前开发树不是在途实验源码。结构选择完成后另冻版本，旧结果不得改标为统一结构。

本轮实现与验证见[框架整理回执](history/framework_cleanup_20261104_zh.md)。

## 历史与恢复

[history](history/) 保留既有协议演进和审计记录。[本次归档](../../archive/repository/20261104/framework-cleanup/README.txt) 保存整理前导航与退役源码原字节；更早迁移见 `archive/repository/`。
旧记录只解释当时的证据，不是当前机器状态或派发许可。新结果写入独立运行目录，不向本页追加逐轮日志。

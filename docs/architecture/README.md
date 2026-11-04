# 框架架构

主线边界：π0.5 / GR00T / RLDX，LIBERO / GR1 tabletop。目标是共享搜索算法和执行基础设施，允许模型原生输入与机器人控制适配不同；不声称所有模型已在两个 benchmark 全部准入。

| 层 | 代码归属 | 允许的职责 |
|---|---|---|
| 公共原语 | `AlphaBrain/common` | 相机数学、图像、动作集成；不依赖研究或 benchmark |
| 冻结模型接入 | `AlphaBrain/model` | 原生特征读取，保留模型输出合同 |
| 研究核心 | `AlphaBrain/research/dsol/{data,metrics,selectors,acquisition}` | 评分、输入合同、共享 Actor/Critic、PPO、搜索交互 |
| 执行基础 | `AlphaBrain/research/dsol/execution` | 账本、所有权、实时许可、进程身份、发布和恢复 |
| benchmark 适配 | `benchmarks/LIBERO`、`benchmarks/Robocasa_tabletop` | 环境、动作、输入、任务成功判据及组合 |
| 分析与报告 | 研究包 `analysis/plotting`、`reports/paper1` | 数值统计与报告组合分离 |
| CLI、测试、治理 | `scripts`、`tests`、`tools` | 参数/组合、无 GPU 默认测试、依赖与来源检查 |
| 历史证据 | `archive`、`docs/dsol_paper1/history` | 按哈希保留，不作为当前运行依赖 |

## 强制依赖方向

核心包不得 import `benchmarks`、`scripts`、`reports` 或 `archive`。
LIBERO 与 GR1 不能相互导入，也不能反向依赖执行脚本。
共同代码放入明确所有者：进程生命周期是 `execution/process_backend.py`，相机数学是 `common/camera_pose.py`；LIBERO 仿真输入是 `benchmarks/LIBERO/simulation_inputs.py`。
`tools/repository/check.py` 对这些边界进行静态检查。

旧脚本中少量兼容导出保留原调用者，它们导入新实现，不复制算法。旧 RoboCasa365 ProbeBackend 仅保留自己的科学准入/命令/结果验收，复用公共进程后端。

## 合同与扩展

- [共享搜索核心](shared_search_core_zh.md)：输入适配与核心结构严格分开，Actor/Critic 独立权重；同形状不同来源拒载。
- [执行维护](distributed_execution_zh.md)：controller/worker/预算/恢复边界。
- [安装与复现](reproduction_zh.md)：源码与 wheel、CPU 测试、环境和发布物。
- [当前就绪状态](framework_readiness_zh.md)：已有证据与真实准入缺口。

新增模型必须声明 feature producer、真实相机/时序、动作与评分合同；新增本体适配控制和观测，不能按模型名称另写 PPO 或改变隐含网络。

## 变更规则

运行配置、评分、奖励、任务/初态/噪声和成功判据不是重构参数。改动开发树前保存在途源码哈希；数值抽取须做对照，生产切换另冻版本并验收。
源码身份统一由 `execution/source_release.py` 收集显式目录的递归 Python 文件清单。它不取代环境、资产和权重哈希。冻结发布需隔离导入，禁止从开发树补缺失模块。

本次已清除目标后端对其他 benchmark/脚本的静态反向导入。旧实验工具仍按其历史 schema 保留，未把整个 AlphaBrain 上游项目或所有历史研究删除。此前长篇架构沿革保存在[维护归档](../../archive/repository/20261104/framework-cleanup/README.txt)。

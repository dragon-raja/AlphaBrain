# 当前框架就绪状态

2026-11-06 更新。整理证据为 `runs/framework-cleanup-20261104/`；LIBERO 主线发布与真实准入见 `runs/unified-libero-mainline-20261106/`。本页本身不是 GPU 派发许可。

## 本轮已处理

| 原问题 | 当前实现 |
|---|---|
| LIBERO 依赖 RoboCasa365 的进程生命周期 | 提取 `execution/process_backend.py`，目标两个后端不再跨 benchmark 导入 |
| 仿真输入反向导入执行脚本 | `benchmarks/LIBERO/simulation_inputs.py` 单一实现；旧脚本只是显式兼容导出 |
| 相机数学/动作集成/旧绘图重复 | `AlphaBrain/common/camera_pose.py`、`action_ensemble.py`、`rollout_visualization.py` 统一所有者 |
| GR00T/RLDX 借用 FastWAM 的原始观测工具 | 原始观测合同移到 LIBERO 输入模块；FastWAM 模型实现隔离在 `legacy/` |
| 两个后端用不同的手写源码哈希方法 | 统一 `execution/source_release.py` 的递归目录 manifest；新增模块自动进入明确源码域 |
| 旧 baseline 静态派发入口和无消费者的重复客户端 | 关闭入口保留最小拒绝壳，完整历史实现归档；不再携带机器路径默认派发 |
| 历史进展混入当前导航 | 当前研究/架构/后端 README 重写为职责导航，原文按哈希保留 |
| benchmark 不随 wheel 安装、CPU CI 跳过网络测试 | wheel 明确包含两个目标后端；增加 search-test extra、独立 CPU Torch CI 和隔离 wheel smoke |
| GR1 实现可读性和任务块恢复 | 模块格式整理、结构化 ports、短块逐调用意图及不明结果恢复；CPU 合同接入实际块执行层 |
| 字体依赖固定系统路径 | 图像报告支持显式字体及默认回退，不再要求单一路径 |

数值原语抽取使用原函数体 AST 对照；另有相机、输入、模型适配、Actor/Critic、PPO 和恢复测试。静态规则禁止目标后端重新导入另一 benchmark 或 scripts/archive。

## 必须保留的真实边界

- **LIBERO 在途运行**：使用共享盘独立冻结源码，维护不热切换。新的开发版本还需按具体科学协议完成运行准入后才能替换生产版本。
- **GR1**：CPU 模型加载、24 任务有限重放、搜索/操作/短块及远端模拟授权有证据。真实生产注册、运行绑定和 GPU 数值/渲染/轨迹/容量准入仍未完成，不能描述成只等 LIBERO 分数或翻一个开关即可运行。
- **GR1 整块组合**：`trial_plan`/`prepared_execution` 先核对所有试验、噪声、任务资源身份及最坏调用上限，再开资源；实际 CPU authority 与真实任务＋模拟模型组合已测试（`gr1-cpu-preparation-20261104/r15`）。`r16` 补齐父子进程逐调用输入/噪声摘要、超时 GO 与结果返回时的 continuation 检查，并以注入 CPU runtime 测试实际模型子进程循环。真实 GPU 子进程 scope、正式注册与运行绑定仍未接通，不能把 CPU 通信验证写成生产工程已结束。
- **LIBERO 最终结构**：U12、无额外 T、G、Q8；三个模型在原生输入适配后使用相同共享核心。RLDX 前语言逐相机捕获已通过同输入动作/flow/RNG、完整轨迹及实际采集器准入。发布合同为主线目录 `core-release.json`，包含初始化与源哈希；旧结构成绩不改标为统一结果。这不替代 GR1/真机验收。
- **输入布局**：共同核心支持一个 movable/joint 主流及最多一个独立辅助流；双独立腕部没有实现。真实缺失由 mask 表示，不伪造相机。
- **分布式能力**：当前共享 PPO 采样/独立评测不是通用 DDP/FSDP 或批推理；这些不是当前三模型两 benchmark 主线的已完成能力。
- **兼容代码**：历史分析与有消费者的旧 schema 保留，明确不属于新主线。原冻结源码仍可能有旧入口，当前拒绝检查不是操作系统级全局隔离。
- **公开发布**：本地提交/安装验证不等于上传权重或公开结果。仍需按具体发布范围核对模型与 benchmark 许可、私有配置和科学发布物。

## 继续工作的顺序

1. 在模拟后端基础上完成 GR1 真实执行组合所需的独立生产 binding 与环境合同；不能借 LIBERO 的许可或冒用 CPU 模拟完成状态。
2. GR1 接已发布共同核心，冻结其任务/初态/相机/场景保持协议，明确空资产例外与调用预算。
3. 在真实资源获准后做最小 GPU 准入和计时，按实测吞吐安排矩阵。
4. 用新发布物切换后续实验；原在途版本自然完成，原结果和预算保留。

[架构](README.md) · [复现](reproduction_zh.md) · [搜索核心](shared_search_core_zh.md) · [研究入口](../dsol_paper1/README.md)

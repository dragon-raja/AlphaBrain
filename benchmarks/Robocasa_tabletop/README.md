# GR1 tabletop 搜索后端

当前准备目标为已冻结的 π0.5 GR1 操作模型；24 个任务的 CPU 资产/重放证据不等于完整 GPU 评测准入。GR00T/RLDX 在该本体上的正式矩阵尚未建立。

| 层 | 文件 |
|---|---|
| 包与动作合同 | `gr1_package.py`、`model_runtime.py`、`task_runtime.py` |
| 独立模型进程 | `model_process.py`：parent 按调用授权，模型与任务环境隔离 |
| 头部运动与观测 | `head_layout.py`、`head_geometry.py`、`head_servo.py`、`head_control.py`、`head_io.py` |
| 原生输入与共享 PPO | `search_runtime.py`、`policy_binding.py`、`episode_runtime.py` |
| 搜索到操作 | `task_composition.py`、`operation_backend.py`、`operation_rollout.py`、`trial_runtime.py` |
| 可恢复短块 | `block_runtime.py`：先记调用意图，失败保留已完成结果和不明消耗，不重发 |
| 整块准备与组合 | `trial_plan.py`、`prepared_execution.py`：全块输入预检、完整预算、噪声哈希与任务/运行身份；`native_resources.py` 组合实际任务和独立模型进程 |
| center 接入 | `continuous_authority.py`：复用共同生命周期，CPU 合同入口；真实派发关闭 |
| 重放 | `task_state.py`、`snapshot_io.py`：状态与合同哈希，无任意 pickle |

## 环境隔离

所核验的上游模型环境使用 JAX 并锁 MuJoCo 2.3.7，任务环境要求 MuJoCo 3.2.6；不能合并覆盖系统环境。模型 CPU 加载与 task CPU 组合分别验收，通过本地进程协议连接。

TaskSpec 显式规定任务、布局、风格、seed、控制频率和 horizon。重放还要求启动时固定 PYTHONHASHSEED，并核对实际 Python hash fingerprint；仅任务 seed 不够。

## 协议和剩余真实验收

`PROJECT_NONHEAD_INTEGRATION_V1` 是显式场景冻结候选，不是自然动力学或真机控制的等价物。四个上游空资产类别必须显式冻结为例外或补齐，不能静默改变采样分布。

真实生产注册、运行绑定与 GPU 数值/渲染/轨迹/容量准入仍需完成；CPU 模拟测试不得冒充这些结果。所有科学参数由未来批准的 manifest 提供，不从 LIBERO 偷用许可、速度或预算。

`synthetic_prepared` 已接入现有 center CPU authority/ledger，可跑两个完整的模拟 trial；36 次为完整搜索与操作上限，固定 STOP 测试实际结算 8 次。`NativeResources` 的 CPU 模式也已通过真实任务物理环境、模拟图像/模型的整块组合检查。原生模式在资源构造前拒绝运行：真实 GPU 模型子进程的授权传播与注册仍需另行实现并验收，不能把 CPU 子进程加载权重的能力当作 GPU 推理已接通。

CPU 测试：`tests/dsol_paper1/runtime/test_gr1_*.py` 与 `tests/dsol_paper1/execution/test_gr1_authority.py`。默认不启动真实仿真或模型。
[复现指南](../../docs/architecture/reproduction_zh.md) · [当前准备证据入口](../../docs/dsol_paper1/README.md)

原上游操作模型训练说明在[维护归档](../../archive/repository/20261104/framework-cleanup/README.txt)；本项目不以重训操作模型作为搜索后端接入的前置要求。

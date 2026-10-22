# Paper 1 可复用研究代码

- `data/`：产物 IO、身份核查、标准初态矩阵读取与显式 Flow 噪声库。
- `metrics/`：Accel 定义与推理适配、任务实体可见性、候选视角指标规则。
- `selectors/`：共享 attention actor/critic、PPO，以及历史视觉排序器。
- `analysis/`：Oracle、配对统计、结果汇总、配置驱动分析流水线。
- `plotting/`：实测相机空间、成功率矩阵等可复用绘图。
- `acquisition/`：当前 RL 搜索环境、动态局部候选及连续位姿接口；真实头部适配尚未完成。
- `execution/`：冻结任务身份、跨主机资源槽动态分配、持久状态、失败收尾和显式恢复；不导入模型或仿真。运行细节由显式后端适配器提供。
- `protocol.py`：此前已存在的协议校验，保留接口。

推荐入口：

```bash
python -m AlphaBrain.research.dsol.analysis.pipeline \
  --config configs/dsol_paper1/analysis/standard_initial_v1.json
```

该命令只做离线核查与分析，不启动闭环评测。路径在配置中，核心实现不依赖作者机器。

报告由 `reports/paper1/` 消费结果；薄分析入口在 `scripts/dsol_paper1/analysis/`，原脚本根目录不保留转发文件。

独立恢复确认：`analysis/confirmation_results.py` 只消费验收后的冻结选位及执行结果，
输出窗口1–5、规范和32视角扰动基线的配对统计；主窗口不随结果改变。
RoboCasa原生复位、噪声地址及完整任务验收属于 `benchmarks/Robocasa365/confirmation_audit.py`，
使用CPU读取已有证据，不启动仿真或GPU。报告发布位于 `reports/paper1/confirmation_update.py`。

历史分析、协议构造、训练、诊断与控制器已按职责迁出脚本平铺区，详见 `scripts/dsol_paper1/README.md`。这属于可维护性整理，不意味着所有历史工具都已抽象成通用算法接口。

实际仿真适配在 `benchmarks/`，薄运行入口在 `scripts/dsol_paper1/runtime/`。新通用调度实现位于 `execution/`，标准入口是 `scripts/dsol_paper1/operations/manage_work_queue.py`；`operations/controllers/` 保留历史兼容控制器。新布局不沿用旧源码身份；正式运行前仍需协议等价与轨迹验收，旧冻结实验使用自身的源码快照。接口、恢复和当前接入边界见 [执行管理协议](../../../../docs/dsol_paper1/protocols/execution_management_zh.md)。

## 当前 RL 搜索代码

- `acquisition/search_env.py` / `local_view_search.py`：固定表和动态局部搜索，查询计费与停止。
- `selectors/search_attention.py` / `search_attention_ppo.py`：共享编码、attention、actor/critic 与 PPO。
- `analysis/search_attention_training.py`：已有训练流程；`data/local_search_plan.py`：下一轮划分/预算核查。
- `analysis/search_task_alignment.py` / `execution_trace_audit.py`：配对任务统计、回放边界、已有执行记录核查。
- `benchmarks/LIBERO/search_runtime.py` / `search_simulator.py` / `continuous_camera.py`：仓库外层的模型/仿真适配。

旧 97 点训练和动态候选设计不可混称为同一个已完成方法。当前状态见[项目首页](../../../../docs/dsol_paper1/README.md)。

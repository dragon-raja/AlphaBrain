# 相机获取工程原语

- `search_env.py`：标签隔离的有限预算缓存搜索环境、简单对照与可选 Gymnasium 桥接。
  当前只完成 CPU 接口/数据验证，未训练、未验证头部运动；设计与预算见
  `docs/dsol_paper1/planning/rl_search_interface_20261020_zh.md`。

- `motion.py`：位姿、运动边界、轨迹与预算。
- `inputs.py`：允许传入选择器的观测与输入约束。
- `flow_trace.py`：只读记录原生Euler流采样的中间状态与速度，不增加模型调用或改动作；`TokenTimeFlowTrace`另支持显式浮点逐动作token时间网格，拒绝RTC前缀时间混入和嵌套读取。RNG隔离、实际执行窗口、动作有效维度和评分定义由调用方显式提供；此读取器仅用于串行eager、无引导的采样，不支持CUDA graph重放。CPU模拟采样器或原版循环配合合成部件的测试不等于真实模型GPU准入。
- `executor.py`：端点获取执行、匹配等待、控制器适配。
- `protocol.py`：获取实验协议声明。
- `observation_loop.py`：新主动观察框架的纯 CPU 事件状态转换，记录实际获取/失败、操作历史、时钟与噪声索引；不包含仿真或训练。数据协议在 `data/active_observation.py`，可替换规划接口在 `selectors/observation_planning.py`，统一设计见 `docs/dsol_paper1/planning/outcome_active_observation_20260918_zh.md`。
- `discrete_search.py`：有限候选目录上的查询受限离线图搜索；只向策略暴露已访问点分数，记录返回所选终点的路径。
- `image_geometry.py`：局部图像—相机运动雅可比与任意可微标量指标的梯度组合；已用于独立 Pi0.5 局部扰动诊断，不是搜索或运动控制器。真实适配器位于 `metrics/pi05_differentiable.py`，纯诊断分析位于 `analysis/local_view_derivative.py`，执行环境桥接位于 `scripts/dsol_paper1/runtime/local_view_diagnostic.py`。

实际相机获取原语尚未进入当前固定外部相机闭环评测。`discrete_search.py` 已用于静态矩阵离线重放，但不证明实际移动可达、实时成本或主动相机净收益；当前闭环评测不调用这里。

对应测试在 `tests/dsol_paper1/acquisition/`；显式工程 smoke 入口位于 `scripts/dsol_paper1/operations/controllers/run_view_acquisition_engineering_smoke.py`，不属于默认测试或自动启动流程。离线研究通过 `AlphaBrain.research.dsol.analysis.view_recovery` 运行，不启动模拟器。

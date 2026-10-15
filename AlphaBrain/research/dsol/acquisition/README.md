# 相机获取工程原语

- `motion.py`：位姿、运动边界、轨迹与预算。
- `inputs.py`：允许传入选择器的观测与输入约束。
- `executor.py`：端点获取执行、匹配等待、控制器适配。
- `protocol.py`：获取实验协议声明。
- `observation_loop.py`：新主动观察框架的纯 CPU 事件状态转换，记录实际获取/失败、操作历史、时钟与噪声索引；不包含仿真或训练。数据协议在 `data/active_observation.py`，可替换规划接口在 `selectors/observation_planning.py`，统一设计见 `docs/dsol_paper1/planning/outcome_active_observation_20260918_zh.md`。
- `discrete_search.py`：有限候选目录上的查询受限离线图搜索；只向策略暴露已访问点分数，记录返回所选终点的路径。
- `image_geometry.py`：局部图像—相机运动雅可比与任意可微标量指标的梯度组合；已用于独立 Pi0.5 局部扰动诊断，不是搜索或运动控制器。真实适配器位于 `metrics/pi05_differentiable.py`，纯诊断分析位于 `analysis/local_view_derivative.py`，执行环境桥接位于 `scripts/dsol_paper1/runtime/local_view_diagnostic.py`。

实际相机获取原语尚未进入当前固定外部相机闭环评测。`discrete_search.py` 已用于静态矩阵离线重放，但不证明实际移动可达、实时成本或主动相机净收益；当前闭环评测不调用这里。

对应测试在 `tests/dsol_paper1/acquisition/`；显式工程 smoke 入口位于 `scripts/dsol_paper1/operations/controllers/run_view_acquisition_engineering_smoke.py`，不属于默认测试或自动启动流程。离线研究通过 `AlphaBrain.research.dsol.analysis.view_recovery` 运行，不启动模拟器。

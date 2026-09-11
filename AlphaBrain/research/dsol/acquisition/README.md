# 相机获取工程原语

- `motion.py`：位姿、运动边界、轨迹与预算。
- `inputs.py`：允许传入选择器的观测与输入约束。
- `executor.py`：端点获取执行、匹配等待、控制器适配。
- `protocol.py`：获取实验协议声明。

这是尚未进入当前固定外部相机评测的工程基础，不是已验证的主动感知方法，也不代表论文已获得主动相机收益。当前闭环评测不调用这里。

对应测试在 `tests/dsol_paper1/test_view_acquisition_*.py`；显式工程 smoke 入口位于 `scripts/dsol_paper1/operations/controllers/run_view_acquisition_engineering_smoke.py`，不属于默认测试或自动启动流程。

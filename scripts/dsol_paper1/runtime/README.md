# LIBERO 运行适配

本层连接具体仿真环境、数据恢复、策略输入与闭环评测，不定义可复用的 Accel 或噪声算法。

- `evaluate_dsol_libero_hdf5_views.py`：闭环评测适配；`standard_initialization_v1.py`：标准初态初始化与输入核对。
- `audit_libero_hdf5_restore.py`、`libero_constructed_view.py`、`scan_libero_hdf5_views.py`：HDF5 恢复、场景／相机适配、候选扫描。
- `run_view_value_expectation_accel_ensemble.py`、`run_matched_view_accel_v1.py`：评分任务的环境、模型和产物编排。
- `shared_runtime_paths.py`、`render_bridge_common_v1.py`：运行来源身份与历史渲染协议适配。

通用实现分别在 `AlphaBrain/research/dsol/metrics/` 和 `data/flow_noise.py`，不得再复制到这里。控制器位于 `../operations/controllers/`，Shell 配方位于 `../operations/launchers/`。

版本边界：此处代码已完成目录和导入迁移，不等同于通过了新源码版本的 GPU 整轨迹验收。旧 release 的固定路径／哈希必须由对应冻结 checkout 满足；不得重新计算旧哈希掩盖版本变化。正式运行前按研究协议执行独立一致性验收，不在默认测试中启动仿真或模型。

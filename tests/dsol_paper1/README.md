# Paper 1 测试导航

测试按被验证的职责归类，而不是全部平铺在论文目录。文件统一采用 `test_*.py` 命名。

| 目录 | 验证对象 |
| --- | --- |
| `data` | 数据 schema、初始化、相机输入与显式噪声 |
| `metrics` | Accel、指标计算、指标规则与评分接口 |
| `selectors` | 选择器与候选筛选 |
| `analysis` | 成功率统计、训练条件对比与数值回归 |
| `protocols` | 候选空间、噪声预算、评测协议与冻结包 |
| `execution` | 调度器、运行控制器及恢复逻辑；通过 mock 验证，不启动正式任务 |
| `diagnostics` | 渲染／轨迹一致性、来源身份和审计 |
| `acquisition` | 尚未作为正式研究结论的相机获取工程原语 |
| `reporting` | 图表与报告生成、研究范围表述 |
| `architecture` | 模块边界、迁移证据、目录约束 |
| `helpers` | 共享合成测试样例，不导入其他测试模块 |

`paths.py` 统一提供仓库根目录，新增测试不再通过自身文件层数推测仓库位置。`conftest.py` 仅配置测试导入环境，不启动模型、GPU 或仿真器。

从仓库根目录运行单模块，例如：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/dsol_paper1/metrics -q
```

完整环境和便携 CI 的区别见 [测试总入口](../README.md)。历史协议测试保留原科学语义；按目录归类并不代表历史结果属于当前论文的有效证据。

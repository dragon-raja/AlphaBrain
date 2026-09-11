# Paper 1 可复用研究代码

- `data/`：产物 IO、身份核查、标准初态矩阵读取与显式 Flow 噪声库。
- `metrics/`：Accel 定义与推理适配、任务实体可见性、候选视角指标规则。
- `selectors/`：冻结视觉编码、特征处理、开发折选择、岭回归排序器。
- `analysis/`：Oracle、配对统计、结果汇总、配置驱动分析流水线。
- `plotting/`：实测相机空间、成功率矩阵等可复用绘图。
- `acquisition/`：获取运动、输入和执行工程原语，尚非当前评测方法。
- `protocol.py`：此前已存在的协议校验，保留接口。

推荐入口：

```bash
python -m AlphaBrain.research.dsol.analysis.pipeline \
  --config configs/dsol_paper1/analysis/standard_initial_v1.json
```

该命令只做离线核查与分析，不启动闭环评测。路径在配置中，核心实现不依赖作者机器。

报告由 `reports/paper1/` 消费结果；薄分析入口在 `scripts/dsol_paper1/analysis/`，原脚本根目录不保留转发文件。

历史分析、协议构造、训练、诊断与控制器已按职责迁出脚本平铺区，详见 `scripts/dsol_paper1/README.md`。这属于可维护性整理，不意味着所有历史工具都已抽象成通用算法接口。

实际仿真适配在 `scripts/dsol_paper1/runtime/`，运行调度在 `operations/controllers/`。新布局不沿用旧源码身份；正式运行前仍需协议等价与轨迹验收，旧冻结实验使用自身的源码快照。

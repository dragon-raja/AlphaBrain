# Paper 1 可复用研究代码

- `data/`：产物 IO、身份核查、标准初态矩阵读取。
- `metrics/`：候选视角指标规则。
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

报告由 `reports/paper1/` 消费结果；旧 `scripts/dsol_paper1/analyze_standard_initial_results_v1.py` 等仅作兼容入口。

历史分析、协议构造、训练、诊断与控制器已按职责迁出脚本平铺区，详见 `scripts/dsol_paper1/README.md`。这属于可维护性整理，不意味着所有历史工具都已抽象成通用算法接口。

尚未迁移的范围：实际仿真闭环、噪声底层及活动控制器所需的 37 个入口／依赖保持原路径与原字节。进一步拆分需要独立源码版本、协议等价与轨迹验收。

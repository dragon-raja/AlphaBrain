# LIBERO 搜索后端

当前三模型主线：π0.5、GR00T N1.5、RLDX-1。共享核心与 PPO 在 `AlphaBrain/research/dsol`，不在这里复制网络或训练算法。

| 接入点 | 职责 |
|---|---|
| `continuous_authority.py` | 绑定校验，受控科学入口，复用共享 model_agent/model_plan |
| `prefix_campaign.py` / `prefix_continuation.py` / `prefix_evaluation.py` | 实验阶段组合及匹配预算的训练/评测 |
| `model_adapters.py` | 目标模型适配表；π0.5 既有入口在 search_runtime |
| `native_feature_runtime.py` / `native_view_source.py` | 原生特征合同和采集 |
| `simulation_inputs.py` / `initialization.py` / `continuous_camera.py` | 仿真输入、初态与相机；不反向 import scripts |
| `search_simulator.py` / `simulator_session.py` | 显式环境构造与生命周期 |
| `legacy/` | 排除主线的历史模型兼容，不作为新实验默认选择 |

安装与无 GPU 验证见[复现指南](../../docs/architecture/reproduction_zh.md)。权重、任务数据、环境与运行许可须使用具体冻结清单，安装代码不授予执行许可。

`baseline_dispatch.py` 是关闭的历史入口；其实现与两个旧静态运行脚本已归档。原上游训练/评测使用说明及早期接入状态保存在[维护归档](../../archive/repository/20261104/framework-cleanup/README.txt)。历史默认成功率不等于本论文扰动条件下的 Stay。

当前状态、原始结果与论文入口见[Paper 1](../../docs/dsol_paper1/README.md)。开发树改动不自动应用到在途 frozen source。

# KYC × Pi0.5 视觉对齐执行状态

更新时间：2026-08-03

## 已完成

- 三种子场景线索 × 腕部相机确认实验已完成；
- wrist-on 的 Control/KYC 成功率约 30%，KYC 无稳定增益；
- wrist-off 的所有方法成功率约 0–1%，判为 baseline invalid；
- 新增 SigLIP MLP rank-16 原生低秩适配，共 54 层、4,713,984 参数；
- 基础 SigLIP 权重键名保持不变，adapter 键进入自包含 checkpoint；
- 视觉基础权重、语言模型和多模态 projector 保持冻结；
- 两步真实 GPU 训练成功；54 个 `adapter_B` 张量全部得到非零更新；
- 自包含 checkpoint 在 `strict_checkpoint=True` 下完整重载成功；
- 视觉适配、相机分支与统计脚本共 11 项相关测试通过；
- LIBERO-Plus 代码固定在 `/projects/LIBERO-plus`，commit
  `4976dc30028e805ff8094b55501d532c48fec182`。

## 正在执行的筛选

条件：随机场景线索、wrist-on、10 个全局相机训练位姿、seed 41。

| 方法 | updates | GPU | 关键差异 |
|---|---:|---:|---|
| PoseAug-RGB+FLA | 2,000 | 2 | 无 ray |
| PoseAug-Control+FLA | 2,000 | 3 | 固定默认 ray |
| KYC+FLA | 2,000 | 4 | 实时匹配 ray |

每个方法使用相同的 5 个 test snapshot group、4 条任务边和 7 个训练支持边界内
相机位姿，共 140 个闭环 episode。KYC 另做相同观测下正确、默认和错配 ray 的
动作因果诊断。

## 进入完整训练的门槛

- RGB/Control 中至少一个完整任务成功率达到 20%；
- KYC+FLA 相对 Control+FLA 提升至少 5 个百分点；
- 错配 ray 相对正确 ray 的动作块 RMS 至少为 0.005；
- 默认视角能力没有明显退化。

未通过时不自动增加训练步数或 seeds；先区分 baseline invalid、视觉适配有效但
KYC 无增量，以及真实几何已被模型因果使用三种情况。

## LIBERO-Plus 资源

代码已经到位。以下大文件尚未通过代理下载：

- `assets.zip`：6,395,849,578 bytes；
- `libero_plus_camparam_rlds.zip`：16,607,835,331 bytes。

目标目录：

```text
/share/longjunyu/alphabrain/datasets/libero-plus/archives/
```

Hugging Face 直连测试超时，代理测试可用。为避免消耗约 23 GB 代理流量，等待
平台下载或跨机器传输。

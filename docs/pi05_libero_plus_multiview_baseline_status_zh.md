# Pi0.5 × LIBERO-Plus 强多视角基线：执行状态

## 研究问题

在官方 Pi0.5 已出现明显视角缺口（canonical 98.8%，官方相机扰动 80.0%）的前提下，检验：

1. 仅使用多视角成功示范训练动作专家，能否缩小视角泛化缺口；
2. 允许 SigLIP 视觉编码器进行低秩适配，是否比冻结视觉编码器更有效；
3. 固定训练算力时，25% 与 100% 可采样数据池的视角多样性是否产生可测差异。

这一步只建立强基线，不引入 KYC、Ray Map、相机选择、主动感知或新的方法模块。

## 数据

- 来源：LIBERO-Plus Camera Parameter RLDS v1。
- 成功 episode：2,876。
- 动作窗口：449,053。
- 任务：40。
- 精确外部相机位姿组：1,178。
- 图像：256×256 agent view 与 wrist view，训练时由 Pi0.5 处理到 224×224。
- 动作：7 维，预测长度 10。

按完整 4×4 外部相机位姿矩阵分组后做 80/10/10 划分；同一相机位姿组不会跨 train/val/test：

| Split | Episode | 窗口 | 相机位姿组 | 覆盖任务 |
|---|---:|---:|---:|---:|
| Train | 2,303 | 361,006 | 948 | 40/40 |
| Val | 293 | 45,471 | 121 | 40/40 |
| Test | 280 | 42,576 | 109 | 40/40 |

25% 训练数据池包含 580 个 episode、90,474 个窗口，并保持 40 个任务全覆盖。

## 训练对照

| 名称 | 数据池 | 视觉编码器 | 动作专家 | 训练参数量 |
|---|---:|---|---|---:|
| `action_b100` | 100% | 冻结 | 微调 | 693.37M |
| `action_b025` | 25% | 冻结 | 微调 | 693.37M |
| `visual_b100` | 100% | SigLIP LoRA rank 16 | 微调 | 698.08M |
| `visual_b025` | 25% | SigLIP LoRA rank 16 | 微调 | 698.08M |

四组统一使用：

- Pi0.5 LIBERO PyTorch 初始化；
- seed 41；
- 33,000 次 optimizer update；
- batch size 1、gradient accumulation 2；
- AdamW 与统一 cosine 学习率计划；
- agent view + wrist view；
- 不输入相机内外参；
- 不使用 KYC/CABI/Ray Map。

每组约消费 66,000 个窗口。100%/25% 指可采样数据池，而非完整遍历一轮；因此本轮回答的是固定算力下的数据多样性效应。

## 已完成链路验证

- TFRecord 随机访问与 4-worker DataLoader：通过，约 155 sample/s。
- Action-only 真实 GPU 训练：通过，约 2.17 update/s。
- 视觉 LoRA 真实 GPU 训练：通过，约 1.54 update/s。
- 自包含 17.6GB checkpoint 保存与严格重载：通过。
- AlphaBrain checkpoint 到 OpenPI WebSocket 协议：通过。
- LIBERO-Plus 真实闭环 episode：通过；2-step smoke checkpoint 在首个 canonical episode 成功。
- 视频：WebM 容器、AV1 编码；不再输出旧 MP4 编码。

Smoke 只证明接口正确，不用于判断模型效果。

## 正式评测

第一阶段对四组模型分别运行：

- 40 个基础任务；
- 每个任务 2 个相同初始状态；
- canonical 与官方相机扰动成对执行；
- 共 160 episode/模型；
- 固定 replan K=5、固定 flow seed、相同 episode seed。

统计单位是 40 个 `suite × 基础任务`，先在任务内平均两个初始状态，再进行 paired group-level bootstrap 95% CI。

主要指标：

- canonical full-task success；
- official-camera full-task success；
- 视角泛化缺口；
- 相对官方 Pi0.5 的绝对百分点变化；
- canonical 退化是否超过 5 个百分点。

完成四组门控后，仅对相机扰动成功率最高的模型运行 480 个候选视角 episode，并与其 gap 结果合并生成最终视角优化/主动感知量化报告。

## 自动执行

- 训练会话：`plus-mv-a100-s41`、`plus-mv-a025-s41`、`plus-mv-v100-s41`、`plus-mv-v025-s41`。
- 自动评测接力：`plus-mv-gate-eval`。
- 训练输出：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs`。
- 评测输出：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/gate-v1`。
- 闭环 smoke：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/eval-smoke-action2-v1`。

自动接力只在四组 checkpoint 完整写入且训练会话退出后开始评测，避免读到尚未写完的权重文件。

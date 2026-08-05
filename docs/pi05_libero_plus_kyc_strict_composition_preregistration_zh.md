# Pi0.5 多视角、KYC 与严格相机—背景组合验证预注册

## 1. 核心问题

本阶段只回答两个相互独立的问题：

1. **模型增量问题**：Pi0.5 已使用随机多视角 RGB 训练并保留腕部相机时，输入当前真实相机几何的 KYC 是否仍比等容量 Control 提供稳定闭环收益？
2. **数据充分性问题**：相机因素和背景因素都在训练中出现、但特定相机×背景配对被留出时，强多视角 RGB 是否仍存在组合泛化缺口？

不能用第一个问题的结果替代第二个问题，也不能把当前相机训练加未见背景的联合域外测试称为严格组合泛化。

## 2. 已有证据直接复用

- LIBERO-Plus 已用大规模配对实验证明相机与背景同时变化比单因素更困难，并报告相机和背景之间的显著交互；不重复验证“联合变化更难”。
- LIBERO-Plus 的 mix-SFT 已证明扩充扰动数据能显著提高 OpenVLA-OFT 的单因素相机与背景鲁棒性；它没有给出 Pi0.5、KYC 或 mix-SFT 后的相机×背景联合结果。
- KYC 已在 ACT、Diffusion Policy 和 SmolVLA、无腕部相机条件下提供匹配比较，并研究训练相机数量；不重复官方正对照。原文报告更多训练相机能缩小纯 RGB 与 KYC 的差距，但在其 Pick scaling 中仍保留小幅 KYC 增益。
- 现有本地 Pi0.5 结果已证明随机多视角 RGB 是主要增益，固定场景下 raw-ray KYC 没有稳定超过 Control；新实验只补 Pi0.5、视觉 LoRA、腕部相机保留、相机×背景联合变化这一缺口。
- 本地 `scene cue × wrist` 因子实验已完成三随机种子，不再重跑。保留腕部时，固定场景 Control/KYC 为 `34.34%/31.03%`，视觉线索随机场景为 `30.03%/29.60%`，均无稳定 KYC 增量；关闭腕部后两组成功率均不超过 `1.15%`，属于基线失效，不能用于裁决 KYC。机器结果位于 `/share/longjunyu/cabi-vla/kyc-scaling-v3/eval/factorial/n10/analysis/confirmed/summary.json`。

KYC 原文明确关闭腕部相机，以隔离第三方相机条件化；LIBERO-Plus 则发现腕部相机提供关键的近场几何与接触线索。当前门控保留腕部 RGB，但只给第三方相机分支真实/规范 ray，因而检验的是：已有腕部线索和强多视角训练后，第三方显式几何是否仍有独立增量。此前本地 Dual-KYC 与正式腕部关闭因子实验都因低基线而只作为边界证据复用，不重复扩展腕部 ray 变体。

## 3. 实验 A：完全匹配的 KYC 增量门控

### 训练

| 项目 | Control | KYC |
|---|---|---|
| 多视角训练数据 | 相同 25% 数据 | 相同 25% 数据 |
| Pi0.5 初始化 | 相同 | 相同 |
| 视觉适配 | SigLIP LoRA rank 16 | SigLIP LoRA rank 16 |
| 相机分支 | 开启，规范位姿 ray | 开启，当前真实位姿 ray |
| 融合 | residual-zero | residual-zero |
| 更新步数 | 33,000 | 33,000 |
| seeds | 41、42、43 | 41、42、43 |

两组唯一差异为 ray 使用规范相机位姿还是与当前 RGB 对应的真实位姿。

### 闭环评测

每个 checkpoint 使用相同 36 个基础任务、2 个初态和四个条件，共 288 条 episode：

- 原始相机 + 原始背景；
- 扰动相机 + 原始背景；
- 原始相机 + 新背景；
- 扰动相机 + 新背景。

同一四条件组共享任务、语言、物体布局、机器人初态和物理状态。KYC 的相机内外参由当前 MuJoCo 相机逐 episode 计算并随每次推理请求传入。

主比较为 KYC - Control 的相机条件和相机+背景条件成功率，以及两类缺口缩小量。基础任务是独立统计单位；先在种子内聚合初态，再同时重采样训练种子与基础任务，做 20,000 次 crossed paired bootstrap，避免忽略训练随机性。

### 门控

- `KYC_INCREMENTAL_VALUE_CONFIRMED`：相机或联合条件至少提高 5 个百分点，任务级 95% CI 下界大于 0，且原始条件退化不超过 5 点。
- `KYC_NO_MEANINGFUL_INCREMENTAL_VALUE`：相机和联合条件的 95% CI 上界均小于 5 点。
- 其余结果标为不确定或退化，不以单 seed 或离线损失裁决。

## 4. 实验 B：见因子、留组合门控

### 4.1 最强版本：精确因子配对

先审计 LIBERO-Goal 官方逐套件数据。只有同时满足以下条件，才允许把结果称为“精确相机位姿×背景纹理留组合”：

1. 每条示范可恢复明确的相机位姿 ID 与背景/桌面 ID；
2. 每个测试相机在训练中至少与其他背景出现；
3. 每个测试背景在训练中至少与其他相机出现；
4. 测试相机×背景配对在训练中出现次数为 0；
5. 任务身份、初态组和示范来源按组隔离，无帧级泄漏；
6. RGB、动作、状态和相机标定能被当前 Pi0.5 数据链路无损读取。

推荐训练集合为 canonical、camera-only 和 background-only，留出 noncanonical-camera × noncanonical-background。

全量归档审计已经确认：16.64 GB Goal RLDS 完整且包含 256 个 TFRecord 分片，但其 episode 字段没有逐条相机位姿或背景纹理 ID；来源标识只保留 `camera_view`、`env` 等扰动类别和基础任务。因此该公开归档不能证明具体测试位姿和纹理分别在训练边缘分布中出现，也不能支持最强的精确配对声明。此门控结论固定为：

`STRICT_EXACT_PAIR_COMPOSITION_DATA_INVALID`

这不表示数据不可用，而是限制结论的粒度。

### 4.2 可执行版本：因子类别分离组合

补充实验采用较弱但仍严格隔离联合条件的数据设计：

- 相机单因素：使用含逐 episode 标定的官方 camera-view RLDS；
- 背景单因素：使用 Goal 归档中来源类别为 `env` 的轨迹，并按规范外部相机标定；
- 训练中不包含任何同时改变相机和背景的 episode；
- 评测仅在 9 个 LIBERO-Goal 基础任务上进行，使用原始、相机单因素、背景单因素和相机+背景四条件；
- Control 与 KYC 使用相同的按任务×因子分层 25% 数据、Pi0.5 初始化、视觉 LoRA、33,000 步和 seeds 41/42/43；
- 每条评测保存 AV1/WebM，统计单位为基础任务，并使用种子×任务 crossed paired bootstrap。

已构建的数据视图含 1,417 个 episode、172,459 步、9 个任务：相机单因素 657 条，背景单因素 760 条；训练 split 为 518/610 条，正式 25% 视图为 131/154 条（15,233/18,975 步），训练联合相机+背景 episode 数为 0。机器清单位于 `/share/longjunyu/alphabrain/datasets/libero-plus/views/pi05-goal-factor-separated-v1/manifest.json`。

该结果只能称为 `FACTOR_SEPARATED_CATEGORY_COMPOSITION`。它可以回答“分别见过两类变化后，联合变化是否仍失败，以及 KYC 是否减轻该失败”，但不能回答“精确见过测试位姿和测试纹理的各自边缘实例后，未见配对是否失败”。后者需要重新采集带参数标识的数据。

## 5. 结论边界

即使两个门控均通过，也只说明静态第三方相机与桌面/背景外观范围内的结果；不能外推到新厨房几何、新物体布局、执行中移动相机或真机标定误差。RoboCasa 的厨房/布局组合验证仍作为外部有效性证据独立进行。

## 6. 主要依据

- [KYC 论文](https://arxiv.org/abs/2510.02268)、[项目页](https://ripl.github.io/know_your_camera/)与[官方代码](https://github.com/ripl/CamPoseOpensource)
- [LIBERO-Plus CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)与[官方仓库](https://github.com/sylvestf/LIBERO-plus)
- [VLA Models Are More Generalizable Than You Think](https://arxiv.org/abs/2512.02902)：作为视觉适配近邻，但其单次适配实验不替代本预注册的 KYC 匹配比较或严格因素配对。

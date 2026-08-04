# 4×A800 视角泛化研究总交接 Prompt

> 使用方式：将本文全文直接交给 4×A800 开发机会话。本文同时包含研究背景、已完成证据、8×RTX 5090 机器现有资源、RoboCasa 验证要求、跨机协作规则和接手后的首轮输出格式，不需要再附带其他交接文档。

## 0. 安全与操作边界

你正在接手 AlphaBrain 中与相机视角、相机几何和视角泛化有关的研究。必须遵守：

- 不读取、打印或泄露任何 API key、token、`auth.json`、HF token、SSH 私钥或代理配置中的敏感字段；
- 不删除、覆盖或重置用户已有代码、数据、checkpoint 和 Git 改动；
- 修改代码前先检查 Git 分支与 dirty 状态；
- 不使用 `git reset --hard` 或会丢失改动的 checkout；
- 不自动下载、复制或跨区域传输大模型和大数据集；
- 先盘点本机已有资源，再列出缺口，由用户决定是否通过 Mac 中转；
- 视频统一保存为 VS Code/浏览器可播放的 AV1/WebM，必要时附 H.264 兼容副本；
- 所有结论必须区分“已有证据”“新的实验结果”和“尚待验证的假设”。

## 1. 你的角色

4×A800 会话是本视角方向的研究主导者，负责：

- 独立完成问题定义、文献近邻审计和算法构思；
- 把本文提供的实验结果作为先验，而不是照抄一套固定方法；
- 使用本机 RoboCasa/RoboCasa365 资源做主要外部验证；
- 决定最终模型、训练方法、数据设计和实验矩阵；
- 完成 Pi0.5 训练、闭环评测、统计分析和继续/转向/终止判断；
- 在已有研究方向与本文证据冲突时，明确指出冲突，并用实验区分。

8×RTX 5090 会话作为支持端，已经完成 LIBERO/LIBERO-Plus 的一批诊断，能够提供小规模复现、代码补丁、统计、图表和已有资产。不要假定 4 卡机必须沿用 8 卡机提出的候选算法。

研究范围覆盖所有真正与“视角”有关的问题，包括但不限于：

- 第三方相机、腕部相机及二者联合扰动；
- 固定视角、episode 级随机视角和执行过程中动态移动相机；
- 相机内参、外参、标定误差、未知位姿和缺失相机；
- 多相机融合、相机 dropout、腕部/外部视角依赖；
- 相机与背景、厨房布局、物体、任务之间的组合泛化；
- 低数据量多视角学习和训练分布外的相机位姿外推；
- KYC、ray map、Plücker ray、几何等变表示、跨视角特征对齐；
- 冻结 VLM、adapter、LoRA、视觉骨干部分/完整微调的差异；
- 视角优化和主动感知，但只有在 oracle 或受控实验显示存在可恢复收益时才作为主方向；
- LIBERO-Plus、KYC/RoboSuite、RoboCasa365 以及后续跨环境验证。

Pi0.5 是当前主要模型与强基线，但新方法可以来自 VLA、VLM、视觉表征、几何归纳偏置、训练目标或数据机制，只要它解决的是经验证仍然存在的视角能力缺口。

## 2. 当前核心科学问题

已知相机变化会显著影响策略，但“随机多视角 RGB 训练”本身可能已经解决一部分同仿真器、同场景内的相机插值。因此真正需要回答的是：

> 在强多视角 RGB 训练之后，面对训练位姿支持之外的新相机、腕部与外部相机联合变化、新厨房/背景/布局以及相机与场景组合变化时，是否仍存在稳定的视角泛化缺口？如果存在，模型级几何或表征方法能否在鲁棒性、位姿外推或数据效率上稳定超过纯 RGB 多视角训练和 KYC？

这不是预设答案。若强多视角数据已经充分解决问题，应停止强行发明相机模块，转向仍有证据支持的子问题。

## 3. 已停止的路线

以下路线已有负面裁决，不要作为主项目重新启动，除非出现新的直接证据：

- FRESH suffix loss weighting：`STOP_TRAINING_WEIGHTING_ROUTE`；
- CORA routing/candidate-support；
- tail weighting、动态 horizon、random/shuffled weighting 和 recovery-support 扩展；
- 围绕恢复、重规划、重采样、world model 或 RL 的旧路线；
- 仅使用任务开始时动作方差或图像质量分数做主动视角选择。

## 4. 8 卡机已经得到的视角证据

### 4.1 原始 LIBERO 相机敏感性

只改变外部 `agentview` 相机，保持任务、物理初始状态、腕部相机、语言和机器人状态不变：

- canonical 视角成功率为 75%；
- 相机半径缩放为 0.925 时降至 30%；
- 相机半径缩放为 1.075 时降至 50%；
- 早期四任务测试中，水平偏航角变化 ±30° 时从 4/4 降至 0/4；
- 一些失败样本中目标仍清晰可见，因此问题不只是物体离开视野。

结论：Pi0.5 对外部相机位姿存在真实敏感性，但这组实验没有证明多视角训练后问题仍然存在。

### 4.2 Pi0.5 多视角训练与 KYC 对照

各方法使用匹配的随机多视角 RGB 数据：

| 方法 | 训练视角 | 相机几何分支 | 成功率 |
| --- | --- | --- | ---: |
| Base | 仅 canonical | 无 | 12.89% |
| PoseAug-RGB | 随机多视角 | 无 | 42.03% |
| PoseAug-Control | 相同随机多视角 | 固定/默认 ray | 44.49%（三 seed） |
| KYC | 相同随机多视角 | 与图像对应的真实 ray | 43.33%（三 seed） |

- `KYC - Control = -1.16` 个百分点；95% CI 为 `[-7.03, +4.43]`；
- 固定 RGB、只替换 ray map 时，动作 chunk RMS 变化仅 0.00071；
- 说明当前 Pi0.5 raw-ray adapter 几乎没有使用动态几何输入；
- 这不等于 KYC 论文无效：官方 KYC ACT/RoboSuite 复现从 24.67% 提升到 62.67%，增加 38 个百分点。

合理解释是：

- 随机多视角 RGB 是当前观察到的主要增益来源；
- 固定 LIBERO 场景让模型能从 RGB 猜测相机位置；
- 保留腕部相机可能降低显式外部相机位姿的边际价值；
- 当前视觉骨干/adapter 的训练方式也可能没有让 Pi0.5 学会使用 ray；
- 因此必须在更强场景变化、腕部相机消融和不同视觉适配强度下重新判断，而不能宣布“KYC 没用”。

### 4.3 LIBERO-Plus 零样本闭环视角缺口

使用 Pi0.5-LIBERO PyTorch checkpoint、外部 RGB + 腕部 RGB、robot state、action horizon 10、固定执行 `K=5`：

- 40 个基础任务：Spatial、Object、Goal、Long 各 10 个；
- 每任务两个物理初始状态；
- 相机差距矩阵 160 个闭环 episode；
- 六候选视角矩阵 480 个闭环 episode；
- 合计 640 个完整 episode。

结果：

- canonical：98.75%（79/80）；
- LIBERO-Plus 官方相机扰动：80.00%（64/80）；
- 成对绝对下降 18.75 个百分点，95% CI `[10.00, 28.75]`；
- difficulty-5 条件下降 50 个百分点；
- 在两个视角中关键物体都清晰可见的 66 对样本上，仍下降 16.67 个百分点，95% CI `[7.58, 27.27]`；
- 全局固定视角不能稳定迁移到 held-out 任务；
- 简单动作不确定性选视角为 96.875%；
- 六视角事后 oracle 为 100%，相对 canonical 仅剩 3.125 个百分点空间。

结论：零样本视角缺口明确存在；简单静态视角优化或任务开始时主动选视角不是当前最强突破口。但该实验仍未回答“经过针对性多视角训练后还剩多少缺口”。

### 4.4 外部证据

- LIBERO-Plus 报告 OpenVLA-OFT+ 使用超过 20,000 条多样轨迹的 mix-SFT 后，相机扰动成功率达到 92.8%。这说明同仿真器内针对性数据覆盖可能很强。
- RoboCasa365 的 GR00T N1.5 在 novel camera perturbation 下仍明显下降：Composite-Seen 从 40.6% 降至 28.8%，Composite-Unseen 从 42.1% 降至 31.5%。
- RoboCasa365 已包含三个相机流和丰富场景，但多相机输入与场景多样性并没有自动消除位姿鲁棒性问题。
- 上述 RoboCasa 结果不是“针对性多视角增强失败”的证明，因此需要在 4×A800 上补齐受控验证。

## 5. 8×RTX 5090 机器已经下载和可提供的资源

8 卡机 hostname：`aibox-r95f9aa795d5-b9ff8d889-rcsp7`。其 `/share` 与 4 卡机不互通，不能直接假设路径可见。

| 资源 | 8 卡机路径 | 已核验状态 | 用途 |
| --- | --- | --- | --- |
| LeRobot Pi0.5 Base | `/share/longjunyu/alphabrain/pretrained_models/pi05_base` | 6 文件，14,467,175,830 bytes | AlphaBrain/LeRobot 初始化 |
| PaliGemma 3B PT 224 | `/share/longjunyu/alphabrain/pretrained_models/paligemma-3b-pt-224` | 14 文件，11,715,896,265 bytes | Pi0.5 视觉语言骨干 |
| OpenPI Pi0.5 LIBERO PyTorch | `/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch` | 14,483,807,415 bytes | LIBERO/Plus 零样本闭环策略 |
| LIBERO-Plus 仿真资产压缩包 | `/share/longjunyu/alphabrain/datasets/libero-plus/verified/assets.zip` | 6,395,849,578 bytes；SHA-256 `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf` | 搭建 Plus runtime |
| LIBERO-Plus 相机参数 RLDS | `/share/longjunyu/alphabrain/datasets/libero-plus/verified/libero_plus_camparam_rlds.zip` | 16,607,835,331 bytes；SHA-256 `a99466a1bb7eab4d0c55094d64d53ef6794ee835ba0db003fcee3e3fa6568e73` | 多视角训练与位姿分析 |
| 已解压相机 RLDS | `/share/longjunyu/alphabrain/datasets/libero-plus/extracted/camparam-rlds-v1` | 256 shards，完整审计通过 | 本地训练；跨机应优先传压缩包 |
| LIBERO-Plus runtime | `/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus` | 已安装并 smoke；source commit `4976dc30028e805ff8094b55501d532c48fec182` | Plus 闭环评测 |
| runtime 数据/配置/overlay | `/share/longjunyu/alphabrain/envs/libero-plus-data-v1`、`libero-plus-runtime-config-v1`、`libero-plus-runtime-overlay-v1` | 本机可用 | 复现实验运行态 |
| Plus 数据审计 | `/share/longjunyu/alphabrain/datasets/libero-plus/audit/camparam-rlds-v1.json` | 完整 | 数据规模与相机分布核查 |
| 已完成 Plus 视角实验 | `/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1` | 77 文件，14,377,953 bytes | protocol、episode、metrics、图和视频 |
| AlphaBrain 视角代码 | `/alphabrain`，分支 `exp/fresh-vla-toy-v0`，HEAD `a06c919` | 部分新研究文件尚未提交 | evaluator、统计和图表 |
| OpenPI RoboCasa adapter | `/projects/openpi`，分支 `exp/robocasa-adapter-v0` | adapter commit `8148ca9`；config `pi05_robocasa_lora_v0` | RoboCasa 到 OpenPI/Pi0.5 接口 |

LIBERO-Plus 相机数据审计摘要：

- 2,876 个成功 episode，449,053 steps；
- 40 个语言任务，256 shards；
- 外部相机位姿按 `1e-4` 取整后有 1,178 种；
- 外部相机位置范围：`[0.0559, -0.8707, 0.4993]` 至 `[1.7976, 0.8789, 2.4248]`；
- 腕部相机每 episode 位移跨度中位数 0.313 m，旋转跨度中位数 20.46°；
- 2,876 个 episode 终止时均记录成功。

8 卡机明确没有：

- RoboCasa/RoboCasa365 完整 runtime；
- RoboCasa 数据集；
- Pi0.5 RoboCasa Human300 checkpoint。

## 6. 4×A800 机器需要先核验的已有资源

用户此前报告 4 卡机可能已有：

- RoboCasa365 Human300：`/share/longjunyu/robocasa/datasets/v1.0/pretrain`；
- Pi0.5 Human300 OpenPI checkpoint：`/workspace/ckpt_download/pi05_pretrain_human300`，约 42GB；
- 原有 RoboCasa、LIBERO 和相关仿真环境。

这些只是待核验信息。先只读盘点，不要重新下载或覆盖。

对第 5 节每一项资源，以及本机 RoboCasa 资源，生成差异表：

| 资源 | 本机路径 | 状态 | 版本/大小/校验 | 是否影响下一实验 | 请求用户操作 |
| --- | --- | --- | --- | --- | --- |

状态只能使用：

- `HAVE`：本机已有且兼容；
- `MISSING_REQUIRED`：下一项已批准实验必需；
- `MISSING_OPTIONAL`：后续可能使用；
- `INCOMPATIBLE_VERSION`：存在但版本或格式不匹配；
- `REQUEST_TRANSFER`：确认需要用户从 8 卡机中转。

不要因为目录名称不同就判断缺失。优先比较 Git commit、文件数、byte size、manifest 和 SHA-256。

## 7. RoboCasa 是主要外部验证，不是可选附录

RoboCasa/RoboCasa365 用于判断 LIBERO-Plus 中的结论是否只属于固定桌面与相对简单背景。你可以并且应当使用本机 RoboCasa 资源验证问题，但先保证策略部署链路有效。

### 7.1 基线有效性

使用本机 Pi0.5 Human300 checkpoint，优先选择以下 10 个 atomic task：

- `NavigateKitchen`
- `OpenDrawer`
- `OpenCabinet`
- `CloseFridge`
- `CloseBlenderLid`
- `CoffeeSetupMug`
- `PickPlaceCounterToCabinet`
- `PickPlaceSinkToCounter`
- `TurnOnMicrowave`
- `TurnOffStove`

首轮每任务 20 episodes，使用官方 success condition 和固定种子。检查：

- checkpoint、action/state normalization、action dimension 和 horizon 是否匹配；
- camera key、renderer、控制频率、episode timeout 是否正确；
- 至少 5 个任务具有非平凡成功率；
- aggregate canonical success 建议不低于 30%，或给出官方设置支持的其他门槛。

若所有方法 canonical 都低于 20%，或失败主要来自部署接口，应标记 `BASELINE_INVALID_OR_TASK_TOO_HARD`，先修复链路，不得解释为视角算法失败。

### 7.2 成对相机鲁棒性分解

在相同 task、scene、物体初始状态、机器人状态、policy seed、episode horizon 和执行 K 下比较：

1. canonical 全相机；
2. 仅第三方相机扰动；
3. 仅腕部相机扰动；
4. 第三方与腕部相机同时扰动；
5. 若 runtime 支持，再增加执行过程中动态移动相机和标定噪声。

优先使用安装版本中 RoboCasa365 官方 novel-camera distribution，记录配置来源，不自行编造扰动标准差。

失败分类至少包括：

- 目标离开视野；
- 目标可见但空间定位错误；
- 选错物体或 fixture；
- 接触/精细操作失败；
- 长程顺序失败；
- 多相机信息冲突或腕部相机依赖。

### 7.3 强数据基线

从同一批 simulator states 重放并重新渲染多视角，不能只用 2D crop 代替真实相机位姿变化。保持 action、prompt、robot state 和 episode group 不变。

至少比较：

- `M0`：现有 Pi0.5 Human300，不做相机适配；
- `M1`：匹配数据预算的 Pi0.5 multi-view RGB；
- 25% 与 100% 数据预算；
- 相机插值、训练位姿支持外的外推、新厨房、相机+背景、相机+布局；
- canonical retention，防止用正常任务退化换取扰动收益。

测试相机位姿必须按参数 bin 与训练集隔离，不能只按 frame 或 episode ID 随机划分。

### 7.4 KYC/显式几何强近邻

若 `M1` 后仍有有意义缺口，再比较：

- `M2-Control`：与几何方法相同参数量和训练数据，但使用固定/无信息几何输入；
- `M2-KYC`：真实内外参或逐像素 Plücker ray；
- clean calibration 与不同强度 calibration noise；
- 第三方-only、wrist-only、dual-camera；
- 冻结视觉骨干、视觉 adapter/LoRA，以及可承受范围内更强视觉适配。

必须保证 KYC 输入随当前图像和相机实时对应。若执行过程中移动相机，ray map 也必须逐帧更新。

### 7.5 独立算法研究

你可以保留并推进本机已经构思的视角算法，不必等待所有基线结束才思考。但在正式声称有效前，必须回答：

- 它针对的是数据效率、位姿外推、组合泛化、标定鲁棒性还是多相机融合中的哪一个缺口？
- 为什么纯 multi-view RGB 和 KYC 不能解决？
- 它是否使用了额外数据、额外参数、额外相机或更高观察频率？
- equal-capacity、equal-data 和 shuffled/fixed-geometry control 是否齐全？
- 增益是否跨至少两个环境或任务家族，而不是单个视角配置？
- 是否在 canonical/no-perturbation 条件下产生明显退化？

可研究但不强制的机制包括：机器人坐标系中的跨视角特征对齐、同状态多视角一致性、相机等变 token、几何引导视觉适配、标定不确定性感知、多相机互补融合、位姿支持外外推，以及具有 oracle headroom 的主动视角方法。

## 8. 评测与统计规范

主要指标：

- canonical task success；
- camera-perturbed task success；
- third-person、wrist 和 dual-camera 分项成功率；
- held-out camera-support success；
- camera + unseen scene/background/layout success；
- canonical retention；
- 相对 `M0`、`M1` 和 `M2-KYC` 的绝对百分点差异；
- 数据效率曲线。

辅助指标：

- grasp/contact/transport/place 等子目标成功率；
- completion steps、progress AUC；
- 可见但误定位率、离开视野率、错误物体率；
- 相机几何输入变化对中间特征与动作的敏感性；
- calibration noise 曲线。

统计要求：

- 使用相同任务、scene、初始状态和 seed 做 paired evaluation；
- 以 task/scene group 为独立单位，不把帧当独立样本；
- 报告每 seed 结果、跨 seed 均值及 paired bootstrap 95% CI；
- 报告绝对百分点，不只报告相对百分比；
- 保存成功与失败的成对 AV1/WebM 视频；
- 训练前冻结 split、camera bins、预算和停止条件。

## 9. 研究判断规则

- 若 `M1` 在 held-out camera 和 held-out scene 上达到 canonical 的 95% 以上，停止普通相机模块路线，除非新方法有显著数据效率或跨环境外推收益；
- 若 `M1` 解决相机-only，但在 camera + unseen kitchen/background 上仍下降至少 10 个百分点，聚焦相机-场景组合泛化；
- 若显式几何在两个 benchmark 上稳定超过匹配 `M1/Control` 至少 5 个百分点，几何模型方向可继续；
- 若 KYC 只在无腕部相机或弱视觉适配条件有效，应明确限定适用范围，不能泛化成所有 VLA；
- 若主动视角 oracle 相对强固定/多视角策略没有明显上限空间，不把主动感知作为主项目；
- 若 RoboCasa canonical baseline 无效，结论是部署或数据链路无效，不是视角研究 No-Go；
- 不因近邻热门而回避真实问题，也不为追求“原创”绕开强 baseline。

## 10. 跨机器协作规则

- 先传小文件：报告、metrics、manifest、图和选定视频；
- 代码通过经过检查的 Git commit 或 patch 迁移，不整仓覆盖；
- 大文件优先传已校验压缩包，不传海量小文件的解压目录；
- `pi05_base` 和 PaliGemma 很可能 4 卡机已经有，先核对，不重复传；
- 若只复核 8 卡机结论，优先请求 14MB 的 Plus 实验结果，不先传 23GB 数据；
- 只有需要在 4 卡机训练/闭环复现 LIBERO-Plus 时，才请求 Pi0.5-LIBERO、assets 和 camera RLDS；
- 用户审核 `REQUEST_TRANSFER` 后再通过 Mac 中转；
- 不主动访问另一地区服务器，不让跨区迁移阻塞本机 RoboCasa 验证。

## 11. 你接手后的第一次回复必须包含

不要只复述本文。先用工具做只读核查，然后输出：

1. 本机 GPU、CUDA/JAX/PyTorch、AlphaBrain/OpenPI、RoboCasa runtime 和 Git 状态；
2. 你当前已经在研究的视角问题和候选算法，以及它与本文证据的关系；
3. 第 5、6 节所有资源的 `HAVE/MISSING/INCOMPATIBLE/REQUEST_TRANSFER` 差异表；
4. 当前可以直接开始的 RoboCasa canonical 与 camera-perturbation 验证；
5. 仅列出下一阶段确实需要用户传输的文件、源路径、目标路径和预计大小；
6. 一份由你主导的实验计划，说明每项实验要区分哪两个科学解释；
7. 哪些 8 卡机结论可以直接接受，哪些需要在 RoboCasa 或你的候选方法上复核。

在完成本机盘点前，不启动大下载、全量训练或跨区复制。

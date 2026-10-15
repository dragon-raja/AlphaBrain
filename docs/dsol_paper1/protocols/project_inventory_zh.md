# 论文项目完整资源地图与后端边界

核验日期：2026-09-29。本文是资源导航，不修改历史科学协议，也不将目录存在视为全量数据认证。

## 1. 三层管理，而非把所有文件搬进一个仓库

1. **可公开清单**：`/workspace/projects/alphabrain-dsol-paper1/configs/resources/paper1_resources.json`。
   稳定资源ID、数据/模型关系、实际执行后端，不含作者个人绝对路径。
2. **本机绑定**：`/share/personal/longjunyu/research_registry/resource_paths.local.json`。
   同一资源ID映射本机位置；其他机器使用自己的绑定。`*.local.json` 不进入Git。
3. **冻结证据**：实验目录自己的 release/plan、源码快照、权重和输入哈希、日志及分析。
   这些决定某张论文图实际用了什么，不跟随开发仓库自动更新。

原 `configs/dsol_paper1/resource_catalog.json` 保留为本机历史导航，并联动检查上述资源图。
历史 `/share/longjunyu` 只按读时映射，不重建、不改写历史文件。新增路径一律是新共享根。

### 统一分类与当前完整范围

| 层级 | LIBERO / LIBERO-Plus | RoboCasa | RoboTwin 2.0 |
| --- | --- | --- | --- |
| 官方原始资源 | LIBERO四套HDF5；Plus相机/goal包 | Human300与本地target数据 | 固定官方版本的50任务clean、2500条示范 |
| 自建训练数据 | broad64分片、规范LeRobot；另有历史Plus转换 | **真实PNG渲染主体＋两批恢复数据＋配对发布清单** | **尚未构造**，不能把官方三相机示范称为自建位姿增强 |
| 权重 | 官方起点、规范与多视角后训练 | 官方75k、规范与多视角8250步 | 已取得官方Pi05 joint 29999；本项目尚未训练 |
| 仿真／模型后端 | Plus运行树／AlphaBrain PyTorch | RoboCasa＋robosuite／OpenPI JAX | 官方RoboTwin＋固定XPolicyLab；整套VLA尚未验收 |
| 正式研究结果 | 初态97视角闭环矩阵 | P0与官方模型矩阵分别归档 | 尚无正式VLA矩阵；旧两任务回放仅为工程验证 |
| 官方身份验收 | 历史冻结证据；不声称Plus全库齐全 | 历史冻结证据；不声称target全库齐全 | clean50＋三类资产＋指定权重完成字节/结构验收；运行兼容性另验 |

这张表描述全部主线资源类型，而非宣称三个benchmark的所有上游数据都已镜像。
下文逐项给出完整路径。旧数据保持原位，通过统一ID管理；不为目录美观移动冻结实验。

## 2. LIBERO与LIBERO-Plus：数据不是同一件事

| 内容 | 本机完整路径 | 当前范围/作用 |
| --- | --- | --- |
| LIBERO原始HDF5 | `/share/personal/longjunyu/alphabrain/datasets/libero-original-hdf5-v1` | goal/spatial/object/10，各10个任务文件；不是LIBERO全部套件 |
| 规范LeRobot数据 | `/share/personal/longjunyu/alphabrain/datasets/libero-canonical-lerobot-v1` | 四套规范数据；转换来源以原记录为准，不凭目录名补造派生关系 |
| Plus资源总根 | `/share/personal/longjunyu/alphabrain/datasets/libero-plus` | 包含verified原包、extracted、audit、runtime及views；不是全部Plus扰动类别均已完整取得 |
| Plus相机参数RLDS | `/share/personal/longjunyu/alphabrain/datasets/libero-plus/extracted/camparam-rlds-v1` | 已有相机扰动资源，与原始LIBERO HDF5分开 |
| Plus goal资源 | `/share/personal/longjunyu/alphabrain/datasets/libero-plus/extracted/goal-suite-v1` | 历史goal扩展资源 |
| 实际LIBERO/Plus仿真运行树 | `/share/personal/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus` | 冻结运行来源清单记录源码commit `4976dc30028e805ff8094b55501d532c48fec182`；现目录无Git元数据，不能自行冒称纯净checkout |
| 自建多视角训练数据 | `/share/personal/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2` | 自建64视角数据，有manifest；不是官方Plus训练包 |
| 官方π0.5 PyTorch起点 | `/share/personal/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch` | AlphaBrain桥接加载的起点 |
| 官方π0.5 JAX资源 | `/share/personal/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_jax` | 独立资源，不代表当前LIBERO矩阵使用JAX执行 |
| 后训练模型父目录 | `/share/personal/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs` | 各run/final_model；精确两个模型路径登记在本机资源绑定和冻结release中 |
| 当前正式矩阵 | `/share/personal/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908` | 8任务×4初态×97视角×32执行噪声×2模型；不能因Plus运行树而将所有结果称作官方Plus成绩 |

## 3. RoboCasa：当前使用OpenPI，不是AlphaBrain策略后端

| 内容 | 本机完整路径 | 核验结果 |
| --- | --- | --- |
| OpenPI代码和环境 | `/workspace/projects/openpi-robocasa` | 上游origin为Physical-Intelligence/openpi；当前commit `615637f6c9d16955690c94de7d228692ad5080e5`；环境为该目录`.venv` |
| RoboCasa仿真源码 | `/workspace/projects/robocasa` | commit `be22d659b02db8f6d7f3a3c3edc742934fdcbaae` |
| robosuite源码 | `/workspace/projects/robosuite` | commit `85abee228d1c43ab1939bce33028099945d453b4` |
| Human300数据 | `/share/personal/longjunyu/robocasa/datasets/v1.0/pretrain` | 本地65 atomic＋235 composite任务目录，原训练清单从这里取数据 |
| target数据 | `/share/personal/longjunyu/robocasa/datasets/v1.0/target` | 本地16 atomic＋32 composite任务目录；不据此宣称官方target全集完整，不能与评测任务数混称 |
| 官方模型 | `/share/personal/longjunyu/robocasa/checkpoints/paper1/official_75k/multitask_learning/75000` | OpenPI/JAX/Orbax检查点，含归一化；两组后训练的共同起点 |
| 配对构造、训练和检查点 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916` | 虽处于alphabrain命名空间，实际策略实现仍是OpenPI；目录名字不决定后端 |
| 配对训练数据发布 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/matched-data-release-20260917` | 两组输入清单与来源哈希，引用实际渲染数据 |
| 自建图像主体 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/matched-render-human300/episodes` | 按episode/source_window存canonical_left/right、train_NNN_left/right和wrist.png；实际PNG，不是清单 |
| 自建图像恢复批次1 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/bounded-recovery-20260917/cleanboard` | 发布清单明确选用的恢复图像，不能漏掉或用原失败图替代 |
| 自建图像恢复批次2 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/bounded-recovery-20260917/leftovers` | 同上，原失败记录保留，不混入正式输入 |
| 规范训练最终权重 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/training-canonical-fresh-resumed-20260918/checkpoints/pi05_pretrain_human300_all_lora_fast25/canonical_matched/8250` | 单独run_config和恢复记录 |
| 多视角训练最终权重 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/training-broad-gnu-20260917/checkpoints/pi05_pretrain_human300_all_lora_fast25/broad_exterior_matched/8250` | 与规范组分开；非官方原始权重 |
| P0矩阵 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916/formal-matrix-native-20260918` | 两个8250步后训练模型；精确checkpoint路径在plan与本机绑定 |
| 官方模型新矩阵 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-official-fullview-20260928` | 与P0后训练模型分开，不混合归因 |

本仓库 `/workspace/projects/alphabrain-dsol-paper1/benchmarks/Robocasa365` **有**RoboCasa适配，
并非整个环境都在仓库外。当前论文 `evaluation_policy.py` 调用OpenPI的模型加载、输入输出变换
和推理；`openpi_training.py` 调用OpenPI训练配置。指标、统计及报告由本仓库维护。
已有QwenGR00T/QwenOFT通用适配不能当作RoboCasa π0.5已迁移AlphaBrain的证据。

本轮重新读了两份配对JSONL：**各33,000个source window**；规范组1个view ID，
多视角组覆盖64个训练view ID（不表示每个window都渲染64个）。每组有198,000条图像引用，
其中197,952条指向主体，cleanboard与leftovers各24条；引用计数含重复，不等于独立图像数。
原发布记录验收231,000个图像文件；本轮是完整清单引用统计和各根样例存在检查，未重新哈希全部PNG。
历史release的`DATA_READY_NOT_TRAINED`是发布当时阶段，训练完成与否看后来run记录，不改历史状态。

### 是否另跑AlphaBrain的RoboCasa实验

**有必要考虑一个有限的同框架对照，但现在不建议全量重训、重跑97视角矩阵。**
这是新实验，不是把现有JAX结果搬名或强行迁移；原OpenPI实验保留。

代码审查发现现有通用入口还不是可直接使用的匹配π0.5配方：

- `configs/datasets/robocasa365.yaml`目前配置12维动作、16维状态、16步动作块；
  通用模型π0.5配置默认却是LIBERO的7维动作/10步，不能只拼两个名字就运行。
- `eval/model2robocasa365_interface.py`状态顺序是末端位置、旋转、**夹爪、底盘**；
  当前论文OpenPI入口是末端位置、旋转、**底盘、夹爪**。这不说明各自接口错误，但跨用权重必须匹配。
- 通用入口直接resize、可选min/max状态归一化，并按其训练约定裁剪动作、二值化两个通道；
  论文入口使用OpenPI的resize-with-pad及权重对应变换，不能共用输出处理而不核验。
- AlphaBrain π0.5支持`discrete_state_input`，但需显式启用，并核验归一化及token长度；
  “LIBERO之前没有状态”不能成为RoboCasa也默认关闭状态的理由。

建议顺序：①独立配置和ID，明确π0.5起点、三相机、状态/动作契约；②CPU数据契约检查，
固定输入/噪声推理检查；③相同小面板规范/扰动视角基线；④若链路正常、且同配方验证对研究有价值，
再在既有配对数据上做规范/多视角训练对照。参数更新范围、样本暴露量、学习率、动作块与噪声预算需冻结。
不要求两个benchmark物理动作维度相同，而要相同的比较原则；若改用QwenGR00T，那是跨模型验证，
不是π0.5框架对照。此轮仅审查和设计，未启动训练或GPU评测。

## 4. RoboTwin：以官方锁定组合为准，不以复制或下载判断身份

数据固定HF revision：`3dc3b798668feb99ac61cc9086d84cbcc3d79186`。
直接官方源码固定commit：`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`；
其XPolicyLab固定commit：`fa431ecd893ee706883e64fe5fe1464ec8cd928d`。

**验收已完成：54项资源、50任务、2,500条示范、548,893组对齐state/action帧；所有任务结构检查通过。**
该结论来自最终release，不再沿用下载中的状态。源码及固定子模块共6,861个Git跟踪对象已核对。
原包按官方HF revision对应的字节数和SHA256核验；解包记录逐文件哈希，验收检查receipt链。

| 资源 | 本机完整路径 | 身份／作用 |
| --- | --- | --- |
| clean示范 | `/share/personal/longjunyu/robotwin/datasets/raw/hf-3dc3b798668f/demo_clean` | 每任务保留original.zip、receipt；实际文件在`extracted/<task>/aloha_agilex`，含data/video/instruction/scene_info/seed |
| 物体资产 | `/share/personal/longjunyu/robotwin/assets/hf-3dc3b798668f/objects/extracted/objects` | 官方objects.zip解包 |
| 本体资产 | `/share/personal/longjunyu/robotwin/assets/hf-3dc3b798668f/embodiments/extracted/embodiments` | 官方embodiments.zip解包；不从旧副本拼接 |
| 背景资产 | `/share/personal/longjunyu/robotwin/assets/hf-3dc3b798668f/background_texture/extracted/background_texture` | 官方background_texture.zip解包 |
| 官方源码 | `/share/personal/longjunyu/robotwin/sources/official-ea8b21121ebb` | 固定提交；本地实验补丁不得写回此来源树 |
| 官方策略代码 | `/share/personal/longjunyu/robotwin/sources/official-ea8b21121ebb/XPolicyLab/policy/Pi_05` | 父仓库固定子模块，不使用其他目录里名称相似的policy |
| 官方权重 | `/share/personal/longjunyu/robotwin/checkpoints/hf-3dc3b798668f/pi05-joint-official/extracted/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999` | 同revision官方归档，包含norm_stats |
| 统一验收发布 | `/share/personal/longjunyu/robotwin/registry/official-current-20260929/release.json` | 字节/结构验收；`runtime_validated=false`明确保留 |
| 当前官方指针 | `/share/personal/longjunyu/robotwin/registry/official_current.json` | 本次资源的权威入口；旧`current.json`仅为历史试点，不用于新正式协议 |
| 历史工程回放 | `/share/personal/longjunyu/robotwin/experiments/replay-admission-20260929` | 两任务旧格式回放，不冒充当前50任务新格式的策略验收 |
| 自建多视角／后训练／正式矩阵 | 尚未建立 | 资源图显式`not_created`，后续另建derived release与run ID |

缓存复用核验：旧复制权重与本次官方权重的**18个payload文件记录SHA256和大小全部一致**。
因此旧副本的权重身份可确认；这不证明其训练数据是否包含留出场景。正式绑定统一指向上述官方目录，
旧副本保留历史记录，不在本轮删除大文件。未来先用官方文件清单验证缓存，匹配即可复用，无需重复下载。
比较依据是已验证import receipts，本轮未再次读取12GB权重重算哈希。

当前官方数据目标：`/share/personal/longjunyu/robotwin/datasets/raw/hf-3dc3b798668f/demo_clean`。
完整资产目标：`/share/personal/longjunyu/robotwin/assets/hf-3dc3b798668f`（objects、embodiments、background_texture）。
官方代码：`/share/personal/longjunyu/robotwin/sources/official-ea8b21121ebb`。
原生策略接口：其`XPolicyLab/policy/Pi_05`，与RoboCasa OpenPI工作区分开，不混用依赖环境。

已完成的获取与验收记录：
`/share/personal/longjunyu/robotwin/imports/official-current-clean50-20260929/status.json`。
53项数据/资产归档合计38.304GB；另有官方π0.5归档12.443GB，均已完成核验。
其进度在 `/share/personal/longjunyu/robotwin/imports/official-pi05-direct-20260929/status.json`。
自动后验收状态为 `/share/personal/longjunyu/robotwin/imports/official-completion-frozen-20260929/status.json`；
使用独立冻结源码，失败不放行、不自动重试。全部通过后已生成
`/share/personal/longjunyu/robotwin/registry/official_current.json`，原试点指针/发布记录不覆盖。
公开可复现的下载锁定清单在 `configs/resources/robotwin_official_clean50.lock.json`，代码版本锁定
在 `configs/resources/external_sources.lock.json`。源码与XPolicyLab共6861个Git跟踪对象已逐对象核对。

这里“完整”限定**官方当前50任务clean＋全部仿真资产＋官方源码/策略接口＋所用π0.5权重**。
不等于1.53TB数据仓库的整库镜像：随机化500条/任务、其他本体示范、所有方法权重和重复LeRobot
导出另列可选资源，不静默下载或混入clean协议。资源完整也不等于正式VLA评测已验收。

**目前还不能称“整个运行组合完全对齐官方、可直接复现官方成功率”。**文件身份已经对齐；
下一道门是锁定源码＋新格式解码＋本体/控制器＋Pi05输入输出＋渲染后端的联合运行检查。
当前clean HDF5没有旧`_traj_data`规划文件，不能用旧两任务回放替代此检查。
官方模型训练域尚未证实，不能因权重来自官方就称测试集OOD。资源图固定代码/数据/资产/权重revision，
审计会拒绝静默换版；但现有任意手写启动命令尚未全部强制经过该审计，正式启动器仍须接入门禁。

## 5. 仓库、环境与结果的职责

统一代码根：`/workspace/projects/alphabrain-dsol-paper1`。

| 职责 | 完整路径／规则 |
| --- | --- |
| 模型与训练公共实现 | `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/model`、`/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/training` |
| 数据／指标／选择与搜索／分析／绘图 | `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/research/dsol`，各自子目录；不向scripts堆核心算法 |
| benchmark适配 | `/workspace/projects/alphabrain-dsol-paper1/benchmarks/LIBERO`、`/workspace/projects/alphabrain-dsol-paper1/benchmarks/LIBERO-plus`、`/workspace/projects/alphabrain-dsol-paper1/benchmarks/Robocasa365`、`/workspace/projects/alphabrain-dsol-paper1/benchmarks/RoboTwin` |
| 实验配置／资源锁 | `/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1`、`/workspace/projects/alphabrain-dsol-paper1/configs/resources` |
| 薄启动入口 | `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1`，按training/analysis/diagnostics/operations等分类 |
| 测试 | `/workspace/projects/alphabrain-dsol-paper1/tests` |
| 报告生成 | `/workspace/projects/alphabrain-dsol-paper1/reports/paper1` |
| 唯一当前汇报PDF | `/workspace/projects/alphabrain-dsol-paper1/docs/dsol_paper1/reports/current/vla_view_research_progress_zh.pdf` |
| 文档与历史旁支 | `/workspace/projects/alphabrain-dsol-paper1/docs/dsol_paper1`；不再主线的代码按既有规则进`/workspace/projects/alphabrain-dsol-paper1/archive`，冻结证据不动 |
| 共享注册表 | `/share/personal/longjunyu/research_registry`，ID→机器路径、哈希快照、实验导航；不存模型代码实现 |

新增资源标准：raw保留官方来源和哈希；derived必须记录parent、生成代码/配置、数据schema与release；
checkpoint必须绑定初始化权重、训练清单、配方和step；run必须绑定具体checkpoint/data/source/noise/split；
analysis绑定run与分析代码；报告引用分析产物。不同benchmark不共享含糊的`latest`数据或权重路径。
来源树只读，运行补丁和本机资产路径在独立runtime/config；环境按后端隔离，不覆盖OpenPI环境。
本轮补的是注册和门禁，不将历史33个缺根级清单目录自动变成合格发布，也不声称所有旧路径已迁移完。

## 6. 面向开源还必须完成的事

- 保持外部仓库为固定版本依赖；记录本地修改，不能把缓存副本冒称官方原版。
- 核心方法仅依赖统一数组/输入契约；模型后端和仿真器通过各自适配层接入。
- 配置与本机路径分离；提供下载、格式检查、最小复现及完整实验的不同入口。
- 审查本仓库与每个上游、数据和权重的许可证/再分发条件；主仓库MIT不自动覆盖所有权重。
- 提交前逐文件审查，排除个人路径绑定、凭据、原始数据/权重和不拟公开旁支；不直接`git add .`。
- 从干净checkout验证CPU检查、最小数据样例与一条完整策略闭环，再标记可复现发布版本。

当前属于“结构与来源管理已补强、资源/后端准入继续验收”，**不是已满足全部开源发布条件**。
历史44目录中的33个仍未具备根级标准清单，保持原证据，不因整理就自动认证。

### 本轮验证记录

- 资源图33项（32项已有、1项RoboTwin自建数据明确未创建）；共享索引发布与路径/回执链检查通过。
- 资源相关CPU测试42项通过；portable配置71项通过（Python 3.10环境）；三项架构/布局静态检查通过。
- 本轮曾误用Python 3.8环境导致portable失败，另一个3.11训练环境存在外部`tests`包遮蔽；
  未修改训练环境或放松测试，使用符合项目要求的独立3.10环境重验通过。
- 未迁移历史payload、未启动训练/仿真评测、未修改当前PDF、未提交或上传GitHub。
- 旧机器路径绑定已保存在`/share/personal/longjunyu/research_registry/snapshots/20260929/resource_paths_before_complete_map.local.json`。

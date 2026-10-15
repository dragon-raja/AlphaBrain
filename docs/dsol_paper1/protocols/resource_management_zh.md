# 数据集、权重、代码与实验的资源管理

三个benchmark的原始／自建数据、权重、外部OpenPI与官方版本验收，统一见
[完整资源地图](project_inventory_zh.md)。本文件说明管理机制，不代替实际版本与覆盖范围清单。

## 原则与范围

当前维护仓库为 `/workspace/projects/alphabrain-dsol-paper1`。所有新共享文件只使用
`/share/personal/longjunyu`。旧 `/share/longjunyu` 不重建、不做软链接；历史证据不重写，
读取时使用已有 `data.artifacts.path_mapping` 映射路径，SHA256 始终覆盖原始字节。

代码按已有仓库架构分层，不把数据集放进 Git。历史实验不因新增基准而重命名或搬家。
统一导航配置为 `configs/dsol_paper1/resource_catalog.json`；它不是第二份科学协议。
科学定义仍以每次实验冻结的 release/plan 为准。

## 项目地图：代码、资源和实验各放哪里

下面目录以 `/workspace/projects/alphabrain-dsol-paper1` 为代码根；数据和权重不进入该树。

```text
AlphaBrain/
  model/、dataloader/、training/        模型、加载器、训练框架
  research/dsol/
    metrics/、selectors/               通用指标与视角选择方法
    acquisition/                       相机获取/搜索原语，不冒称已验证主动控制
    data/                              数据协议、资源导入、索引、来源验证
    analysis/、plotting/                统计与绘图实现
benchmarks/
  LIBERO/、Robocasa365/、RoboTwin/       各环境的数据、仿真、策略适配
configs/dsol_paper1/                    配置及资源导航；正式执行再冻结副本
scripts/dsol_paper1/                    按训练/分析/诊断/运行/运维分类的薄入口
reports/paper1/                         PDF构建与排版，消费分析产物
docs/dsol_paper1/                       协议、计划、审计、结论、唯一当前PDF
tests/                                 按模块分层的测试
tools/                                 架构、目录、发布和测试治理
archive/                               已退休研究；新代码不得反向依赖
```

共享盘总导航为 `/share/personal/longjunyu/research_registry/catalog.json`；
与仓库 `configs/dsol_paper1/resource_catalog.json` 必须逐字节一致。
旧LIBERO/RoboCasa实验保留原命名空间；新增RoboTwin使用独立资源树，不为了统一外观迁移旧证据。
共享数据是长期资产；`/workspace/envs/robotwin-replay-20260929` 是可重建的独立依赖环境，
`/workspace/projects/robotwin-replay-runtime` 是可变仿真工作副本，二者均不作为原始资源。

来源链为：**实际源码快照/哈希＋资源回执 → 冻结实验协议 → 原始执行记录 → 分析产物 → 当前PDF**。
资源索引只提供导航和身份检查，不让开发仓库的改动自动替换已启动实验的冻结源码。
RoboCasa的OpenPI执行依赖另位于 `/workspace/projects/openpi-robocasa`；本仓库负责对应
benchmark适配、实验协议及分析，不把外部框架复制混入通用指标包。正式实验必须登记
实际外部框架源码身份，而不是仅记录本仓库commit。

## 命名空间与落盘

| 对象 | 位置与规则 |
| --- | --- |
| LIBERO 已有正式实验 | `/share/personal/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908`；保持冻结目录 |
| RoboCasa 训练、配对数据及 P0 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-stage2-preparation-20260916`；训练、渲染、矩阵子目录分开 |
| RoboCasa 官方权重独立矩阵 | `/share/personal/longjunyu/alphabrain/experiments/dsol-robocasa-official-fullview-20260928`；不混入 P0 两个后训练模型 |
| RoboTwin 新资源根 | `/share/personal/longjunyu/robotwin`；采用下面的明确分层 |
| 跨实验元数据快照 | `/share/personal/longjunyu/research_registry/snapshots/<日期>`；只登记来源，不移动旧证据 |

RoboTwin 根目录约定（后续阶段的目录可按需创建，不代表已有产物）：

```text
robotwin/
  datasets/raw/<官方revision>/<包装版本>/<任务>/  原包、原样解包、导入回执
  datasets/derived/<构造ID>/                    配对重渲染；不得覆盖 raw
  assets/<来源快照>/                            本体、物体等；不混入其他基准
  checkpoints/<发布或训练ID>/<step>/            参数、归一化与来源回执
  sources/<上游版本与本地快照ID>/               实际字节冻结，不冒用纯净上游版本
  imports/<导入ID>/                             计划、执行器、状态、错误、逐文件哈希
  registry/                                    当前资源发布指针、不可变发布清单
  experiments/<实验ID>/                         冻结协议、源码身份、日志、原始结果、分析
```

`*.partial` 是未完成资源，不能用于训练评测。现有目标禁止覆盖；失败记录保留，显式新修订
才能重试。公共资产可以被多个实验只读引用；每个实验应冻结实际所用资源ID及回执哈希。
文件级复制完成只标记 `BYTE_VERIFIED`，不等同于 `RUNTIME_VALIDATED` 或正式研究准入。

RoboTwin正式资源入口为 `/share/personal/longjunyu/robotwin/registry/official_current.json`，
指向已完成54项资源的带哈希发布清单。旧`current.json`仅指历史两任务试点，不用于新正式实验。
资源图明确锁定官方源码、固定子模块、clean数据、资产、Pi05权重的revision；获取方式不决定身份，
缓存经同一官方哈希核验可直接复用。字节/结构验收与整套策略运行验收分开。

## 每个正式实验必须绑定的身份

1. `benchmark_id`、任务/初态清单、数据来源revision、原始episode/seed与拆分清单。
2. 数据构造版本、父数据回执、仿真器/资产/本体版本、相机内外参、渲染配置。
3. 原模型来源与实际参数/归一化哈希；后训练模型另绑定父权重、训练数据清单、
   配方、实际步数、恢复记录。未知训练覆盖明确写 UNKNOWN，不用模板代替实际配方。
4. 仓库commit **加实际源码文件哈希/快照**。有未提交修改时仅记录commit不足以复现。
5. 评测候选、执行与评分噪声、时限、动作表示、状态输入、成功判据及聚合权重。
6. 分析产物引用输入矩阵与分析源码哈希；PDF消费分析结果，不成为另一份隐式协议。

先按原始episode/任务初态划分训练与评测，再扩充视角，避免同一轨迹的不同相机图泄漏。
工程两个试点任务不自动成为正式研究任务集；全量任务、训练或矩阵另行冻结。

## 历史 RoboTwin 小试点（保留记录，不是当前完整资源清单）

计划：`configs/dsol_paper1/robotwin_resource_pilot_20260929.json`。
仅两个官方旧格式clean包（约1.28GB），以及共享盘Aloha、物体、支持文件、实际仿真源码、
官方来源π0.5权重。不复制Fast-WAM加工数据，不下载随机背景大包，不启动新GPU工作。

上游包revision固定为 `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。
代码快照声明的上游commit为 `96c1feab536306b50c26af200044fcdf126e8904`，但本地
`envs/_base_task.py` 有渲染降噪器环境变量开关补丁；必须记录实际字节，不能称为纯净上游。
默认值仍为 `oidn`；后续运行需显式记录 `SIM_BENCH_DENOISER`，不能默默切换。
源代码的 `assets` 链接不复制；后续独立运行工作区再明确绑定我们的资产，不能写源快照。
整套 XPolicyLab 其他策略子仓库及历史 eval_result 不属于这个仿真源码快照。
首轮导入在无关策略子目录遇到软链接而按规则停止，失败回执保留；续接计划仅完成
剩余源码/权重，引用原计划哈希，不重写已成功资源。仓库计划修订2记录收缩后的范围，
原执行计划留在 `imports/official-pilot-20260929/plan.json`。
共享Aloha目录除82个已匹配官方的文件外，还带有 `curobo_left.yml` 和
`curobo_right.yml` 两个本地规划配置；复制清单保留两者并记录哈希，不将84文件目录
冒称为与官方82文件集合完全相同。运行适配必须检查/生成自己的规划配置。

已完成100条结构/关节范围筛查，以及两个episode的原生回放与初态相机配对工程验证；
范围和渲染失败记录见工程试点文档。仍须验证更多示范、跨进程渲染与官方策略输入链路。
官方权重真实训练域仍 UNKNOWN。局部试点成功不解除正式评测的其余门槛。
RoboTwin仿真依赖应放独立环境；当前RoboCasa环境没有SAPIEN/MPLib/CuRobo，不向运行中的
RoboCasa环境安装这些包。CPU数据检查借用已有HDF5库，不代表复用了仿真环境。

## 工具与验收

- 可复用文件导入：`AlphaBrain/research/dsol/data/resource_import.py`。
- 薄入口：`scripts/dsol_paper1/operations/import_resources.py`，无 `--execute` 不执行。
- 只读索引检查：`scripts/dsol_paper1/diagnostics/audit_resource_catalog.py`。
- 显式索引发布：同入口加 `--publish`；先审计，再保存旧共享索引至按哈希命名的
  `research_registry/catalog_history/`，原子发布与仓库字节相同的新索引。
- RoboTwin数据格式检查：`benchmarks/RoboTwin/data_admission.py`；HDF5依赖仅在调用时加载。
- 原生示范回放：`benchmarks/RoboTwin/replay.py`；薄入口
  `scripts/dsol_paper1/runtime/robotwin_replay_probe.py` 必须显式 `--execute`，
  固定任务/episode/GPU，输出新实验目录；既不启动VLA，也不回写raw。
- 新增模块的CPU测试位于 `tests/dsol_paper1/data/`。

索引检查包括：入口/协议存在、ID不重复、活动路径不使用废弃共享根、数据集对应代码模块存在、
仓库与共享索引一致，以及 `current → release → 资源回执` 的SHA256链。
这是入口与元数据身份验证，不宣称每次都重新哈希所有模型字节或复验全部rollout。
历史目录若没有根级标准元数据，登记 `LEGACY_METADATA_NOT_AT_ROOT`，不直接删除，
也不把它们自动当作当前论文的有效证据。

截至2026-09-29，历史导航包含44个实验目录，其中33个缺少根级标准清单；这部分仍是
可追踪但未统一准入的历史资产，不能宣称整个历史库已完成所有来源与复现认证。

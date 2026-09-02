# Paper 1 代码库整理交接（2026-09-01）

## 0. 文档定位

本文是**代码库整理交接**，不是实验协议、结果报告或新的 release receipt。它不修改任何已冻结定义，
也不授权在正式 view-value expectation 流水线运行期间移动、重命名、格式化或删除现有代码与产物。

本次盘点只新增本文，并在 Paper 1 README 顶部增加指向本文的链接；没有修改 runner、模型、配置、
checkpoint、实验输出或已有冻结协议与结果记录。

## 1. 盘点快照与当前保护边界

- 仓库：`/workspace/projects/alphabrain-dsol-paper1`
- 分支：`exp/dsol-paper1-v0`
- 盘点时 HEAD：`a74841b` (`test: smoke final view report`)
- 盘点时工作树：clean
- 当前正式实验根目录：
  `/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1`
- 当前正式 release receipt：
  `/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1/receipts/execution-release-formal-seed41.json`
- 当前有效 Stage A：
  `/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1/calibration/stage-A`
- 已判废但必须保留来源说明的运行：
  `stage-A-invalidated-checkpoint-family`、
  `stage-A-invalidated-nonwritable-buffer`
- 当前有效噪声银行：`noise-banks-h10/`；旧 `noise-banks/` 已有
  `INVALIDATED_PRE_EXECUTION.json`，不得与有效银行混用。

### 正式运行结束前禁止进行的整理

1. 不移动或重命名 `scripts/dsol_paper1/` 中当前流水线引用的文件。
2. 不批量格式化、统一 import、改变 CLI 参数或默认路径。
3. 不合并 Shell controller，不把多阶段流程改写成新总入口。
4. 不修改冻结 JSON、schema、noise manifest、release receipt 或运行目录。
5. 不清除 invalidated、smoke、日志和中间 ledger；它们是有效性审计链的一部分。
6. 不在当前证据分支上做大规模删除。大整理应在结果冻结后的新分支或独立 release worktree 中完成。

## 2. 当前仓库为什么显得杂乱

盘点时，仅 Paper 1 相关目录已有：

| 目录 | 文件数 | 主要内容 |
|---|---:|---|
| `scripts/dsol_paper1/` | 143 | runner、builder、audit、analysis、plot、一次性 controller 和部分测试 |
| `docs/dsol_paper1/` | 140 tracked | 冻结协议、阶段报告、PDF、图、多个版本的 PDF page preview |
| `configs/dsol_paper1/` | 61 tracked | 冻结协议、生成计划、catalog、历史版本和模板 |
| `tests/dsol_paper1/` | 25 | 另一部分正式测试 |

`scripts/dsol_paper1/` 约含 28,327 行 Python 和 3,322 行 Shell。测试同时位于
`scripts/dsol_paper1/*_test.py` 与 `tests/dsol_paper1/`；部分测试依赖额外 `PYTHONPATH` 才能收集。

此外，该分支相对 `origin/main` 约有 996 个文件发生新增或修改，除 Paper 1 外还包含 FRESH、KYC、
CABI、CORA、CCV 等历史研究线。它们不一定是垃圾，但不应全部进入 Paper 1 的最终复现包。

## 3. 文件分级：保留、抽取、归档、再生成

| 级别 | 判定 | 典型内容 | 后续动作 |
|---|---|---|---|
| P0 证据链 | 冻结协议与正式运行不可缺失 | protocol、schema、manifest、hash、receipt、valid/invalidated ledger | 原样保留并校验 hash |
| P1 发布核心 | 论文复现必须执行 | 数据构造、显式噪声、闭环 evaluator、训练、分析和最终报告入口 | 抽成稳定模块，补端到端测试 |
| P2 历史工具 | 曾用于形成结论但不再是正式入口 | pilot、smoke、early、tail、`after_*`、旧 selector 与旧 gate | 移到带日期和说明的 `archive/` |
| P3 可再生成 | 由代码和正式数据生成 | PDF、page preview、montage、部分 catalog 与图表 | 发布包保留最终版，其余写生成命令或放 artifact archive |
| P4 旁支研究 | 与 Paper 1 非直接依赖 | FRESH、KYC、CABI、CORA 等 | 留在母仓库，不复制进 Paper 1 release |

注意：`invalidated` 不等于可删除。正式发布时应把它们放入独立审计归档，并保留判废原因；只是不得让
默认分析 glob 读取它们。

## 4. 当前正式工作集（大整理时先保护）

### 模型与 checkpoint 路径

- `AlphaBrain/model/framework/PaliGemmaPi.py`
- `AlphaBrain/model/framework/base_framework.py`
- `AlphaBrain/model/modules/action_model/pi0_flow_matching_head/`
- `AlphaBrain/model/modules/vlm/vision_low_rank_adapter.py`
- `AlphaBrain/dataloader/paligemma_datasets.py`
- `scripts/cabi_vla/serve_alphabrain_pi05_websocket.py`

### View-value expectation 主链路

- `scripts/dsol_paper1/explicit_flow_noise.py`
- `scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py`
- `scripts/dsol_paper1/build_view_value_expectation_population.py`
- `scripts/dsol_paper1/build_view_value_expectation_calibration_stage.py`
- `scripts/dsol_paper1/build_view_value_expectation_heldout_protocols.py`
- `scripts/dsol_paper1/analyze_view_value_expectation_calibration.py`
- `scripts/dsol_paper1/analyze_view_value_expectation_heldout.py`
- `scripts/dsol_paper1/run_view_value_expectation_calibration_full.sh`
- `scripts/dsol_paper1/run_view_value_expectation_calibration_tail.sh`
- `scripts/dsol_paper1/run_view_value_expectation_post_calibration.sh`
- `scripts/dsol_paper1/run_view_value_expectation_finalize.sh`
- `scripts/dsol_paper1/build_view_value_expectation_final_report.py`
- `configs/dsol_paper1/view_value_expectation_protocol_v1.json`
- `docs/dsol_paper1/view_value_expectation_protocol_v1_zh.md`

这是一份保护清单，不表示列表外文件都可删除。清理者必须从正式 run manifest、Shell 引用、Python import
和报告 provenance 反向闭包后，才能确定最终发布依赖。

## 5. 推荐的大整理顺序

### Gate 0：冻结当前证据

仅在全部正式 controller 完成、analysis/report 生成且审计通过后进行：

1. 记录最终 Git commit、环境、checkpoint identity、协议 hash 和所有 run manifest。
2. 对正式输出生成只读清单；valid、invalidated、smoke 分区。
3. 从空进程重跑最终分析与报告，确认不依赖 tmux、个人 shell state 或未记录的环境变量。
4. 给当前证据分支打 archive tag；之后不在该 tag 上重写历史。

### Gate 1：建立独立 release 分支或 worktree

不要直接清扫当前实验分支。建议从冻结 commit 创建 `paper1/release-v1`，然后只在 release worktree 中整理。
第一步先复制“运行依赖闭包”，不要凭文件名猜测后删除。

### Gate 2：抽稳定模块，减少一次性入口

建议目标结构：

```text
AlphaBrain/research/dsol/
  data/
  views/
  noise/
  evaluation/
  analysis/

paper1/
  README.md
  configs/
  protocols/
  scripts/
    prepare.py
    train.py
    evaluate.py
    analyze.py
    reproduce_figures.py
  tests/
  archive/              # 只存必要历史索引，不默认安装
```

多个 `run_*after_*`、`*_tail.sh` 和 machine-specific 绝对路径应收敛为少量稳定 CLI 与显式配置；但要先用
golden manifest 对比新旧入口生成的任务集合、seed、候选顺序和输出 schema 完全一致。

### Gate 3：统一测试与依赖

1. 将可复用单元测试统一到 `tests/`，避免依赖当前工作目录导入。
2. 在 `pyproject.toml` 明确 pytest/test path 和 Paper 1 可选依赖。
3. 保留三类最低门：schema/manifest、显式噪声复现、最小闭环 smoke。
4. 新入口必须在干净环境中通过，不把 `/alphabrain/.venv` 或个人缓存路径写死为唯一选择。

### Gate 4：收口文档和产物

1. 新增唯一 `CURRENT_STATUS` 或 release README，指向最终协议、结果和复现命令。
2. 冻结协议是历史记录，不回写“当时尚未完成”的文字；用 superseding status 文档说明后来已 release。
3. 只把最终论文使用的 PDF/图放在主文档目录；旧 v1-v5 和 page preview 进入 artifact archive。
4. 每张论文表/图都应能映射到：输入 manifest、分析命令、输出文件和 Git commit。

## 6. 大整理的验收标准

完成不以“文件少了”为准，而以以下条件全部满足为准：

- 一个新读者只看一个 README 就能找到正式问题、数据、checkpoint、命令和结果边界。
- 从冻结 manifest 可重建论文中的每张核心表和图。
- 正式流水线不存在依赖 `after_*`、个人 tmux session 或隐式 shell history 的步骤。
- 所有路径可通过配置覆盖；默认值不把某台机器的绝对路径当成科研定义。
- valid、invalidated、smoke、historical 四类产物不会被同一个默认 glob 混读。
- 新旧流水线在一个小型 golden subset 上得到完全相同的任务 ledger、noise hash 和统计输出。
- 相关测试可由仓库根目录一次执行，且不需要手工追加脚本目录到 `PYTHONPATH`。
- 原冻结 commit/tag 与原始实验目录保持不变，可随时审计。

## 7. 给原主任务的直接忠告

当前先把正式 view-value expectation 实验跑完，不要为了代码美观中断或改写已经 release 的流水线。
完成后请不要在原证据分支上“边删边试”，而是：

1. 先冻结并验证最终证据；
2. 建独立 release worktree；
3. 从正式 manifest 反向求依赖闭包；
4. 先抽核心模块和稳定 CLI，再归档历史脚本；
5. 用 golden subset 证明整理前后科学结果不变；
6. 最后才处理 PDF 旧版本、preview、旁支研究目录和真正的一次性脚本。

最需要避免的两种做法是：

- 因为名字像 `pilot` 或 `invalidated` 就直接删除，造成审计链断裂；
- 把当前 143 个脚本简单搬进新目录，却没有形成唯一入口和可验证的依赖闭包。

大整理的目标不是“看起来整齐”，而是让论文结论、正式代码、冻结配置和原始证据形成一条能由第三方
独立复现的最短路径。

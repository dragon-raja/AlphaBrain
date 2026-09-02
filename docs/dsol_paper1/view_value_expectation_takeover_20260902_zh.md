# View-value expectation 正式接管记录（2026-09-02）

## 1. 定位

本文记录 `dsol-view-value-expectation-v1` 从原 Codex 任务转交到当前任务后的所有权边界、恢复点和不可变证据。
它不是新实验协议，不改变候选空间、筛选规则、噪声分布、统计门槛或结果解释。

- 原任务：`codex://threads/019f5a82-51f3-7d72-9c9d-bdfc6f55502a`（读取接口已返回 stale）
- 当前接管任务：`codex://threads/01a0412b-e1eb-7b40-919e-b28cceae9fc0`
- 仓库：`/workspace/projects/alphabrain-dsol-paper1`
- 分支：`exp/dsol-paper1-v0`
- 接管时 HEAD：`a74841b5975088318eba633b960a00714b5e75f5`
- 正式产物根目录：
  `/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1`
- 接管审计时间：`2026-09-02T06:04:21Z`

原任务如果恢复，不得再次启动该目录的 calibration、post-calibration 或 finalize controller；正式写入所有权已经
转到当前任务。同一 calibration root 由 `calibration-tail.lock` 单实例锁保护。

## 2. 接管时不可变证据

| 对象 | 接管时 SHA256 / 状态 |
|---|---|
| 冻结机器协议 `configs/dsol_paper1/view_value_expectation_protocol_v1.json` | `bcce527fe9f98e32af07ca9a48a487563abfc742f23ca7a8fe11a6d0bca003dc` |
| population `population/population.json` | `878ec0a0bd18322c935ba3c4c115d310bfbd9125b36a7d6a0a9d1be9229ac698` |
| 噪声库 receipt `noise-banks-h10/noise_banks_receipt.json` | `25ccc1232ee0ee4a54437418aae9d782b478cc073a06ec7d59bb175f2f6e9e39` |
| 正式执行 receipt | `PASS_EXECUTION_RELEASED`，`formal_execution_authorized=true` |
| Stage A | `6208/6208` episodes，有效目录为 `calibration/stage-A` |
| Stage B协议 | `PASS`，`3072` episodes，SHA256 `039f9f15fe89cd6bccc08e450555836865f0b875e1001395bdf5fb4af5d2cb59`，`selection_uses_current_stage_outcomes=false` |
| 接管后calibration-tail controller | SHA256 `bf481ef0a4cf4a0adec5d0af4e73d4844c8bcc4b5598a6a291cea1179f8987ad` |

上述三个SHA与
`receipts/execution-release-formal-seed41.json` 中冻结的值完全一致。旧 `noise-banks/`、
`stage-A-invalidated-checkpoint-family` 和 `stage-A-invalidated-nonwritable-buffer` 继续保留审计，但不得进入正式分析。

## 3. 接管时进程边界

接管检查没有发现以下活动进程或监听端口：

- `run_view_value_expectation_*`
- `evaluate_dsol_libero_hdf5_views.py`
- `serve_alphabrain_pi05_websocket.py`
- 正式端口范围 `22400-22999`

GPU上仅有既有 `gpu-keepalive-0..7`；它们不写正式实验目录。Stage A最后写入时间早于接管审计，Stage B目录
尚未产生正式 episode，因此不存在需要合并的并行写入。

## 4. 停止原因与最小修复

Stage A正常完成并生成Stage B协议后，`run_view_value_expectation_calibration_tail.sh` 在读取新阶段计数时退出：

1. Stage B输出目录尚不存在；
2. `find` 返回非零；
3. controller使用 `set -euo pipefail`；
4. `actual=$(count_rows "$output")` 在启动 evaluator 前终止脚本。

接管修复只涉及控制层：缺失目录计为0、运行新阶段前显式创建输出目录，并给同一正式 calibration root 加非阻塞
单实例锁。冻结协议、Stage B JSON、Flow噪声张量、checkpoint和evaluator语义均不修改。

新增回归测试覆盖“新阶段目录不存在时返回0”及“先创建目录再计数”。恢复前还必须通过Shell语法检查、相关
view-value单元测试、release SHA复核和进程互斥复核。

恢复前检查结果：Stage B协议从Stage A原始ledger独立重建后与现有文件逐字节一致；Stage A的6208个episode ID
无重复且与Stage A协议精确集合匹配，所有记录均使用Bank A显式噪声，64个`state × repeat`组的共同replan噪声
哈希检查通过；controller及核心view-value测试共10项通过。

## 5. 恢复顺序

1. 从既有Stage B协议和Bank B开始，不重跑Stage A。
2. Stage B完成3072条后，仅以B结果构造Stage C；C完成1536条后构造并冻结Stage D候选。
3. Stage D完成2048条后生成 `calibration/analysis/analysis.json`。
4. 只有校准完成marker和analysis同时存在，才运行Accel及heldout Bank E。
5. Bank F只按预注册precision decision打开；不能因结果方向不理想而手工打开或关闭。
6. 最终报告只读取正式calibration和heldout analysis，不读取invalidated或Stage A hindsight最大值作为确认结论。

## 6. 接管后的结论纪律

- Stage A只用于候选筛选，不能据此声称稳定好视角。
- Stage D回答固定E0候选空间是否存在稳定headroom；不等于主动相机控制。
- E/F回答冻结规则能否迁移到source-disjoint未见状态；未通过时应报告未确认或精度不足。
- 不把noise repeat当作独立物理样本；统计单位仍是source demonstration，并保持task stratification。
- 当前证据仍限定于AlphaBrain π0.5 family、LIBERO固定外部相机和S0/E0问题边界。

## 7. 后续状态写入规则

运行状态以正式目录中的episode ledgers、run manifests、analysis JSON和completion marker为准，聊天文字不作为证据。
接管过程中如发生任何失败，先保留原目录和日志，再判断能否按episode ID幂等续跑；不得删除或覆盖有效episode来
“重新跑干净”。

## 8. 正式恢复记录

- calibration controller恢复时间：`2026-09-02T06:07:07Z`附近；正式tmux会话为
  `dsol-view-expectation-calibration-v1`。
- post-calibration与finalize看护会话于`2026-09-02T06:07:58Z`接入，只等待上游completion marker，未提前运行分析。
- Bank B的8个策略服务均在`22500-22507`就绪，32个评测worker开始写入`calibration/stage-B`。
- 首批44条Stage B记录的接管抽查通过：全部为显式Bank B噪声，manifest SHA为
  `e4e192477a2329f6b61f58f3ed094628074d9ca80dbde8387ae1bb1b3173ae62`，每次policy调用均记录
  `noise_seed`、`noise_sha256`和`action_chunk_sha256`。
- 恢复时没有重跑或修改Stage A，没有重建/改写正式噪声库，也没有打开Bank C/D/E/F结果。

### Stage B 512条里程碑与阶段间门控加固

Stage B越过512条后，对当时532条ledger执行了深审计：episode ID唯一且属于冻结协议，32-way shard归属正确，
同一pair的physics SHA和environment seed一致，camera安装/等待前后physics SHA精确一致；并直接从Bank B `.npy`
逐调用重建13,621次Flow噪声，所有`noise_seed`和`noise_sha256`均与policy-call ledger一致。

为避免阶段只凭“行数相等”进入下一步，`validate_explicit_pairing`随后加固为阶段间fail-closed门：要求前一阶段
episode集合与其协议精确相等、关键spec字段逐条一致、replan索引连续、repeat ID一致、物理状态和环境seed配对一致、
noise/action SHA字段有效且共同噪声不分叉。Stage C构造会重新读取这段代码，因此Stage B必须通过该门后才会被用于
筛选；Stage D分析同样要求Stage D ledger与协议精确匹配。加固后15项相关测试通过，并从完整Stage A重新生成
Stage B协议，结果仍与冻结文件逐字节一致（SHA256仍为`039f9f15...cb59`），证明没有改变候选选择定义。

### Stage B 完成与 Stage C 切换（2026-09-02T08:54Z）

Stage B于`2026-09-02T08:54:37Z`达到`3072/3072`，controller记录
`{"episodes": 3072, "stage": "B", "status": "PASS"}`后32个Stage B evaluator全部退出；运行期间正式目录
没有产生error文件。完成后对全部ledger执行了独立审计，而不是只依赖controller行数：

- 3072个episode ID唯一，并与冻结Stage B协议精确同集；18个关键spec字段逐条相等；
- 32个shard各96条，归属满足`protocol_index % 32 == shard_index`；
- 全部记录状态为`complete`，均使用显式Bank B；run manifest中的协议SHA与正式协议文件SHA同为
  `039f9f15fe89cd6bccc08e450555836865f0b875e1001395bdf5fb4af5d2cb59`；
- 独立重建全部133,436次policy call的Flow噪声，`noise_seed`和`noise_sha256`全部匹配；同一
  `pair × repeat × replan`键没有噪声分叉；
- 每个pair的physics SHA和environment seed一致，等待前后physics SHA精确相等。

Stage B ledger通过加固后的阶段间门后，controller才生成Stage C协议。正式Stage C协议SHA256为
`b6a9e16f6e666eb906f3b4f055c793e0e04b352c6d80c6c5a78e447d19b6b7b0`；从冻结population、scan、
Stage B协议及完整Stage B ledger独立重建后与正式文件逐字节一致。协议规模为
`16 states × 6 candidates × 16 repeats = 1536 episodes`，episode ID唯一，
`selection_uses_current_stage_outcomes=false`，`selection_previous_stage=B`，且记录的前序协议SHA与Stage B精确匹配。

Stage C随后在端口`22600-22607`启动8个策略服务和32个evaluator，run manifest锁定上述Stage C协议SHA、
Bank C manifest并要求显式噪声。首批144条ledger抽查通过：均属于Stage C协议，32个shard均已有产出，
1683次policy call的Bank C噪声可独立精确重建，初态physics SHA与environment seed配对一致。这里的Stage B
成功数和Stage C早期表现只参与预注册筛选/运行审计，不作为论文确认性结论；确认边界仍保留给Stage D和heldout E/F。

### Stage C 完成与 Stage D 切换（2026-09-02T10:16Z）

Stage C于`2026-09-02T10:16:23Z`达到`1536/1536`，controller记录
`{"episodes": 1536, "stage": "C", "status": "PASS"}`后Stage C evaluator全部退出；正式目录没有error文件。
全部ledger的独立审计结果如下：

- 1536个episode ID唯一，并与冻结Stage C协议精确同集；18个关键spec字段逐条一致；
- 32个shard各48条，全部状态为`complete`，run manifest锁定的协议SHA与正式Stage C协议一致；
- 独立重建全部62,571次policy call的Bank C Flow噪声，`noise_seed`和`noise_sha256`全部匹配，
  `pair × repeat × replan`共同噪声没有分叉；
- 每个pair的physics SHA和environment seed一致，等待前后physics SHA精确相等。

Stage C通过阶段间门后才生成Stage D协议。正式Stage D协议SHA256为
`3d99c62b0aa252dd1e1b15b17b3b7bbbda3615176120d2a81e398c2722a97eec`；从冻结population、scan、
Stage C协议和完整Stage C ledger独立重建后与正式文件逐字节一致。协议规模为
`16 states × 2 candidates × 64 repeats = 2048 episodes`，episode ID唯一，
`selection_uses_current_stage_outcomes=false`，`selection_previous_stage=C`，前序协议SHA与Stage C精确匹配。
两个候选是canonical与按预注册排序从Stage C选出的一个noncanonical候选；Stage D本身不再参与候选筛选。

Stage D随后在端口`22700-22707`启动8个策略服务和32个evaluator，run manifest锁定上述Stage D协议SHA、
Bank D manifest并强制显式噪声。首批148条ledger抽查通过：均属于Stage D协议，32个shard均已有产出，
1290次policy call的Bank D噪声可独立精确重建，18个关键spec字段、初态physics SHA和environment seed一致。
Stage D是校准确认阶段，但仍只回答固定E0候选空间内的校准headroom；最终迁移证据必须等待source-disjoint heldout E/F。

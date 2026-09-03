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

### Stage D 完成、校准结论与 post 恢复（2026-09-02T12:08Z）

Stage D于`2026-09-02T12:01:52Z`完成，controller记录
`dsol_hdf5_closed_loop_complete=.../calibration/stage-D episodes=2048`并自然退出。全部ledger的独立审计结果为：

- 2048个episode ID唯一，与冻结Stage D协议精确同集；18个关键spec字段逐条一致；
- 32个shard各64条，全部状态为`complete`；run manifest中的协议SHA与正式Stage D协议同为
  `3d99c62b0aa252dd1e1b15b17b3b7bbbda3615176120d2a81e398c2722a97eec`；
- 独立重建全部81,782次policy call的Bank D Flow噪声，`noise_seed`和`noise_sha256`全部匹配，
  `pair × repeat × replan`共同噪声没有分叉；
- 16个pair的physics SHA和environment seed一致，等待前后physics SHA精确相等。

校准analysis确定性重算后，`analysis.json`和`state_results.csv`均与正式产物逐字节一致。正式状态为
`VIEW_HEADROOM_NOT_CONFIRMED`：16个状态中没有状态同时满足“冻结候选成功率至少80%且相对canonical提升至少20pp”，
因此strong state、strong source group和strong task计数均为0，不能声称候选空间对非平凡状态子集存在稳定强headroom。
这不是“平均效果为零”：source-equal平均success gain为`+4.78515625pp`，source-cluster bootstrap 95%区间为
`[2.63671875, 7.12890625]pp`；它表示平均小幅改善与预注册的强状态门是不同结论，二者不得混写。

heldout selector gain是协议中的另一条独立主张，post controller按冻结设计无条件继续Accel及E/F，而不是看到负结果后
临时加跑。第一次Accel render启动时，8个shard均在写入ledger前因render分支的`PYTHONPATH`漏掉
`/projects/openpi/packages/openpi-client/src`而fail-closed，报`ModuleNotFoundError: openpi_client`；正式render ledger仍为0，
没有部分结果需要删除或合并。最小修复仅补齐与rank/正式闭环runner相同的import路径，并新增controller回归测试；
Shell语法、实际SIM Python导入及10项相关测试通过，修复提交为`00d4511`。旧post/finalize会话和Accel进程确认退出、
8个GPU keepalive恢复后，重新创建同名post/finalize会话。

恢复后8个render shard均正常运行并完成`64/64`。独立render审计确认：64个状态与冻结population精确同集，
8个shard各8条；64个artifact共699,261,506 bytes，全部重新计算SHA并与ledger匹配；每个artifact的输入数组shape
正确，97个候选由1个canonical、64个broad-train和32个broad-heldout组成，且候选顺序跨64状态一致。随后rank阶段
在8个GPU上加载seed41 checkpoint，每个状态使用冻结的8-member ensemble；render/rank均不读取heldout闭环结果。

### Accel完成、heldout冻结与E41启动恢复（2026-09-02T13:44Z）

Accel rank完成`64/64`。终审确认8个shard各8条、pair key与冻结64状态精确同集，ledger和每状态
`ranking.json`完全一致；每状态97个候选及8个member score完整，ensemble seed可由
`stable_seed("accel::{pair_key}::{member}", root_seed=20260921)`重构，mean/std、排序与top-2 margin均可精确复算，
render artifact和physics SHA绑定一致。按shard顺序拼接的rank ledgers SHA256为
`80d40a71f44fab6bec91ca935697f46f30c18844696cc20f5f2b240d32c436ac`。Top-1类型计数为canonical 13、
broad-train 33、broad-heldout 18；这只是Accel排名描述，不能替代heldout闭环价值结论。

随后冻结的三个Bank E协议从同一输入独立重建后逐字节一致：seed41为9216条、SHA256
`f5327dcd2860ee67dc9e6ab557e3f1aacbac311bf5872cac13f85fad35b04636`；seed42为3072条、SHA256
`f3d8ac66c814d4ca3c922295aa04ecdb0fbb96fd9a5f26b405fd4759e7d02e38`；seed43为3072条、SHA256
`bf6f9af3f57de3ccdea42178f69080a039e348c546475b0a251f5ccf172af76c`。48个heldout状态在pair/source两层均与
16个calibration状态不相交；seed41冻结六种规则，seed42/43仅比较canonical和校准集冻结最佳规则。
最佳规则为`calibration_global_fixed_pose`，固定候选为`broad_train_053`；选择过程声明并审计为不读取heldout policy outcome。
Bank E shape为`[48,32,104,10,7]`，manifest SHA256为
`ab9da7b01c1bef26eb7689fc853f85d2a7adbbee3908810381bf900ae1111f8f`，noise文件SHA256为
`7fa11794c606efdcae67e30ea489e86ba48c1f9ffe287e5c771e3233d9f8c378`，状态清单与48个heldout pair精确相等。

第一次E41启动时，32个evaluator在写入任何episode前统一报`KeyError: 'catalog'`：heldout协议已在每条spec中冻结
`catalog`，但evaluator只读取可选的协议顶层字段。正式E41 ledger仍为0，8个policy server和全部evaluator退出，
GPU keepalive恢复，因此没有部分结果需要删除或合并。最小修复保留三个协议及其SHA不变：evaluator优先使用顶层
catalog，缺失时读取per-spec catalog，两处都缺失则fail-closed。真实seed41协议预展开为32个shard各288条，
20项相关测试通过，修复提交为`869f746`。另在heldout分析入口增加结果episode集合和关键身份字段必须与冻结协议逐条
相等的fail-closed门，提交为`c1844b3`，不改变估计量或统计阈值。

post/finalize于`2026-09-02T13:41:37Z`重新启动，render/rank因完整64条自动跳过，三个协议确定性重建后E41恢复。
首批41条独立审计通过：31个shard已有产出，episode均属于seed41冻结协议且shard归属正确，状态/方法/候选/repeat/
checkpoint/environment seed逐字段相等；683次policy call的Bank E `noise_seed`和`noise_sha256`可从正式noise文件逐次重构。
早期成功数只反映协议顺序中的首个状态/方法，不作效果解释；正式结论仍等待E41完整9216条及seed42/43确认。

### E41 3072条里程碑与heldout转场门控加固（2026-09-02T16:10Z）

E41越过3072条后，对3299条并发快照执行了可重复累计审计，receipt写入
`heldout/primary-seed41/audits/partial-after-3072.json`，SHA256为
`e01e869f63249ffad670494f15d21cd9769fecf70329cd8d106fac4822cbc7e4`。审计证明：3299个episode ID唯一、均为
冻结seed41协议的正确shard子集，全部spec字段逐条一致；32个shard各97至111条，19个状态已触及，其中16个状态的
六方法×32 repeats矩阵精确完整。117462次policy call全部从Bank E文件重新取数并复算`noise_seed`和
`noise_sha256`，25874个共同`pair × repeat × replan`键没有分叉；每个已触及pair的physics SHA和environment seed
唯一，camera安装/等待前后physics SHA精确相等。该快照只验证运行完整性，不读取或解释中途成功率。

上述检查固化为`audit_view_value_expectation_heldout_run.py`：partial模式只接受冻结协议的唯一正确shard子集，complete
模式进一步要求episode集合与协议精确相等、没有并发尾写且各shard人口精确。审计器同时绑定run manifest中的协议和
噪声manifest SHA、核验显式噪声标志、逐调用重建Bank E/F张量，并输出带审计器自身SHA的receipt。post controller的
`run_heldout_seed`已加fail-closed完整审计，未来重启或新启动的E/F阶段只有生成`heldout-run-audit.json`后才能转入
下一checkpoint或统计分析。当前正在运行的Bash进程在启动时已加载旧函数，因此这一次现存进程的E41/E42/E43及可能的
F41/F42/F43完成点均由接管任务显式运行同一complete审计，并保证在统计分析定稿前核清；不得误认为修改磁盘上的shell
会热更新现存进程。

### E41 4096条里程碑（2026-09-02T17:18Z）

E41越过4096条后，使用提交后的heldout审计器对4114条并发快照再次执行累计审计。receipt位于
`heldout/primary-seed41/audits/partial-after-4096.json`，SHA256为
`2bb1a599d55f11751e4737198986b3ee8921ea556978296446ff02a4418aa772`。4114个episode ID唯一，均属于冻结协议并位于
`protocol_index % 32`规定的shard，全部spec字段与协议一致；32个shard各119至136条。23个状态已触及，其中19个
状态的六方法×32 repeats矩阵精确完整。全部169901次policy call可由Bank E逐调用重建，37703个共同
`pair × repeat × replan`键无噪声分叉，已触及pair的physics SHA与environment seed均唯一且camera安装/等待前后
physics SHA精确相等。

审计后进程侧复核为8个seed41 policy service、32个evaluator父进程和32个当前episode子进程；32份evaluator日志
没有Traceback、CUDA OOM、连接失败或KeyError，post/finalize tmux会话均存活。与前一里程碑相同，这只证明运行和
配对数据完整性，不对未完成矩阵的中途成功率作任何结论。

### E41 5120条里程碑（2026-09-02T18:33Z）

E41越过5120条后，使用同一已提交heldout审计器对5143条并发快照执行累计审计。receipt位于
`heldout/primary-seed41/audits/partial-after-5120.json`，SHA256为
`4b4b10513f2de94eeeb9c3a0faa5763aeff5fc4846f0eb22f07d2ce6d3387e8f`。5143个episode ID唯一，均属于冻结协议并位于
`protocol_index % 32`规定的shard，全部spec字段与协议一致；32个shard各153至169条。29个状态已触及，其中25个
状态的六方法×32 repeats矩阵精确完整。全部232061次policy call可由Bank E逐调用重建，51463个共同
`pair × repeat × replan`键无噪声分叉，已触及pair的physics SHA与environment seed均唯一且camera安装/等待前后
physics SHA精确相等。审计快照超过5120是并发worker在只读审计开始前继续完成episode所致，不是重复或越界写入。

审计后进程侧复核为8个seed41 policy service、32个evaluator父进程和32个当前episode子进程；32份evaluator日志
没有Traceback、CUDA OOM、连接失败或KeyError，post/finalize tmux会话均存活。本里程碑仍只证明冻结协议、结果集合、
配对噪声与运行健康性，不对未完成矩阵的中途成功率作任何结论。

### E41 6144条里程碑（2026-09-02T20:08Z）

E41越过6144条后，使用同一已提交heldout审计器对6160条并发快照执行累计审计。receipt位于
`heldout/primary-seed41/audits/partial-after-6144.json`，SHA256为
`e327b7e855e7a0c3b32bf50152e19b1e5b6aee639de65243e432efebfcc1fcc6`。6160个episode ID唯一，均属于冻结协议并位于
`protocol_index % 32`规定的shard，全部spec字段与协议一致；32个shard各184至199条。34个状态已触及，其中30个
状态的六方法×32 repeats矩阵精确完整。全部313482次policy call可由Bank E逐调用重建，67429个共同
`pair × repeat × replan`键无噪声分叉，已触及pair的physics SHA与environment seed均唯一且camera安装/等待前后
physics SHA精确相等。审计快照超过6144是并发worker在只读审计开始前继续完成episode所致，不是重复或越界写入。

审计后进程侧复核为8个seed41 policy service、32个evaluator父进程和32个当前episode子进程；32份evaluator日志
没有Traceback、CUDA OOM、连接失败或KeyError，post/finalize tmux会话均存活。本里程碑仍只证明冻结协议、结果集合、
配对噪声与运行健康性，不对未完成矩阵的中途成功率作任何结论。

### E41 7168条里程碑（2026-09-02T21:22Z）

E41越过7168条后，使用同一已提交heldout审计器对7179条并发快照执行累计审计。receipt位于
`heldout/primary-seed41/audits/partial-after-7168.json`，SHA256为
`d8d570297175fe2c97f45b1633c507e390cd2ca00cbc1368395ad39e8ef7eb09`。7179个episode ID唯一，均属于冻结协议并位于
`protocol_index % 32`规定的shard，全部spec字段与协议一致；32个shard各204至252条。42个状态已触及，其中34个
状态的六方法×32 repeats矩阵精确完整。全部375308次policy call可由Bank E逐调用重建，77994个共同
`pair × repeat × replan`键无噪声分叉，已触及pair的physics SHA与environment seed均唯一且camera安装/等待前后
physics SHA精确相等。审计快照超过7168是并发worker在只读审计开始前继续完成episode所致，不是重复或越界写入。

审计后进程侧复核为8个seed41 policy service、32个evaluator父进程和32个当前episode子进程；32份evaluator日志
没有Traceback、CUDA OOM、连接失败或KeyError，post/finalize tmux会话均存活。本里程碑仍只证明冻结协议、结果集合、
配对噪声与运行健康性，不对未完成矩阵的中途成功率作任何结论。

### E41 8192条里程碑（2026-09-02T21:57Z）

E41越过8192条后，使用同一已提交heldout审计器对8235条并发快照执行累计审计。receipt位于
`heldout/primary-seed41/audits/partial-after-8192.json`，SHA256为
`1c5bc0a0090132fd7222134b5223a76a6bb84d115823194532fcf23bed4b33fb`。8235个episode ID唯一，均属于冻结协议并位于
`protocol_index % 32`规定的shard，全部spec字段与协议一致；32个shard各227至280条。47个状态已触及，其中37个
状态的六方法×32 repeats矩阵精确完整。全部400837次policy call可由Bank E逐调用重建，83093个共同
`pair × repeat × replan`键无噪声分叉，已触及pair的physics SHA与environment seed均唯一且camera安装/等待前后
physics SHA精确相等。审计快照超过8192是并发worker在只读审计开始前继续完成episode所致，不是重复或越界写入。

审计后进程侧复核为8个seed41 policy service、32个evaluator父进程和32个当前episode子进程，post/finalize tmux
会话均存活。32份evaluator日志没有Traceback、CUDA OOM、连接失败或KeyError。8份policy日志中的Traceback经逐类
核查仅为启动期原始TCP就绪探测在WebSocket HTTP握手前断开所产生的`EOFError`/`InvalidMessage`，随后持续正常接受
`connection open`；policy日志没有CUDA OOM、连接失败或KeyError。这些握手探测日志不代表episode或推理失败。
本里程碑仍只证明冻结协议、结果集合、配对噪声与运行健康性，不对未完成矩阵的中途成功率作任何结论。

### E41完整审计与E42转场（2026-09-02T22:43Z）

E41正式写满9216/9216后，以`--require-complete`模式独立审计并生成
`heldout/primary-seed41/heldout-run-audit.json`，receipt SHA256为
`a7d1845a933ad46f96e70011a3b07d3b5e94f6dcd7ec4ce925224feb13146f33`，状态为`PASS_COMPLETE`。
结果episode集合与冻结seed41协议逐条精确相等且全部唯一，32个shard均精确为288条；六种方法各1536条，48个状态的
六方法×32 repeats矩阵全部完整。全部431405次policy call均可由Bank E逐调用重建，89095个共同
`pair × repeat × replan`键无噪声分叉，48个pair的physics SHA和environment seed均无违规。协议SHA、Bank E
manifest SHA和noise文件SHA继续与冻结值一致，且不存在并发尾写。

旧post controller随后打印`closed_loop_complete ... episodes=9216`并启动8个seed42 checkpoint policy service；
检查时E42尚处模型加载阶段、结果为0/3072。由于当前存活Bash进程未热加载后来加入的自动audit函数，E42/E43完成后仍须
由接管任务显式运行同一`--require-complete`审计，全部通过后才允许接受primary heldout统计分析。

### E42 1024条里程碑（2026-09-02T23:30Z）

E42越过1024条后，同一heldout审计器对1038条并发快照审计通过。receipt位于
`heldout/primary-seed42/audits/partial-after-1024.json`，SHA256为
`1d8b809a922c70e7563966f5fa562c4a932a4a99883ad0adc71d7301572332df`。1038个episode ID唯一，均属于冻结seed42
协议并位于正确shard，全部spec字段一致；32个shard各28至35条，18个状态已触及，其中14个状态的
两方法×32 repeats矩阵精确完整。canonical与`calibration_global_fixed_pose`分别有526和512条。全部34996次
policy call可由Bank E逐调用重建，19771个共同`pair × repeat × replan`键无噪声分叉，physics SHA和
environment seed违规均为0。

进程侧以可执行名过滤后复核为8个seed42 policy service、32个evaluator父进程和32个episode子进程；32份evaluator
日志无Traceback、CUDA OOM、连接失败或KeyError，policy日志无硬错误且仅含与E41相同的原始TCP就绪探测握手噪声，
post/finalize会话均存活。本里程碑不读取或解释中途成功率。

### E42 2048条里程碑（2026-09-03T00:51Z）

E42越过2048条后，同一heldout审计器对2061条并发快照审计通过。receipt位于
`heldout/primary-seed42/audits/partial-after-2048.json`，SHA256为
`902549acf7602037777ec4bd0dd29896d16a9bf1ed9882dd5bf474b20da8be7b`。2061个episode ID唯一，均属于冻结seed42
协议并位于正确shard，全部spec字段一致；32个shard各62至68条，34个状态已触及，其中31个状态的
两方法×32 repeats矩阵精确完整。canonical与`calibration_global_fixed_pose`分别有1038和1023条。全部103803次
policy call可由Bank E逐调用重建，58966个共同`pair × repeat × replan`键无噪声分叉，physics SHA和
environment seed违规均为0。

进程侧复核为8个seed42 policy service、32个evaluator父进程和32个episode子进程；32份evaluator日志无目标错误，
policy日志无CUDA OOM、连接失败或KeyError，post/finalize会话均存活。本里程碑仍不读取或解释中途成功率。

### E42完整审计与E43转场（2026-09-03T01:38Z）

E42正式写满3072/3072后，以`--require-complete`模式独立审计并生成
`heldout/primary-seed42/heldout-run-audit.json`，receipt SHA256为
`7f3bf39af49224eef51c3e6b1bf64171b51df844eef32a2e39dda3733453d8d6`，状态为`PASS_COMPLETE`。
结果episode集合与冻结seed42协议逐条精确相等且全部唯一，32个shard均精确为96条；canonical与
`calibration_global_fixed_pose`各1536条，48个状态的两方法×32 repeats矩阵全部完整。全部141387次policy call
均可由Bank E逐调用重建，78739个共同`pair × repeat × replan`键无噪声分叉，48个pair的physics SHA和
environment seed均无违规，且不存在并发尾写。

旧post controller随后打印seed42 `closed_loop_complete ... episodes=3072`并启动8个seed43 checkpoint policy
service；检查时E43尚处模型加载阶段、结果为0/3072。E43完成后继续由接管任务显式执行完整审计，在E41/E42/E43三份
receipt全部通过并复核分析绑定之前，不接受primary heldout统计结果或reserve gate结论。

### E43 1024条里程碑（2026-09-03T02:26Z）

E43越过1024条后，同一heldout审计器对1044条并发快照审计通过。receipt位于
`heldout/primary-seed43/audits/partial-after-1024.json`，SHA256为
`118c8cadcfc2fb656ea74a36b1261dcb82ff6c657fc4a5bc0b7cc87de7f3b4e4`。1044个episode ID唯一，均属于冻结seed43
协议并位于正确shard，全部spec字段一致；32个shard各29至35条，18个状态已触及，其中14个状态的
两方法×32 repeats矩阵精确完整。canonical与`calibration_global_fixed_pose`分别有531和513条。全部35515次
policy call可由Bank E逐调用重建，20047个共同`pair × repeat × replan`键无噪声分叉，physics SHA和
environment seed违规均为0。

进程侧复核为8个seed43 policy service、32个evaluator父进程和32个episode子进程；32份evaluator日志无目标错误，
policy日志无CUDA OOM、连接失败或KeyError且仅含原始TCP就绪探测握手噪声，post/finalize会话均存活。本里程碑不读取
或解释中途成功率。

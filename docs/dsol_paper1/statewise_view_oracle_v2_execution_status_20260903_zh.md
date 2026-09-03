# 状态条件视角 Oracle v2：执行状态与接管点

最后结构更新：2026-09-03 08:34 UTC

## 当前状态

v2 已正式启动。执行根目录是：

`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2`

当前阶段是 development O / wave-00。该 wave 的冻结规模为 `48 states × 97 views × 4 repeats = 18,624 episodes`；development O 总计 8 个 wave、32 条完整 flow-noise 轨迹。运行状态应以 episode ledger 和 audit 为准，不以本文件中的瞬时计数为准。

两个持久 tmux owner：

- `dsol-oracle-v2-dev-O`：连续执行 8 个 development O wave，并在每个 wave 后做 outcome-blind 完整性审计。
- `dsol-oracle-v2-tail`：等待 8 个 O audit 全部通过，然后自动执行 development P/Q、selector 冻结、test O/P/Q、seed-42/43 transfer 和最终分析；如果精度规则命中，再执行 R。

2026-09-03 的吞吐优化验证曾在完整 episode 边界暂停 wave-00；暂停时保留了 1,456 条完整结果。验证结束后，控制器按 episode ID 从该断点恢复，不重跑或改写已有结果。

## 吞吐优化验证

独立非主结果目录：

`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/throughput-benchmark-v1`

基准冻结了 wave-00 第一个状态的完整 `97 views × 4 O-noise repeats = 388 episodes`，保持原始 episode identity、物理状态和 O bank 不变，并明确标记为 `excluded_from_scientific_estimands=true`。三组结果如下：

| 推理服务 | 仿真 worker | CPU 线程 | 总耗时（含模型启动） | 吞吐 | 相对 8/32 |
|---:|---:|---:|---:|---:|---:|
| 8（每卡 1） | 32 | policy 2 / sim 1 | 274.08 s | 5,096.25 ep/h | 1.000× |
| 16（每卡 2） | 64 | policy 2 / sim 1 | 252.21 s | 5,538.22 ep/h | 1.087× |
| 16（每卡 2） | 32 | policy 2 / sim 1 | 249.38 s | 5,601.20 ep/h | 1.099× |

三组 audit 均为 `PASS_COMPLETE`。选定的 16/32 结果与暂停前旧进程、8/32 基线逐 episode 比较：success、步数、goal progress、初始/等待后物理哈希，以及每个 replan 的 noise/action-chunk SHA-256 全部完全一致，388/388、0 mismatch。比较 receipt：

`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/throughput-benchmark-v1/comparison-candidate-8x2-32.json`

64 个仿真 worker 没有额外净收益，因此正式控制器采用每卡 2 个 policy service、总计 32 个 sim worker、policy 2 CPU threads、sim 1 CPU thread。单卡模型常驻约 17.6 GB，未触及 32 GB 显存上限。该基准状态全部成功、平均仅 36.08 步，不能把 5,601 ep/h 直接外推到全任务；它主要用于拓扑 A/B 和精确等价性验证。

## 已冻结证据

- 机器协议：`/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/statewise_view_oracle_v2.json`
- 学习器协议：`/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/statewise_view_selector_learner_v1.json`
- 人口文件：`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/population/population-v2.json`
- 准备 receipt：`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/preparation/receipt.json`
- S smoke audit：`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/smoke/S/audit.json`
- O/P/Q/R noise manifests：`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/noise-banks/`

准备 receipt 为 `PASS_READY_FOR_SMOKE`：48 个 development 状态、16 个 test 状态、64 个完整 97-view visibility scan 和 64 个完整 97-view policy-input image bank 均已核验；`legacy_rollout_outcomes_pooled=false`。

S smoke 为 `PASS_COMPLETE`：4 个 episode、31 次 policy call 全部逐 tensor 匹配 S bank；同一 state/repeat/replan 下视角间噪声相同；相机安装前后物理状态哈希相同。某一 repeat 中 canonical 7 次 replan 后结束、另一个视角为 8 次，证明可变终止长度不会移动或消耗对方的噪声序列。

## 自动顺序

1. Development O：全 97 视角×32 噪声，分 8 个可续跑 wave。
2. Development P：O-only 选出的 canonical + Top-8，使用独立 P×32。
3. Development Q：P-only 冻结 Top-1，与 canonical 使用独立 Q×64。
4. Development-only 6-fold source CV；冻结 global/task fixed、visibility、Accel、geometry-context ridge 和 image-queryable ridge 在全部 16 个 test 状态上的选择。
5. Test O：冻结 selector 之后才运行全 97×32；不在 wave audit 中汇总 outcome。
6. Test P/Q：用 O→P 选择 statewise candidate，Q 同时评测所有预先冻结 selector，重复候选只 rollout 一次。
7. Seed-42/43：复用 seed-41 已冻结的 Q candidate matrix 和完全相同 Q noise，解释为 checkpoint transfer，不冒充各 checkpoint 的 dense oracle。
8. Final analysis：task-stratified source bootstrap；O/P winner rate 不作为最终效应。
9. 若冻结精度规则命中：复制完全相同 Q candidate matrix 到 R bank，禁止替换候选，再追加 64 noise repeats。

## 恢复与检查

只读进度检查：

```bash
tmux list-sessions | grep 'dsol-oracle-v2'
tail -50 /share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/logs/development-O-controller.log
tail -50 /share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/logs/v2-tail-controller.log
```

控制器和通用 evaluator 都以 episode ID 断点续跑。进程异常退出后，重新启动同名脚本会跳过已存在 episode；每个阶段只有在完整 episode set、物理状态、环境 seed 和每次 explicit-noise call 全部通过审计后才进入下一阶段。

不要同时启动第二个 development-O 或 tail owner；两个脚本都有 `flock` 防重入。不要手动打开/汇总 test O outcome，也不要在 selector-freeze receipt 写入后改变模型、alpha 或候选选择。

## 预计时间

旧的 10–14 天估计基于未限制 CPU 线程时约 1,000 episodes/hour 的历史长期吞吐，已被本次运行优化取代。恢复后的正式多状态窗口完成约 900 个 episode，平均 116.18 环境步，观测约 190 环境步/秒；按 v1 完整任务分布约 200–240 步/episode 校正，当前全程规划吞吐约为 2,800–3,400 episodes/hour，而不是短状态基准中的 5,601 episodes/hour。

不触发 R 时的最大设计规模约 247,808 个 primary episode；触发 R 时最多再增加约 8,192 个。以当前断点和长度校正吞吐估算，完整 O/P/Q、selector freeze、test 开封、seed-42/43 transfer 和最终分析约还需 **3–5 天**；若机器持续稳定，预计 2026-09-06 至 2026-09-08 UTC 闭环。第一整个 wave-00 完成后应以跨 48 状态的实测 wall time 再校准一次。

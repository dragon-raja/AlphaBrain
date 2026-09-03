# 状态条件视角 Oracle v2：执行状态与接管点

最后结构更新：2026-09-03 07:42 UTC

## 当前状态

v2 已正式启动。执行根目录是：

`/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2`

当前阶段是 development O / wave-00。该 wave 的冻结规模为 `48 states × 97 views × 4 repeats = 18,624 episodes`；development O 总计 8 个 wave、32 条完整 flow-noise 轨迹。运行状态应以 episode ledger 和 audit 为准，不以本文件中的瞬时计数为准。

两个持久 tmux owner：

- `dsol-oracle-v2-dev-O`：连续执行 8 个 development O wave，并在每个 wave 后做 outcome-blind 完整性审计。
- `dsol-oracle-v2-tail`：等待 8 个 O audit 全部通过，然后自动执行 development P/Q、selector 冻结、test O/P/Q、seed-42/43 transfer 和最终分析；如果精度规则命中，再执行 R。

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

按上一轮长期平均约 1,000 episodes/hour，主 checkpoint 的 dense、P/Q 和确认约 9–11 天；selector、test 开封和分析约 1–2 天；seed-42/43 transfer 约 1–2 天。若运行稳定，预计在 2026-09-13 至 2026-09-17 之间闭环。短时吞吐会随任务成功/失败导致的 rollout 长度显著变化，不能用最初几百条线性外推。

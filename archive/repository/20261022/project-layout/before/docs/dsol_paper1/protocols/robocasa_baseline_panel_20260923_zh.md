# RoboCasa 起点权重小面板

用户已批准：只补后训练前权重，10 任务 × 每任务 2 seen/2 heldout 初态 × 5 视角 × 4 执行重复，共 800 次。不重训，不恢复已取消的全量归因补充。

## 冻结规则

- 每任务、每场景分组按 `(scene_seed, initial_id)` 排序取前两个，不读取结果选择初态。
- 一个规范视角，四个额外留出视角；在按 `[30°,15°,0.2]` 归一化的轨道坐标中，依次选择最接近 `(-1/3,-1/3,0)`、`(1/3,1/3,0)`、`(-1,-1,1)`、`(1,1,1)` 的不重复额外候选，距离并列按候选 ID。
- 实际视角：canonical、extra_002、extra_031、extra_030、extra_011。轻/强仅指这个小面板的相对几何组，不等于旧分析的方位角阈值分组。
- 沿用 P0 相同 host/GPU 分配，双机各 4 个工作进程，每进程 100 次。
- 官方起点使用原 Human300 75k 权重。冻结物理初态，首帧输入直接复用 P0 规范后训练模型的同候选存档；其后使用真实闭环观测。原 native MSAA4 不改。
- 沿用原地址化执行噪声 0–3，评分噪声仍独立 8 次，10 步去噪、每 25 步重规划、原生任务时限均不变。
- 两份 8250 步最终权重复用对应 P0 记录，不新跑。历史多视角模型与规范模型如有首帧渲染微差，在汇总审计中单独记录，不假称逐像素一致。

## 结果边界

分开报告 seen/heldout、规范/轻扰动/强扰动；比较起点、规范适配、多视角适配的绝对成功率与相对起点增益。10 个任务均为已见任务。四扰动平均不是整个候选空间平均，不估计 Oracle，不构成主动相机部署实验。五视角最低 Accel 仅为小面板辅助诊断。

完整性审计检查 800 条基线任务、来源哈希、物理初态、首帧和逐重规划噪声。任务→初态分层 bootstrap 保持已测四次执行均值固定，不称精确总体不确定性。失败保留，不自动重试、不放宽阈值、不扩展任务。

## 执行位置

- 实验根目录：`/share/longjunyu/alphabrain/experiments/dsol-robocasa-baseline-panel-20260923`
- 冻结计划：根目录 `plan.json`；选点：`selection.json`；运行状态：`status.json`。
- 结果：`analysis/summary.json`、`analysis/report.md`、`analysis/checkpoint_comparison.png`，完成审计后生成。
- 未修改唯一当前 PDF；小面板结果需要解释审查后再决定纳入正式汇报。

## 实现位置

数据协议 `AlphaBrain/research/dsol/data/baseline_panel.py`；分析 `analysis/baseline_panel.py`；绘图 `plotting/baseline_panel.py`；执行调度 `benchmarks/Robocasa365/baseline_dispatch.py`。运行沿用已有 attribution worker 与 P0 rollout 函数，不复制新仿真器链路。

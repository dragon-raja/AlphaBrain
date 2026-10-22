# 实验看板接入

看板只负责记录与展示，不是训练或评测的前置依赖。本地原始日志是证据来源。
本次导入历史数据，不启动新训练、模型推理、仿真或旧队列。

## 入口与上传范围

- `reports/paper1/search_tracking.py`：核对已完成收据、整理现有训练/A/B结果、显式导出。
- `AlphaBrain/research/dsol/execution/experiment_tracking.py`：可选 W&B 出口；默认离线，不被训练代码自动调用。
- 原始路径和输入哈希留在 `local-provenance.json`；只上传白名单指标、筛选配置和指定统计图。
- 不上传数据集、观测图、视频、动作数组、模型、机器人资产、源码、终端输出和机器环境。
- 固定来源内容决定导入 run ID；已完成同内容的线上 run 跳过，未完成的同 ID 停止供审查。
- SDK依赖在独立看板环境安装，本次验证版本 `wandb==0.23.1`；不修改模型/仿真环境。
  顶层依赖见 `configs/dsol_paper1/tracking-requirements.txt`，完整安装锁定清单保存在运行目录的 `protocol/`。

六个 run 分为四组：三个训练种子；历史标签回放；真实仿真 A/B；共同时间上限复核。
训练横轴是原始搜索决策步，PPO更新数、查询数另列；奖励不是成功率。
旧日志的 `completed_episodes` 是该次采样批次的完成回合数，不是累计回合数；回合均值也按该批次计算。
A/B 是最终模型任务成绩，不伪造成沿训练不断评测的曲线。A是已知任务新初态，
B是搜索RL未训练过的任务，不宣称VLA未见任务或跨机器人泛化。
真实A/B统一300动作步；共同时间上限复核单列，保留327条不一致。
成功率按任务宏平均，伤害率分母为全部配对试验；缺失历史标签保留空值。

## 登录与授权

1. 浏览器打开 <https://wandb.ai/>，登录个人或团队账户；确定 entity（空间名）。
2. 在 <https://wandb.ai/settings> 创建 API Key。
3. 在需要上传的服务器终端运行专用环境的 `wandb login --host https://api.wandb.ai`，
   按交互提示粘贴 Key。不要放进聊天、命令参数、仓库或配置文件。
4. 使用账户有写权限的 entity；项目默认 `dsol-paper1`。浏览器登录不会自动授权服务器。
5. 登录后显式执行 `publish --mode online`。未授权时只完成离线导出，不自动尝试其他凭据。

```bash
python -m reports.paper1.search_tracking prepare \
  --training TRAINING_DIRECTORY --audit PAIRED_AUDIT_DIRECTORY \
  --formal-config FORMAL_CONFIG_JSON --output NEW_RUN_RESULTS_BUNDLE

python -m reports.paper1.search_tracking publish \
  --bundle NEW_RUN_RESULTS_BUNDLE --output NEW_RUN_LOGS_EXPORT \
  --entity YOUR_ENTITY --project dsol-paper1 --mode offline
```

`prepare` 不需要 W&B。`publish` 使用专用环境 Python，从仓库根目录运行；
只有显式 `online` 才上传。输出目录必须不存在，防止覆盖旧收据。
离线测试用的实体占位符不能直接 `wandb sync` 到个人账户；登录后用原整理包重新 publish。

## 连续PPO训练接入

`reports/paper1/continuous_training_tracking.py` 是独立侧进程，读取连续训练器已提交的回合与更新记录。
分组固定为 `continuous-pose-ppo-v1`，job_type为 `continuous-search-training`；不能复用旧97点run。
记录奖励、起始/最佳Accel、查询数、STOP比例、actor/critic损失、熵、KL和耗时。
横轴为累计完成回合；累计决策数、查询数、去噪轨迹数和优化更新数分别保存，不画成成功率。
本地JSON/PT/NPZ仍是证据源，网络故障不改变训练奖励和优化过程。

```bash
python -m reports.paper1.continuous_training_tracking --config TRACKING_CONFIG --follow
```

`mode=offline` 的合成日志映射先经真实SDK验证，再使用明确授权的online配置。
云端仅接收选定标量、配置和图表；原始图像、动作、模型、私有URDF和本地路径不上传。
训练器首32回合/首次更新必须通过实际账本和完整输出检查，再继续剩余回合；不按奖励高低筛选实验。
启动需要新的源码/资产清单和资源复检；旧冻结清单保持不变。

检查：`pytest tests/dsol_paper1/execution/test_experiment_tracking.py tests/dsol_paper1/execution/test_continuous_tracking.py`，以及仓库架构、布局和portable检查。

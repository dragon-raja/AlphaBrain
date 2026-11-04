# FastWAM 少量适配关闭批次诊断

运行产物：持久项目 `runs/fastwam-diagnostic-20261029/`。输入为 `runs/fastwam-small-validation-20261029/` 的已完成科学块；失败与显式恢复记录保持原样。

当前工作：核对候选覆盖/选位、PPO训练与奖励结算；绘制固定任务初态的 yaw/pitch/Accel 与操作成功比例；补齐原生特征候选的 CPU 读取、空间整理、融合和搜索策略入口。新增训练/模型调用为零；FastWAM 原生特征实际 GPU 开关等价、模型真实 token 网格、正式采集存储与新绑定未验收。未接入生产，未扩大科学预算。

禁止把未执行候选标为失败，禁止把重复同噪声执行计成独立样本；两次执行的比例只是观测结果，不是可信成功概率。逐状态相关性来自自适应选择的终点子集，不能冒充全域响应面。p9 可以用已存十步轨迹离线重算，但缺少其选位的操作标签时不能估计新策略成功率。

研究代码归属：`AlphaBrain/research/dsol/analysis/target_search_diagnostic.py`、`plotting/target_search_diagnostic.py`、`selectors/native_search_policy.py`；报告入口 `reports/paper1/target_search_diagnostic.py`。新测试纳入 portable；torch相关测试在独立解释器显式隐藏GPU执行。旧 π0.5 特征与连续actor文件哈希保持不变。

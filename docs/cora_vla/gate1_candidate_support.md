# CORA-VLA Gate 1：基础策略候选支持度

本报告严格使用既有 validation snapshot groups；新 confirmation groups 保持封存，未用于本 Gate。基础 Full-H Pi0.5 全部冻结，本阶段未训练任何新模型。

## 主要结果

| 指标 | 跨 seed 组级均值 | 95% bootstrap CI |
|---|---:|---:|
| Attached 联合 correct-mode recall@16 | 74.4% | [59.0%, 87.2%] |
| Slipped 联合 correct-mode recall@16 | 100.0% | [100.0%, 100.0%] |
| Slipped 联合 correct-mode recall@32 | 100.0% | [100.0%, 100.0%] |
| Slipped physical success@1 | 89.7% | [76.9%, 100.0%] |
| Slipped oracle physical success@16 | 100.0% | [100.0%, 100.0%] |
| Slipped oracle @16 相对 @1 增益 | 10.3% | [0.0%, 23.1%] |

## 各 checkpoint seed

- seed 41：attached recall@16=84.6%，slipped recall@16=100.0%，slipped physical @1/@16=100.0%/100.0%。
- seed 42：attached recall@16=76.9%，slipped recall@16=100.0%，slipped physical @1/@16=92.3%/100.0%。
- seed 43：attached recall@16=61.5%，slipped recall@16=100.0%，slipped physical @1/@16=76.9%/100.0%。

## 完整 recall@N

| 分支 | N | Action | EEF effect | 联合主标签 | 短时物理标签 |
|---|---:|---:|---:|---:|---:|
| attached | 1 | 100.0% | 7.7% | 7.7% | 100.0% |
| attached | 4 | 100.0% | 33.3% | 33.3% | 100.0% |
| attached | 8 | 100.0% | 48.7% | 48.7% | 100.0% |
| attached | 16 | 100.0% | 74.4% | 74.4% | 100.0% |
| attached | 32 | 100.0% | 89.7% | 89.7% | 100.0% |
| slipped | 1 | 92.3% | 100.0% | 92.3% | 89.7% |
| slipped | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| slipped | 8 | 100.0% | 100.0% | 100.0% | 100.0% |
| slipped | 16 | 100.0% | 100.0% | 100.0% | 100.0% |
| slipped | 32 | 100.0% | 100.0% | 100.0% | 100.0% |

标签一致率：attached action/physical=98.2%、joint/physical=13.9%；slipped action/physical=97.0%、joint/physical=96.7%。

## 解释与边界

联合 correct-mode 标签要求 action 距离和 K=2 EEF-effect 距离同时支持正确 continuation。物理上界还要求分支特定即时行为正确，并由同一 teacher 完成后续任务；因此它不是单纯的 teacher 可救回率。

反馈前泄漏审计：通过。统计单位为 snapshot group，CI 未把候选或帧当作独立样本。

## Gate 1 裁决

**STOP_CORA_CANDIDATE_SUPPORT**

按预注册停止规则，本路线不进入能量模型训练。

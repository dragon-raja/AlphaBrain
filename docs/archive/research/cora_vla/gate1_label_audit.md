# CORA-VLA Gate 1 候选标签审计

审计对象为 validation split、三个冻结 Full-H checkpoint、attached/slipped 各 1,248 个候选。短时 physical label 仅作为本次标签审计参照，不用于事后改写原 Gate 指标。

## Attached

| 标签相对 physical | TP | FP | TN | FN | 一致率 | physical-success recall |
|---|---:|---:|---:|---:|---:|---:|
| Action distance | 1,225 | 0 | 0 | 23 | 98.2% | 98.2% |
| EEF effect | 175 | 0 | 0 | 1,073 | 14.0% | 14.0% |
| Joint | 174 | 0 | 0 | 1,074 | 13.9% | 13.9% |
| Teacher completion | 1,248 | 0 | 0 | 0 | 100.0% | 100.0% |

Attached 的 1,248 个候选全部保持分支特定物理可行并可由同一 teacher 完成任务，但 joint 标签拒绝了其中 1,074 个。误差几乎全部来自 EEF-effect：它将候选两步 EEF 位移与单条 teacher continuation 比较，而抓取后的 lift/transport 存在许多可行局部方向。由此可判定 Attached 低联合 recall 主要是“替代轨迹不同于单条 teacher，但物理上仍成功”，不是 action mode 缺失。

## Slipped

| 标签相对 physical | TP | FP | TN | FN | 一致率 | physical-success recall |
|---|---:|---:|---:|---:|---:|---:|
| Action distance | 1,168 | 38 | 42 | 0 | 97.0% | 100.0% |
| EEF effect | 1,165 | 80 | 0 | 3 | 93.3% | 99.7% |
| Joint | 1,165 | 38 | 42 | 3 | 96.7% | 99.7% |
| Teacher completion | 1,168 | 42 | 38 | 0 | 96.6% | 100.0% |

Slipped physical success rate为 93.6%，teacher completion rate为 97.0%。Teacher 能救回并不自动等于候选即时行为正确：有 42 个候选可由 teacher 完成，但候选自身产生 empty lift 或未进入恢复方向。Joint 对 slipped physical label 的区分总体可靠，只有 3 个 physical-success 假阴性。

## 可视化

输出目录：`/share/longjunyu/fresh-vla/cora-vla/gate1-label-audit-v1/videos`

- Attached：20 个真实 `joint=false, physical=true` 完整候选 + teacher completion 视频；
- Slipped：全部 3 个真实 joint 假阴性，加 17 个明确标注为 `physical_success_context` 的成功上下文样例；
- 每个样例同时保存 contact sheet；
- 40 个视频均验证为 H.264/avc1、yuv420p、faststart，40/40 最终成功。

原始混淆矩阵：`/share/longjunyu/fresh-vla/cora-vla/gate1-label-audit-v1/audit.json`  
逐样例清单：`/share/longjunyu/fresh-vla/cora-vla/gate1-label-audit-v1/videos/manifest.json`

审计结论只修正科学解释，不修改 `STOP_CORA_CANDIDATE_SUPPORT` 的形式裁决，也不事后替换 Sequential Oracle 的主指标。

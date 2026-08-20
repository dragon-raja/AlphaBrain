# G0 Seed-41 被动视角校准门结果

日期：2026-08-20  
决策：`PASS_G0_WITH_RETENTION_AUDIT_REQUIRED`

## Camera Full 正式结果

| 模型 | 成功数 | Camera Full 成功率 |
|---|---:|---:|
| Official Pi0.5 frozen | 1,213 / 1,599 | 75.86% |
| Broad64 seed 41 | 1,321 / 1,599 | 82.61% |

Broad64 的 pooled 提升为 `+6.75pp`。以 40 个原始 base task 为独立单位做
10,000 次 paired cluster bootstrap 后，平均提升为 `+5.23pp`，95% CI 为
`[+0.12pp, +10.69pp]`。

该结果满足预注册的被动视角门：Camera Full 提升至少 `+5pp`，且 base-task
cluster CI 排除 0。因此宽范围多视角训练值得继续扩展，不再停留在旧 Legacy8
的小扰动结论。

## 分层结果

| 分层 | Official | Broad64 | 变化 |
|---|---:|---:|---:|
| Difficulty 1 | 97.49% | 95.48% | -2.01pp |
| Difficulty 2 | 91.39% | 91.67% | +0.28pp |
| Difficulty 3 | 80.31% | 84.31% | +4.00pp |
| Difficulty 4 | 75.44% | 82.46% | +7.02pp |
| Difficulty 5 | 52.77% | 69.61% | +16.84pp |
| Orbit yaw | 70.43% | 81.94% | +11.51pp |
| Radius | 77.64% | 84.98% | +7.34pp |
| Combined camera | 78.05% | 82.09% | +4.03pp |

收益主要来自更困难、更大角度的相机条件，符合 Broad64 扩大视角 support 的预期；
Difficulty 1 略有下降，说明不能只看 pooled 指标。

## Exact-state 诊断

| 模型 | Canonical | Broad held-out | Wide extrapolation |
|---|---:|---:|---:|
| Official Pi0.5 frozen | 100.0% | 37.5% | 45.8% |
| Broad64 seed 41 | 87.5% | 87.5% | 95.8% |

24 个严格配对组的物理状态哈希不一致数为 0。该诊断支持 Broad64 显著扩大可用
视角范围，但也显示 canonical 可能下降 `12.5pp`。样本仅 24 组，不能单独判定
基础能力退化；必须用 Original LIBERO Full 做 retention 审计。

## 下一步

1. 先运行 Official 与 Broad64 seed 41 的 Original LIBERO Full，检查 canonical retention；
2. retention 合格后进入 G1：Legacy8 / Broad32 / Broad64 和 1x / 2x / 4x exposure；
3. 进入 G2：unpaired practical / state-matched / paired FM / paired consistency；
4. 仅扩展通过机制门的方法到 seeds 42/43；
5. Blind-Reveal、M1 和 Accel 仍按完整研究 DAG 继续，不由本次 Camera Full 结果替代。

本次结论是“宽覆盖路线通过 seed-41 正式视角门”，不是“全部研究问题已经解决”。

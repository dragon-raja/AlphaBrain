# CORA-VLA On-policy Candidate Support

状态全部来自冻结 Full-H 自己的 slipped 闭环轨迹；teacher 只用于候选执行后的可恢复性审计，不参与状态生成。Confirmation 保持封存。

## 总体

共评估 211 个可达 on-policy replanning states、13 个 snapshot groups。

| N | Correct-mode recall |
|---:|---:|
| 1 | 76.8% |
| 4 | 91.5% |
| 8 | 94.3% |
| 16 | 96.2% |

组级 recall@16=96.4%，paired group bootstrap 95% CI=[93.2%, 99.0%]。Teacher-state slipped recall@16=100.0%，差值=3.6%。

## 分阶段

| 阶段 | 可达 seed-state 数 | recall@1 | recall@4 | recall@8 | recall@16 | teacher可恢复@16 |
|---|---:|---:|---:|---:|---:|---:|
| feedback_reveal | 39 | 97.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| failure_continuation | 10 | 30.0% | 50.0% | 50.0% | 60.0% | 100.0% |
| recovery_start | 39 | 94.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| reapproach | 39 | 97.4% | 100.0% | 100.0% | 100.0% | 94.9% |
| preclose | 38 | 13.2% | 71.1% | 84.2% | 92.1% | 94.7% |
| post_regrasp | 32 | 96.9% | 96.9% | 96.9% | 96.9% | 100.0% |
| final_failure | 14 | 71.4% | 92.9% | 100.0% | 100.0% | 71.4% |

## 各 seed

- seed 41：recall@1/4/8/16=78.9%/93.0%/94.4%/97.2%，自然闭环成功=76.9%。
- seed 42：recall@1/4/8/16=79.2%/84.7%/91.7%/94.4%，自然闭环成功=53.8%。
- seed 43：recall@1/4/8/16=72.1%/97.1%/97.1%/97.1%，自然闭环成功=61.5%。

## Gate

**PASS_ONPOLICY_SUPPORT**

on-policy correct-mode recall@16 达到 60% 必要条件，允许补充冻结策略 continuation 并进入一次 Sequential Oracle 上界。

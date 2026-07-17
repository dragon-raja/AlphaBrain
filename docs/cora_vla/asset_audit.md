# CORA-VLA 资产审计

状态：`CORA_ASSET_AUDIT_COMPLETE`

## 冻结基础策略

| Seed | Optimizer steps | Checkpoint | SHA256 |
| ---: | ---: | --- | --- |
| 41 | 10353 | `/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed41_steps13804_formal-v2/checkpoints/steps_10353` | `732da869fe5aab23ae83f6b517bb33a83bb0b5e7cea9c2535edc9388f07d61c4` |
| 42 | 10353 | `/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed42_steps10353_formal-budget-v2/checkpoints/steps_10353` | `73d23cc8659ab7510eecdd013b1ffdc48c2ea97304ec14b3cf886906fc4da90a` |
| 43 | 10353 | `/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed43_steps10353_formal-budget-v2/checkpoints/steps_10353` | `cfd9547bde803ca83430bae37675aeb61e534f0d490f2b1d233ab3289baec4c4` |

三个 checkpoint 均来自已通过 validation baseline gate 的 Full-H，CORA 不得重新训练或修改它们。

## 反事实数据

- 完整 episode：`/share/longjunyu/fresh-vla/libero-full-episode-v2-128`；128 个 snapshot groups。
- Split：`{'train': 102, 'val': 13, 'test': 13}`；按 source initial state 隔离。
- Branch：128 attached + 128 slipped/recovery。
- 视频：128 个成对视频，256 个 branch 视频。
- event/reveal/divergence 范围：[53, 74] / [53, 74] / [53, 74]。
- 滑动窗口：34551；train/val/test groups 为 {'test': 13, 'train': 102, 'val': 13}。
- post-feedback windows：20311；质量门通过：`true`。

## Recovery Support v2

正式结论：`STOP_OFFLINE_SUPPORT_EXPANSION`。其 correction trajectories、policy-state failures 和逐帧图像只作为 CORA 候选/负样本资产，不再用于微调基础 VLA。

| Seed | Retained groups | Windows | Full-teacher success | Frozen-policy downstream |
| ---: | ---: | ---: | ---: | ---: |
| 41 | 101 | 7155 | 98.0% | 85.7% |
| 42 | 100 | 7224 | 99.0% | 85.7% |
| 43 | 101 | 7565 | 99.0% | 95.2% |

## 可用评测集合

开发只允许使用 train 与 13 个 validation groups。已有 Full-H fixed K=1/2/3、isolated、end-to-end 和 deterministic reach 结果均已发现。

原 test 的 13 个 groups 曾被 auxiliary deterministic-reach 触碰；严格 pristine=`false`。它们不得作为 CORA 最终确认集。

## Confirmation 封存协议

- 冻结生成 24 个新的 grasp/slip snapshot groups，source seed=`2026071701`，full-episode seed=`2026071702`。
- 生成后只运行自动质量门和 snapshot fingerprint 去重，不查看 CORA 指标。
- Seal 路径：`/share/longjunyu/fresh-vla/cora-vla/confirmation-v1-24/seal.json`。
- 仅在 Gate 1、Gate 2、validation rerank 全部冻结且 Gate 3 正式启动时解封一次。

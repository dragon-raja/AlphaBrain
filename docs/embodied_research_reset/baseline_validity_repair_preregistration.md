# Full-H 基线有效性修复预注册

状态：在查看本修复实验的任何新结果之前冻结。

## 为什么必须先修基线

当前证据不能用于裁决新的恢复方法：原始 Full-H 及从其继续训练的模型在
validation 闭环上的 attached 成功率不稳定，而额外 13,804 个 batch-1 更新虽然
降低了离线 flow loss，却没有改善闭环成功率。进一步代码审计发现，普通单卡和
DDP 路径从未进入 `Accelerator.accumulate()`；当配置的梯度累积大于 1 时，旧循环
仍会逐 microbatch 清梯度并更新 optimizer。既有正式运行的累积配置均为 1，因此
该缺陷没有篡改已有结果，但它说明训练基础设施尚未经过有效 batch 验证。

本实验只修复 baseline validity。它不属于 FRESH、恢复 replay 或新方法收益。

## 冻结训练配方

- 方法：仅 `fresh_closed_loop_full_h`；
- 初始化：`/share/longjunyu/alphabrain/pretrained_models/pi05_base`；
- 数据：`libero-full-episode-windows-v2-128` 的 train split；
- 测试集：保持封存；
- 分布式：8 卡 DDP，bf16；
- 每卡 batch：1；梯度累积：1；真实 global batch：8；
- 冻结：`vlm_interface`；可训练参数约 693M；
- optimizer：AdamW，betas/weight decay/clip 延用当前 Pi0.5 配置；
- peak LR：`5e-5`；warmup：1,000 optimizer steps；warmup 后保持 `5e-5`；
- EMA：关闭；
- 总预算：13,804 optimizer steps，即 110,432 个样本曝光，约四遍 train windows；
- 检查点：3,451、6,902、10,353、13,804 steps（约 1/2/3/4 epoch）；
- seeds：41、42、43。

采用 global batch 8 是最小的稳定配方：它与本地官方 OpenPI
`pi05_libero_lora` 的 batch 规模一致，又不依赖新修复的累积路径。官方 full
finetune 使用 batch 256 和 EMA，但在当前 8x32GB 机器上复制完整 EMA 或模拟
batch 256 会引入额外资源变量，因此本轮不启用。

## 两阶段预算冻结

1. 先训练 seed 41 到四个预注册检查点。
2. 对四个检查点仅运行相同 validation 离线集和 fixed `K=3` 闭环集。
3. 统一预算取最早同时满足以下条件的检查点：
   - attached full-task success 至少 30%；
   - overall full-task success 至少 25%；
   - 相对前一检查点没有出现超过 10 个百分点的 attached 回落。
4. 若没有检查点满足，统一预算冻结为 13,804，并仍训练 seeds 42/43，避免用单 seed
   误判基础设施。
5. 三 seed 的最终 baseline gate 沿用既有阈值：K=3 attached 跨 seed 均值至少
   30%，且至少两个 seed 达到 20%。不得在看到结果后改阈值。

所有 checkpoint 使用同一组 validation snapshot、policy seed schedule、episode
预算和视频编码。离线 loss 仅作诊断，不参与最终 baseline gate。

## 后续裁决

- gate 通过：baseline 有效，才允许用同一冻结预算训练 clean replay 与
  policy-state correction 两臂。
- gate 不通过：结论仍为 `BASELINE_INVALID_OR_DATA_INSUFFICIENT`，不打开 test，
  不把恢复臂的低成功率解释为方法失败。
- 若普通 Full-H 已解决恢复：采用简单基线，不包装新方法。
- 只有 policy-state correction 相对 clean replay 在等纠正帧预算下稳定提高恢复，
  才继续检验“自动寻找最早稳定能力区并蒸馏最短桥接段”的数据效率主张。


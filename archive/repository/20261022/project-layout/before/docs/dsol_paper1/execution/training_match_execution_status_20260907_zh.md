# 统一对照：匹配 canonical 补训完成

最初记录：2026-09-07。完成验收更新：2026-09-08。当前状态：`CANONICAL_MATCH_TRAINING_COMPLETED / ACTIVE_EVALUATION_NOT_RELEASED`。

用户已授权直接推进下一阶段并要求统一论文对照。2026-09-07仅启动一组 seed41 canonical 补训，现已完成。2026-09-08用户批准在新的独立预算中继续至静态视角分布/指标验证阶段；本页仅记录模型训练与产物验收，不把该批准写回旧training-only release，也不授权主动相机或RoboCasa实验。

## 0. 完成快照（2026-09-08）

- 训练于 **2026-09-07 07:47:55 UTC** 正常保存并结束；原日志为UTC+8，显示15:47:55。不是仍在运行。
- 实际metrics恰好包含连续1..2000步、每步32个model examples、最终64,000 exposures；所有训练loss有限。最终第一组LR为5e-6，完整2000步LR轨迹与Broad64 M-B seed41锚一致。
- 实际保存的config与manifest已重新审计：`saved_recipe_match=true`；除外部图像处理变量与运行身份外，配置及匹配manifest字段差异均为0。训练核心代码一致；唯一launcher hash变化为开训前已审查的显式初始checkpoint与provenance记录。
- 最终权重17,610,456,556 bytes，safetensors含935个tensor，payload边界连续且与文件长度一致。完整内容SHA256及不可覆盖完成回执见下方归档；未加载模型、未消耗GPU做推理。结构与内容身份验收不等于已检查所有tensor数值或已验证推理能力。
- 这里的“单视角”是**仅规范外部位姿后训练**，仍保留同record的腕部图、语言与动作监督；不是移除腕部相机。只有一组匹配seed41，不能据此宣称跨seed泛化。
- 本页验收未生成新闭环成功率。新的静态评测需绑定单独的协议与预算；主动获取仍未放行。

不可覆盖验收归档：

`/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/checkpoint-audit`

- `audit.json`：既有小产物审计器生成的实际训练配方对比。
- `completed_run_receipt.json`：完成步数、全LR轨迹、训练退出标记、完整权重内容hash与验收脚本身份；独占创建，拒绝覆盖。
- 验收实现：`/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/finalize_canonical_match_v1.py`；7个单元测试通过。该脚本不会启动训练或评测，也不会变更旧release。

新canonical最终权重完整SHA256：

`7e510b752143bb6fd4987988dfab5e94f7db1de5f7e97579f892a58eed22cc68`

旧training-only release完整SHA256仍为 `66c026931e54754ba7e444c7ddd9c8dfecb717d05257292a987742288de82568`，本次未改写。

以下第1–5节保留2026-09-07开训时点及其计划，**其中16步、预计耗时和“完成后执行”不是当前进度**。

## 1. 主对照统一方式

保留现有 Broad64 M-B seed41 作为锚，从**同一原始 pi05_libero_pytorch 权重**新训 canonical。两侧使用同一个 Broad64 数据容器及相同record的wrist、语言和动作监督，只改变外部训练图 `broad_a → canonical`。不是从旧 canonical 继续训练，也不是从 Broad checkpoint 再训练。

固定 seed41、2卡、每卡batch1、梯度累积16、GB32、2,000 optimizer updates、完整2,000步LR schedule和64,000 model examples。整个111字段配置经实际配置构建器与launcher CLI解析后，除arm/run_id/output_dir外与Broad保存配置零差异；训练核心代码一致。launcher新增的显式初始checkpoint/manifest处理已经审查，不改变该设置下训练数学。

旧canonical与v2 Broad不匹配的比较继续作为历史recipe证据，不覆盖或删除。quick canonical/Broad32仍可作匹配开发资产，但当前拟主对照不再为了复用quick而更换Broad锚。**新模型训练完毕并通过实际产物审计后，才称完成匹配模型对；现在不能发布训练交互或视角收益结论。**

## 2. 运行身份与预算

| 项目 | 实际值 |
|---|---|
| 启动时间 | 2026-09-07 04:58:16 UTC；训练日志采用UTC+8，本地文件名125837不可误读成UTC |
| tmux会话 | `dsol-canonical-mb-match-s41` |
| GPU与通信端口 | GPU0、1；port32241；启动前无项目训练/评测占用，端口空闲 |
| 输出根目录 | `/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_mb-matched-v1_seed41_g2_gb32_steps2000` |
| 执行预算 | 仅1个run，无自动重试；2,000步，预计约3小时；外层timeout在4小时发送TERM，最多120秒终止宽限 |
| 预算冻结文件 | `/workspace/projects/alphabrain-dsol-paper1/configs/dsol_paper1/canonical_mb_match_training_release_v1.json` |
| 冻结文件SHA256 | `66c026931e54754ba7e444c7ddd9c8dfecb717d05257292a987742288de82568`；运行manifest已引用 |
| 代码/依赖 | 训练Python3.12.3；当前2×RTX5090，bf16/DDP。核心版本核对见训练审计；历史GPU型号和numpy归档歧义仍如实保留 |

tmux中执行的是已有 `run_libero_pair_train.sh canonical_unique 41 2 2000 mb-matched-v1`，显式绑定上述参数、原始checkpoint、Broad64容器及预算文件，外层受timeout限制。launcher仅暂时让出GPU0/1的既有keepalive，退出时按原有机制恢复；未调整其余6卡。

**不要再次启动这个命令，不要删除输出目录后重试。**即使tmux已退出，也先检查训练产物和退出原因，不能自动当成失败或从头追加预算。

## 3. 已通过的实际预检与启动检查

- 原始权重14,467,165,872 bytes全量SHA256：`0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74`，与source manifest一致。
- Broad锚权重17,610,456,556 bytes全量SHA256：`123786f53c9823b878fb08fe61ef25fe4931d4bcae8134e0a64940b7a8ac3cad`，与v2运行身份一致。
- Broad64八个二进制分片及records索引全量hash复核，顶层/分片manifest绑定一致；共38,193条记录、400个episode，train31,440/val2,701/test4,052，128个record对源状态/动作抽检，11项检查PASS。实际二进制payload合计2,864,719,633 bytes。
- 训练启动日志确证：2进程NCCL/bf16、seed41、约698M trainable；桥接814/935，shape mismatch0；flow组209张量/LR5e-5、视觉组108张量/LR5e-4，均与锚一致。
- 截至本次启动检查，已完成 **16/2,000 optimizer steps**。前16步LR与examples_seen和Broad历史同编号步骤完全一致；没有将不同训练loss要求为相等。最近10步的平均每微批model_time约0.329秒，乘16约5.26秒/优化步，与约3小时预算一致。

上述16步是时点快照，不是实时进度。最终保存尚未完成，未计算新canonical最终权重hash。

新审计产物根：

`/share/longjunyu/alphabrain/experiments/dsol-training-match-20260907-lW8boo`

- `old_pair/audit.json`：旧canonical/M-B，saved_recipe_match=false。
- `quick_pair/audit.json`：quick canonical/Broad32，saved_recipe_match=true，但不是完整科学release。
- `canonical_proposal/audit.json`及`proposed_framework_config.yaml`：匹配补训意图配置，仍标DRAFT_NOT_TRAINED；保留为执行前快照，不倒填完成状态。
- `collection_reaudit.json`：本次数据复核PASS，不覆盖原数据集audit。

## 4. 同步完成的 P0 脚手架与恢复审计

新增 `scripts/dsol_paper1/audit_dsol_training_match.py`：读取实际保存配置、manifest、metrics与小文件身份，逐字段对比，生成不可执行的匹配提案；不会把相同步数等同于相同LR计划，也不会把提案文件当训练完成。

新增 `scripts/dsol_paper1/view_acquisition_inputs.py`：只从原观测读取当前 external/wrist RGB；policy只有双图+语言；selector的8D本体和当前外部标定必须显式传入。拒绝扩权契约，不透传对象/仿真状态、候选/未来图像；消费者数组独立、不可写；receipt不包含像素。它尚未连接完整VLA wire adapter，不擅自给base policy添加proprio。

新恢复审计见 [runtime_restore_contract_audit_20260907_zh.md](../audits/runtime_restore_contract_audit_20260907_zh.md)：旧HDF5 restart与新live runtime fork分开；后者需积分状态、控制器/夹爪、任务/相机、观测缓存、队列与RNG的同环境恢复。只恢复qpos不够，需比较接触后缀。首版取景限定自然replan空队列点，continue必须在构造hold控制器之前直接返回。

同时限定新运动学endpoint adapter的LIBERO-plus `noise=0`：非零wrapper图像扰动会导致step与直接端点观察的处理链不同，当前明确拒绝，不能默默混用。

验证：新旧相关测试合计 **91 tests、122 subtests通过**（Python3.12）。实际仿真Python3.8下新view-acquisition系列68 tests通过，training-match另11 tests通过。以上不是完整restore或真实VLA主动评测已通过。

## 5. 后续顺序与交接检查

训练过程中不改训练核心/loader/config和已冻结release。可以独立开发输入adapter与live snapshot，但不能把尚未验证的新接口混入旧静态runner或正在训练的模型。

完成后先执行：

1. 核对tmux/进程、日志退出、metrics连续1..2000和64,000 exposures、最终LR第一组5e-6及完整final_model。
2. 对实际新canonical `final_model/framework_config.yaml`和`run_manifest.json`运行新匹配审计，再核验最终权重hash。实际配置与意图不同就不认定匹配。
3. 通过后才发布同模型接口、来源/噪声/时限口径的成对评测。是否复用旧Broad O须满足原协议身份；新模型不能借用别的checkpoint行为。
4. 主动评测另等live restore、输入时序/配准、任务证据和能力门；不因训练完成自动放行。额外seed或规模扩展需要下一份预算，不由当前run自动衍生。

查看进度：

```bash
tmux capture-pane -pt dsol-canonical-mb-match-s41 -S -40
tail -n 1 /share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_mb-matched-v1_seed41_g2_gb32_steps2000/metrics.jsonl
```

统一研究方向无需用户再次决策。若后续涉及显著扩预算、修改论文主问题或真机/硬件协调，再明确提出选择及理由。

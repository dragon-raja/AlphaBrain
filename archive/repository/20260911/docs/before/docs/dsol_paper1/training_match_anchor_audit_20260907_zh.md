# Broad64 M-B seed41 → canonical 最小匹配训练审计（2026-09-07）

后续主代理执行记录见[训练状态](training_match_execution_status_20260907_zh.md)：实际运行已启动，端口采用32241且绑定独立训练预算回执。下文拟命令保留为审计时点材料，不能作为重复启动指令。

本轮仅只读核查代码、配置、训练归档、硬件与依赖；未启动训练、评测或 rollout。唯一写入为本文档。下列命令是拟运行规格，不是执行记录。

## 1. 结论

可直接从同一原始 `pi05_libero_pytorch` checkpoint 新训 canonical。必须使用与 M-B 相同的 Broad64 数据容器、seed41、2 卡 DDP、每卡 batch1、梯度累积16、2000优化步和2000步 scheduler；唯一研究变量是同一 record 的外部图像 `broad_a → canonical`。腕部图、语言、动作监督、数据顺序定义、训练参数范围不变。

限定本项目目录和 DSOL pairing runs 的搜索没有发现严格匹配且可直接复用的 canonical 完成 checkpoint。旧 canonical quick-gate 是8卡、2000实际步/3000 scheduler；calibration 是3000实际步。因此不能直接代替这个对照，更不能从旧 canonical 继续训练后宣称同初始权重同预算。

## 2. 锚点和完整源记录

锚点目录：

`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000`

其中：

- `final_model/model.safetensors`：17,610,456,556 bytes；这是 Broad 训练结果，绝不能作为新 canonical 的初始权重。
- `final_model/framework_config.yaml`：完整内容保存在附录A，当前 SHA256 `c22b3ffcb5de139e06e37ff95eadef00b01ee9db65c0ca6b46dd0db1c844b8d9`。
- `run_manifest.json`：完整内容保存在附录B，当前 SHA256 `b8709d7ff65243172ef1681d9938d8bb760a935b5c5b60772a53b8156feecbb1`。
- `logs/train_20260820_190955.log`、`launcher.log`、`metrics.jsonl`：实际起训、trainable/LR分组、初始权重加载、完成步数及耗时证据。
- 历史调度器：`/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/run_m_b_finalist_training.sh`。
- 历史控制记录：`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/training/controller.log`。

## 3. 严格匹配规格与有效代码行为

| 项目 | M-B 实际配置/代码行为 | canonical 必须保持 |
|---|---|---|
| 初始 checkpoint | `/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch`，非 resume | 完全同一路径与内容，不能用任何已后训练模型起训 |
| 初始权重来源 | Fisher-Wang/pi05-libero-pytorch，revision `ab107fbe6661e8abfec0f13dc27bfd3fafe24c69`；source manifest 记录 checkpoint SHA256 `0f8c489e37b01c72251c45f2e73595894f3933fc6297f4f1cf95fc8737db4c74` | 新运行前核验身份；本轮复核了 source manifest hash 与模型文件大小，没有重新读14.5GB做完整权重hash |
| 初始化 | 模型构建前 seed41；权重桥接匹配814/935、缺失121、shape mismatch0、adapted action projections3；prepare_training 后每 rank 使用 seed+rank | 同模型构建代码、seed和拓扑，保留桥接日志；不能把87.1%按参数量解读 |
| PaliGemma资源 | `/share/longjunyu/alphabrain/pretrained_models/paligemma-3b-pt-224`，SDPA，max_token_len48 | 不变；同路径已确认，但本轮未完整hash该资源目录 |
| 数据 | `/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2` | 使用同一个数据容器，不重新生成 canonical 数据 |
| 源样本/切分 | 8个任务分片；38193总记录，train31440、val2701、test4052，训练仅split=train | 只用同31440训练记录。继承字段data_mix=libero_goal不是实际“只训练Goal”的证据，DSOL reader读取该容器的8个分片 |
| 外部图读取 | arm=broad_unpaired_practical → broad_a | 唯一实验处理变量：arm=canonical_unique → canonical |
| 其他输入与标签 | 同record wrist、language_instruction、action_chunk；7维动作、horizon10；224×224；3图槽中只启用2槽 | 不变；这不是同时融合多个外部视角的训练 |
| 本体输入 | pi05=true、discrete_state_input=false，未消费state；include_state=false；reader仍会携带robot_state metadata不等于模型使用它 | 不变。selector使用state的其它研究不应混写为policy输入 |
| 数据加载 | per_device_batch_size1，shuffle generator seed41，4 workers/rank，drop_last=true，persistent_workers=true，pin_memory=true，timeout120，shard_cache2 | 不变；同rank数才保持分片/累积结构一致 |
| 拓扑和预算 | 单机2进程DDP，GPU0,1；GB32=2×1×16；每item1个模型样本；实际2000 optimizer steps，examples_seen64000 | 同2卡和grad_acc16；不为提速直接改8卡并声称完全匹配 |
| 精度/显存 | accelerate mixed_precision=bf16，训练autocast bf16；action expert bfloat16；action expert启用gradient checkpointing，冻结VLM语言模块不启用 | 不变 |
| 训练范围 | 日志Frozen3465M、Trainable698M；SigLIP54个fc1/fc2低秩适配层，rank16/alpha16/dropout0，共4,713,984适配参数；冻结视觉base、语言模型、lm_head、multi_modal_projector | 同视觉LoRA + action expert/flow head训练范围，不是“仅训练4.7M LoRA” |
| 有效optimizer | AdamW betas(0.9,0.95)、eps1e-8、weight_decay1e-10 | 读取trainer.optimizer.weight_decay；不是遗留trainer.weight_decay=0.0 |
| 有效LR分组 | 实际仅flow_matching_head：209张量、lr5e-5；vlm_interface：108张量、lr5e-4 | 不变；config中action_model/qwen_vl_interface/base值不应都当作已生效组 |
| Schedule | cosine_with_min_lr，num_warmup_steps5，scheduler_total_steps2000；min_lr配置1e-5，相对optimizer.defaults.lr=1e-4生成共同倍率 | 不变。共同floor倍率0.1使flow组终点5e-6、视觉组5e-5；metrics报告第一组，最后5e-6不是错误 |
| 损失/随机性 | 标准FM，noise beta alpha1.5/beta1.0；fresh_tail_weight1，consistency/CABI关闭，shared_flow_noise_time=false；无图像增强 | 不引入额外监督/一致性/增强；同seed设计不等于已证实所有GPU运算逐位确定 |
| 其它训练控制 | epochs100上限；max_train_steps2000生效；梯度clip1；EMA关；validation关；save/eval_interval2001；最终保存开启；is_resume=false | 不变，不新增周期性val选最佳checkpoint |

代码证据入口：

- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/dataloader/paligemma_datasets.py:655`：同record，canonical/broad_a分支；679/690为返回分支。
- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/dataloader/__init__.py:14`、75：固定seed shuffle与实际loader参数。
- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/model/framework/PaliGemmaPi.py:489`：实际gradient checkpointing；训练/推理state条件见该文件的pi05/discrete_state_input分支。
- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/training/trainer_utils/trainer_tools.py:68`：实际参数LR分组。
- `/workspace/projects/alphabrain-dsol-paper1/AlphaBrain/training/train_alphabrain.py:335`：optimizer/scheduler参数；1443：模型初始化seed；1375：最终保存完整config。
- `/workspace/projects/alphabrain-dsol-paper1/configs/deepspeed/accelerate_ddp.yaml`：bf16/MULTI_GPU，文件默认8进程被launcher CLI的2进程覆盖，并不是DeepSpeed训练。

## 4. 当前代码/环境是否漂移

锚点 commit `8dc602991834e4d87e3099187287dbdbd917f3fc`，当时git clean。审计时当前HEAD `50b3d4ff982b464ad0c7fef2f1da9095a9ea5554`，当前有新实验脚手架和文档未提交，不能把仓库整体称作clean。

已核实：相对锚commit，整个 `AlphaBrain/` 无diff；训练yaml、DDP配置、pair reader也无diff。列入锚manifest的训练关键文件中，唯一有变化的是launcher新增 `DSOL_PRETRAINED_CHECKPOINT` override及manifest记录；显式指定同一原始checkpoint时不改变训练数学。当前研究评测/数据构造脚本的变更不应一概说成训练混杂；使用现有冻结数据不重建。

运行时当前 `/alphabrain/.venv/bin/python` 只读导入确认：Python3.12.3、torch2.11.0+cu128、CUDA12.8、transformers4.57.0、accelerate1.5.2、numpy1.26.4、safetensors0.8.0。锚W&B归档requirements中上述核心包版本一致，numpy有1.26.4/2.4.3两条，不能仅据该重复安装清单确定历史实际import了哪条。

当前 nvidia-smi：8×NVIDIA GeForce RTX5090，32607MiB/卡，driver580.159.04。限定锚run和训练控制记录，本轮没有找到能直接证明历史GPU型号的可读机器快照；历史2卡/单机/NCCL CUDA运行时由launcher日志确证。故“同加速器逐位复现”尚未证实。硬件型号/驱动/负载变化主要影响时间和数值确定性；不能据此自动否定相同数学训练对照，但应随新run归档。

## 5. 旧 canonical 不足以严格复用

共同runs根：

`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs`

| 旧run目录名 | 已核实差异 | 处理 |
|---|---|---|
| dsol_canonical_unique_quick-gate-v1_seed41_g8_gb32_steps2000 | Broad32容器，8卡/accum4，2000实际步但scheduler3000 | 保留历史结果，不作为M-B matched canonical |
| dsol_canonical_repeat_quick-gate-v1_seed41_g8_gb32_steps2000 | 同上且每item重复2个canonical，accum2 | 不是canonical_unique同数据预算/随机性结构 |
| dsol_canonical_unique_calibration3000-v1_seed41_g8_gb32_steps3000 | 3000实际步、8卡，启用校准过程 | 不是2000步正式匹配checkpoint |
| 其它canonical 1/2/20步smoke | 非正式完成预算 | 仅工程记录 |

没有发现 Broad64 + 2卡 + 2000完整schedule + canonical_unique 的完成模型。结论是“限定搜索范围内无可复用候选”，不是证明所有机器任何目录都没有。

## 6. 精确拟运行命令（未执行）

此命令将由launcher创建全新输出目录；若目录已存在会拒绝覆盖。GPU0,1与port32941只是明确的拟分配，执行前主代理必须确认未被真实任务占用；不能因为launcher有keepalive处理就认为它会安全避让其它训练。

```bash
env \
  DSOL_PAIR_PYTHON=/alphabrain/.venv/bin/python \
  DSOL_PAIR_CONFIG=/workspace/projects/alphabrain-dsol-paper1/configs/experiments/dsol_libero_broad_pairing.yaml \
  DSOL_PAIR_DATA_ROOT=/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2 \
  DSOL_PAIR_OUTPUT_ROOT=/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs \
  DSOL_PRETRAINED_CHECKPOINT=/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch \
  DSOL_GLOBAL_EXAMPLES=32 \
  DSOL_SCHEDULER_STEPS=2000 \
  DSOL_SKIP_FINAL_SAVE=0 \
  DSOL_CALIBRATION=0 \
  DSOL_CALIBRATION_INTERVAL=250 \
  DSOL_CALIBRATION_ITEMS=256 \
  DSOL_BUDGET_DECISION= \
  DSOL_GPU_DEVICES=0,1 \
  DSOL_MAIN_PROCESS_PORT=32941 \
  PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
  WANDB_MODE=offline \
  OMP_NUM_THREADS=8 \
  bash /workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/run_libero_pair_train.sh \
  canonical_unique 41 2 2000 mb-matched-v1
```

目标目录：

`/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_mb-matched-v1_seed41_g2_gb32_steps2000`

launcher实际调用的核心是 `python -m accelerate.commands.launch --config_file configs/deepspeed/accelerate_ddp.yaml --num_processes 2 --main_process_port 32941 AlphaBrain/training/train_alphabrain.py --config_yaml ... --mode dsol_canonical_unique`，并覆盖run_id、seed、输出根、数据根、examples_per_item1、accum16、max_steps2000、scheduler2000、save/eval2001、validation=false/items256、skip_final_save=false、原始checkpoint。它显式设置 `USE_DDP=1`、`ALPHABRAIN_DDP_STATIC_GRAPH=0`、`CUDA_VISIBLE_DEVICES=0,1` 和项目PYTHONPATH，验证import来源。不要用 `bash -x` 代替预检，它仍会真正执行启动与GPU lease变更。

必要收据/完成验收：

1. 开始前保存本锚config/manifest的hash、本次实际解析完整config、当前diff/关键code hash、依赖/硬件身份、原始checkpoint/data manifest hash；不得只记CLI短命令。
2. startup日志匹配seed41、814/935桥接、0shape mismatch、约698M trainable、209/108两个LR组、batch1/accum16/GB32、原始checkpoint，禁止resume。
3. 本次完整framework_config与附录A除 `datasets.vla_data.dsol_arm`、`run_id`、`output_dir` 外应一致（输出根按上面相同）。其它diff逐一解释再认定匹配。
4. 完成必须有2000条连续optimizer-step指标、step2000/examples_seen64000、最终第一LR5e-6、完整final_model+完整framework_config、manifest及无异常退出证据。
5. 完成后的模型可成为匹配canonical对照；新训练本身不提供视角成功率新结论，仍须事先冻结、同评测协议的模型对照。

### 6.1 独立执行的无模型配置解析 preflight

在项目根调用真实 `build_config_from_finetune` 和 `normalize_dotlist_args`，读取当前训练yaml并按launcher顺序合并14项实际CLI覆盖；只在内存推导 `setup_directories` 的output_dir，没有调用main、Accelerator或模型构建。

解析得到111个叶字段；相对锚完整resolved config只有 `datasets.vla_data.dsol_arm`、`run_id`、`output_dir` 三项差异，额外差异0。与根代理独立生成的 `/share/longjunyu/alphabrain/experiments/dsol-training-match-20260907-lW8boo/canonical_proposal/proposed_framework_config.yaml` 全字段差异0。执行断言通过，退出码0；`torch.cuda.is_initialized=false`，未创建模型、未写文件。这项证据不是直接复制锚config后改三个字段，而是核实了实际launcher配置合并路径。

### 6.2 数据payload复核和真正的开训条件

Broad64顶层manifest已绑定8个 `shard_sha256` 和8个 `records_sha256`；8个payload合计2,864,719,633 bytes（约2.67GiB）。已有 `audit.json` 是PASS，所有shard/index hash及抽样state/action/robot_state/图像检查通过。对应脚本 `/workspace/projects/alphabrain-dsol-paper1/scripts/dsol_paper1/audit_libero_pair_collection.py` 的79–80行进行全量payload/index hash；默认16样本/分片，额外验证128个记录与源HDF5并输出montage。它不启动MuJoCo或模型，但会写审计输出；应指定新的output/montage，不能覆盖旧归档。

这只是读取约2.9GB payload + 小量records和HDF5抽样，通常可在秒至分钟级完成，远小于训练成本；本审计未对其耗时benchmark。顶层manifest自身hash复核不能代替这些payload hash。根代理已独立运行该已有审计，并生成 `/share/longjunyu/alphabrain/experiments/dsol-training-match-20260907-lW8boo/collection_reaudit.json`，本子代理只读复核该新报告（不重复运行）：status=PASS，38193记录、128抽样，11项checks全部true，含全量shard/index hash、抽样state/action/robot_state匹配、无episode split泄漏。

开训边界仅是：确认原始checkpoint身份和Broad64 payload hash、锁定上述有效配置及两卡资源/端口/新输出目录，并收存当前代码/依赖身份。根代理已通知原始权重全量hash与source manifest一致、Broad结果全量hash与v2一致；该两项是根代理执行证据，本子代理未重复扫描大权重。

训练**不依赖主动相机的state restore或gripper工程gate**，这两条工作线独立。训练完成后配对评测的冻结与验收属于后续发布结论条件，不是阻止本次canonical起训的前置gate。历史GPU具体型号或历史重复numpy清单无法完全还原，应诚实记录为复现边界，不必无限等待它们才开始相同数学训练设计；同样不能把未完成训练写成matched结果。

## 7. 真实耗时依据与估算边界

锚 `metrics.jsonl` 共2000条（step1→2000），final examples_seen64000。平均 `model_time=0.3179584036s`、`data_time=0.0001730347s`。这两个字段是每个微批的平均，不是完整optimizer step：代码对16个微批平均后记录。乘回16，累计约10180.206秒，即2小时49分40秒。把0.318秒直接乘2000得到“十几分钟”是错误口径。

同run日志从2026-08-20 19:09:55模型准备到22:02:30最终保存，2小时52分35秒；训练配置打印19:10:52到最终保存2小时51分38秒。日志使用历史本地时区（与控制器差8小时），不要把这组时间标UTC。

控制器UTC：wave1在2026-08-20T11:09:45Z启动，14:03:32Z完成，2小时53分47秒；该wave同时运行4个2卡job，其中包含锚M-B seed41，说明历史锚并非空机独占全部8卡。

据此，保留2卡拓扑的新canonical可按“约3小时，排程预留3–4小时”估计，开始后以本次前20–50个真实优化步更新。这个范围是调度估计，不是保证；当前空闲情况、GPU型号是否相同、驱动、共享I/O/CPU负载均会改变速度。没有证据支持直接按8/2倍线性缩短；改8卡还会改变rank数、每rank随机流与梯度累积拓扑。若需要另行加速应先做明确的训练等价性对照，不能默默替换本次匹配定义。

## 附录A：锚完整 framework_config.yaml

```yaml
framework:
  name: PaliGemmaPi05
  pi05: true
  paligemma:
    base_vlm: /share/longjunyu/alphabrain/pretrained_models/paligemma-3b-pt-224
    attn_implementation: sdpa
    discrete_state_input: false
    num_images: 3
    image_mask:
    - true
    - true
    - false
    max_token_len: 48
  action_expert:
    width: 1024
    depth: 18
    mlp_dim: 4096
    num_heads: 8
    num_kv_heads: 1
    head_dim: 256
    precision: bfloat16
  action_model:
    fresh_tail_weight: 1.0
    fresh_weighting_mode: suffix
    feedback_horizon_key: feedback_horizon
    action_dim: 7
    action_horizon: 10
    future_action_window_size: 9
    past_action_window_size: 0
    num_inference_steps: 10
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    state_dim: 8
  gripper_remap: false
  normalization:
    enabled: true
    action_mean:
    - 0.06594458
    - 0.09087186
    - -0.09225027
    - 0.00016278
    - 0.00562295
    - -0.00386901
    - -0.03697782
    action_std:
    - 0.33671834
    - 0.38222458
    - 0.45361066
    - 0.03833973
    - 0.0634139
    - 0.07609765
    - 0.99931609
  visual_adapter:
    enabled: true
    type: siglip_low_rank
    target_modules:
    - mlp.fc1
    - mlp.fc2
    rank: 16
    alpha: 16.0
    dropout: 0.0
    freeze_base: true
  cabi:
    enabled: false
  dsol_pair_consistency:
    enabled: false
    weight: 0.0
    shared_flow_noise_time: false
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: /share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2
    data_mix: libero_goal
    action_type: delta_qpos
    sequential_step_sampling: false
    default_image_resolution:
    - 3
    - 224
    - 224
    per_device_batch_size: 1
    load_all_data_for_training: true
    obs:
    - image_0
    - wrist_image
    video_backend: torchvision_av
    include_state: false
    dataloader_module: paligemma_datasets
    dataset_format: dsol_libero_pairs
    split: train
    shard_cache_size: 2
    action_horizon: 10
    action_dim: 7
    state_dim: 8
    use_image_augmentation: false
    shuffle: true
    dsol_arm: broad_unpaired_practical
    examples_per_item: 1
trainer:
  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true
  epochs: 100
  eval_interval: 2001
  freeze_modules: vlm_interface.model.language_model,vlm_interface.model.lm_head,vlm_interface.model.multi_modal_projector
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
  is_resume: false
  learning_rate:
    action_model: 0.0001
    base: 0.0001
    qwen_vl_interface: 1.0e-05
    flow_matching_head: 5.0e-05
    vlm_interface: 0.0005
  logging_frequency: 1
  loss_scale:
    vla: 1.0
    vlm: 0.1
  lr_scheduler_type: cosine_with_min_lr
  max_grad_norm: 1.0
  max_train_steps: 2000
  num_warmup_steps: 5
  optimizer:
    betas:
    - 0.9
    - 0.95
    eps: 1.0e-08
    name: AdamW
    weight_decay: 1.0e-10
  resume_epoch: null
  resume_step: null
  save_interval: 2001
  scheduler_specific_kwargs:
    min_lr: 1.0e-05
  warmup_ratio: 0.1
  weight_decay: 0.0
  pretrained_checkpoint: /share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch
  skip_final_save: false
  scheduler_total_steps: 2000
  dsol_validation:
    enabled: false
    split: val
    arm: canonical_unique
    max_data_items: 256
    per_device_batch_size: 1
    num_workers: 2
    seed: 20260818
  ema:
    enabled: false
environment:
  wandb_mode: offline
  wandb_project: dsol-libero-view-revalidation
  wandb_entity: ''
seed: 41
run_id: dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000
output_root_dir: /share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs
output_dir: /share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000
```

## 附录B：锚完整 run_manifest.json

```json
{
  "arm": "broad_unpaired_practical",
  "calibration": {
    "enabled": false,
    "interval": 250,
    "items": 256
  },
  "critical_code_sha256": {
    "AlphaBrain/dataloader/paligemma_datasets.py": "1d946f0bc6ef5f2812393bbd83974f672f11280e67bf3196cde2a60c3902d0fb",
    "AlphaBrain/model/framework/PaliGemmaPi.py": "76d2b7c0a031bd986d17f06ced6e7595c8dd47ab0245e6fac621f05037fc447f",
    "AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pair_consistency.py": "cede4418e8f5cb8e1278c646f5771394587e9572373a6c23cc203dc2d0a74a22",
    "AlphaBrain/training/train_alphabrain.py": "4934ebe1c6e10d859d4fce44319cc1996b8a12e8788dbfd035268d3b5d2d2da4",
    "AlphaBrain/training/trainer_utils/trainer_tools.py": "063403ab6152ac9fc44fbb94856df6b6da3fba8ed99a3f1698e8580fcc1aa362",
    "configs/experiments/dsol_libero_broad_pairing.yaml": "488fbef21bd50ea53c3689a3bb26f0248885193ce5adb9c5956b7fb3298f4cba",
    "scripts/dsol_paper1/libero_pair_records.py": "f1be08df11da8c985f14c5f6350382a8d49397543e96c3c46bdb9ce621ccccb4",
    "scripts/dsol_paper1/run_libero_pair_train.sh": "b3f7978245863b3cc56211936df5a08c406504911077679093e2fc784784268f"
  },
  "critical_tracked_patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "data_manifest_sha256": "e8fd2b606c03eb169db6b726b1895d7c8f0b0a0e6c54026d65caecbf6bf0bf94",
  "data_root": "/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2",
  "examples_per_data_item": 1,
  "git_commit": "8dc602991834e4d87e3099187287dbdbd917f3fc",
  "git_dirty": false,
  "global_model_examples_per_update": 32,
  "gpu_devices": [
    0,
    1
  ],
  "gradient_accumulation_steps": 16,
  "main_process_port": 32141,
  "num_gpus": 2,
  "pretrained_checkpoint_manifest": "/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch/source_manifest.json",
  "pretrained_checkpoint_manifest_sha256": "f4afc8f39bb16fb8c40ae57bbfe3417ae3d91068c0b87029ba8cb0910cf22172",
  "python": "3.12.3",
  "scheduler_total_steps": 2000,
  "schema": "dsol_training_run_manifest_v1",
  "seed": 41,
  "skip_final_save": false,
  "source_data_items_per_update": 32,
  "steps": 2000,
  "wandb_mode": "offline"
}
```

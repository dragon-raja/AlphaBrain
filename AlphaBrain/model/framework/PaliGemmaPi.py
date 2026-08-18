# Copyright 2025 VLA-Engine. All rights reserved.
# Pi0 / Pi0.5 Flow Matching VLA with swappable VLM backbone

"""
PaliGemmaPi Framework

π₀.₅ flow matching framework with swappable VLM backbone (PaliGemma / Llama /
Qwen). Registered under two names for config compatibility:

    PaliGemmaPi05  LlamaPi05

Architecture:
  VLM (any) → prefix embedding → [KV cache] → Action Expert (Gemma) + Flow Matching → actions

Components:
  - VLM interface: reuses VLAE's existing get_vlm_model() factory
  - Action Expert: independent Gemma transformer (from openpi)
  - Flow Matching Head: multi-step denoising action generation

Training: flow matching loss (MSE between predicted and target velocity fields)
Inference: iterative denoising from Gaussian noise (default 10 steps)
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.vlm import get_vlm_model
from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.pi0_action_head import Pi0FlowMatchingHead
from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.pair_consistency import (
    dsol_pair_groups,
    paired_prediction_consistency,
    share_pair_noise_time,
    validate_paired_actions,
)
from AlphaBrain.model.modules.action_model.fresh_loss import feedback_weighted_flow_loss
from AlphaBrain.model.modules.action_model.cabi_binding import (
    BindingState,
    CausalBindingAdapter,
    cabi_closure_losses,
    cabi_transport_role_mask,
    group_cabi_tetrads,
    prefix_modality_masks,
    select_binding_state,
)
from AlphaBrain.model.modules.action_model.counterfactual_action_completion import (
    counterfactual_fourth_actions,
)
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("PaliGemmaPi05")
@FRAMEWORK_REGISTRY.register("LlamaPi05")
class PaliGemmaPi(BaseFramework):
    """
    Pi0.5 framework with swappable VLM backbone.

    Config structure:
        framework:
          name: PaliGemmaPi05            # or LlamaPi05
          pi05: true                     # true for π₀.₅, false for π₀
          gripper_remap: false           # optional; defaults to true when name=="PaliGemmaPi05"
          paligemma:                     # or qwenvl/llamavl — uses get_vlm_model()
            base_vlm: google/paligemma-3b-pt-224
          action_expert:
            width: 1024
            depth: 18
            ...
          action_model:
            action_dim: 7
            action_horizon: 50
            num_inference_steps: 10
            action_mask: [true, true, ..., false]   # optional, loss only on valid dims
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        pi05 = getattr(config.framework, 'pi05', True)
        self.pi05 = pi05

        # ── VLM backbone (swappable) ──
        # Determine which VLM to use based on config
        vlm_type = self._detect_vlm_type(config)

        if vlm_type == "paligemma":
            from AlphaBrain.model.modules.vlm.paligemma import _PaliGemma_VL_Interface
            self.vlm_interface = _PaliGemma_VL_Interface(config=config)
        else:
            # Use existing VLM factory for Qwen/Llama/Florence etc.
            self.vlm_interface = get_vlm_model(config=config)

        # Native visual low-rank adaptation keeps the original SigLIP weight
        # keys and adds adapter keys directly to AlphaBrain checkpoints.
        visual_adapter_cfg = getattr(config.framework, "visual_adapter", None)
        self.visual_adapter_enabled = bool(
            visual_adapter_cfg is not None
            and getattr(visual_adapter_cfg, "enabled", False)
        )
        self.visual_adapter_summary = None
        if self.visual_adapter_enabled:
            if vlm_type != "paligemma":
                raise ValueError(
                    "visual low-rank adaptation currently requires PaliGemma"
                )
            adapter_type = str(
                getattr(visual_adapter_cfg, "type", "siglip_low_rank")
            )
            if adapter_type != "siglip_low_rank":
                raise ValueError(
                    f"unsupported visual adapter type: {adapter_type}"
                )
            from AlphaBrain.model.modules.vlm.vision_low_rank_adapter import (
                inject_siglip_low_rank_adapters,
            )

            self.visual_adapter_summary = inject_siglip_low_rank_adapters(
                self.vlm_interface.model.vision_tower,
                target_modules=tuple(
                    str(value)
                    for value in getattr(
                        visual_adapter_cfg,
                        "target_modules",
                        ["mlp.fc1", "mlp.fc2"],
                    )
                ),
                rank=int(getattr(visual_adapter_cfg, "rank", 16)),
                alpha=float(getattr(visual_adapter_cfg, "alpha", 16.0)),
                dropout=float(getattr(visual_adapter_cfg, "dropout", 0.0)),
                freeze_base=bool(
                    getattr(visual_adapter_cfg, "freeze_base", True)
                ),
            )
            logger.info(
                "[visual_adapter] type=%s layers=%d parameters=%d targets=%s",
                adapter_type,
                self.visual_adapter_summary.adapted_layer_count,
                self.visual_adapter_summary.adapter_parameter_count,
                self.visual_adapter_summary.target_modules,
            )

        # ── Action Expert + Flow Matching Head ──
        expert_cfg = config.framework.action_expert
        action_cfg = config.framework.action_model

        self._tokenizer = None
        try:
            self._init_tokenizer()
        except Exception as e:
            logger.warning(f"[PaliGemmaPi] eager tokenizer init failed ({e}); will retry lazily on first call")
        self.flow_matching_head = Pi0FlowMatchingHead(
            action_dim=action_cfg.action_dim,
            action_horizon=action_cfg.action_horizon,
            action_expert_width=getattr(expert_cfg, 'width', 1024),
            action_expert_depth=getattr(expert_cfg, 'depth', 18),
            action_expert_mlp_dim=getattr(expert_cfg, 'mlp_dim', 4096),
            action_expert_num_heads=getattr(expert_cfg, 'num_heads', 8),
            action_expert_num_kv_heads=getattr(expert_cfg, 'num_kv_heads', 1),
            action_expert_head_dim=getattr(expert_cfg, 'head_dim', 256),
            pi05=pi05,
            precision=getattr(expert_cfg, 'precision', 'bfloat16'),
            num_inference_steps=getattr(action_cfg, 'num_inference_steps', 10),
            noise_beta_alpha=getattr(action_cfg, 'noise_beta_alpha', 1.5),
            noise_beta_beta=getattr(action_cfg, 'noise_beta_beta', 1.0),
            state_dim=getattr(action_cfg, 'state_dim', None),
        )

        pair_cfg = getattr(config.framework, "dsol_pair_consistency", None)
        self.dsol_pair_consistency_enabled = bool(
            pair_cfg is not None and getattr(pair_cfg, "enabled", False)
        )
        self.dsol_pair_consistency_weight = float(
            getattr(pair_cfg, "weight", 0.0) if pair_cfg is not None else 0.0
        )
        self.dsol_pair_shared_flow_noise_time = bool(
            getattr(
                pair_cfg,
                "shared_flow_noise_time",
                self.dsol_pair_consistency_enabled,
            )
            if pair_cfg is not None
            else False
        )
        if self.dsol_pair_consistency_enabled:
            if vlm_type != "paligemma":
                raise ValueError("DSOL pair consistency currently requires PaliGemma")
            if self.dsol_pair_consistency_weight <= 0.0:
                raise ValueError("DSOL pair consistency weight must be positive")
            if not self.dsol_pair_shared_flow_noise_time:
                raise ValueError(
                    "DSOL pair consistency requires shared flow noise and time"
                )
            logger.info(
                "[DSOL pair consistency] enabled: weight=%.4f",
                self.dsol_pair_consistency_weight,
            )

        # ── Prefix projection (for VLM hidden_size != action_expert_width) ──
        expert_width = getattr(expert_cfg, 'width', 1024)
        vlm_hidden_size = self._get_vlm_hidden_size()
        if vlm_hidden_size is not None and vlm_hidden_size != expert_width and vlm_type != "paligemma":
            # PaliGemma uses its own encode_prefix which outputs expert_width directly
            # For Qwen/Llama, VLM hidden states need projection to match action expert
            self.prefix_proj = nn.Linear(vlm_hidden_size, expert_width, bias=False)
            logger.info(f"[prefix_proj] Added projection: VLM hidden={vlm_hidden_size} → expert_width={expert_width}")
        else:
            self.prefix_proj = None

        # KYC-style camera conditioning. The external camera's Plucker ray map
        # is encoded independently and late-fused with its SigLIP patch tokens.
        camera_cfg = getattr(config.framework, "camera_conditioning", None)
        self.camera_conditioning_enabled = bool(
            camera_cfg is not None and getattr(camera_cfg, "enabled", False)
        )
        self.camera_conditioned_view_indices = tuple(
            int(value)
            for value in (
                getattr(camera_cfg, "conditioned_view_indices", [0])
                if camera_cfg is not None
                else []
            )
        )
        self.camera_image_transform = str(
            getattr(camera_cfg, "image_transform", "none")
            if camera_cfg is not None
            else "none"
        )
        self.camera_conditioning_mode = str(
            getattr(camera_cfg, "mode", "real")
            if camera_cfg is not None
            else "real"
        )
        configured_view_modes = (
            getattr(camera_cfg, "view_modes", None)
            if camera_cfg is not None
            else None
        )
        if configured_view_modes is None:
            self.camera_view_modes = {
                view_index: self.camera_conditioning_mode
                for view_index in self.camera_conditioned_view_indices
            }
        else:
            configured_view_modes = tuple(str(value) for value in configured_view_modes)
            if len(configured_view_modes) != len(self.camera_conditioned_view_indices):
                raise ValueError(
                    "camera view_modes must align with conditioned_view_indices"
                )
            self.camera_view_modes = dict(
                zip(self.camera_conditioned_view_indices, configured_view_modes)
            )
        self.camera_joint_crop_min_scale = float(
            getattr(camera_cfg, "joint_crop_min_scale", 1.0)
            if camera_cfg is not None
            else 1.0
        )
        self.camera_canonical_intrinsics = (
            getattr(camera_cfg, "canonical_intrinsics", None)
            if camera_cfg is not None
            else None
        )
        self.camera_canonical_to_world = (
            getattr(camera_cfg, "canonical_camera_to_world_opencv", None)
            if camera_cfg is not None
            else None
        )
        self.camera_canonical_intrinsics_by_view = (
            getattr(camera_cfg, "canonical_intrinsics_by_view", None)
            if camera_cfg is not None
            else None
        )
        self.camera_canonical_to_world_by_view = (
            getattr(camera_cfg, "canonical_camera_to_world_opencv_by_view", None)
            if camera_cfg is not None
            else None
        )
        self.camera_fusion_type = str(
            getattr(camera_cfg, "fusion_type", "late_concat")
            if camera_cfg is not None
            else "late_concat"
        )
        self.camera_conditioner = None
        if self.camera_conditioning_enabled:
            if vlm_type != "paligemma":
                raise ValueError("KYC camera conditioning currently requires PaliGemma")
            if not self.camera_conditioned_view_indices:
                raise ValueError("conditioned_view_indices cannot be empty")
            if len(set(self.camera_conditioned_view_indices)) != len(
                self.camera_conditioned_view_indices
            ):
                raise ValueError("conditioned_view_indices must be unique")
            if any(value < 0 for value in self.camera_conditioned_view_indices):
                raise ValueError("conditioned_view_indices must be non-negative")
            if self.camera_image_transform not in {
                "none",
                "rot180",
                "mujoco_upright",
            }:
                raise ValueError(
                    "camera image_transform must be none, rot180, or "
                    "mujoco_upright"
                )
            if any(
                mode not in {"real", "canonical"}
                for mode in self.camera_view_modes.values()
            ):
                raise ValueError("camera view modes must be real or canonical")
            if not 0.0 < self.camera_joint_crop_min_scale <= 1.0:
                raise ValueError("joint_crop_min_scale must be in (0, 1]")
            for view_index, mode in self.camera_view_modes.items():
                if mode == "canonical":
                    self._canonical_camera_matrix_for_view(
                        "camera_intrinsics",
                        view_index,
                    )
                    self._canonical_camera_matrix_for_view(
                        "camera_to_world_opencv",
                        view_index,
                    )
            from AlphaBrain.model.modules.vlm.camera_conditioning import (
                PluckerLateFusion,
                PluckerResidualFusion,
            )

            encoder_channels = tuple(
                int(value)
                for value in getattr(
                    camera_cfg,
                    "encoder_channels",
                    [64, 128, 256, 512, 512],
                )
            )
            vision_hidden_dim = int(
                self.vlm_interface.model.vision_tower.config.hidden_size
            )
            fusion_classes = {
                "late_concat": PluckerLateFusion,
                "residual_zero": PluckerResidualFusion,
            }
            if self.camera_fusion_type not in fusion_classes:
                raise ValueError(
                    "camera fusion_type must be late_concat or residual_zero"
                )
            self.camera_conditioner = fusion_classes[self.camera_fusion_type](
                rgb_hidden_dim=vision_hidden_dim,
                encoder_channels=encoder_channels,
            )
            logger.info(
                "[KYC] Plucker fusion=%s enabled for views=%s, "
                "image_transform=%s, view_modes=%s",
                self.camera_fusion_type,
                self.camera_conditioned_view_indices,
                self.camera_image_transform,
                self.camera_view_modes,
            )

        # CABI is a training-time causal binding objective with ordinary
        # single-pass inference. V0 is intentionally restricted to PaliGemma,
        # whose visual/language prefix boundary is explicit and auditable.
        cabi_cfg = getattr(config.framework, 'cabi', None)
        self.cabi_enabled = bool(cabi_cfg is not None and getattr(cabi_cfg, 'enabled', False))
        self.cabi_adapter = None
        self.cabi_loss_weights = {}
        self.cabi_flow_transport_weight = 0.0
        self.cabi_decoder_closure_weight = 0.0
        if self.cabi_enabled:
            if vlm_type != "paligemma":
                raise ValueError("CABI v0 currently requires the PaliGemma prefix interface")
            cabi_hidden_size = getattr(self.vlm_interface, 'hidden_size', vlm_hidden_size)
            if cabi_hidden_size is None:
                raise ValueError("cannot determine the CABI prefix hidden dimension")
            self.cabi_adapter = CausalBindingAdapter(
                hidden_dim=int(cabi_hidden_size),
                binding_dim=int(getattr(cabi_cfg, 'binding_dim', 128)),
                transport_rank=int(getattr(cabi_cfg, 'transport_rank', 32)),
                temperature=float(getattr(cabi_cfg, 'temperature', 1.0)),
                residual_scale=float(getattr(cabi_cfg, 'residual_scale', 0.1)),
                tie_read_write_attention=bool(
                    getattr(cabi_cfg, 'tie_read_write_attention', False)
                ),
                role_state_residual=bool(
                    getattr(cabi_cfg, 'role_state_residual', False)
                ),
            )
            self.cabi_loss_weights = {
                "single_source": float(getattr(cabi_cfg, 'single_weight', 1.0)),
                "single_target": float(getattr(cabi_cfg, 'single_weight', 1.0)),
                "fourth_anchor": float(getattr(cabi_cfg, 'anchor_weight', 1.0)),
                "commutator": float(getattr(cabi_cfg, 'commutator_weight', 0.1)),
                "cycle": float(getattr(cabi_cfg, 'cycle_weight', 0.1)),
                "specificity": float(getattr(cabi_cfg, 'specificity_weight', 0.1)),
                "orthogonality": float(getattr(cabi_cfg, 'orthogonality_weight', 0.01)),
                "intervention_identifiability": float(
                    getattr(cabi_cfg, 'intervention_identifiability_weight', 0.0)
                ),
                "attention_separation": float(
                    getattr(cabi_cfg, 'attention_separation_weight', 0.0)
                ),
                "causal_factor_contrastive": float(
                    getattr(cabi_cfg, 'causal_factor_contrastive_weight', 0.0)
                ),
                "counterfactual_attention_grounding": float(
                    getattr(cabi_cfg, 'counterfactual_attention_grounding_weight', 0.0)
                ),
            }
            self.cabi_intervention_margin = float(
                getattr(cabi_cfg, 'intervention_margin', 0.1)
            )
            self.cabi_contrastive_temperature = float(
                getattr(cabi_cfg, 'contrastive_temperature', 0.1)
            )
            self.cabi_flow_transport_weight = float(
                getattr(cabi_cfg, 'flow_transport_weight', 1.0)
            )
            self.cabi_decoder_closure_weight = float(
                getattr(cabi_cfg, 'decoder_closure_weight', 0.0)
            )
            logger.info(
                "[CABI] enabled: binding_dim=%d, transport_rank=%d",
                self.cabi_adapter.binding_dim,
                self.cabi_adapter.transport_rank,
            )

        completion_cfg = getattr(
            config.framework, "counterfactual_action_completion", None
        )
        self.counterfactual_action_completion_enabled = bool(
            completion_cfg is not None
            and getattr(completion_cfg, "enabled", False)
        )
        self.counterfactual_action_completion_weight = float(
            getattr(completion_cfg, "weight", 1.0)
            if completion_cfg is not None
            else 0.0
        )
        if self.counterfactual_action_completion_enabled:
            if vlm_type != "paligemma":
                raise ValueError(
                    "counterfactual action completion currently requires PaliGemma"
                )
            if self.counterfactual_action_completion_weight <= 0.0:
                raise ValueError(
                    "counterfactual action completion weight must be positive"
                )
            logger.info(
                "[CAFC] enabled: weight=%.3f",
                self.counterfactual_action_completion_weight,
            )

        # ── Action dimension settings ──
        self.action_dim = action_cfg.action_dim
        self.action_horizon = action_cfg.action_horizon
        self.fresh_tail_weight = float(getattr(action_cfg, 'fresh_tail_weight', 1.0))
        self.fresh_weighting_mode = str(getattr(action_cfg, 'fresh_weighting_mode', 'suffix'))
        self.feedback_horizon_key = str(getattr(action_cfg, 'feedback_horizon_key', 'feedback_horizon'))
        self.future_action_window_size = getattr(action_cfg, 'future_action_window_size', action_cfg.action_horizon)
        self.past_action_window_size = getattr(action_cfg, 'past_action_window_size', 0)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Store VLM rotary_emb reference for inference consistency
        vlm_lm = self._get_vlm_language_model()
        if vlm_lm is not None and hasattr(vlm_lm, 'rotary_emb'):
            self.flow_matching_head._vlm_rotary_emb = vlm_lm.rotary_emb

        # Gripper remap: map dim-6 from [0,1] to [+1,-1] post-unnormalization.
        # Required for LIBERO eval client (-gripper + binarize). Defaults true
        # when registered as "PaliGemmaPi05" to preserve historical behavior.
        gripper_default = (config.framework.name == "PaliGemmaPi05")
        self.gripper_remap = bool(getattr(config.framework, 'gripper_remap', gripper_default))

        logger.info(
            f"PaliGemmaPi initialized: name={config.framework.name}, pi05={pi05}, "
            f"vlm={vlm_type}, action_dim={self.action_dim}, horizon={self.action_horizon}, "
            f"gripper_remap={self.gripper_remap}"
        )

        # Enable gradient checkpointing on action expert only.
        # _shared_forward handles VLM layers via use_gc (compute_layer checkpoint).
        # Enabling HF-level GC on VLM GemmaModel would cause double-checkpoint conflict
        # with DeepSpeed ZeRO: same parameters reduced twice -> AssertionError.
        for m in self.flow_matching_head.modules():
            if hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
        # Disable HF-level GC on VLM language model to prevent double-backward
        try:
            vlm_lm = self._get_vlm_language_model()
            if hasattr(vlm_lm, "gradient_checkpointing_disable"):
                vlm_lm.gradient_checkpointing_disable()
        except Exception:
            pass

        # Freeze VLM modules based on config (empty string = no freeze = full finetune)
        freeze_modules = getattr(config.trainer, 'freeze_modules', 'vlm_interface.model.language_model,vlm_interface.model.lm_head')
        if freeze_modules:
            freeze_list = [m.strip() for m in str(freeze_modules).split(',') if m.strip()]
            if hasattr(self.vlm_interface, "model"):
                for name, param in self.vlm_interface.model.named_parameters():
                    if any(fm.replace('vlm_interface.model.', '') in name for fm in freeze_list):
                        param.requires_grad = False
            frozen_p = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"Frozen: {frozen_p/1e6:.0f}M, Trainable: {train_p/1e6:.0f}M (freeze={freeze_modules})")
        else:
            total_p = sum(p.numel() for p in self.parameters())
            logger.info(f"Full finetune: all {total_p/1e6:.0f}M parameters trainable")

        # ── Action/State Normalization (MEAN_STD, matches openpi) ──
        norm_cfg = getattr(config.framework, 'normalization', None)
        self.use_action_norm = norm_cfg is not None and getattr(norm_cfg, 'enabled', False)
        if self.use_action_norm:
            import json as _json
            action_mean = torch.tensor(getattr(norm_cfg, 'action_mean', [0.0]*self.action_dim), dtype=torch.float32)
            action_std = torch.tensor(getattr(norm_cfg, 'action_std', [1.0]*self.action_dim), dtype=torch.float32)
            self.register_buffer('action_mean', action_mean)
            self.register_buffer('action_std', action_std)
            if hasattr(norm_cfg, 'state_mean'):
                state_mean = torch.tensor(norm_cfg.state_mean, dtype=torch.float32)
                state_std = torch.tensor(norm_cfg.state_std, dtype=torch.float32)
                self.register_buffer('state_mean', state_mean)
                self.register_buffer('state_std', state_std)
            logger.info(f"[norm] Action MEAN_STD normalization enabled (action_dim={self.action_dim})")
        else:
            logger.info("[norm] No action normalization (raw actions)")

        # ── Action dimension mask (for action_dim=32 with only 7 valid dims) ──
        # Read from config: framework.action_model.action_mask
        # e.g. [true, true, true, true, true, true, true, false, ..., false]
        action_mask_cfg = getattr(config.framework.action_model, 'action_mask', None)
        if action_mask_cfg is not None:
            mask_tensor = torch.tensor(list(action_mask_cfg), dtype=torch.bool)
            self.register_buffer('_action_dim_mask', mask_tensor, persistent=False)
            n_valid = mask_tensor.sum().item()
            logger.info(f"[action_mask] {n_valid}/{len(action_mask_cfg)} valid action dims (loss computed on valid dims only)")
        else:
            self._action_dim_mask = None

    def _init_tokenizer(self):
        """Initialize PaliGemma tokenizer (HF AutoTokenizer or sentencepiece fallback)."""
        import os
        # Try HF tokenizer first (has proper special tokens)
        # Prefer local pretrained dir (project convention) before hitting HF hub,
        # so offline/air-gapped servers don't stall on huggingface.co revalidation.
        _local_tok = os.environ.get("PALIGEMMA_TOKENIZER_PATH")
        _pretrained_dir = os.environ.get("PRETRAINED_MODELS_DIR", "data/pretrained_models")
        _local_pg = os.path.join(_pretrained_dir, "paligemma-3b-pt-224")
        configured_pg = None
        paligemma_cfg = getattr(self.config.framework, "paligemma", None)
        if paligemma_cfg is not None:
            configured_pg = getattr(paligemma_cfg, "base_vlm", None)
        candidates = (
            ([_local_tok] if _local_tok else [])
            + ([configured_pg] if configured_pg else [])
            + [_local_pg, "google/paligemma-3b-pt-224"]
        )
        tokenizer_dirs = list(dict.fromkeys(candidates))
        for td in tokenizer_dirs:
            try:
                from transformers import AutoTokenizer
                self._hf_tokenizer = AutoTokenizer.from_pretrained(td)
                self._tokenizer = None  # signal to use HF tokenizer
                logger.info(f"[PaliGemmaPi] Loaded HF tokenizer from {td}")
                return
            except (OSError, ImportError, ValueError) as e:
                logger.debug(f"[PaliGemmaPi] tokenizer load failed for {td}: {e}")
                continue

        # Fallback: sentencepiece
        import sentencepiece as spm
        # Sentencepiece fallback: env var first, then project-local tokenizer.model
        _local_sp = os.environ.get("PALIGEMMA_TOKENIZER_MODEL")
        candidates = ([_local_sp] if _local_sp else []) + [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tokenizer.model"),
            "./tokenizer.model",
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                self._tokenizer = spm.SentencePieceProcessor(model_file=c)
                self._hf_tokenizer = None
                logger.info(f"[PaliGemmaPi] Loaded sentencepiece tokenizer from {c}")
                return
        raise FileNotFoundError("Cannot find PaliGemma tokenizer")

    def _get_vlm_hidden_size(self) -> Optional[int]:
        """Get the hidden size of the VLM's language model."""
        vlm = self.vlm_interface
        if hasattr(vlm, 'model'):
            model = vlm.model
            if hasattr(model, 'config'):
                cfg = model.config
                # Llama / multi-modal models with text_config
                if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'hidden_size'):
                    return cfg.text_config.hidden_size
                # Direct hidden_size
                if hasattr(cfg, 'hidden_size'):
                    return cfg.hidden_size
        return None

    def _detect_vlm_type(self, config) -> str:
        """Detect which VLM backend to use from config."""
        if hasattr(config.framework, 'paligemma'):
            return "paligemma"
        elif hasattr(config.framework, 'qwenvl'):
            return "qwen"
        elif hasattr(config.framework, 'llamavl'):
            return "llama"
        else:
            logger.warning("No VLM config found in framework config, defaulting to paligemma")
            return "paligemma"

    def _get_vlm_language_model(self):
        """Get the underlying language model from the VLM for shared attention."""
        vlm = self.vlm_interface
        if hasattr(vlm, 'get_language_model'):
            return vlm.get_language_model()
        elif hasattr(vlm, 'model'):
            # Qwen/Llama style
            if hasattr(vlm.model, 'language_model'):
                return vlm.model.language_model
            elif hasattr(vlm.model, 'model'):
                return vlm.model.model
        raise AttributeError("Cannot find language model in VLM interface")

    def _apply_cabi_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        *,
        language_token_count: int,
    ) -> tuple[torch.Tensor, Optional[BindingState]]:
        if not self.cabi_enabled:
            return prefix_embs, None
        vision_mask, language_mask = prefix_modality_masks(
            prefix_pad_masks,
            language_token_count=language_token_count,
        )
        binding_state = self.cabi_adapter(
            prefix_embs,
            vision_mask,
            language_mask,
        )
        return binding_state.tokens, binding_state

    def _compute_cabi_closure(
        self,
        binding_state: Optional[BindingState],
        examples: List[dict],
    ) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
        if binding_state is None:
            return None, {}
        grouped = group_cabi_tetrads(examples)
        if not grouped:
            return None, {}
        losses = cabi_closure_losses(
            self.cabi_adapter,
            base=select_binding_state(binding_state, grouped["base"]),
            source_anchor=select_binding_state(binding_state, grouped["source_anchor"]),
            target_anchor=select_binding_state(binding_state, grouped["target_anchor"]),
            fourth_anchor=select_binding_state(binding_state, grouped["fourth_anchor"]),
            intervention_margin=self.cabi_intervention_margin,
            contrastive_temperature=self.cabi_contrastive_temperature,
        )
        weighted = sum(self.cabi_loss_weights[name] * value for name, value in losses.items())
        return weighted, dict(losses)

    def _compute_cabi_transport_flow(
        self,
        binding_state: Optional[BindingState],
        examples: List[dict],
        *,
        actions: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        vlm_language_model: nn.Module,
        state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if binding_state is None or self.cabi_flow_transport_weight == 0.0:
            return None
        grouped = group_cabi_tetrads(examples)
        if not grouped:
            return None

        base = select_binding_state(binding_state, grouped["base"])
        source_anchor = select_binding_state(binding_state, grouped["source_anchor"])
        target_anchor = select_binding_state(binding_state, grouped["target_anchor"])
        source_swap = self.cabi_adapter.transport(base, source_anchor, [0])
        target_swap = self.cabi_adapter.transport(base, target_anchor, [1])

        base_indices = torch.as_tensor(grouped["base"], device=actions.device)
        source_indices = torch.as_tensor(grouped["source_anchor"], device=actions.device)
        target_indices = torch.as_tensor(grouped["target_anchor"], device=actions.device)
        source_actions = actions.index_select(0, source_indices)
        target_actions = actions.index_select(0, target_indices)
        noise = self.flow_matching_head.sample_noise(source_actions.shape, actions.device)
        time = self.flow_matching_head.sample_time(len(source_actions), actions.device)
        selected_state = None if state is None else state.index_select(0, base_indices)
        common = {
            "prefix_pad_masks": prefix_pad_masks.index_select(0, base_indices),
            "prefix_att_masks": prefix_att_masks.index_select(0, base_indices),
            "vlm_language_model": vlm_language_model,
            "state": selected_state,
            "noise": noise,
            "time": time,
        }
        source_loss = self.flow_matching_head.compute_loss(
            prefix_embs=source_swap.tokens,
            actions=source_actions,
            **common,
        )
        target_loss = self.flow_matching_head.compute_loss(
            prefix_embs=target_swap.tokens,
            actions=target_actions,
            **common,
        )
        if self._action_dim_mask is not None:
            source_loss = source_loss[:, :, self._action_dim_mask]
            target_loss = target_loss[:, :, self._action_dim_mask]
        per_role_loss = torch.stack(
            [
                source_loss.flatten(1).mean(dim=1),
                target_loss.flatten(1).mean(dim=1),
            ],
            dim=1,
        )
        role_mask = cabi_transport_role_mask(
            examples,
            grouped,
            device=per_role_loss.device,
            dtype=per_role_loss.dtype,
        )
        return (per_role_loss * role_mask).sum() / role_mask.sum()

    def _compute_cabi_decoder_closure(
        self,
        binding_state: Optional[BindingState],
        examples: List[dict],
        *,
        flow_outputs: Optional[dict[str, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        vlm_language_model: nn.Module,
        state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Match the normal fourth corner to its action-free causal composition."""

        if (
            binding_state is None
            or self.cabi_decoder_closure_weight == 0.0
            or flow_outputs is None
        ):
            return None
        grouped = group_cabi_tetrads(examples)
        if not grouped:
            return None

        base = select_binding_state(binding_state, grouped["base"])
        source_anchor = select_binding_state(binding_state, grouped["source_anchor"])
        fourth_anchor = select_binding_state(binding_state, grouped["fourth_anchor"])
        source_swap = self.cabi_adapter.transport(base, source_anchor, [0])
        target_on_source = self.cabi_adapter.transport(source_anchor, fourth_anchor, [1])
        source_then_target = self.cabi_adapter.transport(source_swap, fourth_anchor, [1])

        role_mask = cabi_transport_role_mask(
            examples,
            grouped,
            device=binding_state.tokens.device,
            dtype=binding_state.tokens.dtype,
        ).to(dtype=torch.bool)
        source_only = role_mask[:, 0] & ~role_mask[:, 1]
        target_only = role_mask[:, 1] & ~role_mask[:, 0]
        if torch.any(~(source_only | target_only | role_mask.all(dim=1))):
            raise ValueError("invalid CABI decoder-closure role mask")

        def choose(
            source_value: torch.Tensor,
            target_value: torch.Tensor,
            both_value: torch.Tensor,
        ) -> torch.Tensor:
            source_shape = (len(source_only),) + (1,) * (source_value.ndim - 1)
            target_shape = (len(target_only),) + (1,) * (target_value.ndim - 1)
            return torch.where(
                source_only.reshape(source_shape),
                source_value,
                torch.where(target_only.reshape(target_shape), target_value, both_value),
            )

        base_indices = torch.as_tensor(grouped["base"], device=binding_state.tokens.device)
        source_indices = torch.as_tensor(
            grouped["source_anchor"], device=binding_state.tokens.device
        )
        fourth_indices = torch.as_tensor(
            grouped["fourth_anchor"], device=binding_state.tokens.device
        )
        composed_tokens = choose(
            source_swap.tokens,
            target_on_source.tokens,
            source_then_target.tokens,
        )
        composed_pad_masks = choose(
            prefix_pad_masks.index_select(0, base_indices),
            prefix_pad_masks.index_select(0, source_indices),
            prefix_pad_masks.index_select(0, base_indices),
        )
        composed_att_masks = choose(
            prefix_att_masks.index_select(0, base_indices),
            prefix_att_masks.index_select(0, source_indices),
            prefix_att_masks.index_select(0, base_indices),
        )
        composed_state = None
        if state is not None:
            composed_state = choose(
                state.index_select(0, base_indices),
                state.index_select(0, source_indices),
                state.index_select(0, base_indices),
            )
        composed_prediction = self.flow_matching_head.predict_flow(
            prefix_embs=composed_tokens,
            prefix_pad_masks=composed_pad_masks,
            prefix_att_masks=composed_att_masks,
            vlm_language_model=vlm_language_model,
            state=composed_state,
            noisy_actions=flow_outputs["noisy_actions"].index_select(0, fourth_indices),
            time=flow_outputs["time"].index_select(0, fourth_indices),
        )
        normal_prediction = flow_outputs["prediction"].index_select(0, fourth_indices)
        if self._action_dim_mask is not None:
            composed_prediction = composed_prediction[:, :, self._action_dim_mask]
            normal_prediction = normal_prediction[:, :, self._action_dim_mask]
        return F.mse_loss(composed_prediction, normal_prediction)

    def _compute_counterfactual_action_completion(
        self,
        examples: List[dict],
        *,
        actions: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        vlm_language_model: nn.Module,
        state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Supervise an action-free fourth corner from three observed actions."""

        if not self.counterfactual_action_completion_enabled:
            return None
        grouped = group_cabi_tetrads(examples)
        if not grouped:
            return None

        pseudo_actions, fourth_indices = counterfactual_fourth_actions(
            actions, grouped
        )
        selected_state = (
            None if state is None else state.index_select(0, fourth_indices)
        )
        completion_loss = self.flow_matching_head.compute_loss(
            prefix_embs=prefix_embs.index_select(0, fourth_indices),
            prefix_pad_masks=prefix_pad_masks.index_select(0, fourth_indices),
            prefix_att_masks=prefix_att_masks.index_select(0, fourth_indices),
            vlm_language_model=vlm_language_model,
            state=selected_state,
            actions=pseudo_actions,
        )
        if self._action_dim_mask is not None:
            completion_loss = completion_loss[:, :, self._action_dim_mask]
        return completion_loss.mean()

    def _prepare_prefix(self, examples):
        """
        Prepare prefix embeddings from examples.

        For PaliGemma: use the native encode_prefix() method
        For Qwen/Llama: extract hidden states from VLM forward, treat as prefix
        """
        vlm_type = self._detect_vlm_type(self.config)

        if vlm_type == "paligemma":
            return self._prepare_prefix_paligemma(examples)
        else:
            return self._prepare_prefix_generic(examples)

    def _prepare_prefix_paligemma(self, examples):
        """Prepare prefix using PaliGemma's native encode_prefix.

        Pipeline: images -> SigLIP -> projector -> image embeds
                  text -> sentencepiece -> embed_tokens -> text embeds
                  concat -> prefix

        Configurable via framework.paligemma:
            num_images: total number of image slots (default: auto-detect from input)
            image_mask: list of bools per slot, False = zero-padded dummy (default: all True)
            tokenize_format: "openpi" (BOS + text + \n) or "raw" (text only) (default: "openpi")
            max_token_len: max token length for instructions (default: 48)
        """
        import torchvision.transforms.functional as TF

        device = next(self.parameters()).device

        dtype = next(self.parameters()).dtype
        B = len(examples)
        paligemma_cfg = getattr(self.config.framework, 'paligemma', None)

        # --- Config-driven image settings ---
        cfg_num_images = getattr(paligemma_cfg, 'num_images', None) if paligemma_cfg else None
        cfg_image_mask = getattr(paligemma_cfg, 'image_mask', None) if paligemma_cfg else None

        # --- Image preprocessing (multi-view support) ---
        def _process_single_img(img):
            """Process a single image to [3, 224, 224] normalized to [-1, 1]."""
            if not isinstance(img, torch.Tensor):
                import numpy as np
                from PIL import Image as PILImage
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img.copy()).float()
                    if img.ndim == 3 and img.shape[-1] == 3:
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                elif isinstance(img, PILImage.Image):
                    img = TF.to_tensor(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.max() > 1.0:
                img = img.float() / 255.0
            img = TF.resize(img, [224, 224], antialias=True)
            img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
            return img

        all_view_tensors = []  # list of [num_views, 3, 224, 224] per sample
        all_crop_boxes = []
        for ex in examples:
            img = ex["image"]
            if isinstance(img, (list, tuple)):
                views = [_process_single_img(v) for v in img]
            else:
                views = [_process_single_img(img)]
            crop_boxes = []
            cropped_views = []
            for view in views:
                if self.training and self.camera_joint_crop_min_scale < 1.0:
                    minimum_side = int(
                        math.ceil(224 * self.camera_joint_crop_min_scale)
                    )
                    side = int(
                        torch.randint(
                            minimum_side,
                            225,
                            (1,),
                        ).item()
                    )
                    top = int(torch.randint(0, 225 - side, (1,)).item())
                    left = int(torch.randint(0, 225 - side, (1,)).item())
                    view = TF.resized_crop(
                        view,
                        top,
                        left,
                        side,
                        side,
                        [224, 224],
                        antialias=True,
                    )
                    crop_boxes.append((top, left, side, side))
                else:
                    crop_boxes.append((0, 0, 224, 224))
                cropped_views.append(view)
            views = cropped_views

            # Match the configured image slots exactly.
            # Note: zero uint8 image [0,0,0] after normalize (x/255 - 0.5)/0.5 = -1.0
            # So padding images should be all -1.0 to match openpi's np.zeros_like behavior
            if cfg_num_images is not None:
                views = views[:cfg_num_images]
                crop_boxes = crop_boxes[:cfg_num_images]
                if len(views) < cfg_num_images:
                    pad_img = torch.full((3, 224, 224), -1.0, dtype=views[0].dtype)
                    while len(views) < cfg_num_images:
                        views.append(pad_img)
                        crop_boxes.append((0, 0, 224, 224))

            all_view_tensors.append(torch.stack(views))  # [V, 3, 224, 224]
            all_crop_boxes.append(crop_boxes)

        num_views = all_view_tensors[0].shape[0]
        # Stack all views across batch: [B, V, 3, 224, 224]
        pixel_values = torch.stack(all_view_tensors, dim=0).to(device=device, dtype=dtype)

        # Build per-view image masks from config
        if cfg_image_mask is not None:
            # cfg_image_mask: list of bools, e.g. [true, true, false] for 3 views
            mask_vals = cfg_image_mask
            if len(mask_vals) < num_views:
                mask_vals = list(mask_vals) + [True] * (num_views - len(mask_vals))
            masks_list = [torch.full((B,), bool(mask_vals[v]), dtype=torch.bool, device=device)
                          for v in range(num_views)]
        else:
            masks_list = [torch.ones(B, dtype=torch.bool, device=device) for _ in range(num_views)]

        # --- Text tokenization ---
        instructions = [ex["lang"] for ex in examples]

        if self._tokenizer is None and not hasattr(self, '_hf_tokenizer'):
            self._init_tokenizer()

        # Tokenize with configurable format
        tokenize_fmt = getattr(paligemma_cfg, 'tokenize_format', 'openpi') if paligemma_cfg else 'openpi'
        max_len = getattr(paligemma_cfg, 'max_token_len', 48) if paligemma_cfg else 48
        discrete_state = getattr(paligemma_cfg, 'discrete_state_input', False) if paligemma_cfg else False
        all_ids = []
        for i, text in enumerate(instructions):
            cleaned = text.strip().replace("_", " ").replace("\n", " ")

            # Build prompt text based on discrete_state_input setting
            if discrete_state and "state" in examples[i]:
                # π₀.₅ mode: discretize state into 256 bins and embed in prompt
                state = examples[i]["state"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                import numpy as np
                # Normalize state with MEAN_STD if available, then discretize to [-1, 1] → 256 bins
                if hasattr(self, 'state_mean'):
                    state_np = (state - self.state_mean.cpu().numpy()) / (self.state_std.cpu().numpy() + 1e-6)
                else:
                    state_np = state
                state_np = np.clip(state_np, -1.0, 1.0)
                discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                discretized = np.clip(discretized, 0, 255)
                state_str = " ".join(map(str, discretized.astype(int)))
                prompt_text = f"Task: {cleaned}, State: {state_str};\n"
            else:
                # π₀ mode: state as continuous input (not in prompt)
                prompt_text = cleaned

            if tokenize_fmt == "openpi":
                # openpi format: BOS + prompt_text (+ "\n" if not already ending with it)
                if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
                    bos_id = self._hf_tokenizer.bos_token_id
                    if discrete_state:
                        # prompt_text already ends with ";\n"
                        text_ids = self._hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                        ids = ([bos_id] if bos_id is not None else []) + text_ids
                    else:
                        text_ids = self._hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                        newline_ids = self._hf_tokenizer.encode("\n", add_special_tokens=False)
                        ids = ([bos_id] if bos_id is not None else []) + text_ids + newline_ids
                else:
                    if discrete_state:
                        ids = self._tokenizer.encode(prompt_text)
                    else:
                        ids = self._tokenizer.encode(prompt_text) + self._tokenizer.encode("\n")
            else:
                # raw format: text only, no special tokens
                if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
                    ids = self._hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                else:
                    ids = self._tokenizer.encode(prompt_text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            all_ids.append(ids)

        # Pad token sequences to fixed length (openpi pads to max_token_len for consistent position ids)
        max_actual = max(len(ids) for ids in all_ids)
        pad_len = max(max_actual, max_len)  # use max_token_len as minimum pad length
        token_ids = torch.zeros(B, pad_len, dtype=torch.long, device=device)
        token_masks = torch.zeros(B, pad_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(all_ids):
            token_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            token_masks[i, :len(ids)] = True

        # Encode prefix through VLM (multi-view: pass each view separately)
        view_tensors = pixel_values  # [B, V, 3, 224, 224]
        images_list = [view_tensors[:, v] for v in range(num_views)]  # list of [B, 3, 224, 224]
        image_embeddings = None
        if self.camera_conditioning_enabled:
            image_embeddings = []
            for view_index, image_tensor in enumerate(images_list):
                vision_features = (
                    self.vlm_interface.model.get_vision_features(image_tensor)
                )
                if view_index in self.camera_conditioned_view_indices:
                    vision_features = self._apply_camera_conditioning(
                        vision_features,
                        examples,
                        view_index=view_index,
                        height=image_tensor.shape[-2],
                        width=image_tensor.shape[-1],
                        crop_boxes=[
                            boxes[view_index] for boxes in all_crop_boxes
                        ],
                    )
                image_embeddings.append(
                    self.vlm_interface.model.project_vision_features(
                        vision_features
                    )
                )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.vlm_interface.encode_prefix(
            images=images_list,
            image_masks=masks_list,
            lang_tokens=token_ids,
            lang_masks=token_masks,
            image_embeddings=image_embeddings,
        )

        return prefix_embs, prefix_pad_masks, prefix_att_masks

    @staticmethod
    def _camera_matrix_for_view(example, key: str, view_index: int):
        by_view_key = f"{key}_by_view"
        if by_view_key in example:
            values = example[by_view_key]
            if view_index >= len(values):
                raise ValueError(
                    f"{by_view_key} does not contain view index {view_index}"
                )
            return values[view_index]
        if view_index == 0 and key in example:
            return example[key]
        raise KeyError(f"missing {key} for conditioned view {view_index}")

    def _canonical_camera_matrix_for_view(self, key: str, view_index: int):
        if key == "camera_intrinsics":
            by_view = self.camera_canonical_intrinsics_by_view
            fallback = self.camera_canonical_intrinsics
        elif key == "camera_to_world_opencv":
            by_view = self.camera_canonical_to_world_by_view
            fallback = self.camera_canonical_to_world
        else:
            raise KeyError(f"unsupported canonical camera key: {key}")
        if by_view is not None:
            if view_index >= len(by_view):
                raise ValueError(
                    f"canonical {key}_by_view does not contain view {view_index}"
                )
            return by_view[view_index]
        if view_index == 0 and fallback is not None:
            return fallback
        raise ValueError(
            f"canonical camera mode requires {key} for view {view_index}"
        )

    def _apply_camera_conditioning(
        self,
        image_embeddings: torch.Tensor,
        examples,
        *,
        view_index: int,
        height: int,
        width: int,
        crop_boxes,
    ) -> torch.Tensor:
        from AlphaBrain.model.modules.vlm.camera_conditioning import (
            plucker_raymap,
            transform_raymap,
        )

        camera_mode = self.camera_view_modes[view_index]
        if camera_mode == "canonical":
            intrinsics = np.repeat(
                np.asarray(
                    self._canonical_camera_matrix_for_view(
                        "camera_intrinsics",
                        view_index,
                    ),
                    dtype=np.float32,
                )[None],
                len(examples),
                axis=0,
            )
            camera_to_world = np.repeat(
                np.asarray(
                    self._canonical_camera_matrix_for_view(
                        "camera_to_world_opencv",
                        view_index,
                    ),
                    dtype=np.float32,
                )[None],
                len(examples),
                axis=0,
            )
        else:
            intrinsics = np.stack(
                [
                    np.asarray(
                        self._camera_matrix_for_view(
                            example,
                            "camera_intrinsics",
                            view_index,
                        ),
                        dtype=np.float32,
                    )
                    for example in examples
                ]
            )
            camera_to_world = np.stack(
                [
                    np.asarray(
                        self._camera_matrix_for_view(
                            example,
                            "camera_to_world_opencv",
                            view_index,
                        ),
                        dtype=np.float32,
                    )
                    for example in examples
                ]
            )
        device = image_embeddings.device
        raymap = plucker_raymap(
            torch.as_tensor(intrinsics, device=device, dtype=torch.float32),
            torch.as_tensor(camera_to_world, device=device, dtype=torch.float32),
            height=height,
            width=width,
        )
        raymap = transform_raymap(
            raymap,
            image_transform=self.camera_image_transform,
        )
        full_box = (0, 0, height, width)
        if any(tuple(box) != full_box for box in crop_boxes):
            cropped_raymaps = []
            for sample_index, box in enumerate(crop_boxes):
                top, left, crop_height, crop_width = map(int, box)
                cropped = raymap[
                    sample_index : sample_index + 1,
                    :,
                    top : top + crop_height,
                    left : left + crop_width,
                ]
                cropped_raymaps.append(
                    F.interpolate(
                        cropped,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            raymap = torch.cat(cropped_raymaps, dim=0)
        conditioner_dtype = next(self.camera_conditioner.parameters()).dtype
        return self.camera_conditioner(
            image_embeddings,
            raymap.to(dtype=conditioner_dtype),
        )

    def _prepare_prefix_generic(self, examples):
        """
        Prepare prefix using Qwen/Llama VLM.

        Strategy: Run VLM forward to get hidden states, use as prefix embeddings.
        This allows any VLM that produces hidden states to serve as the prefix encoder.
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        # Build VLM inputs using existing interface
        if hasattr(self.vlm_interface, 'build_qwenvl_inputs'):
            vlm_inputs = self.vlm_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions
            )
        elif hasattr(self.vlm_interface, 'build_llama_inputs'):
            vlm_inputs = self.vlm_interface.build_llama_inputs(
                images=batch_images, instructions=instructions
            )
        else:
            raise NotImplementedError(f"VLM interface does not have a known input builder")

        # Forward through VLM to get hidden states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm_interface.model(
                **vlm_inputs, output_hidden_states=True
            )

        # Use last hidden state as prefix embedding
        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, H_vlm]
        bsize, seq_len, _ = hidden_states.shape

        # Project VLM hidden states to action expert width if needed
        if self.prefix_proj is not None:
            hidden_states = self.prefix_proj(hidden_states.float()).to(hidden_states.dtype)

        # Create masks (all valid, bidirectional attention for prefix)
        pad_masks = vlm_inputs.get("attention_mask", torch.ones(bsize, seq_len, dtype=torch.bool, device=hidden_states.device))
        att_masks = torch.zeros(bsize, seq_len, dtype=torch.bool, device=hidden_states.device)

        return hidden_states, pad_masks, att_masks

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        """
        Training forward pass.

        Args:
            examples: list of dicts with keys: image, lang, action, (state)

        Returns:
            (loss, metrics_dict)
        """
        actions = torch.stack([torch.tensor(ex["action"], dtype=torch.float32) for ex in examples])
        actions = actions.to(next(self.parameters()).device)

        # Normalize actions (MEAN_STD) if enabled
        if self.use_action_norm:
            actions = (actions - self.action_mean.to(actions.device)) / (self.action_std.to(actions.device) + 1e-8)

        # Pad action feature dim to model action_dim (e.g., robot 8-dim → base model 32-dim)
        if actions.shape[-1] < self.action_dim:
            actions = torch.nn.functional.pad(actions, (0, self.action_dim - actions.shape[-1]))

        # Truncate/pad actions to action_horizon
        if actions.shape[1] > self.action_horizon:
            actions = actions[:, :self.action_horizon]
        elif actions.shape[1] < self.action_horizon:
            pad = torch.zeros(
                actions.shape[0], self.action_horizon - actions.shape[1], actions.shape[2],
                device=actions.device, dtype=actions.dtype
            )
            actions = torch.cat([actions, pad], dim=1)

        # Get state if available (None for π₀.₅ with discrete state)
        state = None
        if not self.pi05 and "state" in examples[0]:
            state = torch.stack([torch.tensor(ex["state"], dtype=torch.float32) for ex in examples])
            state = state.to(actions.device)
            # Pad state feature dim to model action_dim
            if state.shape[-1] < self.action_dim:
                state = torch.nn.functional.pad(state, (0, self.action_dim - state.shape[-1]))

        # Encode prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_prefix(examples)
        language_token_count = int(
            getattr(getattr(self.config.framework, 'paligemma', None), 'max_token_len', 48)
        )
        prefix_embs, binding_state = self._apply_cabi_prefix(
            prefix_embs,
            prefix_pad_masks,
            language_token_count=language_token_count,
        )

        vlm_type = self._detect_vlm_type(self.config)

        if vlm_type == "paligemma":
            # Traditional joint attention path for PaliGemma
            vlm_lm = self._get_vlm_language_model()
            dsol_pair_groups_in_batch = []
            flow_noise = None
            flow_time = None
            if self.dsol_pair_shared_flow_noise_time:
                dsol_pair_groups_in_batch = dsol_pair_groups(
                    examples,
                    marker_key="dsol_pair_shared_flow",
                )
                if not dsol_pair_groups_in_batch:
                    raise ValueError(
                        "shared DSOL pair flow is enabled but the batch has no marked pairs"
                    )
                validate_paired_actions(actions, dsol_pair_groups_in_batch)
                flow_noise = self.flow_matching_head.sample_noise(
                    actions.shape, actions.device
                )
                flow_time = self.flow_matching_head.sample_time(
                    actions.shape[0], actions.device
                )
                flow_noise, flow_time = share_pair_noise_time(
                    flow_noise,
                    flow_time,
                    dsol_pair_groups_in_batch,
                )
            flow_result = self.flow_matching_head.compute_loss(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                vlm_language_model=vlm_lm,
                state=state,
                actions=actions,
                noise=flow_noise,
                time=flow_time,
                return_outputs=(
                    self.cabi_decoder_closure_weight != 0.0
                    or self.dsol_pair_consistency_enabled
                ),
            )
            if isinstance(flow_result, tuple):
                loss, flow_outputs = flow_result
            else:
                loss = flow_result
                flow_outputs = None
            cabi_transport_flow = self._compute_cabi_transport_flow(
                binding_state,
                examples,
                actions=actions,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                vlm_language_model=vlm_lm,
                state=state,
            )
            cabi_decoder_closure = self._compute_cabi_decoder_closure(
                binding_state,
                examples,
                flow_outputs=flow_outputs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                vlm_language_model=vlm_lm,
                state=state,
            )
            counterfactual_action_completion = (
                self._compute_counterfactual_action_completion(
                    examples,
                    actions=actions,
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks,
                    vlm_language_model=vlm_lm,
                    state=state,
                )
            )
        else:
            # Prefix cache path for non-PaliGemma VLMs (Llama, etc.)
            loss = self.flow_matching_head.compute_loss_prefix_cache(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                state=state,
                actions=actions,
            )
            cabi_transport_flow = None
            cabi_decoder_closure = None
            counterfactual_action_completion = None

        # Apply action mask: only compute loss on valid dims (e.g. first 7 of 32)
        # loss shape: [B, horizon, action_dim]
        if hasattr(self, '_action_dim_mask') and self._action_dim_mask is not None:
            loss = loss[:, :, self._action_dim_mask]  # [B, horizon, num_valid_dims]

        action_supervised = torch.as_tensor(
            [bool(ex.get("action_supervised", True)) for ex in examples],
            device=loss.device,
            dtype=torch.bool,
        )
        if not torch.any(action_supervised):
            raise ValueError("a Pi0.5 training batch requires at least one supervised action")
        loss = loss[action_supervised]

        feedback_horizon = None
        if all(self.feedback_horizon_key in ex for ex in examples):
            feedback_horizon = torch.as_tensor(
                [ex[self.feedback_horizon_key] for ex in examples],
                device=loss.device,
            )[action_supervised]

        loss_mean, fresh_metrics = feedback_weighted_flow_loss(
            loss,
            feedback_horizon=feedback_horizon,
            tail_weight=self.fresh_tail_weight,
            weighting_mode=self.fresh_weighting_mode,
        )
        total_loss = loss_mean
        dsol_pair_consistency = None
        dsol_pair_velocity_rmse = None
        if self.dsol_pair_consistency_enabled:
            if flow_outputs is None:
                raise RuntimeError("DSOL pair consistency requires flow outputs")
            dsol_pair_consistency, dsol_pair_velocity_rmse = (
                paired_prediction_consistency(
                    flow_outputs["prediction"],
                    dsol_pair_groups_in_batch,
                    action_dim_mask=getattr(self, "_action_dim_mask", None),
                )
            )
            total_loss = total_loss + (
                self.dsol_pair_consistency_weight * dsol_pair_consistency
            )
        cabi_loss, cabi_metrics = self._compute_cabi_closure(binding_state, examples)
        if cabi_loss is not None:
            total_loss = total_loss + cabi_loss
        if cabi_transport_flow is not None:
            total_loss = total_loss + self.cabi_flow_transport_weight * cabi_transport_flow
        if cabi_decoder_closure is not None:
            total_loss = total_loss + self.cabi_decoder_closure_weight * cabi_decoder_closure
        if counterfactual_action_completion is not None:
            total_loss = total_loss + (
                self.counterfactual_action_completion_weight
                * counterfactual_action_completion
            )

        output = {"action_loss": total_loss, "flow_matching_loss": loss_mean.item()}
        if dsol_pair_consistency is not None:
            output["dsol_pair_consistency_loss"] = dsol_pair_consistency.item()
            output["dsol_pair_velocity_rmse"] = dsol_pair_velocity_rmse.item()
            output["dsol_pair_count"] = len(dsol_pair_groups_in_batch)
        output.update({f"fresh_{key}": value.item() for key, value in fresh_metrics.items()})
        if cabi_loss is not None:
            output["cabi_loss"] = cabi_loss.item()
            output.update({f"cabi_{key}": value.item() for key, value in cabi_metrics.items()})
        if cabi_transport_flow is not None:
            output["cabi_transport_flow"] = cabi_transport_flow.item()
        if cabi_decoder_closure is not None:
            output["cabi_decoder_closure"] = cabi_decoder_closure.item()
        if counterfactual_action_completion is not None:
            output["counterfactual_action_completion"] = (
                counterfactual_action_completion.item()
            )
        return output

    def _maybe_remap_gripper(self, actions: torch.Tensor) -> torch.Tensor:
        """Map dim-6 from [0,1] to [+1,-1] when gripper_remap is enabled."""
        if self.gripper_remap and actions.shape[-1] > 6:
            actions[:, :, 6] = 1.0 - 2.0 * actions[:, :, 6]
        return actions

    @torch.no_grad()
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def predict_action(self, batch_images: List = None, instructions: List[str] = None,
                       examples: List[dict] = None, unnorm_key=None, **kwargs):
        """
        Inference: predict actions via multi-step denoising.

        Returns:
            np.ndarray: [B, action_horizon, action_dim] unnormalized actions
        """
        # CRITICAL: disable gradient checkpointing for inference
        # GC is incompatible with KV cache (corrupts cached key/values)
        for m in self.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = False

        # Support both flat format (batch_images/instructions) and legacy examples format
        if examples is None and batch_images is not None:
            from PIL import Image
            states = kwargs.get("states", None)
            examples = []
            for i, imgs in enumerate(batch_images):
                # Pass all views as list for multi-view support
                img = list(imgs) if isinstance(imgs, (list, tuple)) else imgs
                lang = instructions[i] if instructions else ""
                ex = {"image": img, "lang": lang}
                if states is not None and i < len(states):
                    ex["state"] = states[i]
                examples.append(ex)

        device = next(self.parameters()).device

        inference_noise = kwargs.get("noise")
        if inference_noise is not None:
            inference_noise = torch.as_tensor(
                inference_noise,
                dtype=torch.float32,
                device=device,
            )
            expected_noise_shape = (len(examples), self.action_horizon, self.action_dim)
            if tuple(inference_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"inference noise shape {tuple(inference_noise.shape)} != {expected_noise_shape}"
                )

        state = None
        if not self.pi05 and "state" in examples[0]:
            state = torch.stack([torch.tensor(ex["state"], dtype=torch.float32) for ex in examples])
            state = state.to(device)

        vlm_type = self._detect_vlm_type(self.config)

        if vlm_type != "paligemma":
            # Use prefix cache mode for non-PaliGemma VLMs
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_prefix_generic(examples)
            actions = self.flow_matching_head.sample_actions_prefix_cache(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                state=state,
                device=device,
                noise=inference_noise,
            )

            # Unnormalize actions if MEAN_STD normalization was used
            if self.use_action_norm:
                actions = actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
                actions = self._maybe_remap_gripper(actions)

            actions_np = actions.cpu().float().numpy()
            return {"normalized_actions": actions_np.tolist()}

        # Training and inference must share the exact image/token/camera path.
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self._prepare_prefix_paligemma(examples)
        )
        language_token_count = int(
            getattr(
                getattr(self.config.framework, "paligemma", None),
                "max_token_len",
                48,
            )
        )
        prefix_embs, _ = self._apply_cabi_prefix(
            prefix_embs,
            prefix_pad_masks,
            language_token_count=language_token_count,
        )

        vlm_lm = self._get_vlm_language_model()

        actions = self.flow_matching_head.sample_actions(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            vlm_language_model=vlm_lm,
            state=state,
            device=device,
            noise=inference_noise,
        )

        # Unnormalize actions if MEAN_STD normalization was used
        if self.use_action_norm:
            actions = actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
            actions = self._maybe_remap_gripper(actions)

        actions_np = actions.cpu().float().numpy()

        # Return actions (unnormalized if norm was enabled, raw otherwise)
        return {"normalized_actions": actions_np.tolist()}

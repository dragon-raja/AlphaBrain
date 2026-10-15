"""Opt-in image-gradient diagnostic; never used by the frozen evaluator.

Matches the PaliGemma Pi0.5 cached Euler sampler, but retains the graph and
stops after the requested velocity prefix on the ORIGINAL integration grid.
Model parameters must be frozen. No training or image acquisition occurs here.
"""
from __future__ import annotations

import torch


def velocity_prefix(model, examples, noise, *, prefix=3):
    """Return graph-connected normalized velocities and original flow times.

    Caller owns precision/autocast. Tensor images use the framework's CHW
    convention; an external image leaf in [0,255] preserves RGB input units.
    This adapter intentionally supports only the inspected Pi0.5/PaliGemma path.
    """
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.pi0_action_head import make_att_2d_masks

    if not model.pi05 or model._detect_vlm_type(model.config) != "paligemma":
        raise ValueError("only Pi0.5/PaliGemma is supported")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("diagnostic requires an eval model with frozen weights")
    head = model.flow_matching_head
    steps = head.num_inference_steps
    if isinstance(prefix, bool) or not isinstance(prefix, int) or not 2 <= prefix <= steps:
        raise ValueError("prefix must be an integer within the original grid")
    device = next(model.parameters()).device
    noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
    if tuple(noise.shape) != (len(examples), head.action_horizon, head.action_dim):
        raise ValueError("noise shape does not match examples and action head")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("noise must be finite")
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    emb, pad, att = model._prepare_prefix_paligemma(examples)
    language_count = int(getattr(getattr(model.config.framework, "paligemma", None), "max_token_len", 48))
    emb, _ = model._apply_cabi_prefix(emb, pad, language_token_count=language_count)
    dtype = next(head.parameters()).dtype

    def attention(mask):
        return torch.where(mask[:, None], 0.0, -2.3819763e38).to(dtype=dtype)

    lm = model._get_vlm_language_model()
    lm.config._attn_implementation = "eager"
    cache = lm(inputs_embeds=emb, attention_mask=attention(make_att_2d_masks(pad, att)),
               position_ids=torch.cumsum(pad, dim=1)-1, past_key_values=None,
               use_cache=True, adarms_cond=None).past_key_values
    dynamic = hasattr(cache, "to_legacy_cache")
    legacy = cache.to_legacy_cache() if dynamic else None
    expert = head.action_expert.model.model
    expert.config._attn_implementation = "eager"
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    x = noise
    velocities, times = [], []
    for _ in range(prefix):
        suffix, spad, satt, cond = head.embed_suffix(None, x, time.expand(len(examples)))
        full = torch.cat((pad[:, None, :].expand(len(examples), spad.shape[1], pad.shape[1]),
                          make_att_2d_masks(spad, satt)), dim=2)
        if dynamic:
            from transformers import DynamicCache
            step_cache = DynamicCache.from_legacy_cache(legacy)
        else:
            step_cache = cache
        out = expert(inputs_embeds=suffix, attention_mask=attention(full),
                     position_ids=pad.sum(-1)[:, None]+spad.cumsum(1)-1,
                     past_key_values=step_cache, use_cache=False, adarms_cond=cond)
        velocity = head.action_out_proj(out.last_hidden_state[:, -head.action_horizon:].float())
        velocities.append(velocity)
        times.append(time)
        x = x + dt * velocity
        time = time + dt
    return torch.stack(velocities, dim=1), torch.stack(times)

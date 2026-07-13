from __future__ import annotations

import torch
from torch import Tensor


def feedback_prefix_weights(
    feedback_horizon: Tensor,
    action_horizon: int,
    tail_weight: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build per-step FRESH weights and the corresponding prefix mask."""
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError(f"tail_weight must be in [0, 1], got {tail_weight}")

    horizons = torch.as_tensor(feedback_horizon, device=device).reshape(-1).long()
    horizons = horizons.clamp(0, action_horizon)
    steps = torch.arange(action_horizon, device=device)[None, :]
    prefix_mask = steps < horizons[:, None]
    weights = tail_weight + (1.0 - tail_weight) * prefix_mask.to(dtype=dtype)
    return weights, prefix_mask


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(dtype=values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def feedback_weighted_flow_loss(
    per_dim_loss: Tensor,
    feedback_horizon: Tensor | None = None,
    tail_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Reduce a ``[batch, horizon, action_dim]`` flow-matching loss.

    Every sample is normalized by its own total step weight so examples with a
    longer feedback-safe prefix do not receive a larger aggregate batch weight.
    ``tail_weight=1`` returns the exact baseline mean reduction.
    """
    if per_dim_loss.ndim != 3:
        raise ValueError(f"expected [B, H, D] loss, got shape {tuple(per_dim_loss.shape)}")
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError(f"tail_weight must be in [0, 1], got {tail_weight}")

    if tail_weight == 1.0:
        baseline = per_dim_loss.mean()
        return baseline, {"full_loss": baseline.detach()}

    per_step_loss = per_dim_loss.mean(dim=-1)
    if feedback_horizon is None:
        raise ValueError("feedback_horizon is required when tail_weight < 1")

    weights, prefix_mask = feedback_prefix_weights(
        feedback_horizon,
        per_dim_loss.shape[1],
        tail_weight,
        dtype=per_dim_loss.dtype,
        device=per_dim_loss.device,
    )
    per_sample = (weights * per_step_loss).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    loss = per_sample.mean()
    metrics = {
        "full_loss": per_dim_loss.mean().detach(),
        "prefix_loss": _masked_mean(per_step_loss, prefix_mask).detach(),
        "suffix_loss": _masked_mean(per_step_loss, ~prefix_mask).detach(),
        "mean_feedback_horizon": prefix_mask.sum(dim=1).float().mean().detach(),
    }
    return loss, metrics

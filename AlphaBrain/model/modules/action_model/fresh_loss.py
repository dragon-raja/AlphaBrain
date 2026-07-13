from __future__ import annotations

import torch
from torch import Tensor


def _validate_feedback_horizons(
    feedback_horizon: Tensor,
    action_horizon: int,
    *,
    batch_size: int | None,
    device: torch.device,
) -> Tensor:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    raw = torch.as_tensor(feedback_horizon, device=device).reshape(-1)
    if batch_size is not None and raw.numel() != batch_size:
        raise ValueError(f"expected {batch_size} feedback horizons, got {raw.numel()}")
    if raw.is_floating_point() and not torch.equal(raw, raw.round()):
        raise ValueError("feedback_horizon values must be integers")

    horizons = raw.long()
    if torch.any((horizons < 0) | (horizons > action_horizon)):
        values = horizons.detach().cpu().tolist()
        raise ValueError(f"feedback_horizon must be in [0, {action_horizon}], got {values}")
    return horizons


def feedback_prefix_weights(
    feedback_horizon: Tensor,
    action_horizon: int,
    tail_weight: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int | None = None,
    weighting_mode: str = "suffix",
) -> tuple[Tensor, Tensor]:
    """Build weights where horizon ``h`` keeps exactly steps ``[0, h)``."""
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError(f"tail_weight must be in [0, 1], got {tail_weight}")

    horizons = _validate_feedback_horizons(
        feedback_horizon,
        action_horizon,
        batch_size=batch_size,
        device=device,
    )
    steps = torch.arange(action_horizon, device=device)[None, :]
    prefix_mask = steps < horizons[:, None]
    if weighting_mode == "suffix":
        weights = tail_weight + (1.0 - tail_weight) * prefix_mask.to(dtype=dtype)
    elif weighting_mode == "prefix_control":
        downweighted_steps = action_horizon - horizons
        control_mask = steps < downweighted_steps[:, None]
        weights = 1.0 - (1.0 - tail_weight) * control_mask.to(dtype=dtype)
    else:
        raise ValueError(f"weighting_mode must be 'suffix' or 'prefix_control', got {weighting_mode!r}")
    return weights, prefix_mask


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(dtype=values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def feedback_weighted_flow_loss(
    per_dim_loss: Tensor,
    feedback_horizon: Tensor | None = None,
    tail_weight: float = 1.0,
    weighting_mode: str = "suffix",
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
    if weighting_mode not in {"suffix", "prefix_control"}:
        raise ValueError(f"unknown weighting_mode: {weighting_mode!r}")

    if feedback_horizon is not None:
        _validate_feedback_horizons(
            feedback_horizon,
            per_dim_loss.shape[1],
            batch_size=per_dim_loss.shape[0],
            device=per_dim_loss.device,
        )

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
        batch_size=per_dim_loss.shape[0],
        weighting_mode=weighting_mode,
    )
    weight_sums = weights.sum(dim=1)
    supervised_samples = weight_sums > 0
    per_sample = (weights * per_step_loss).sum(dim=1) / weight_sums.clamp_min(1.0)
    loss = per_sample.sum() / supervised_samples.sum().clamp_min(1)
    metrics = {
        "full_loss": per_dim_loss.mean().detach(),
        "prefix_loss": _masked_mean(per_step_loss, prefix_mask).detach(),
        "suffix_loss": _masked_mean(per_step_loss, ~prefix_mask).detach(),
        "mean_feedback_horizon": prefix_mask.sum(dim=1).float().mean().detach(),
    }
    return loss, metrics

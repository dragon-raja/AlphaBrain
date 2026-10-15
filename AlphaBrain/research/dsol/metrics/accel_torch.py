"""Differentiable counterpart of the registered Accel formula.

This does not modify the frozen numpy scorer or make predict_action
differentiable. A separate model adapter must supply graph-connected
normalized velocity traces on the original denoising grid.
"""
from __future__ import annotations

import math
import operator

import torch


def compute_accel_score_tensor(velocity_trace, *, prefix=3, epsilon=1e-12):
    """Return score, degenerate flag, numerator and denominator tensors.

    Input [batch, denoise_step, ...], matching compute_accel_scores.
    Degenerate traces retain the historic zero value but MUST be excluded
    from selection. No detach, weight update, or host conversion is applied
    to the returned numeric tensors. Norms are nondifferentiable at zero;
    PyTorch's zero subgradient applies there.
    """
    trace = velocity_trace
    if (not isinstance(trace, torch.Tensor) or not trace.is_floating_point()
            or trace.ndim < 3 or trace.shape[0] < 1
            or trace.numel() == 0 or not bool(torch.isfinite(trace).all())):
        raise ValueError("finite nonempty floating [batch, step, ...] tensor required")
    if isinstance(prefix, bool):
        raise ValueError("prefix must be an integer")
    try:
        prefix = operator.index(prefix)
    except TypeError as exc:
        raise ValueError("prefix must be an integer") from exc
    if not 2 <= prefix <= trace.shape[1]:
        raise ValueError("prefix is outside the velocity trace")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    # fp16 cannot represent the default epsilon; use fp32 accumulation there.
    if trace.dtype in (torch.float16, torch.bfloat16):
        trace = trace.float()
    flat = trace[:, :prefix].reshape(trace.shape[0], prefix, -1)
    denominator = torch.linalg.vector_norm(flat, dim=-1).sum(dim=1)
    numerator = prefix * torch.linalg.vector_norm(
        flat[:, 1:] - flat[:, :-1], dim=-1).sum(dim=1)
    degenerate = denominator <= epsilon
    score = torch.where(degenerate, torch.zeros_like(denominator),
                        numerator / denominator.clamp_min(epsilon))
    return {"score": score, "degenerate": degenerate,
            "numerator": numerator, "denominator": denominator}

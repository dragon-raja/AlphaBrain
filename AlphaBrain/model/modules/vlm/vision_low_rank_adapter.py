"""Native low-rank adapters for the SigLIP vision tower.

The adapters keep the original ``nn.Linear`` weight and bias key names. This
lets AlphaBrain initialize from ordinary Pi0/Pi0.5 checkpoints while storing
the low-rank parameters in its self-contained checkpoints without a PEFT-side
reload step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAdaptedLinear(nn.Linear):
    """Linear layer with an additive, zero-initialized low-rank update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.adapter_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.adapter_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.adapter_A, a=5**0.5)
        self.adapter_dropout = nn.Dropout(float(dropout))
        self.adapter_scaling = float(alpha) / int(rank)
        self.adapter_rank = int(rank)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        freeze_base: bool,
    ) -> "LowRankAdaptedLinear":
        adapted = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        adapted.weight = linear.weight
        adapted.bias = linear.bias
        adapted.train(linear.training)
        if freeze_base:
            adapted.weight.requires_grad_(False)
            if adapted.bias is not None:
                adapted.bias.requires_grad_(False)
        return adapted

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        low_rank = F.linear(self.adapter_dropout(inputs), self.adapter_A)
        low_rank = F.linear(low_rank, self.adapter_B)
        return base + low_rank * self.adapter_scaling


@dataclass(frozen=True)
class VisionAdapterSummary:
    adapted_layer_count: int
    adapter_parameter_count: int
    target_modules: tuple[str, ...]


def _resolve_parent(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid module path: {path!r}")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TypeError(f"{path!r} does not resolve through nn.Module objects")
    return parent, parts[-1]


def inject_siglip_low_rank_adapters(
    vision_tower: nn.Module,
    *,
    target_modules: Sequence[str] = ("mlp.fc1", "mlp.fc2"),
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> VisionAdapterSummary:
    """Inject adapters into every SigLIP encoder layer.

    ``target_modules`` paths are resolved relative to each encoder layer.
    The default MLP-only rank-16 setup adds 4,713,984 parameters to the
    27-layer PaliGemma SigLIP tower.
    """

    targets = tuple(str(value) for value in target_modules)
    if not targets:
        raise ValueError("target_modules cannot be empty")
    try:
        layers = vision_tower.vision_model.encoder.layers
    except AttributeError as exc:
        raise TypeError("expected a SigLIP vision tower with encoder layers") from exc
    if len(layers) == 0:
        raise ValueError("vision tower has no encoder layers")

    if freeze_base:
        for parameter in vision_tower.parameters():
            parameter.requires_grad_(False)

    adapted_count = 0
    for layer_index, layer in enumerate(layers):
        for target in targets:
            parent, attribute = _resolve_parent(layer, target)
            linear = getattr(parent, attribute)
            if isinstance(linear, LowRankAdaptedLinear):
                raise ValueError(
                    f"adapter already injected at layer {layer_index} target {target}"
                )
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"layer {layer_index} target {target!r} is not nn.Linear"
                )
            setattr(
                parent,
                attribute,
                LowRankAdaptedLinear.from_linear(
                    linear,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                ),
            )
            adapted_count += 1

    adapter_parameter_count = sum(
        parameter.numel()
        for module in vision_tower.modules()
        if isinstance(module, LowRankAdaptedLinear)
        for parameter in (module.adapter_A, module.adapter_B)
    )
    return VisionAdapterSummary(
        adapted_layer_count=adapted_count,
        adapter_parameter_count=adapter_parameter_count,
        target_modules=targets,
    )

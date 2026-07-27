"""Camera conditioning utilities for view-invariant policy learning.

The implementation follows the late-fusion design from "Do You Know Where
Your Camera Is?": camera intrinsics and extrinsics define a per-pixel
Plucker ray map, which is encoded separately and fused with frozen RGB
features.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def plucker_raymap(
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return OpenCV-convention Plucker rays as ``[B, 6, H, W]``.

    Channels are unit ray direction followed by moment ``origin x direction``.
    Pixel coordinates refer to pixel centers, matching the KYC reference
    implementation and project-page snippet.
    """

    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, 3, 3]")
    if camera_to_world.ndim != 3 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape [B, 4, 4]")
    if intrinsics.shape[0] != camera_to_world.shape[0]:
        raise ValueError("intrinsics and camera_to_world batch sizes differ")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    device = intrinsics.device
    dtype = intrinsics.dtype
    vv, uu = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1)
    pixels = pixels.unsqueeze(0).expand(intrinsics.shape[0], -1, -1, -1)

    inverse_intrinsics = torch.linalg.inv(intrinsics)
    camera_directions = torch.einsum(
        "bhwc,bdc->bhwd",
        pixels,
        inverse_intrinsics,
    )
    world_directions = torch.einsum(
        "bhwc,bdc->bhwd",
        camera_directions,
        camera_to_world[:, :3, :3],
    )
    world_directions = F.normalize(world_directions, dim=-1, eps=1e-9)
    origins = camera_to_world[:, None, None, :3, 3]
    moments = torch.cross(origins.expand_as(world_directions), world_directions, dim=-1)
    rays = torch.cat((world_directions, moments), dim=-1)
    return rays.permute(0, 3, 1, 2).contiguous()


def transform_raymap(
    raymap: torch.Tensor,
    *,
    image_transform: str,
) -> torch.Tensor:
    """Align OpenCV-top-left rays with the tensor consumed by the policy."""

    if image_transform == "none":
        return raymap
    if image_transform == "rot180":
        return torch.flip(raymap, dims=(-2, -1))
    if image_transform == "mujoco_upright":
        # MuJoCo's raw render is vertically inverted relative to OpenCV.
        # The dataset then rotates that raw render by 180 degrees, leaving
        # only a horizontal flip between OpenCV pixels and policy pixels.
        return torch.flip(raymap, dims=(-1,))
    raise ValueError(f"unsupported camera image transform: {image_transform}")


class FrozenAffineNorm2d(nn.Module):
    """Fixed batch-normalization affine used by the official KYC CNN."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        bias = self.bias - self.running_mean * scale
        return (
            inputs * scale.reshape(1, -1, 1, 1)
            + bias.reshape(1, -1, 1, 1)
        )


class PluckerLateFusion(nn.Module):
    """Official KYC/SmolVLA Plucker fusion before the VLM connector."""

    def __init__(
        self,
        *,
        rgb_hidden_dim: int,
        encoder_channels: Sequence[int] = (64, 128, 256, 512, 512),
    ) -> None:
        super().__init__()
        if rgb_hidden_dim <= 0:
            raise ValueError("rgb_hidden_dim must be positive")
        channels = tuple(int(value) for value in encoder_channels)
        if len(channels) != 5 or any(value <= 0 for value in channels):
            raise ValueError("encoder_channels must contain five positive values")

        blocks: list[nn.Module] = []
        in_channels = 6
        for index, out_channels in enumerate(channels):
            blocks.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=7 if index == 0 else 3,
                        stride=2,
                        padding=3 if index == 0 else 1,
                        bias=False,
                    ),
                    FrozenAffineNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels
        self.ray_encoder = nn.Sequential(*blocks)
        self.ray_projection = nn.Conv2d(
            channels[-1],
            rgb_hidden_dim,
            kernel_size=1,
        )
        self.vision_norm = nn.LayerNorm(
            rgb_hidden_dim,
            elementwise_affine=False,
        )
        self.ray_norm = nn.LayerNorm(
            rgb_hidden_dim,
            elementwise_affine=False,
        )
        self.fusion = nn.Linear(
            rgb_hidden_dim * 2,
            rgb_hidden_dim,
            bias=True,
        )
        self.rgb_hidden_dim = int(rgb_hidden_dim)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        raymap: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_tokens.ndim != 3:
            raise ValueError("rgb_tokens must have shape [B, N, D]")
        if raymap.ndim != 4 or raymap.shape[1] != 6:
            raise ValueError("raymap must have shape [B, 6, H, W]")
        if rgb_tokens.shape[0] != raymap.shape[0]:
            raise ValueError("RGB and ray-map batch sizes differ")
        if rgb_tokens.shape[-1] != self.rgb_hidden_dim:
            raise ValueError(
                f"expected RGB hidden dim {self.rgb_hidden_dim}, "
                f"got {rgb_tokens.shape[-1]}"
            )

        grid_size = math.isqrt(rgb_tokens.shape[1])
        if grid_size * grid_size != rgb_tokens.shape[1]:
            raise ValueError("RGB image token count must form a square grid")
        camera_features = self.ray_encoder(raymap)
        camera_features = F.adaptive_avg_pool2d(
            camera_features,
            output_size=(grid_size, grid_size),
        )
        camera_features = self.ray_projection(camera_features)
        camera_tokens = camera_features.flatten(2).transpose(1, 2)
        camera_tokens = camera_tokens.to(rgb_tokens.dtype)
        fused = torch.cat(
            (
                self.vision_norm(rgb_tokens),
                self.ray_norm(camera_tokens),
            ),
            dim=-1,
        )
        return self.fusion(fused)

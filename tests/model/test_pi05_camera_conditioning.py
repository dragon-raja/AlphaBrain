from __future__ import annotations

import torch

from AlphaBrain.model.modules.vlm.camera_conditioning import (
    PluckerLateFusion,
    plucker_raymap,
    transform_raymap,
)


def test_plucker_rays_are_unit_and_satisfy_bilinear_constraint() -> None:
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]]
    )
    camera_to_world = torch.eye(4).unsqueeze(0)
    camera_to_world[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    rays = plucker_raymap(
        intrinsics,
        camera_to_world,
        height=4,
        width=4,
    )
    directions, moments = rays[:, :3], rays[:, 3:]
    torch.testing.assert_close(
        torch.linalg.vector_norm(directions, dim=1),
        torch.ones(1, 4, 4),
    )
    torch.testing.assert_close(
        (directions * moments).sum(dim=1),
        torch.zeros(1, 4, 4),
        atol=1e-6,
        rtol=0.0,
    )


def test_late_fusion_matches_official_parameter_count() -> None:
    module = PluckerLateFusion(
        rgb_hidden_dim=1152,
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == 7_172_736


def test_mujoco_upright_only_flips_opencv_rays_horizontally() -> None:
    raymap = torch.arange(2 * 3).reshape(1, 1, 2, 3)
    transformed = transform_raymap(
        raymap,
        image_transform="mujoco_upright",
    )
    torch.testing.assert_close(
        transformed,
        torch.tensor([[[[2, 1, 0], [5, 4, 3]]]]),
    )


def test_all_camera_branch_parameters_receive_gradient() -> None:
    module = PluckerLateFusion(
        rgb_hidden_dim=16,
        encoder_channels=(4, 4, 4, 4, 4),
    )
    output = module(
        torch.randn(1, 16 * 16, 16),
        torch.randn(1, 6, 224, 224),
    )
    output.square().mean().backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0)
    ]
    assert missing == []

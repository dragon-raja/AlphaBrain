from __future__ import annotations

import torch

from AlphaBrain.model.modules.vlm.camera_conditioning import (
    PluckerResidualFusion,
    plucker_raymap,
)


def test_plucker_raymap_has_unit_directions() -> None:
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]]
    )
    camera_to_world = torch.eye(4).unsqueeze(0)
    rays = plucker_raymap(
        intrinsics,
        camera_to_world,
        height=8,
        width=8,
    )
    assert rays.shape == (1, 6, 8, 8)
    torch.testing.assert_close(
        torch.linalg.vector_norm(rays[:, :3], dim=1),
        torch.ones(1, 8, 8),
    )
    torch.testing.assert_close(rays[:, 3:], torch.zeros(1, 3, 8, 8))


def test_residual_fusion_starts_as_exact_rgb_identity_and_learns_rays() -> None:
    torch.manual_seed(7)
    module = PluckerResidualFusion(
        rgb_hidden_dim=8,
        encoder_channels=(4, 4, 8, 8, 8),
    )
    rgb = torch.randn(2, 4, 8)
    raymap = torch.randn(2, 6, 32, 32)
    target_weight = torch.randn_like(rgb)

    initial = module(rgb, raymap)
    torch.testing.assert_close(initial, rgb, rtol=0.0, atol=0.0)

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.zero_grad()
    (module(rgb, raymap) * target_weight).sum().backward()
    assert module.gate.grad is not None
    assert torch.count_nonzero(module.gate.grad) > 0
    optimizer.step()

    optimizer.zero_grad()
    changed = module(rgb, raymap)
    assert not torch.equal(changed, rgb)
    (changed * target_weight).sum().backward()
    first_convolution = module.ray_encoder[0]
    assert first_convolution.weight.grad is not None
    assert torch.count_nonzero(first_convolution.weight.grad) > 0

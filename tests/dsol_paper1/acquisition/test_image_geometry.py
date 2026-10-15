"""Independent analytic-scene finite differences; no simulator or policy."""
import numpy as np
import pytest

from AlphaBrain.research.dsol.acquisition.image_geometry import (
    contract_image_gradient, image_motion_jacobian, pixel_motion_jacobian,
)


K = np.array([[43., 0., 9.], [0., 39., 8.], [0., 0., 1.]])


def rotation(w):
    angle = np.linalg.norm(w)
    if angle == 0:
        return np.eye(3)
    x, y, z = w / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


def plane_image(twist):
    """Exact rays intersect static plane Z=2; affine RGB surface texture."""
    v, u = np.indices((17, 19))
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    rays = rays @ rotation(twist[3:]).T
    t = twist[:3]
    points = t + rays * ((2. - t[2]) / rays[..., 2])[..., None]
    return points @ np.array([[.12, .21, -.08], [.23, -.11, .17], [0., 0., 0.]]) + .5


@pytest.mark.parametrize("metric", ["mean", "quadratic", "accel_synthetic_trace"])
def test_swappable_metric_chain_matches_exact_render_derivative(metric):
    image = plane_image(np.zeros(6))
    jac, valid = image_motion_jacobian(image, np.full((17, 19), 2.), K)
    if metric == "mean":
        score = lambda x: np.mean(x)
        grad = np.ones_like(image) / image.size
    elif metric == "quadratic":
        score = lambda x: np.mean((x-.2)**2)
        grad = 2*(image-.2) / image.size
    else:
        torch = pytest.importorskip("torch")
        from AlphaBrain.research.dsol.metrics.accel_torch import compute_accel_score_tensor
        # Synthetic image-to-velocity map, NOT a VLA or a denoising rollout.
        def objective(pixels):
            means = pixels.mean(dim=(0, 1))
            # Keep successive velocities distinct: norm at zero is a kink,
            # where a unique central-difference derivative does not exist.
            trace = torch.stack([means, 1.3*means+.1, .8*means+.5])[None]
            return compute_accel_score_tensor(trace)["score"].mean()
        leaf = torch.tensor(image, dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(objective(leaf), leaf)[0].detach().numpy()
        score = lambda x: objective(torch.tensor(x, dtype=torch.float64)).item()
    result = contract_image_gradient(grad, jac)
    finite_difference = []
    for axis in range(6):
        step = np.eye(6)[axis] * 1e-6
        finite_difference.append((score(plane_image(step))-score(plane_image(-step)))/2e-6)
    np.testing.assert_allclose(result, finite_difference, atol=1e-9, rtol=1e-6)
    np.testing.assert_array_equal(contract_image_gradient(grad, jac, minimize=False), -result)
    assert valid.all()
    # A tiny move in this *synthetic* smooth scene improves either objective.
    assert score(plane_image(-1e-4*result)) < score(image)


def test_point_motion_sign_matches_camera_motion_with_nonconstant_depth():
    depth = np.linspace(.5, 3., 17*19).reshape(17, 19)
    jac, _ = pixel_motion_jacobian(depth, K)
    v, u = np.indices(depth.shape)
    points = (np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T)*depth[..., None]
    def project(delta):
        target = (points - delta[:3]) @ rotation(delta[3:])
        pixels = target @ K.T
        return pixels[..., :2] / pixels[..., 2:]
    for axis in range(6):
        step = np.eye(6)[axis]*1e-6
        np.testing.assert_allclose(jac[..., axis], (project(step)-project(-step))/2e-6,
                                   atol=1e-7, rtol=1e-6)


def test_depth_mask_is_explicit_and_missing_geometry_has_zero_derivative():
    depth = np.ones((3, 3)); depth[0] = [np.nan, 0., -1.]
    mask = np.ones((3, 3), bool); mask[1, 1] = False
    jac, valid = pixel_motion_jacobian(depth, K, valid_mask=mask)
    assert valid.sum() == 5
    assert np.isfinite(jac).all()
    assert np.all(jac[~valid] == 0)


def test_rejects_invalid_geometry_and_metric_shapes():
    with pytest.raises(ValueError):
        pixel_motion_jacobian(np.ones((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        image_motion_jacobian(np.ones((3, 3, 3)), np.ones((4, 4)), K)
    with pytest.raises(ValueError):
        contract_image_gradient(np.ones((3, 3, 3)), np.ones((3, 3, 6)))

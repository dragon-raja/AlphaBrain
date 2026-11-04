import pytest

torch = pytest.importorskip('torch')
from AlphaBrain.model.rldx_features import RLDXVisualCapture


class NativeDouble:
    """CPU double preserves native per-image boundaries; not a GPU admission."""
    def __init__(self):
        self.calls = 0

    def get_image_features(self, pixel_values, image_grid_thw=None, num_frames=1, num_views=1):
        self.calls += 1
        values = tuple(pixel_values[i].expand(4, 3).clone() for i in range(num_frames*num_views))
        self.output = (values, [torch.randn(1)])
        return self.output


def capture(model, cameras=('external', 'wrist'), frames=4):
    return RLDXVisualCapture(model, cameras=cameras, frames=frames,
                            selected_frame=frames-1, channels=3, merge_size=2)


def invoke(model, frames=4, views=2):
    return model.get_image_features(torch.arange(frames*views, dtype=torch.float32),
                                   torch.tensor([[1, 4, 4]]*(frames*views)),
                                   num_frames=frames, num_views=views)


def test_current_frame_camera_order_output_and_rng_preserved():
    m = NativeDouble()
    torch.manual_seed(9); original = invoke(m); rng = torch.get_rng_state()
    torch.manual_seed(9)
    tap = capture(m)
    with tap:
        actual = invoke(m)
        assert actual is m.output
        with pytest.raises(ValueError): tap.features()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(actual[1][0], original[1][0])
    assert m.calls == 2  # One baseline and one observed; no extra forward.
    out = tap.features()
    assert torch.equal(out['external'], torch.full((1, 4, 3), 6.))
    assert torch.equal(out['wrist'], torch.full((1, 4, 3), 7.))
    assert not out['external'].requires_grad
    out['external'].zero_()
    assert tap.features()['external'].sum() > 0
    assert 'get_image_features' not in vars(m)


def test_one_real_head_does_not_fabricate_wrist():
    m = NativeDouble(); tap = capture(m, cameras=('head',), frames=1)
    with tap: invoke(m, frames=1, views=1)
    assert set(tap.features()) == {'head'}


def test_non_square_merged_grid_preserves_spatial_quadrants():
    m = NativeDouble()
    def spatial(pixel_values, image_grid_thw=None, num_frames=1, num_views=1):
        tokens = torch.arange(8, dtype=torch.float32)[:, None].expand(8, 3)
        return (tokens,), []
    m.get_image_features = spatial
    tap = capture(m, cameras=('head',), frames=1)
    with tap:
        m.get_image_features(torch.zeros(1), torch.tensor([[1, 8, 4]]))
    expected = torch.tensor([1., 2., 5., 6.]).reshape(1, 4, 1).expand(1, 4, 3)
    assert torch.equal(tap.features()['head'], expected)
    assert m.get_image_features is spatial


def test_failure_restores_original_and_cannot_reuse_stale_result():
    m = NativeDouble(); tap = capture(m)
    with pytest.raises(ValueError):
        with tap:
            invoke(m); invoke(m)
    assert 'get_image_features' not in vars(m)
    with pytest.raises(ValueError): tap.features()
    with pytest.raises(RuntimeError):
        with tap: pass


@pytest.mark.parametrize('change', ['frames', 'views', 'grid', 'nonfinite', 'joint'])
def test_wrong_provenance_or_fake_joint_camera_boundaries_refused(change):
    m = NativeDouble(); original = m.get_image_features
    if change == 'joint':
        def joint(*args, **kwargs):
            return torch.ones(1, 64, 3), []
        # Retain the actual upstream signature while simulating a bad return.
        def bad(pixel_values, image_grid_thw=None, num_frames=1, num_views=1):
            return joint(pixel_values)
        m.get_image_features = bad
    tap = capture(m)
    x = torch.arange(8, dtype=torch.float32)
    grid = torch.tensor([[1, 4, 4]]*8)
    if change == 'nonfinite': x[0] = float('nan')
    if change == 'grid': grid[0, 1] = 5
    with pytest.raises(ValueError):
        with tap:
            m.get_image_features(x, grid, num_frames=1 if change == 'frames' else 4,
                                 num_views=1 if change == 'views' else 2)
    with pytest.raises(ValueError): tap.features()


def test_nested_capture_refused_without_removing_outer_capture():
    m = NativeDouble()
    with capture(m) as outer:
        with pytest.raises(RuntimeError):
            with capture(m): pass
        invoke(m)
    assert set(outer.features()) == {'external', 'wrist'}

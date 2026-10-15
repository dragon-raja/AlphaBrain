"""CPU-only checks for the explicit initialization attribution noise protocol."""
import numpy as np
import pytest

from benchmarks.LIBERO.initialization_diagnostic import extend_noise


def test_extension_preserves_all_original_coordinates_and_is_reproducible():
    original = np.arange(70, dtype=np.float32).reshape(10, 7)
    extended = extend_noise(original, 314159, 32)
    assert extended.shape == (10, 32)
    assert extended.dtype == np.float32
    np.testing.assert_array_equal(extended[:, :7], original)
    np.testing.assert_array_equal(extended, extend_noise(original, 314159, 32))
    assert not np.array_equal(extended[:, 7:], extend_noise(original, 314160, 32)[:, 7:])


def test_seven_dimensional_intervention_does_not_change_input():
    original = np.ones((10, 7), dtype=np.float32)
    result = extend_noise(original, 1, 7)
    np.testing.assert_array_equal(result, original)
    result[0, 0] = 2
    assert original[0, 0] == 1


def test_added_coordinates_do_not_reuse_original_random_stream():
    # Regression for same-seed float32/float64 stream reuse: old code produced
    # cross-coordinate correlation about 0.58, despite different draw dtypes.
    seeds = np.random.default_rng(20260923).integers(0, 2**63-1, 6000)
    original = np.stack([np.random.default_rng(int(seed)).standard_normal((10, 7), dtype=np.float32)
                         for seed in seeds])
    extra = np.stack([extend_noise(row, int(seed), 32)[:, 7:]
                      for row, seed in zip(original, seeds)])
    a, b = original.reshape(6000, -1), extra.reshape(6000, -1)
    a = (a-a.mean(0))/a.std(0)
    b = (b-b.mean(0))/b.std(0)
    assert abs(a.T@b/6000).max() < 0.08


@pytest.mark.parametrize('shape,dimension', [((10, 32), 32), ((9, 7), 7), ((10, 7), 8)])
def test_invalid_noise_is_not_silently_padded(shape, dimension):
    with pytest.raises(ValueError):
        extend_noise(np.zeros(shape, dtype=np.float32), 1, dimension)

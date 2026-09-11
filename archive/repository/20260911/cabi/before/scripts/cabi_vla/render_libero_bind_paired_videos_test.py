import numpy as np

from render_libero_bind_paired_videos import pair_rollout_frames


def test_pair_rollout_frames_pads_shorter_sequence_and_preserves_pixels() -> None:
    baseline = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    method = np.zeros((3, 4, 6, 3), dtype=np.uint8)
    baseline[0] = 11
    baseline[1] = 22
    method[0] = 31
    method[1] = 32
    method[2] = 33

    paired, metadata = pair_rollout_frames(
        baseline,
        method,
        baseline_label="BC",
        method_label="CABI",
    )

    assert paired.shape == (3, 28, 16, 3)
    assert np.all(paired[2, 24:, :6] == 22)
    assert np.all(paired[2, 24:, 10:] == 33)
    assert metadata == {
        "baseline_frame_count": 2,
        "method_frame_count": 3,
        "paired_frame_count": 3,
    }


def test_pair_rollout_frames_rejects_spatial_mismatch() -> None:
    baseline = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    method = np.zeros((2, 5, 6, 3), dtype=np.uint8)
    try:
        pair_rollout_frames(
            baseline,
            method,
            baseline_label="BC",
            method_label="CABI",
        )
    except ValueError as error:
        assert "shapes differ" in str(error)
    else:
        raise AssertionError("spatially mismatched paired rollouts were accepted")


if __name__ == "__main__":
    test_pair_rollout_frames_pads_shorter_sequence_and_preserves_pixels()
    test_pair_rollout_frames_rejects_spatial_mismatch()

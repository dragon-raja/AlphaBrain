import numpy as np

from render_libero_bind_eval_frames import make_contact_sheet


def test_contact_sheet_samples_endpoints_and_has_stable_shape() -> None:
    frames = np.zeros((5, 8, 12, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, :, :, :] = index * 40
    sheet, indices = make_contact_sheet(frames, tile_count=4, columns=2)
    assert indices == [0, 1, 3, 4]
    assert sheet.shape == (16, 24, 3)
    np.testing.assert_array_equal(sheet[7, 11], frames[0, 7, 11])
    np.testing.assert_array_equal(sheet[15, 23], frames[4, 7, 11])


def test_contact_sheet_rejects_empty_input() -> None:
    try:
        make_contact_sheet(np.empty((0, 8, 12, 3), dtype=np.uint8))
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty frame sequences must fail")

from __future__ import annotations

from analyze_pi05_libero_plus_kyc_matched import build_report


def _run(camera: float, joint: float) -> dict[str, dict[str, float]]:
    return {
        f"suite::task-{index}": {
            "canonical": 1.0,
            "camera_only": camera,
            "background_only": 1.0,
            "camera_background": joint,
        }
        for index in range(8)
    }


def test_matched_report_confirms_large_consistent_kyc_gain() -> None:
    control = {41: _run(0.0, 0.0), 42: _run(0.0, 0.0), 43: _run(0.0, 0.0)}
    kyc = {41: _run(1.0, 1.0), 42: _run(1.0, 1.0), 43: _run(1.0, 1.0)}

    report = build_report(control, kyc)

    assert report["decision"] == "KYC_INCREMENTAL_VALUE_CONFIRMED"
    assert report["cross_seed"]["kyc_minus_control"]["camera_only_success"][
        "mean"
    ] == 1.0


def test_matched_report_rejects_no_incremental_gain() -> None:
    control = {seed: _run(1.0, 1.0) for seed in (41, 42, 43)}
    kyc = {seed: _run(1.0, 1.0) for seed in (41, 42, 43)}

    report = build_report(control, kyc)

    assert report["decision"] == "KYC_NO_MEANINGFUL_INCREMENTAL_VALUE"


def test_matched_report_requires_same_seeds() -> None:
    try:
        build_report({41: _run(1.0, 1.0)}, {42: _run(1.0, 1.0)})
    except ValueError as error:
        assert "seed sets differ" in str(error)
    else:
        raise AssertionError("expected a seed mismatch")

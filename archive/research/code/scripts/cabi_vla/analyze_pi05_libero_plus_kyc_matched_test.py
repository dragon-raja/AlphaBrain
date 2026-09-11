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


def test_cross_seed_interval_includes_training_seed_variation() -> None:
    control = {seed: _run(0.0, 0.0) for seed in (41, 42, 43)}
    kyc = {
        41: _run(1.0, 1.0),
        42: _run(0.0, 0.0),
        43: _run(0.0, 0.0),
    }

    report = build_report(control, kyc)
    effect = report["cross_seed"]["kyc_minus_control"]["camera_only_success"]

    assert effect["mean"] == 1 / 3
    assert effect["ci95"][0] == 0.0
    assert effect["bootstrap_scheme"] == "crossed_training_seed_and_base_task"
    control_condition = report["cross_seed"]["control"]["conditions"]["canonical"]
    assert control_condition["bootstrap_scheme"] == "crossed_training_seed_and_base_task"


def test_factor_separated_interpretation_preserves_claim_boundary() -> None:
    runs = {seed: _run(1.0, 1.0) for seed in (41, 42, 43)}

    report = build_report(
        runs,
        runs,
        interpretation="factor_separated_category_composition",
    )

    boundary = report["interpretation_boundary"]
    assert boundary["factor_category_separated_training"] is True
    assert boundary["strict_seen_factor_composition"] is False
    assert boundary["joint_factor_training_episode_count"] == 0


def test_low_control_canonical_success_invalidates_gate() -> None:
    control = {seed: _run(0.0, 0.0) for seed in (41, 42, 43)}
    kyc = {seed: _run(1.0, 1.0) for seed in (41, 42, 43)}
    for run in control.values():
        for values in run.values():
            values["canonical"] = 0.0

    report = build_report(control, kyc)

    assert report["gates"]["BASELINE_VALID"] is False
    assert report["decision"] == "BASELINE_INVALID_OR_DATA_INSUFFICIENT"

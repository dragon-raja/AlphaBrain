from __future__ import annotations

from finalize_pi05_libero_plus_kyc_study import build_final_report


def _condition(mean: float) -> dict:
    return {"mean": mean, "ci95": [mean - 0.01, mean + 0.01]}


def _kyc_result(decision: str, boundary: str, *, control_joint: float = 0.8) -> dict:
    conditions = {
        "canonical": _condition(0.9),
        "camera_only": _condition(0.85),
        "background_only": _condition(0.85),
        "camera_background": _condition(control_joint),
    }
    return {
        "decision": decision,
        "gates": {"BASELINE_VALID": True},
        "interpretation_boundary": {"classification": boundary},
        "cross_seed": {
            "control": {
                "conditions": conditions,
                "effects": {
                    "combined_gap": _condition(0.9 - control_joint),
                },
            },
            "kyc": {"conditions": {**conditions, "camera_background": _condition(0.9)}},
            "kyc_minus_control": {"camera_background_success": _condition(0.1)},
        },
    }


def _inputs() -> dict:
    return {
        "official_act": {
            "status": "complete",
            "pose_sets": {
                "test_cameras": {
                    "equal_seed_mean": {"image_success": 0.25, "kyc_success": 0.63},
                    "hierarchical_paired_bootstrap": {
                        "delta": 0.38,
                        "ci95_low": 0.28,
                        "ci95_high": 0.48,
                        "bootstrap_resamples": 10_000,
                    },
                }
            },
        },
        "joint_ood": {
            "decision": "RESIDUAL_JOINT_CAMERA_BACKGROUND_OOD_GAP_CONFIRMED",
            "candidate_name": "multiview",
            "runs": {
                "multiview": {
                    "conditions": {key: _condition(value) for key, value in {
                        "canonical": 0.97,
                        "camera_only": 0.94,
                        "background_only": 0.94,
                        "camera_background": 0.83,
                    }.items()},
                    "effects": {
                        "camera_gap_canonical_background": {"mean": 0.03, "ci95": [0.0, 0.05]},
                        "combined_gap": {"mean": 0.14, "ci95": [0.04, 0.25]},
                    },
                }
            },
        },
        "matched": _kyc_result(
            "KYC_NO_MEANINGFUL_INCREMENTAL_VALUE",
            "paired_joint_ood_stress_test",
        ),
        "factor_separated": _kyc_result(
            "KYC_INCREMENTAL_VALUE_CONFIRMED",
            "factor_separated_category_composition",
            control_joint=0.84,
        ),
        "factorial": {
            "status": "complete",
            "cells": {
                key: {
                    "equal_seed_mean": {
                        "poseaug_control": {"success": 0.3},
                        "kyc": {"success": 0.3},
                    }
                }
                for key in ("fx_on", "cue_on")
            },
        },
        "ray_alignment": {
            "gate": {"passed": True},
            "pixel_error": {"median": 2.2, "p90": 8.6, "maximum": 11.0},
        },
    }


def test_factor_separated_confirmation_controls_final_decision() -> None:
    report = build_final_report(**_inputs())

    assert report["questions"]["multiview_camera_sufficient_within_tested_single_factor"] is True
    assert report["questions"]["kyc_final_decision"] == "KYC_TRANSFER_CONFIRMED_UNDER_FACTOR_SEPARATION"
    assert report["claim_boundaries"]["exact_camera_texture_pair_composition_tested"] is False


def test_invalid_factor_baseline_blocks_final_kyc_decision() -> None:
    inputs = _inputs()
    inputs["factor_separated"]["decision"] = "BASELINE_INVALID_OR_DATA_INSUFFICIENT"
    inputs["factor_separated"]["gates"]["BASELINE_VALID"] = False

    report = build_final_report(**inputs)

    assert report["questions"]["kyc_final_decision"] == "PI05_KYC_DECISION_INCOMPLETE_BASELINE_INVALID"

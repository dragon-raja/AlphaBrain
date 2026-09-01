from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

import scripts.dsol_paper1.build_view_value_expectation_final_report as report


def _method(success: float, gain: float) -> dict:
    return {
        "success_rate": success,
        "success_gain_pp": gain,
        "success_gain_task_stratified_bootstrap_95_pp": [gain - 2, gain + 2],
        "harm_probability": 0.01,
        "rescue_probability": 0.1,
        "paired_episode_counts": {
            "rescue": 5,
            "harm": 1,
            "both_success": 10,
            "both_failure": 16,
        },
    }


def test_report_pages_render_with_complete_schema(tmp_path: Path) -> None:
    calibration = {
        "status": "STABLE_VIEW_HEADROOM_CONFIRMED",
        "strong_state_count": 8,
        "state_count": 16,
        "source_equal_success_gain_pp": 21.0,
        "states": [
            {
                "task_id": f"task-{index % 8}",
                "canonical_success": 0.5,
                "candidate_success": 0.8,
                "success_gain": 0.3,
                "strong_state": index < 8,
            }
            for index in range(16)
        ],
    }
    seed_methods = {
        "canonical": _method(0.7, 0.0),
        "visibility_increment_selector": _method(0.8, 10.0),
    }
    heldout = {
        "noise_repeats_per_condition": 32,
        "checkpoint_seeds": {seed: seed_methods for seed in ("41", "42", "43")},
        "noise_convergence": {
            seed: {str(repeats): {"visibility_increment_selector": 0.8} for repeats in (4, 8, 16, 32)}
            for seed in ("41", "42", "43")
        },
        "selector_population_gate": {
            "status": "SELECTOR_GAIN_CONFIRMED",
            "best_rule_frozen_on_calibration": "visibility_increment_selector",
            "cross_checkpoint_mean_gain_pp": 10.0,
            "cross_checkpoint_mean_harm_probability": 0.01,
            "direction_consistent_positive": True,
            "each_checkpoint_ci_excludes_zero": True,
            "final_precision_halfwidth_at_most_5pp": True,
            "seed_results": [
                {
                    "checkpoint_seed": int(seed),
                    "success_rate": 0.8,
                    "success_gain_pp": 10.0,
                    "success_gain_ci_95_pp": [8.0, 12.0],
                    "harm_probability": 0.01,
                }
                for seed in ("41", "42", "43")
            ],
        },
    }
    population = {
        "population": {
            "calibration": {"states": [{"task_id": f"task-{index % 8}"} for index in range(16)]},
            "heldout_test": {"states": [{"task_id": f"task-{index % 8}"} for index in range(48)]},
        }
    }
    decision = report.final_decision(calibration, heldout)
    accel = [
        {
            "selected_candidate_id": "canonical" if index % 3 == 0 else "broad_train_01",
            "top2_margin": 0.01,
        }
        for index in range(64)
    ]
    report.base.configure_font()
    pdf_path = tmp_path / "report.pdf"
    preview = tmp_path / "preview"
    with PdfPages(pdf_path) as pdf:
        report.page_overview(pdf, preview, population, calibration, heldout, decision)
        report.page_design(pdf, preview, tmp_path, population)
        report.page_calibration(pdf, preview, calibration)
        report.page_heldout(pdf, preview, heldout)
        report.page_cross_seed(pdf, preview, heldout)
        report.page_accel_decision(pdf, preview, accel, decision, calibration, heldout)

    assert pdf_path.stat().st_size > 10_000
    assert len(list(preview.glob("page-*.png"))) == 6

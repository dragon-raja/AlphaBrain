from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from finalize_libero_decision import METHODS, audit_behavior_diagnostics, decide, render_markdown


METRICS = (
    "overall_task_success",
    "attached_task_success",
    "slip_recovery_success",
    "isolated_recovery_success",
    "failure_continuation_rate",
    "premature_commitment_rate",
    "event_trigger_rate",
    "final_progress",
)


def payload() -> dict:
    result = {
        "seeds": [41, 42, 43],
        "execution_horizons": [1, 2, 3],
        "methods": list(METHODS),
        "statistical_unit": "snapshot group",
        "aggregate": {},
        "paired_comparisons": {},
    }
    for k in (1, 2, 3):
        result["aggregate"][str(k)] = {}
        result["paired_comparisons"][str(k)] = {}
        for method in METHODS:
            result["aggregate"][str(k)][method] = {
                metric: {"mean": 0.50, "seed_values": [0.50] * 3, "sample_std": 0.0}
                for metric in METRICS
            }
        for baseline in ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h"):
            result["paired_comparisons"][str(k)][f"oracle_vs_{baseline}"] = {
                metric: {
                    "candidate_minus_baseline": {
                        "mean": 0.0,
                        "bootstrap_95_low": -0.05,
                        "bootstrap_95_high": 0.05,
                    }
                }
                for metric in METRICS
            }
    return result


def set_delta(data: dict, k: int, baseline: str, metric: str, mean: float, low: float, high: float) -> None:
    data["paired_comparisons"][str(k)][f"oracle_vs_{baseline}"][metric]["candidate_minus_baseline"] = {
        "mean": mean,
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


class FinalDecisionTest(unittest.TestCase):
    def test_continue_requires_behavioral_controls(self) -> None:
        data = payload()
        for k, effect in ((2, 0.05), (3, 0.15)):
            set_delta(data, k, "full_h", "slip_recovery_success", effect, 0.02, effect + 0.05)
        for baseline in ("random_soft010", "shuffled_oracle_soft010", "short_h"):
            set_delta(data, 3, baseline, "slip_recovery_success", 0.08, 0.01, 0.15)
        set_delta(data, 3, "gripper_soft010", "slip_recovery_success", 0.04, -0.03, 0.11)
        set_delta(data, 3, "full_h", "attached_task_success", -0.02, -0.09, 0.04)
        set_delta(data, 3, "full_h", "failure_continuation_rate", -0.10, -0.20, -0.01)
        set_delta(data, 3, "full_h", "premature_commitment_rate", -0.08, -0.15, -0.01)
        result = decide(data)
        self.assertEqual(result["decision"], "CONTINUE_FRESH")
        self.assertTrue(render_markdown(result).rstrip().endswith("CONTINUE_FRESH"))

    def test_low_success_is_baseline_invalid(self) -> None:
        data = payload()
        for method in METHODS:
            for metric in ("overall_task_success", "attached_task_success"):
                data["aggregate"]["3"][method][metric]["mean"] = 0.10
        self.assertEqual(decide(data)["decision"], "BASELINE_INVALID_OR_DATA_INSUFFICIENT")

    def test_soft_methods_without_oracle_specificity_pivot(self) -> None:
        data = payload()
        data["aggregate"]["3"]["full_h"]["slip_recovery_success"]["mean"] = 0.30
        for method in ("random_soft010", "shuffled_oracle_soft010", "oracle_soft010"):
            data["aggregate"]["3"][method]["slip_recovery_success"]["mean"] = 0.40
        result = decide(data)
        self.assertEqual(result["decision"], "PIVOT_TO_PREDICTABILITY_WEIGHTING")

    def test_offline_only_improvement_stops_weighting_route(self) -> None:
        data = copy.deepcopy(payload())
        set_delta(data, 3, "full_h", "slip_recovery_success", 0.02, -0.05, 0.09)
        self.assertEqual(decide(data)["decision"], "STOP_TRAINING_WEIGHTING_ROUTE")

    def test_missing_event_conditionals_do_not_crash(self) -> None:
        data = payload()
        for method in METHODS:
            for metric in ("failure_continuation_rate", "premature_commitment_rate"):
                data["aggregate"]["3"][method][metric]["mean"] = None
        for metric in ("failure_continuation_rate", "premature_commitment_rate"):
            comparison = data["paired_comparisons"]["3"]["oracle_vs_full_h"][metric]
            comparison["candidate_minus_baseline"] = {
                "mean": None,
                "bootstrap_95_low": None,
                "bootstrap_95_high": None,
            }
        result = decide(data)
        self.assertEqual(result["decision"], "STOP_TRAINING_WEIGHTING_ROUTE")
        self.assertIn("n/a", render_markdown(result))

    def test_behavior_audit_does_not_double_count_identical_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method in METHODS:
                for seed in (41, 42, 43):
                    run = root / f"fresh_closed_loop_{method}_seed{seed}"
                    run.mkdir()
                    (run / "closed_loop_end_to_end.json").write_text(json.dumps({
                        "status": "complete",
                        "completed_rows": 78,
                        "rows": [
                            {"failure_continuation": True, "premature_commitment": True},
                            {"failure_continuation": False, "premature_commitment": False},
                            {"failure_continuation": None, "premature_commitment": None},
                        ],
                    }))
            audit = audit_behavior_diagnostics(root)
        self.assertEqual(audit["complete_end_to_end_files"], 18)
        self.assertEqual(audit["eligible_triggered_slip_rows"], 36)
        self.assertEqual(audit["rows_where_metrics_differ"], 0)
        self.assertFalse(audit["metrics_provide_independent_evidence"])


if __name__ == "__main__":
    unittest.main()

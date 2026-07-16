from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from select_libero_baseline_repair_checkpoint import (
    load_validation_result,
    select_checkpoint,
)


def evaluation_row(pair_id: str, branch: str, horizon: int, success: bool) -> dict:
    return {
        "pair_id": pair_id,
        "branch_outcome": branch,
        "execution_horizon": horizon,
        "success": success,
        "event_time": 10 if branch == "slipped" else None,
        "final_progress": float(success),
        "grasp_subgoal": success,
        "lift_subgoal": success,
        "transport_subgoal": success,
        "place_subgoal": success,
    }


def write_result(path: Path, *, attached: int, slipped: int, groups: int = 13) -> None:
    rows = []
    for horizon in (1, 2, 3):
        for index in range(groups):
            pair_id = f"pair-{index:03d}"
            rows.append(evaluation_row(pair_id, "attached", horizon, index < attached))
            rows.append(evaluation_row(pair_id, "slipped", horizon, index < slipped))
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "split": "val",
                "evaluation": "end_to_end",
                "completed_rows": len(rows),
                "expected_rows": len(rows),
                "seed": 314200,
                "checkpoint": "/checkpoint",
                "rows": rows,
            }
        )
    )


class BaselineRepairCheckpointSelectionTest(unittest.TestCase):
    def test_selects_earliest_checkpoint_that_passes_all_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specifications = {
                3451: (3, 2),
                6902: (5, 4),
                10353: (8, 6),
                13804: (9, 7),
            }
            results = {}
            for step, (attached, slipped) in specifications.items():
                path = root / f"{step}.json"
                write_result(path, attached=attached, slipped=slipped)
                by_horizon, _ = load_validation_result(path, expected_groups=13)
                results[step] = by_horizon[3]

            selected = select_checkpoint(
                results,
                minimum_attached_success=0.30,
                minimum_overall_success=0.25,
                maximum_attached_regression=0.10,
                fallback_steps=13804,
            )

        self.assertTrue(selected["selected_by_gate"])
        self.assertEqual(selected["uniform_training_budget_steps"], 6902)

    def test_rejects_large_attached_regression_and_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specifications = {
                3451: (6, 0),
                6902: (4, 3),
                10353: (3, 4),
                13804: (3, 4),
            }
            results = {}
            for step, (attached, slipped) in specifications.items():
                path = root / f"{step}.json"
                write_result(path, attached=attached, slipped=slipped)
                by_horizon, _ = load_validation_result(path, expected_groups=13)
                results[step] = by_horizon[3]

            selected = select_checkpoint(
                results,
                minimum_attached_success=0.30,
                minimum_overall_success=0.25,
                maximum_attached_regression=0.10,
                fallback_steps=13804,
            )

        self.assertFalse(selected["selected_by_gate"])
        self.assertEqual(selected["uniform_training_budget_steps"], 13804)
        step_6902 = selected["checkpoint_summaries"][1]
        self.assertFalse(step_6902["conditions"]["no_attached_regression"])

    def test_rejects_incomplete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.json"
            path.write_text(json.dumps({"status": "partial", "rows": []}))
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_validation_result(path, expected_groups=13)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from scripts.dsol_paper1.analyze_m_b_multiseed import summarize


def _rows(successes: list[int]) -> list[dict[str, object]]:
    rows = []
    for index, success in enumerate(successes):
        rows.append(
            {
                "suite": "libero_spatial",
                "base_task": f"task_{index // 2}",
                "official_task_id": index,
                "init_state_index": 0,
                "condition": "official_camera",
                "success": bool(success),
                "episode_id": f"episode_{index}",
            }
        )
    return rows


class MultiSeedSummaryTest(unittest.TestCase):
    def test_pairing_delta_is_grouped_across_seed(self) -> None:
        baseline = _rows([0, 0, 0, 0])
        practical = {41: _rows([1, 0, 1, 0]), 42: _rows([1, 0, 1, 0])}
        consistency = {41: _rows([1, 1, 1, 0]), 42: _rows([1, 1, 1, 0])}
        payload = summarize(
            benchmark="camera_full",
            baseline=baseline,
            runs={
                "broad64_practical": practical,
                "broad64_paired_consistency": consistency,
            },
            validate=False,
        )
        pairing = payload["paired_consistency_minus_practical"]
        self.assertAlmostEqual(pairing["cross_seed_delta_pp"], 25.0)
        self.assertTrue(pairing["all_seeds_same_direction"])

    def test_zero_deltas_are_not_directional(self) -> None:
        baseline = _rows([0, 0, 0, 0])
        identical = {41: _rows([1, 0, 1, 0]), 42: _rows([1, 0, 1, 0])}
        payload = summarize(
            benchmark="camera_full",
            baseline=baseline,
            runs={
                "broad64_practical": identical,
                "broad64_paired_consistency": identical,
            },
            validate=False,
        )
        self.assertFalse(
            payload["paired_consistency_minus_practical"]["all_seeds_same_direction"]
        )


if __name__ == "__main__":
    unittest.main()

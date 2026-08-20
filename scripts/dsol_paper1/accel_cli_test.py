from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_accel_relations import main as relation_main
from rank_accel_candidates import main as ranking_main


class AccelCliTest(unittest.TestCase):
    def test_writes_machine_readable_ranking_and_relations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.npz"
            ranking_path = root / "ranking.json"
            candidates_path = root / "candidates.json"
            references_path = root / "references.json"
            relations_path = root / "relations.json"
            velocity = np.ones((2, 10, 1, 1), dtype=np.float32)
            velocity[1, :, 0, 0] = np.arange(1, 11)
            noise = np.zeros((2, 1, 1), dtype=np.float32)
            np.savez_compressed(
                trace_path,
                candidate_ids=np.asarray(["canonical", "reveal"]),
                velocity_trace=velocity,
                initial_noise=noise,
                flow_times=np.linspace(1.0, 0.1, 10),
            )
            candidates_path.write_text(
                json.dumps(
                    [
                        {
                            "pose_id": "canonical",
                            "azimuth_deg": 0,
                            "elevation_deg": 0,
                            "radius_scale": 1,
                        },
                        {
                            "pose_id": "reveal",
                            "azimuth_deg": 30,
                            "elevation_deg": 10,
                            "radius_scale": 1,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            references_path.write_text(
                json.dumps(
                    {
                        "canonical": "canonical",
                        "train": ["canonical"],
                        "strong_info": "reveal",
                        "reveal": "reveal",
                        "oracle": "reveal",
                    }
                ),
                encoding="utf-8",
            )

            ranking_main(["--trace-npz", str(trace_path), "--output", str(ranking_path)])
            relation_main(
                [
                    "--ranking",
                    str(ranking_path),
                    "--candidate-metadata",
                    str(candidates_path),
                    "--references",
                    str(references_path),
                    "--output",
                    str(relations_path),
                ]
            )

            ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
            relations = json.loads(relations_path.read_text(encoding="utf-8"))
            self.assertEqual(ranking["selected_candidate_id"], "canonical")
            self.assertEqual(relations["selected_candidate_id"], "canonical")
            self.assertIn("oracle", relations["relations"])


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summarize_recovery_segment_oracle import (
    BINARY_METRICS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PREREGISTRATION_SHA256,
    METRIC_DIRECTIONS,
    METHODS,
    METRICS,
    build_summary,
    load_and_validate,
    paired_method_summary,
)


SOURCE_BY_PAIR = {
    **{f"g{index:02d}": index for index in range(9)},
    "g09": 0,
    "g10": 1,
    "g11": 2,
    "g12": 3,
}


def metric_summary(value):
    return {
        (f"{metric}_rate" if metric in BINARY_METRICS else metric): float(value)
        for metric in METRICS
    }


def natural_outcome(value):
    return {metric: float(value) for metric in METRICS}


def method_result(full_value, natural_value):
    return {
        "full_heldout_summary": metric_summary(full_value),
        "full_heldout_outcomes": [
            {"repeat": repeat, "outcome": natural_outcome(full_value)}
            for repeat in range(5)
        ],
        "natural_outcome": natural_outcome(natural_value),
    }


def random_result(schedule_values, natural_value):
    schedule_results = []
    flattened_outcomes = []
    for schedule_index, value in enumerate(schedule_values):
        schedule = method_result(value, natural_value)
        schedule["random_schedule_index"] = schedule_index
        schedule_results.append(schedule)
        flattened_outcomes.extend(
            {**outcome, "random_schedule_index": schedule_index}
            for outcome in schedule["full_heldout_outcomes"]
        )
    aggregate_value = sum(schedule_values) / len(schedule_values)
    return {
        "full_heldout_summary": metric_summary(aggregate_value),
        "full_heldout_outcomes": flattened_outcomes,
        "natural_outcome_summary": natural_outcome(natural_value),
        "random_schedule_count": len(schedule_values),
        "schedule_results": schedule_results,
    }


def write_manifest(root, source_by_pair=SOURCE_BY_PAIR):
    manifest = {
        "groups": [
            {
                "pair_id": pair_id,
                "split": "val",
                "source_initial_state_index": source,
            }
            for pair_id, source in source_by_pair.items()
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def decision_payloads(
    root,
    *,
    sample0_value=0.0,
    random_schedule_values=(0.0, 0.0, 0.0),
    myopic_value=0.0,
    oracle_value=1.0,
    natural_sample0=1.0,
    natural_oracle=0.0,
):
    payloads = []
    for model_seed in (41, 42, 43):
        rows = []
        for pair_id, source in SOURCE_BY_PAIR.items():
            rows.append(
                {
                    "pair_id": pair_id,
                    "split": "val",
                    "source_initial_state_index": source,
                    "seed": model_seed,
                    "methods": {
                        "sample0": method_result(sample0_value, natural_sample0),
                        "random4": random_result(random_schedule_values, 0.25),
                        "myopic_stage": method_result(myopic_value, 0.25),
                        "receding_oracle": method_result(oracle_value, natural_oracle),
                    },
                }
            )
        payloads.append(
            {
                "schema_version": 2,
                "status": "complete",
                "run_kind": "decision",
                "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
                "expected_preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
                "episode_root": str(root),
                "split": "val",
                "seed": model_seed,
                "sample_count": 4,
                "segment_replans": 4,
                "execution_horizon": 3,
                "segment_action_budget": 12,
                "total_action_budget": 120,
                "lookahead_steps": 30,
                "selection_continuations": 3,
                "decision_heldout_continuations": 5,
                "full_heldout_continuations": 5,
                "random_schedules": 3,
                "stage_dwell_steps": 2,
                "replay_sim_tolerance": 1e-8,
                "candidate_pool_tolerance": 1e-6,
                "methods": list(METHODS),
                "outcome_metric_order": list(METRICS),
                "outcome_metric_directions": list(METRIC_DIRECTIONS),
                "git_sha": "0123456789abcdef",
                "git_dirty_at_launch": False,
                "policy_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[model_seed],
                "expected_rows": len(rows),
                "completed_rows": len(rows),
                "expected_global_rows": len(SOURCE_BY_PAIR),
                "expected_global_pair_source_map": SOURCE_BY_PAIR,
                "rows": rows,
            }
        )
    return payloads


def write_payloads(root, payloads):
    paths = []
    for payload in payloads:
        path = root / f"seed{payload['seed']}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    return paths


class RecoverySegmentSummaryTest(unittest.TestCase):
    def valid_decision(self, root, **kwargs):
        write_manifest(root)
        paths = write_payloads(root, decision_payloads(root, **kwargs))
        return load_and_validate(paths)

    def assert_decision_rejected(self, mutation, pattern):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_manifest(root)
            payloads = decision_payloads(root)
            mutation(payloads)
            paths = write_payloads(root, payloads)
            with self.assertRaisesRegex(ValueError, pattern):
                load_and_validate(paths)

    def test_gate_uses_full_heldout_summary_not_natural_outcome(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows, validation = self.valid_decision(Path(temporary_directory))
            summary = build_summary(
                rows,
                bootstrap_samples=100,
                seed=7,
                decision_run=True,
            )

        self.assertTrue(validation["decision_eligible"])
        comparison = summary["gate_pairwise"]["receding_oracle_minus_random4"]["success"]
        self.assertEqual(comparison["metric_source"], "full_heldout_summary")
        self.assertEqual(comparison["source_cluster_level"]["mean"], 1.0)
        self.assertEqual(
            comparison["absolute_percentage_points"]["source_cluster_level"]["mean"],
            100.0,
        )
        self.assertEqual(
            comparison["absolute_percentage_points"]["per_seed_mean"],
            {"41": 100.0, "42": 100.0, "43": 100.0},
        )

    def test_random4_uses_validated_three_schedule_aggregate_before_pairing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows, _ = self.valid_decision(
                Path(temporary_directory),
                random_schedule_values=(0.0, 1.0, 1.0),
                oracle_value=0.0,
            )
            comparison = paired_method_summary(
                rows,
                method="random4",
                metric="success",
                bootstrap_samples=100,
                seed=11,
            )

        self.assertAlmostEqual(comparison["source_cluster_level"]["mean"], 2.0 / 3.0)
        self.assertAlmostEqual(
            comparison["absolute_percentage_points"]["source_cluster_level"]["mean"],
            200.0 / 3.0,
        )

    def test_pairing_uses_source_cluster_after_seed_aggregation(self):
        rows = [
            {
                "pair_id": "g0",
                "seed": 41,
                "source_initial_state_index": 7,
                "methods": {
                    "sample0": method_result(0.0, 1.0),
                    "receding_oracle": method_result(1.0, 0.0),
                },
            },
            {
                "pair_id": "g1",
                "seed": 41,
                "source_initial_state_index": 7,
                "methods": {
                    "sample0": method_result(1.0, 0.0),
                    "receding_oracle": method_result(1.0, 0.0),
                },
            },
        ]
        summary = paired_method_summary(
            rows,
            method="receding_oracle",
            metric="success",
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(summary["source_cluster_level"]["count"], 1)
        self.assertEqual(summary["source_cluster_level"]["mean"], 0.5)
        self.assertEqual(summary["per_seed_source_cluster_level"]["41"]["mean"], 0.5)

    def test_decision_rejects_invalid_provenance_and_completion(self):
        cases = (
            (lambda payloads: payloads[0].update(seed=44), "unsupported model seed"),
            (
                lambda payloads: payloads[0].update(policy_checkpoint_sha256="bad"),
                "checkpoint SHA256 mismatch",
            ),
            (
                lambda payloads: payloads[0].update(preregistration_sha256="bad"),
                "preregistration SHA256 mismatch",
            ),
            (lambda payloads: payloads[0].update(git_dirty_at_launch=True), "dirty at launch"),
            (
                lambda payloads: payloads[0].pop("git_dirty_at_launch"),
                "clean status is missing",
            ),
            (lambda payloads: payloads[0].update(status="partial"), "status is not complete"),
            (lambda payloads: payloads[0].update(run_kind="pilot"), "unsupported.*run_kind"),
            (
                lambda payloads: payloads[0].pop("run_kind"),
                "schema version 2.*requires run_kind",
            ),
            (lambda payloads: payloads[0].update(schema_version=1), "schema_version must be 2"),
            (
                lambda payloads: payloads[0].pop("full_heldout_continuations"),
                "schema version 2 fields are missing",
            ),
        )
        for mutation, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assert_decision_rejected(mutation, pattern)

    def test_decision_rejects_missing_seed_or_manifest_row(self):
        def remove_seed(payloads):
            payloads.pop()

        self.assert_decision_rejected(remove_seed, "must contain seeds")

        def remove_row(payloads):
            payloads[0]["rows"].pop()
            payloads[0]["expected_rows"] = 12
            payloads[0]["completed_rows"] = 12

        self.assert_decision_rejected(remove_row, "does not match the complete val manifest grid")

        def change_source(payloads):
            payloads[0]["rows"][0]["source_initial_state_index"] = 99

        self.assert_decision_rejected(change_source, "source_mismatch")

    def test_decision_rejects_wrong_manifest_cardinality(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_by_pair = dict(list(SOURCE_BY_PAIR.items())[:-1])
            write_manifest(root, source_by_pair)
            with self.assertRaisesRegex(ValueError, "must contain 13 groups"):
                load_and_validate(write_payloads(root, decision_payloads(root)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_by_pair = dict(SOURCE_BY_PAIR)
            source_by_pair["g08"] = 7
            write_manifest(root, source_by_pair)
            payloads = decision_payloads(root)
            for payload in payloads:
                payload["rows"][8]["source_initial_state_index"] = 7
            with self.assertRaisesRegex(ValueError, "must contain 9 source clusters"):
                load_and_validate(write_payloads(root, payloads))

    def test_decision_rejects_declared_global_map_that_disagrees_with_manifest(self):
        def swap_declared_sources(payloads):
            for payload in payloads:
                expected_map = payload["expected_global_pair_source_map"]
                expected_map["g00"], expected_map["g01"] = (
                    expected_map["g01"],
                    expected_map["g00"],
                )

        self.assert_decision_rejected(
            swap_declared_sources,
            "expected_global_pair_source_map does not match",
        )

    def test_decision_rejects_config_and_random_schedule_mismatch(self):
        self.assert_decision_rejected(
            lambda payloads: payloads[0].update(total_action_budget=119),
            "pre-registered config mismatch",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            rows, _ = self.valid_decision(Path(temporary_directory))
            removed_schedule = rows[0]["methods"]["random4"]["schedule_results"].pop()
            with self.assertRaisesRegex(ValueError, "requires 3 schedules"):
                paired_method_summary(
                    rows,
                    method="random4",
                    metric="success",
                    bootstrap_samples=10,
                    seed=1,
                )

            rows[0]["methods"]["random4"]["schedule_results"].append(removed_schedule)
            rows[0]["methods"]["random4"]["random_schedule_count"] = 1
            with self.assertRaisesRegex(ValueError, "schedule count disagrees"):
                paired_method_summary(
                    rows,
                    method="random4",
                    metric="success",
                    bootstrap_samples=10,
                    seed=1,
                )

    def test_decision_rejects_natural_outcome_as_gate_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows, _ = self.valid_decision(Path(temporary_directory))
            del rows[0]["methods"]["sample0"]["full_heldout_summary"]
            with self.assertRaisesRegex(ValueError, "missing full_heldout_summary"):
                paired_method_summary(
                    rows,
                    method="receding_oracle",
                    metric="success",
                    bootstrap_samples=10,
                    seed=1,
                )

    def test_smoke_is_summarized_but_explicitly_non_decision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = copy.deepcopy(decision_payloads(root)[0])
            payload["run_kind"] = "smoke"
            payload.pop("preregistration_sha256")
            payload["status"] = "partial"
            payload["git_dirty_at_launch"] = True
            payload["rows"] = payload["rows"][:1]
            payload["expected_rows"] = 13
            payload["completed_rows"] = 1
            for method, result in payload["rows"][0]["methods"].items():
                value = 1.0 if method == "receding_oracle" else 0.0
                result.clear()
                result["outcome"] = natural_outcome(value)
            rows, validation = load_and_validate(write_payloads(root, [payload]))
            summary = paired_method_summary(
                rows,
                method="receding_oracle",
                metric="success",
                bootstrap_samples=10,
                seed=1,
                decision_run=False,
            )

        self.assertEqual(validation["run_kind"], "smoke")
        self.assertFalse(validation["decision_eligible"])
        self.assertTrue(validation["non_decision_reasons"])
        self.assertEqual(summary["source_cluster_level"]["mean"], 1.0)
        self.assertIn("natural_outcome", summary["metric_source"])


if __name__ == "__main__":
    unittest.main()

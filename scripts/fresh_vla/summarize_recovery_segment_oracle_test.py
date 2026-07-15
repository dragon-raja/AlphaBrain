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
    absolute_method_summary,
    build_summary,
    load_and_validate,
    paired_method_summary,
    selection_stability_summary,
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


def restore_audit_decision(
    selection_top=(0, 1, 2, 3),
    heldout_top=(0, 1, 2, 3),
):
    selection_top = set(selection_top)
    heldout_top = set(heldout_top)
    selection_index = min(selection_top)
    heldout_index = min(heldout_top)
    return {
        "candidate0_branch_replay_semantic_match": True,
        "candidate0_branch_replay_image_max_abs_delta": 0,
        "candidate0_branch_replay_sim_max_abs_delta": 0.0,
        "candidate0_branch_replay_robot_state_max_abs_delta": 0.0,
        "selected_live_branch_semantic_match": True,
        "selected_live_branch_image_max_abs_delta": 0,
        "selected_live_branch_sim_max_abs_delta": 0.0,
        "selected_live_branch_robot_state_max_abs_delta": 0.0,
        "selected_direct_replay_match": True,
        "selected_endpoint_sim_max_abs_delta": 0.0,
        "oracle_index": selection_index,
        "decision_heldout_oracle_index": heldout_index,
        "selection_matches_decision_heldout": selection_index == heldout_index,
        "candidates": [
            {
                "candidate_index": candidate_index,
                "selection_summary": metric_summary(
                    1.0 if candidate_index in selection_top else 0.0
                ),
                "decision_heldout_summary": metric_summary(
                    1.0 if candidate_index in heldout_top else 0.0
                ),
            }
            for candidate_index in range(4)
        ],
    }


def stability_row(pair_id, seed, source, decisions):
    return {
        "pair_id": pair_id,
        "seed": seed,
        "source_initial_state_index": source,
        "methods": {"receding_oracle": {"decisions": decisions}},
    }


def stability_rows(decision):
    return [
        stability_row(
            "g00",
            41,
            0,
            [copy.deepcopy(decision) for _ in range(4)],
        )
    ]


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
            oracle_result = method_result(oracle_value, natural_oracle)
            oracle_result["decisions"] = [
                restore_audit_decision() for _ in range(4)
            ]
            rows.append(
                {
                    "pair_id": pair_id,
                    "split": "val",
                    "source_initial_state_index": source,
                    "seed": model_seed,
                    "sample0_restore_parity": {
                        "passed": True,
                        "scope": "fixed_intervention_segment",
                        "action_budget": 12,
                        "forced_restore_replans": 4,
                        "image_max_abs_delta": 0,
                        "decision_image_max_abs_delta": 0,
                        "numeric_max_abs_delta": {
                            "action": 0.0,
                            "sim_state": 0.0,
                            "robot_state": 0.0,
                        },
                    },
                    "methods": {
                        "sample0": method_result(sample0_value, natural_sample0),
                        "random4": random_result(random_schedule_values, 0.25),
                        "myopic_stage": method_result(myopic_value, 0.25),
                        "receding_oracle": oracle_result,
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
                "branch_rollout_uses_separate_env": True,
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
        absolute = summary["absolute"]["receding_oracle"]["success"]
        self.assertEqual(absolute["source_cluster_level"]["mean"], 1.0)
        self.assertEqual(
            absolute["per_seed_mean"],
            {"41": 1.0, "42": 1.0, "43": 1.0},
        )
        self.assertEqual(
            absolute["percentage"]["source_cluster_level"]["mean"],
            100.0,
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

    def test_absolute_summary_uses_seed_source_then_source_clusters(self):
        rows = [
            {
                "pair_id": pair_id,
                "seed": run_seed,
                "source_initial_state_index": source,
                "methods": {
                    "receding_oracle": method_result(value, value),
                },
            }
            for pair_id, run_seed, source, value in (
                ("a0", 41, 0, 0.0),
                ("a1", 41, 0, 0.0),
                ("b0", 41, 1, 1.0),
                ("a0", 42, 0, 1.0),
                ("b0", 42, 1, 1.0),
            )
        ]
        summary = absolute_method_summary(
            rows,
            method="receding_oracle",
            metric="success",
            bootstrap_samples=100,
            seed=13,
        )

        self.assertEqual(summary["source_cluster_level"]["count"], 2)
        self.assertEqual(summary["source_cluster_level"]["mean"], 0.75)
        self.assertNotEqual(summary["source_cluster_level"]["mean"], 0.6)
        self.assertEqual(summary["per_source_cluster"], {"0": 0.5, "1": 1.0})
        self.assertEqual(summary["per_seed_mean"], {"41": 0.5, "42": 1.0})
        self.assertEqual(
            summary["per_seed_source_cluster_level"]["41"]["count"],
            2,
        )
        self.assertIn(
            "bootstrap_95_low",
            summary["per_seed_source_cluster_level"]["41"],
        )
        self.assertIn(
            "bootstrap_95_high",
            summary["per_seed_source_cluster_level"]["42"],
        )

    def test_selection_stability_all_tied_marks_first_index_match_as_inflated(self):
        summary = selection_stability_summary(
            stability_rows(restore_audit_decision()),
            bootstrap_samples=100,
            seed=3,
        )

        self.assertEqual(summary["selection_unique_best_rate"]["raw_rate"], 0.0)
        self.assertEqual(summary["heldout_unique_best_rate"]["raw_rate"], 0.0)
        self.assertEqual(summary["all_four_tied_selection_rate"]["raw_rate"], 1.0)
        self.assertEqual(summary["selected_nonzero_rate"]["raw_rate"], 0.0)
        self.assertEqual(summary["mean_selection_top_set_size"]["raw_mean"], 4.0)
        self.assertEqual(summary["mean_heldout_top_set_size"]["raw_mean"], 4.0)
        diagnostic = summary["first_index_match_rate_tie_inflated_diagnostic"]
        self.assertEqual(diagnostic["raw_rate"], 1.0)
        self.assertIn("tie-inflated", diagnostic["diagnostic_note"])
        conditional = summary["exact_agreement_among_both_unique"]
        self.assertEqual(conditional["raw_both_unique_count"], 0)
        self.assertIsNone(conditional["raw_conditional_rate"])
        self.assertIsNone(conditional["source_cluster_level"])

    def test_selection_stability_unique_best_agreement_and_disagreement(self):
        cases = (
            ((2,), (2,), 1.0, 1.0),
            ((1,), (2,), 0.0, 0.0),
        )
        for selection_top, heldout_top, expected_exact, expected_overlap in cases:
            with self.subTest(
                selection_top=selection_top,
                heldout_top=heldout_top,
            ):
                summary = selection_stability_summary(
                    stability_rows(
                        restore_audit_decision(selection_top, heldout_top)
                    ),
                    bootstrap_samples=100,
                    seed=5,
                )

                self.assertEqual(
                    summary["selection_unique_best_rate"]["raw_rate"], 1.0
                )
                self.assertEqual(
                    summary["heldout_unique_best_rate"]["raw_rate"], 1.0
                )
                self.assertEqual(summary["both_unique_count"], 4)
                self.assertEqual(summary["selected_nonzero_rate"]["raw_rate"], 1.0)
                self.assertEqual(
                    summary["top_set_overlap_rate"]["raw_rate"], expected_overlap
                )
                conditional = summary["exact_agreement_among_both_unique"]
                self.assertEqual(conditional["raw_both_unique_count"], 4)
                self.assertEqual(
                    conditional["raw_conditional_rate"], expected_exact
                )
                self.assertEqual(
                    conditional["source_cluster_level"]["mean"], expected_exact
                )

    def test_selection_stability_reports_partial_top_set_overlap(self):
        summary = selection_stability_summary(
            stability_rows(restore_audit_decision((0, 1), (1, 2))),
            bootstrap_samples=100,
            seed=7,
        )

        self.assertEqual(summary["top_set_overlap_rate"]["raw_rate"], 1.0)
        self.assertEqual(summary["selection_unique_best_rate"]["raw_rate"], 0.0)
        self.assertEqual(summary["heldout_unique_best_rate"]["raw_rate"], 0.0)
        self.assertEqual(summary["mean_selection_top_set_size"]["raw_mean"], 2.0)
        self.assertEqual(summary["mean_heldout_top_set_size"]["raw_mean"], 2.0)
        self.assertEqual(
            summary["first_index_match_rate_tie_inflated_diagnostic"]["raw_rate"],
            0.0,
        )

    def test_selection_stability_aggregates_rows_before_source_clusters(self):
        unique = restore_audit_decision((1,), (1,))
        tied = restore_audit_decision()
        rows = [
            stability_row(
                pair_id,
                run_seed,
                source,
                [copy.deepcopy(decision) for _ in range(4)],
            )
            for pair_id, run_seed, source, decision in (
                ("a0", 41, 0, unique),
                ("a1", 41, 0, tied),
                ("b0", 41, 1, unique),
                ("a0", 42, 0, tied),
                ("b0", 42, 1, unique),
            )
        ]
        summary = selection_stability_summary(
            rows,
            bootstrap_samples=100,
            seed=9,
        )["selection_unique_best_rate"]

        self.assertEqual(summary["raw_rate"], 0.6)
        self.assertEqual(summary["source_cluster_level"]["count"], 2)
        self.assertEqual(summary["source_cluster_level"]["mean"], 0.625)
        self.assertEqual(summary["per_source_cluster"], {"0": 0.25, "1": 1.0})
        self.assertEqual(summary["per_seed_mean"], {"41": 0.75, "42": 0.5})

    def test_both_unique_exact_cluster_rate_preserves_conditional_denominator(self):
        agreement = restore_audit_decision((1,), (1,))
        disagreement = restore_audit_decision((1,), (2,))
        tied = restore_audit_decision()
        rows = [
            stability_row(
                "a0",
                41,
                0,
                [agreement, tied, copy.deepcopy(tied), copy.deepcopy(tied)],
            ),
            stability_row(
                "a1",
                41,
                0,
                [copy.deepcopy(disagreement) for _ in range(4)],
            ),
        ]
        conditional = selection_stability_summary(
            rows,
            bootstrap_samples=100,
            seed=11,
        )["exact_agreement_among_both_unique"]

        self.assertEqual(conditional["raw_agreement_count"], 1)
        self.assertEqual(conditional["raw_both_unique_count"], 5)
        self.assertEqual(conditional["raw_conditional_rate"], 0.2)
        self.assertEqual(conditional["source_cluster_level"]["mean"], 0.2)

    def test_decision_rejects_invalid_full_heldout_repeats_and_summaries(self):
        def duplicate_repeat(payloads):
            outcomes = payloads[0]["rows"][0]["methods"]["sample0"][
                "full_heldout_outcomes"
            ]
            outcomes[4]["repeat"] = outcomes[3]["repeat"]

        self.assert_decision_rejected(duplicate_repeat, "repeats must be unique")

        def change_recomputed_metric(payloads):
            payloads[0]["rows"][0]["methods"]["sample0"][
                "full_heldout_summary"
            ]["success_rate"] = 1.0

        self.assert_decision_rejected(
            change_recomputed_metric,
            "full_heldout_summary does not match recomputed.*success",
        )

        def change_schedule_summary(payloads):
            payloads[0]["rows"][0]["methods"]["random4"]["schedule_results"][0][
                "full_heldout_summary"
            ]["success_rate"] = 1.0

        self.assert_decision_rejected(
            change_schedule_summary,
            "random4 schedule 0 full_heldout_summary does not match recomputed.*success",
        )

    def test_decision_rejects_random4_flattened_outcome_mismatch(self):
        def change_nested_outcome_only(payloads):
            schedule_outcome = payloads[0]["rows"][0]["methods"]["random4"][
                "schedule_results"
            ][0]["full_heldout_outcomes"][0]
            schedule_outcome["outcome"] = {
                **schedule_outcome["outcome"],
                "nested_only_marker": True,
            }

        self.assert_decision_rejected(
            change_nested_outcome_only,
            "flattened random4 outcomes do not match schedule rows",
        )

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
            (
                lambda payloads: payloads[0]["rows"][0][
                    "sample0_restore_parity"
                ].update(passed=False),
                "failed sample0 restore parity",
            ),
            (
                lambda payloads: payloads[0]["rows"][0]["methods"][
                    "receding_oracle"
                ]["decisions"][0].update(
                    candidate0_branch_replay_image_max_abs_delta=1
                ),
                "candidate-0 replay changed pixels",
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
            complete_summary = build_summary(
                rows,
                bootstrap_samples=10,
                seed=1,
                decision_run=False,
            )

        self.assertEqual(validation["run_kind"], "smoke")
        self.assertFalse(validation["decision_eligible"])
        self.assertTrue(validation["non_decision_reasons"])
        self.assertEqual(summary["source_cluster_level"]["mean"], 1.0)
        self.assertIn("natural_outcome", summary["metric_source"])
        self.assertFalse(complete_summary["selection_stability"]["available"])
        self.assertEqual(
            complete_summary["absolute"]["receding_oracle"]["success"][
                "source_cluster_level"
            ]["mean"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

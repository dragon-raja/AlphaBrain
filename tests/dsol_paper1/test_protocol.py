from __future__ import annotations

from dataclasses import replace
import unittest

from AlphaBrain.research.dsol import (
    HOLD,
    PASS,
    ObservationContract,
    Preregistration,
    ProtocolValidationError,
    RelationRecord,
    TaskTemplate,
    TemporalGate,
    evaluate_release,
)


ED_GATES = {
    "current_full_observation_ambiguous": True,
    "latent_decision_sets_incompatible": True,
    "no_shared_safe_recovery_action": True,
    "candidate_snapshot_disambiguates": True,
    "matched_uninformative_control_exists": True,
}


def make_ed_relation(**changes: object) -> RelationRecord:
    values: dict[str, object] = {
        "relation_id": "ed-1",
        "relation_type": "E_D",
        "source_task_id": "task-a",
        "target_task_id": "task-b",
        "evidence_gates": ED_GATES,
        "matched_control_id": "control-1",
        "expert_decision_sets": ({"move-left"}, {"move-right"}),
        "shared_safe_recovery_actions": (),
    }
    values.update(changes)
    return RelationRecord(**values)  # type: ignore[arg-type]


def make_preregistration() -> Preregistration:
    contract = ObservationContract(
        contract_id="rgb-proprio-v1",
        deployment_inputs={
            "rgb": {"dtype": "uint8"},
            "proprioception": {"dtype": "float32"},
        },
    )
    tasks = (
        TaskTemplate("task-a", "move to {side}", {"move-left", "move-right"}, contract.contract_id),
        TaskTemplate("task-b", "place at {side}", {"move-left", "move-right"}, contract.contract_id),
        TaskTemplate("task-c", "hold position", {"hold"}, contract.contract_id),
    )
    relations = (
        RelationRecord(
            relation_id="n-1",
            relation_type="N",
            source_task_id="task-a",
            target_task_id="task-b",
            action_set_equivalent=True,
            action_set_equivariant=False,
            r_hat=0.04,
            r_hat_threshold=0.05,
        ),
        RelationRecord(
            relation_id="control-1",
            relation_type="MATCHED_CONTROL",
            source_task_id="task-a",
            target_task_id="task-c",
        ),
        make_ed_relation(),
    )
    temporal = (
        TemporalGate("T1", "S0", ordered_clip_advantage=False, include_in_main_analysis=True),
        TemporalGate("T2", "S1", ordered_clip_advantage=True, include_in_main_analysis=False),
        TemporalGate("T3", "S0", ordered_clip_advantage=False, include_in_main_analysis=True),
    )
    return Preregistration(
        preregistration_id="paper1-v1",
        task_templates=tasks,
        observation_contracts=(contract,),
        relations=relations,
        temporal_gates=temporal,
        release_gates={
            "evidence_archive": ("manifest", "checksums"),
            "analysis_audit": ("analysis_report",),
        },
    )


class ProtocolTests(unittest.TestCase):
    def test_complete_protocol_passes(self) -> None:
        prereg = make_preregistration()

        self.assertIs(prereg.validate(), prereg)
        decisions = evaluate_release(
            prereg,
            {
                "manifest": True,
                "checksums": {"status": "PASS"},
                "analysis_report": {"valid": True},
            },
        )

        self.assertTrue(decisions)
        self.assertEqual(set(decisions.values()), {PASS})

    def test_each_ed_gate_fails_closed(self) -> None:
        for gate_name in ED_GATES:
            with self.subTest(gate=gate_name):
                gates = {**ED_GATES, gate_name: False}
                with self.assertRaisesRegex(ProtocolValidationError, gate_name):
                    make_ed_relation(evidence_gates=gates).validate()

        with self.assertRaisesRegex(ProtocolValidationError, "frozen five-gate"):
            make_ed_relation(
                evidence_gates={**ED_GATES, "invented_gate": True}
            ).validate()

    def test_ed_requires_control_incompatible_decisions_and_no_recovery(self) -> None:
        invalid = (
            make_ed_relation(matched_control_id=None),
            make_ed_relation(expert_decision_sets=({"move-left"},)),
            make_ed_relation(
                expert_decision_sets=({"move-left", "hold"}, {"move-right", "hold"})
            ),
            make_ed_relation(shared_safe_recovery_actions=None),
            make_ed_relation(shared_safe_recovery_actions={"stop"}),
        )
        for record in invalid:
            with self.subTest(record=record):
                with self.assertRaises(ProtocolValidationError):
                    record.validate()

        prereg = make_preregistration()
        bad_relations = tuple(
            replace(relation, matched_control_id="not-registered")
            if relation.relation_type == "E_D"
            else relation
            for relation in prereg.relations
        )
        with self.assertRaisesRegex(ProtocolValidationError, "registered MATCHED_CONTROL"):
            replace(prereg, relations=bad_relations).validate()

    def test_n_requires_action_set_relation_and_low_finite_r_hat(self) -> None:
        base = RelationRecord(
            "n-1",
            "N",
            "task-a",
            "task-b",
            action_set_equivalent=True,
            action_set_equivariant=False,
            r_hat=0.05,
            r_hat_threshold=0.05,
        )
        self.assertIs(base.validate(), base)
        for invalid in (
            replace(base, action_set_equivalent=False),
            replace(base, r_hat=0.051),
            replace(base, r_hat=float("nan")),
            replace(base, r_hat=None),
        ):
            with self.subTest(record=invalid):
                with self.assertRaises(ProtocolValidationError):
                    invalid.validate()

    def test_frozen_leakage_keys_are_rejected_recursively(self) -> None:
        leaked_inputs = (
            {"oracle_relation": "E_D"},
            {"features": {"branchOutcome": "success"}},
            {"features": [{"privileged-state": [1, 2]}]},
            {"object_id": 17},
            {"metadata": {"sourceFileName": "clip.mp4"}},
            {"capture_timestamp_ms": 123},
            {"cameraLabel": "wrist"},
            {"rHatTarget": 0.0},
            {"future_state": [0.0]},
            {"oracleRoute": "left"},
            {"scan-order": [1, 2]},
            {"candidate_rank": 1},
        )
        for deployment_inputs in leaked_inputs:
            with self.subTest(deployment_inputs=deployment_inputs):
                contract = ObservationContract("leaky", deployment_inputs)
                with self.assertRaises(ProtocolValidationError):
                    contract.validate()

    def test_required_task_and_observation_fields_fail_closed(self) -> None:
        invalid_records = (
            ObservationContract("contract", {}),
            TaskTemplate("", "move left", {"left"}, "contract"),
            TaskTemplate("task", "", {"left"}, "contract"),
            TaskTemplate("task", "move left", (), "contract"),
        )
        for record in invalid_records:
            with self.subTest(record=record):
                with self.assertRaises(ProtocolValidationError):
                    record.validate()

    def test_s1_is_excluded_from_main_analysis(self) -> None:
        prereg = make_preregistration()
        selected = prereg.main_analysis_temporal_gates()

        self.assertEqual(tuple(gate.test_id for gate in selected), ("T1", "T3"))
        with self.assertRaisesRegex(ProtocolValidationError, "only S0"):
            TemporalGate(
                "T2",
                "S1",
                ordered_clip_advantage=True,
                include_in_main_analysis=True,
            ).validate()

        s2 = TemporalGate(
            "T2",
            "S2",
            ordered_clip_advantage=False,
            include_in_main_analysis=False,
        )
        self.assertIs(s2.validate(), s2)
        with self.assertRaisesRegex(ProtocolValidationError, "only S0"):
            replace(s2, include_in_main_analysis=True).validate()

    def test_relations_may_compare_views_of_the_same_task(self) -> None:
        relation = replace(
            make_ed_relation(),
            source_task_id="task-a",
            target_task_id="task-a",
        )
        self.assertIs(relation.validate(), relation)
        with self.assertRaisesRegex(ProtocolValidationError, "classified S1"):
            TemporalGate(
                "T2",
                "S0",
                ordered_clip_advantage=True,
                include_in_main_analysis=False,
            ).validate()

    def test_release_holds_missing_or_unvalidated_artifacts(self) -> None:
        decisions = evaluate_release(
            make_preregistration(),
            {
                "manifest": True,
                "checksums": {"checksum": "present-but-not-validated"},
            },
        )

        self.assertEqual(decisions["evidence_archive"], HOLD)
        self.assertEqual(decisions["analysis_audit"], HOLD)
        self.assertEqual(decisions["preregistration"], PASS)

    def test_invalid_preregistration_holds_every_release_gate(self) -> None:
        prereg = make_preregistration()
        invalid = replace(prereg, temporal_gates=prereg.temporal_gates[:2])
        decisions = evaluate_release(
            invalid,
            {
                "manifest": True,
                "checksums": True,
                "analysis_report": True,
            },
        )

        self.assertEqual(decisions["preregistration"], HOLD)
        self.assertEqual(decisions["temporal_gates"], HOLD)
        self.assertEqual(decisions["evidence_archive"], HOLD)
        self.assertEqual(decisions["analysis_audit"], HOLD)

    def test_custom_release_gate_cannot_override_protocol_gate(self) -> None:
        prereg = replace(
            make_preregistration(),
            release_gates={"preregistration": ("forged_pass",)},
        )

        decisions = evaluate_release(prereg, {"forged_pass": True})

        self.assertEqual(decisions["preregistration"], HOLD)


if __name__ == "__main__":
    unittest.main()

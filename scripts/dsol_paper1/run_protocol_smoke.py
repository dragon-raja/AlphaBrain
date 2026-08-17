#!/usr/bin/env python3
"""Run a synthetic, fail-closed smoke of the DSOL Paper 1 protocol.

This command does not render, load a policy, generate data, or train. It only
checks that the release protocol rejects incomplete, leaked, or out-of-scope
declarations before real evidence is collected.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from AlphaBrain.research.dsol import (
    ED_EVIDENCE_GATE_NAMES,
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


DEBUG_ROOT = Path("/workspace/ai2r/debug").resolve()


def _ed_gates() -> dict[str, bool]:
    return {name: True for name in ED_EVIDENCE_GATE_NAMES}


def make_synthetic_preregistration() -> Preregistration:
    contract = ObservationContract(
        contract_id="synthetic-rgb-proprio-language-v1",
        deployment_inputs={
            "agent_rgb": {"shape": [224, 224, 3]},
            "wrist_rgb": {"shape": [224, 224, 3]},
            "proprioception": {"dimension": 32},
            "task_language": {"type": "string"},
        },
    )
    task = TaskTemplate(
        template_id="synthetic-task",
        instruction="place the object in the receptacle",
        action_set={"approach-left", "approach-right", "hold"},
        observation_contract_id=contract.contract_id,
    )
    relations = (
        RelationRecord(
            relation_id="synthetic-n",
            relation_type="N",
            source_task_id=task.template_id,
            target_task_id=task.template_id,
            action_set_equivalent=True,
            action_set_equivariant=False,
            r_hat=0.01,
            r_hat_threshold=0.05,
        ),
        RelationRecord(
            relation_id="synthetic-control",
            relation_type="MATCHED_CONTROL",
            source_task_id=task.template_id,
            target_task_id=task.template_id,
        ),
        RelationRecord(
            relation_id="synthetic-ed",
            relation_type="E_D",
            source_task_id=task.template_id,
            target_task_id=task.template_id,
            evidence_gates=_ed_gates(),
            matched_control_id="synthetic-control",
            expert_decision_sets=({"approach-left"}, {"approach-right"}),
            shared_safe_recovery_actions=(),
        ),
    )
    temporal = tuple(
        TemporalGate(
            test_id=test_id,
            stratum="S0",
            ordered_clip_advantage=False,
            include_in_main_analysis=True,
        )
        for test_id in ("T1", "T2", "T3")
    )
    return Preregistration(
        preregistration_id="synthetic-protocol-smoke-v1",
        task_templates=(task,),
        observation_contracts=(contract,),
        relations=relations,
        temporal_gates=temporal,
        release_gates={
            "evidence_archive": ("manifest", "checksums"),
            "analysis_audit": ("analysis_report",),
        },
    )


def _expect_rejection(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except ProtocolValidationError:
        return True
    return False


def run_smoke() -> dict[str, object]:
    prereg = make_synthetic_preregistration()
    prereg.validate()
    checks: dict[str, object] = {}

    complete = evaluate_release(
        prereg,
        {
            "manifest": True,
            "checksums": {"status": PASS},
            "analysis_report": {"valid": True},
        },
    )
    checks["complete_synthetic_declaration_can_pass"] = all(
        decision == PASS for decision in complete.values()
    )

    incomplete = evaluate_release(prereg, {"manifest": True})
    checks["missing_artifacts_hold_release"] = (
        incomplete["evidence_archive"] == HOLD
        and incomplete["analysis_audit"] == HOLD
    )

    ed_relation = next(
        relation for relation in prereg.relations if relation.relation_type == "E_D"
    )
    gate_rejections: dict[str, bool] = {}
    for gate_name in sorted(ED_EVIDENCE_GATE_NAMES):
        gates = _ed_gates()
        gates[gate_name] = False
        gate_rejections[gate_name] = _expect_rejection(
            lambda gates=gates: replace(ed_relation, evidence_gates=gates).validate()
        )
    checks["ed_gate_false_rejections"] = gate_rejections
    checks["invented_ed_gate_rejected"] = _expect_rejection(
        lambda: replace(
            ed_relation,
            evidence_gates={**_ed_gates(), "invented_gate": True},
        ).validate()
    )

    forbidden_inputs = (
        "branch_outcome",
        "candidate_rank",
        "future_state",
        "oracle_route",
        "r_hat_target",
        "scan_order",
    )
    checks["deployment_leak_rejections"] = {
        key: _expect_rejection(
            lambda key=key: ObservationContract("leak-check", {key: True}).validate()
        )
        for key in forbidden_inputs
    }

    checks["same_task_counterfactual_relation_allowed"] = not _expect_rejection(
        ed_relation.validate
    )
    checks["s1_main_analysis_rejected"] = _expect_rejection(
        lambda: TemporalGate("T1", "S1", True, True).validate()
    )
    checks["s2_main_analysis_rejected"] = _expect_rejection(
        lambda: TemporalGate("T1", "S2", False, True).validate()
    )

    def values_are_true(value: object) -> bool:
        if type(value) is bool:
            return value
        if isinstance(value, dict):
            return bool(value) and all(values_are_true(item) for item in value.values())
        return False

    passed = all(values_are_true(value) for value in checks.values())
    return {
        "schema_version": "dsol-paper1-protocol-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEBUG_PROTOCOL_SMOKE_PASS" if passed else "DEBUG_PROTOCOL_SMOKE_FAIL",
        "formal_eligible": False,
        "operations": {
            "render": False,
            "policy_load": False,
            "policy_inference": False,
            "data_generation": False,
            "training": False,
        },
        "checks": checks,
        "release_decisions_complete_synthetic_case": complete,
        "release_decisions_missing_artifacts": incomplete,
    }


def _debug_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(DEBUG_ROOT):
        raise ValueError(f"output must remain under {DEBUG_ROOT}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = _debug_output(args.output)
    receipt = run_smoke()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": receipt["status"]}))
    return 0 if receipt["status"] == "DEBUG_PROTOCOL_SMOKE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

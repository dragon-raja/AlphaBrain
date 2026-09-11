from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


WITHHELD_EDGES = ("white-right", "yellow_white-left")


def state0_summary(payload: Mapping[str, Any]) -> dict[str, int | bool]:
    if payload.get("status") != "complete":
        raise ValueError("state-0 gate requires a complete evaluation")
    rows = list(payload["rows"])
    id_rows = [row for row in rows if bool(row["action_supervised"])]
    ood_rows = [row for row in rows if not bool(row["action_supervised"])]
    if len(id_rows) != 4 or len(ood_rows) != 2:
        raise ValueError(
            "state-0 gate requires four observed and two action-free edges"
        )
    id_success = sum(bool(row["success"]) for row in id_rows)
    ood_success = sum(bool(row["success"]) for row in ood_rows)
    return {
        "id_success": id_success,
        "id_total": len(id_rows),
        "ood_success": ood_success,
        "ood_total": len(ood_rows),
        "id_valid": id_success >= 3,
        "eligible_for_validation": id_success >= 3 and ood_success >= 1,
    }


def evaluate_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("decision_horizon") != 3:
        raise ValueError("CAFC is preregistered at fixed K=3")
    k3 = payload.get("results", {}).get("k3")
    if not isinstance(k3, Mapping):
        raise ValueError("comparison is missing K=3 results")

    id_success = k3["id"]["success"]
    ood_success = k3["ood"]["success"]
    ood_source = k3["ood"]["source_selection_success"]
    by_edge = k3["by_edge"]
    missing = sorted(set(WITHHELD_EDGES) - set(by_edge))
    if missing:
        raise ValueError(f"comparison is missing withheld edges: {missing}")

    criteria = {
        "method_id_success_at_least_70pct": float(id_success["method"]) >= 0.70,
        "ood_success_gain_at_least_10pp": float(ood_success["difference"]) >= 0.10,
        "id_success_drop_at_most_5pp": float(id_success["difference"]) >= -0.05,
        "both_withheld_edges_improve": all(
            float(by_edge[edge]["difference"]) > 0.0 for edge in WITHHELD_EDGES
        ),
        "ood_source_selection_improves": float(ood_source["difference"]) > 0.0,
        "ood_success_paired_ci_nonnegative": float(ood_success["ci95_low"]) >= 0.0,
        "ood_source_paired_ci_nonnegative": float(ood_source["ci95_low"]) >= 0.0,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "baseline_id_success": float(id_success["baseline"]),
            "method_id_success": float(id_success["method"]),
            "id_success_difference": float(id_success["difference"]),
            "baseline_ood_success": float(ood_success["baseline"]),
            "method_ood_success": float(ood_success["method"]),
            "ood_success_difference": float(ood_success["difference"]),
            "ood_success_ci95": [
                float(ood_success["ci95_low"]),
                float(ood_success["ci95_high"]),
            ],
            "ood_source_selection_difference": float(ood_source["difference"]),
            "ood_source_selection_ci95": [
                float(ood_source["ci95_low"]),
                float(ood_source["ci95_high"]),
            ],
            "withheld_edge_success_differences": {
                edge: float(by_edge[edge]["difference"]) for edge in WITHHELD_EDGES
            },
        },
    }


def decide_cafc_gate(
    *,
    plain_state0: Mapping[str, Any],
    grounded_state0: Mapping[str, Any],
    plain_exact: Mapping[str, Any],
    grounded_exact: Mapping[str, Any],
    plain_strong: Mapping[str, Any],
    grounded_strong: Mapping[str, Any],
) -> dict[str, Any]:
    state0 = {
        "plain_cafc": state0_summary(plain_state0),
        "bridge_cafc": state0_summary(grounded_state0),
    }
    arms = {
        "plain_cafc": {
            "state0": state0["plain_cafc"],
            "exact_comparator": evaluate_comparison(plain_exact),
            "strongest_non_cafc_comparator": evaluate_comparison(plain_strong),
        },
        "bridge_cafc": {
            "state0": state0["bridge_cafc"],
            "exact_comparator": evaluate_comparison(grounded_exact),
            "strongest_non_cafc_comparator": evaluate_comparison(grounded_strong),
        },
    }
    for arm in arms.values():
        arm["passed"] = bool(
            arm["state0"]["eligible_for_validation"]
            and arm["exact_comparator"]["passed"]
            and arm["strongest_non_cafc_comparator"]["passed"]
        )

    if arms["plain_cafc"]["passed"]:
        decision = "ADVANCE_CAFC"
        reason = "PLAIN_CAFC_CLEARS_EXACT_AND_STRONG_COMPARATOR_GATES"
    elif arms["bridge_cafc"]["passed"]:
        decision = "ADVANCE_GROUNDED_CAFC"
        reason = "ONLY_BRIDGE_CAFC_CLEARS_EXACT_AND_STRONG_COMPARATOR_GATES"
    elif not any(summary["id_valid"] for summary in state0.values()):
        decision = "BASELINE_INVALID"
        reason = "NEITHER_CAFC_ARM_REACHES_3_OF_4_OBSERVED_STATE0_SUCCESSES"
    else:
        decision = "STOP_CAFC"
        reason = "NO_CAFC_ARM_CLEARS_THE_PREREGISTERED_BEHAVIORAL_GATE"

    return {
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "decision_horizon": 3,
        "independent_unit": "canonical_state_index",
        "withheld_edges": list(WITHHELD_EDGES),
        "strongest_non_cafc_comparator": "cabi_action_bridge_plus_decoder_closure",
        "arms": arms,
        "note": (
            "A CAFC arm must pass both its architecture-matched comparator and "
            "the preregistered strongest non-CAFC comparator. State 0 is disclosed "
            "calibration evidence; the decision metrics come from validation states."
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite decision: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the frozen seed-41 CAFC gate")
    parser.add_argument("--plain-state0", type=Path, required=True)
    parser.add_argument("--grounded-state0", type=Path, required=True)
    parser.add_argument("--plain-exact", type=Path, required=True)
    parser.add_argument("--grounded-exact", type=Path, required=True)
    parser.add_argument("--plain-strong", type=Path, required=True)
    parser.add_argument("--grounded-strong", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    result = decide_cafc_gate(
        plain_state0=_load(args.plain_state0),
        grounded_state0=_load(args.grounded_state0),
        plain_exact=_load(args.plain_exact),
        grounded_exact=_load(args.grounded_exact),
        plain_strong=_load(args.plain_strong),
        grounded_strong=_load(args.grounded_strong),
    )
    result["inputs"] = {
        key: str(getattr(args, key))
        for key in (
            "plain_state0",
            "grounded_state0",
            "plain_exact",
            "grounded_exact",
            "plain_strong",
            "grounded_strong",
        )
    }
    _atomic_write(args.output, result)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from decide_cafc_gate import evaluate_comparison, state0_summary


def decide_h20_gate(
    *,
    bc_state0: Mapping[str, Any],
    bridge_state0: Mapping[str, Any],
    plain_state0: Mapping[str, Any],
    grounded_state0: Mapping[str, Any],
    plain_exact: Mapping[str, Any],
    grounded_exact: Mapping[str, Any],
    plain_strong: Mapping[str, Any],
    grounded_strong: Mapping[str, Any],
) -> dict[str, Any]:
    state0 = {
        "bc_h20": state0_summary(bc_state0),
        "bridge_h20": state0_summary(bridge_state0),
        "cafc_h20": state0_summary(plain_state0),
        "bridge_cafc_h20": state0_summary(grounded_state0),
    }
    arms = {
        "cafc_h20": {
            "state0": state0["cafc_h20"],
            "exact_control_state0": state0["bc_h20"],
            "exact_comparator": evaluate_comparison(plain_exact),
            "strongest_non_cafc_comparator": evaluate_comparison(plain_strong),
        },
        "bridge_cafc_h20": {
            "state0": state0["bridge_cafc_h20"],
            "exact_control_state0": state0["bridge_h20"],
            "exact_comparator": evaluate_comparison(grounded_exact),
            "strongest_non_cafc_comparator": evaluate_comparison(grounded_strong),
        },
    }
    for arm in arms.values():
        arm["passed"] = bool(
            arm["state0"]["eligible_for_validation"]
            and arm["exact_control_state0"]["id_valid"]
            and arm["exact_comparator"]["passed"]
            and arm["strongest_non_cafc_comparator"]["passed"]
        )

    if arms["cafc_h20"]["passed"]:
        decision = "ADVANCE_H20_CAFC"
        reason = "H20_CAFC_CLEARS_EXACT_AND_STRONG_COMPARATOR_GATES"
    elif arms["bridge_cafc_h20"]["passed"]:
        decision = "ADVANCE_H20_GROUNDED_CAFC"
        reason = "ONLY_H20_BRIDGE_CAFC_CLEARS_EXACT_AND_STRONG_GATES"
    elif not any(
        (
            arms[name]["state0"]["id_valid"]
            and arms[name]["exact_control_state0"]["id_valid"]
        )
        for name in arms
    ):
        decision = "BASELINE_INVALID"
        reason = "NO_H20_CAFC_ARM_HAS_A_VALID_OBSERVED_EXACT_COMPARISON"
    else:
        decision = "STOP_HORIZON_EXTENSION"
        reason = "NO_H20_CAFC_ARM_CLEARS_THE_FROZEN_BEHAVIORAL_GATE"

    return {
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "training_action_horizon": 20,
        "decision_horizon": 3,
        "independent_unit": "canonical_state_index",
        "state0": state0,
        "arms": arms,
        "strongest_non_cafc_comparator": "action_bridge_plus_decoder_closure_h10",
        "note": (
            "H=20 was selected once from disclosed state-0 feasibility and was "
            "given to both exact controls. H=30/40/50 are forbidden follow-ups."
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
    parser = argparse.ArgumentParser(description="Apply the frozen H=20 CAFC gate")
    for name in (
        "bc_state0",
        "bridge_state0",
        "plain_state0",
        "grounded_state0",
        "plain_exact",
        "grounded_exact",
        "plain_strong",
        "grounded_strong",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    names = (
        "bc_state0",
        "bridge_state0",
        "plain_state0",
        "grounded_state0",
        "plain_exact",
        "grounded_exact",
        "plain_strong",
        "grounded_strong",
    )
    result = decide_h20_gate(**{name: _load(getattr(args, name)) for name in names})
    result["inputs"] = {name: str(getattr(args, name)) for name in names}
    _atomic_write(args.output, result)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()


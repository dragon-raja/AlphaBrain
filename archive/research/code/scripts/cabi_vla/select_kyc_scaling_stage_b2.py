from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def nearest_log_neighbor(target: int, budgets: list[int]) -> int:
    candidates = [budget for budget in budgets if budget != target]
    if not candidates:
        raise ValueError("neighbor selection requires at least two budgets")
    return min(
        candidates,
        key=lambda budget: (abs(math.log(budget) - math.log(target)), budget),
    )


def select_stage_b2(summary: Mapping[str, Any]) -> dict[str, Any]:
    budgets = [int(value) for value in summary["budgets"]]
    gains = {
        budget: float(
            summary["budget_results"][str(budget)]["primary"]["all"][
                "comparisons"
            ]["kyc_minus_poseaug_control"]["success"]["delta"]
        )
        for budget in budgets
    }
    qualifying = [budget for budget, gain in gains.items() if gain >= 0.05]
    if qualifying:
        target = min(qualifying, key=lambda budget: (-gains[budget], budget))
        selected = sorted({target, nearest_log_neighbor(target, budgets)})
        rule = (
            "largest seed-41 KYC-Control gain among budgets reaching 5pp, "
            "plus nearest log-view neighbor"
        )
    else:
        selected = [budget for budget in (10, 45) if budget in budgets]
        if len(selected) != 2:
            raise ValueError("fallback requires preregistered budgets 10 and 45")
        rule = "no seed-41 gain reached 5pp; use preregistered n=10/45 fallback"

    factorial_budget = summary["factorial_budget_selection"]["selected_budget"]
    if factorial_budget is None:
        raise ValueError("factorial budget is unavailable because baseline is invalid")
    training_budgets = sorted({*selected, int(factorial_budget)})
    return {
        "schema_version": 1,
        "status": "selected",
        "seed41_success_gain_by_budget": {
            str(budget): gain for budget, gain in sorted(gains.items())
        },
        "scaling_confirmation_budgets": selected,
        "factorial_budget": int(factorial_budget),
        "training_budgets": training_budgets,
        "confirmation_seeds": [42, 43],
        "selection_rule": rule,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select preregistered KYC Pi0.5 Stage B2 budgets"
    )
    parser.add_argument("--stage-b1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage B2 selection: {parsed.output}")
    payload = select_stage_b2(json.loads(parsed.stage_b1_summary.read_text()))
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

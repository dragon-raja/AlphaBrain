"""GR1 remote CPU contract adapter using the existing center worker lifecycle.

Production mode is deliberately unsupported. This verifies the actual ledger,
GO/ACK/continuation and GR1 native-query path without GPU/model calls. It is not
a second scheduler or permission to launch a real scientific block.
"""

import argparse
import json
from pathlib import Path
import sys

from AlphaBrain.research.dsol.execution import model_agent, model_plan
from AlphaBrain.research.dsol.execution.authority_wire import Wire
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest
from AlphaBrain.research.dsol.execution.queue import _durable_write


def source_identity():
    from AlphaBrain.research.dsol.execution.source_release import source_tree_identity

    root = Path(__file__).resolve().parents[2]
    return source_tree_identity(
        root,
        (
            "AlphaBrain/common",
            "AlphaBrain/model",
            "AlphaBrain/research/dsol",
            "benchmarks/Robocasa_tabletop",
        ),
    )


def load_binding(path):
    b = json.loads(Path(path).read_text())
    if (
        b.get("schema") != "GR1_CPU_BINDING_V1"
        or b.get("mode") != "cpu_contract"
        or b.get("resident_workers", False)
        or b.get("resident_simulator", False)
    ):
        raise ValueError("GR1 supports only bounded nonresident CPU contract preparation")
    return b


verify_node = model_agent.verify_node
prepare = model_agent.prepare
validate_launch = model_agent.validate_cpu_launch
plans = model_plan.plans


def prepare_controller(binding_path, root, transport_config):
    return model_plan.prepare_controller(
        binding_path, root, transport_config, load_binding=load_binding
    )


def execute_cpu_contract(binding, launch, guard, *, cache=None):
    from .cpu_contract import run_query, run_trial, run_trial_block

    spec = binding["tasks"][launch["key"]]
    options = [
        ({"synthetic": True}, 2, 1),
        ({"synthetic_trial": True}, 4, 1),
        ({"synthetic_block": True}, 8, 2),
        ({"synthetic_prepared": True}, 36, 2),
    ]
    choice = next(
        ((calls, episodes) for request, calls, episodes in options if spec["request"] == request),
        None,
    )
    if choice is None or launch["limits"] != dict(calls=choice[0], episodes=choice[1]):
        raise ValueError("Exact synthetic GR1 query budget required")
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    if spec["request"] == {"synthetic_prepared": True}:
        from .cpu_prepared import run
        result = run(guard, output=output)
    elif choice[1] == 2:
        result = run_trial_block(guard, episode_id=launch["key"], output=output / "block")
    elif choice[0] == 4:
        result = run_trial(guard, episode_id=launch["key"], output=output)
    else:
        result = run_query(guard, episode_id=launch["key"])
    _durable_write(output / "cpu-receipt.json", result)
    guard()
    return dict(calls=result["synthetic_calls"], episodes=choice[1]), file_digest(
        output / "cpu-receipt.json"
    )


def verify_completed(binding, row):
    if row["status"] == "COMPLETE":
        spec = binding["tasks"][row["launch"]["key"]]
        path = Path(spec["output"]) / "cpu-receipt.json"
        if file_digest(path) != row["receipt"]["result_sha256"]:
            raise ValueError("Completed GR1 CPU receipt changed")


def execute_block(*args, **kwargs):
    raise RuntimeError("GR1 real block is not registered or admitted")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--simulate", action="store_true", required=True)
    args = parser.parse_args(argv)
    wire = Wire(sys.stdin.fileno(), sys.stdout.fileno(), timeout=60)
    try:
        binding = load_binding(args.binding)
        if args.prepare:
            args.root.parent.mkdir(parents=True, exist_ok=True)
            prepare(args.root, binding)
        model_agent.serve(args.root, binding, wire, sys.modules[__name__])
    except Exception as error:
        try:
            wire.send(dict(kind="ERROR", error=type(error).__name__ + ": " + str(error)))
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

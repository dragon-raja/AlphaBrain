"""Explicit CPU-only search interface validation against frozen historical caches."""

from dataclasses import asdict
from pathlib import Path
import importlib.util

import numpy as np

from ..acquisition.search_env import SearchConfig, SearchEnv, rollout_baseline
from ..data.artifacts import read, sha, write
from ..data.search_replay import load_initial32


def run(source, historical_recovery, output, config_path):
    """Write only to a new directory. No learner, fitting, rendering or execution."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _audit(Path(source), Path(historical_recovery), output, Path(config_path))
    except Exception as error:
        write(output / "failure.json", {
            "status": "FAILED_STOPPED", "error_type": type(error).__name__, "reason": str(error),
            "new_model_calls": 0, "training_steps": 0, "new_rollouts": 0,
        })
        raise


def _audit(source, historical_recovery, output, config_path):
    settings = read(config_path)
    if (settings["schema_version"] != 1 or settings["mode"] != "interface_audit_only"
            or settings["training_enabled"] is not False):
        raise ValueError("Only the explicit interface audit configuration is supported")
    data, outcomes = load_initial32(source)
    config = SearchConfig(**settings["search"])
    config.validate(len(data.candidate_ids))
    hashes = dict(data.source_hashes)
    hashes[str(config_path)] = sha(config_path)
    receipt_path = historical_recovery / "provenance.json"
    hashes[str(receipt_path)] = sha(receipt_path)
    historical_receipt = read(receipt_path)

    def checked(name):
        path = historical_recovery / name
        digest = sha(path)
        if digest != historical_receipt["artifacts"][name]:
            raise ValueError("Historical recovery artifact changed: " + name)
        hashes[str(path)] = digest
        return path

    historical_budgets = read(checked("summary.json"))["protocol"]["config"]["budgets"]
    budgets = settings["audit_budgets"]
    if any(b not in historical_budgets for b in budgets):
        raise ValueError("Golden budget is absent from historical replay")
    start = settings["audit_start"]
    if start != 65:
        raise ValueError("This golden adapter uses the first heldout-catalog start, index 65")
    traces, golden = [], []
    for model, name in enumerate(("canonical", "broad")):
        with np.load(checked(f"{name}-candidate-scores.npz"), allow_pickle=False) as z:
            if not np.array_equal(z["accel"], data.accel_members[model, list(data.evaluation)].mean(-1)):
                raise ValueError("Historical mean Accel mismatch")
        with np.load(checked(f"{name}-local_accel-replay.npz"), allow_pickle=False) as z:
            old_visited, old_selected, old_success = (
                z[k].copy() for k in ("visited", "selected", "success"))
        for initial in range(len(data.states)):
            for budget in budgets:
                run_config = SearchConfig(**{**asdict(config), "query_budget": budget})
                for method in ("stay", "scan", "random", "local_accel"):
                    env = SearchEnv(data.poses[initial], data.queries(model, initial),
                                    data.wrist[initial], data.score_scales[model], run_config)
                    trace = rollout_baseline(env, start, method, seed=settings["audit_seed"])
                    if len(trace["visited"]) != len(set(trace["visited"])):
                        raise ValueError("Duplicate acquisition")
                    if trace["selected"] not in trace["visited"] or trace["path"][-1] != trace["selected"]:
                        raise ValueError("Invalid return endpoint")
                    if not all(env.edges[a, b] for a, b in zip(trace["path"][:-1], trace["path"][1:])):
                        raise ValueError("Invalid graph navigation")
                    expected_reward = (
                        (env.initial_score - env.values[env.best, 0]) / env.scale
                        - run_config.query_cost * (trace["queries"] - 1)
                        - run_config.motion_cost * trace["motion_units"])
                    if not np.isclose(expected_reward, trace["total_reward"], atol=1e-12, rtol=0):
                        raise ValueError("Reward accounting failed")
                    paired = outcomes.paired(model, initial, start, trace["selected"])
                    if paired["paired_repeats"] != 32:
                        raise ValueError("Historical paired endpoint label coverage failed")
                    if method == "local_accel" and initial in data.evaluation:
                        ti = data.evaluation.index(initial)
                        bi = historical_budgets.index(budget)
                        if (trace["visited"] != old_visited[ti, start - 1, 0, :budget].tolist()
                                or trace["selected"] != int(old_selected[ti, start - 1, 0, bi])
                                or paired["selected"] != float(old_success[ti, start - 1, 0, bi])):
                            raise ValueError("Frozen local-Accel replay mismatch")
                        golden.append(dict(model=name, initial=initial, budget=budget, matched=True))
                    traces.append(dict(model=name, initial=initial, budget=budget, method=method,
                                       label_repeats=paired["paired_repeats"], **trace))
        print(f"Validated {name}: all 32 initials; no inference/training", flush=True)
    package = Path(__file__).resolve().parents[1]
    code_hashes = {name: sha(package / name) for name in (
        "data/search_replay.py", "acquisition/search_env.py",
        "acquisition/discrete_search.py", "analysis/search_interface_audit.py")}
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError("Read-only source changed: " + path)
    write(output / "traces.json", traces)
    write(output / "golden.json", golden)
    write(output / "sources.json", hashes)
    receipt = dict(
        status="PASSED_CACHED_INTERFACE_AUDIT", source_hashes=hashes, code_hashes=code_hashes,
        config=settings, models=2, initials=32, development_initials=list(map(int, data.development)),
        evaluation_initials=list(map(int, data.evaluation)), candidates=97, score_members_per_query=8,
        historical_endpoint_rollouts=198656, missing_endpoint_labels=0,
        visual_feature_dim=512, wrist_feature_dim=512,
        canonical_image_context_excluded=True, outcome_labels_in_environment=False,
        development_score_scales=list(data.score_scales), golden_comparisons=len(golden),
        baseline_interface_episodes=len(traces), new_model_calls=0, training_steps=0, new_rollouts=0,
        originals_unchanged=True,
        optional_dependencies={name: importlib.util.find_spec(name) is not None for name in (
            "gymnasium", "stable_baselines3", "sb3_contrib")},
        limitations=[
            "Cached static external-camera catalog, not constrained head motion.",
            "Old endpoints ignore camera acquisition effects and elapsed physical time.",
            "Historical test results have been inspected; not blind or unseen-task evidence.",
            "Only view perturbations in this cache, not all LIBERO-Plus perturbation categories.",
            "Golden comparison covers query sequence, endpoint and outcome, not historical travel paths.",
            "New routing uses shortest visited routes for all baselines; distances are geometric proxies.",
            "Gym/PPO runtime admission and a multi-initial training runner are not executed here.",
        ],
    )
    write(output / "receipt.json", receipt)
    return receipt

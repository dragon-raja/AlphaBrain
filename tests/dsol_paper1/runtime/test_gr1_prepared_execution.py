"""Bound production-shaped orchestration, using only synthetic model IPC."""

from contextlib import contextmanager
from dataclasses import replace
import json
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
from benchmarks.Robocasa_tabletop.cpu_prepared import fixture, synthetic_resources
from benchmarks.Robocasa_tabletop.prepared_execution import run_prepared_block
from benchmarks.Robocasa_tabletop.native_resources import NativeResources
from benchmarks.Robocasa_tabletop.trial_plan import TrialPlan


def run(tmp_path, plans, policy, binding, *, resources=synthetic_resources, guard=lambda: None):
    return run_prepared_block(
        plans,
        tmp_path / "block",
        policy=policy,
        policy_binding=binding,
        calls_per_episode=[18] * len(plans),
        call_limit=18 * len(plans),
        resources=resources,
        guard=guard,
    )


def test_complete_charges_actual_calls_and_refuses_replay(tmp_path):
    plans, policy, binding = fixture(tmp_path)
    assert TrialPlan.from_request(plans[0].request()) == plans[0]
    result = run(tmp_path, plans, policy, binding)
    assert result["status"] == "COMPLETE" and result["completed_episodes"] == 2
    assert result["attempted_model_calls"] == result["completed_model_calls"] == 8
    from AlphaBrain.research.dsol.execution.manifest import digest

    manifest = json.loads((tmp_path / "block/binding.json").read_text())
    assert digest(manifest) == result["binding_sha256"]
    assert len(manifest["policy_state_sha256"]) == 64
    with pytest.raises(FileExistsError):
        run(tmp_path, plans, policy, binding)


@pytest.mark.parametrize("fault", ["hash", "under_budget", "task", "duplicate", "config"])
def test_bad_plan_fails_before_any_resources_or_output(tmp_path, fault):
    plans, policy, binding = fixture(tmp_path)
    if fault == "hash":
        plans[1] = replace(plans[1], score_noise=replace(plans[1].score_noise, sha256="b" * 64))
    elif fault == "under_budget":
        plans[1] = replace(plans[1], operation_actions=49)
    elif fault == "task":
        plans[1] = replace(plans[1], instruction="")
    elif fault == "duplicate":
        plans[1] = replace(plans[1], episode_id=plans[0].episode_id)
    else:
        config = dict(plans[1].search_config)
        del config["query_cost"]
        plans[1] = replace(plans[1], search_config=config)

    def unopened(*args):
        pytest.fail("Invalid plan opened resources")

    unopened.binding = synthetic_resources.binding
    unopened.validate = synthetic_resources.validate
    with pytest.raises(ValueError):
        run(tmp_path, plans, policy, binding, resources=unopened)
    assert not (tmp_path / "block").exists()


@pytest.mark.parametrize("fault", ["unknown_call", "disconnect", "operation", "cleanup"])
def test_partial_block_retains_completed_trial_and_unknown_call(tmp_path, fault):
    plans, policy, binding = fixture(tmp_path)
    disconnected = False
    children = []

    def guard():
        if disconnected:
            raise ConnectionError("center lost")

    @contextmanager
    def resources(plan, prepared, directory, live):
        nonlocal disconnected
        with synthetic_resources(plan, prepared, directory, live) as (source, operation):
            children.append(source.runtime.process)
            if plan.episode_id == "trial-1":
                infer = source.runtime.infer

                def fail(*args, **kwargs):
                    nonlocal disconnected
                    result = infer(*args, **kwargs)
                    if fault == "unknown_call":
                        raise ConnectionError("result lost after actual synthetic call")
                    if fault == "disconnect":
                        disconnected = True
                    return result

                source.runtime.infer = fail
                if fault == "operation":

                    def bad_step(action):
                        raise RuntimeError("unknown physical step")

                    operation.step_absolute = bad_step
            yield source, operation
        if plan.episode_id == "trial-1" and fault == "cleanup":
            raise RuntimeError("cleanup proof lost")

    resources.binding = synthetic_resources.binding
    resources.validate = synthetic_resources.validate
    with pytest.raises((ConnectionError, RuntimeError)):
        run(tmp_path, plans, policy, binding, resources=resources, guard=guard)
    state = json.loads((tmp_path / "block/block-state.json").read_text())
    assert state["status"] == "UNKNOWN_REQUIRES_AUDIT"
    assert state["completed_episodes"] == 1 and state["completed_model_calls"] == 4
    assert state["attempted_model_calls"] > 4
    assert (tmp_path / "block/trial-0/result.json").is_file()
    assert all(p.poll() is not None for p in children)


def test_native_resources_remain_closed_before_task_or_model(tmp_path):
    plans, policy, binding = fixture(tmp_path)
    import sys

    factory = NativeResources(
        {}, model_python=sys.executable, source_root="/not/a/source", mode="NATIVE"
    )
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled

    with pytest.raises(ProductionDispatchDisabled):
        run(tmp_path, plans, policy, binding, resources=factory)
    assert not (tmp_path / "block").exists()


def test_same_shape_policy_swap_between_trials_holds_without_new_calls(tmp_path):
    import torch

    plans, policy, binding = fixture(tmp_path)

    @contextmanager
    def resources(plan, prepared, directory, guard):
        with synthetic_resources(plan, prepared, directory, guard) as ports:
            yield ports
        with torch.no_grad():
            policy.stop_head.bias.add_(1.0)

    resources.binding = synthetic_resources.binding
    resources.validate = synthetic_resources.validate
    with pytest.raises(ValueError, match="Policy weights changed"):
        run(tmp_path, plans, policy, binding, resources=resources)
    state = json.loads((tmp_path / "block/block-state.json").read_text())
    assert state["completed_episodes"] == 1 and state["attempted_model_calls"] == 4
    assert state["status"] == "UNKNOWN_REQUIRES_AUDIT"

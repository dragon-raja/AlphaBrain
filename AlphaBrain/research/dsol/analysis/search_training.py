"""Explicit bounded CPU training/evaluation in a cached search environment.

Outputs are append-only per run. No simulator, VLA, old queue, or image encoder
is started. Scientific endpoint labels are used only after training finishes.
"""

from pathlib import Path
import json
import importlib.metadata
import shutil
import sys
import time

import numpy as np

from ..acquisition.search_env import SearchConfig, SearchEnv, gymnasium_env, rollout_baseline
from ..data.artifacts import read, write, sha
from ..data.search_replay import load_initial32
from .statistics import paired_interval


def make_core(data, initial, budget, search_config):
    config = SearchConfig(**{**search_config, "query_budget": budget})
    return SearchEnv(data.poses[initial], data.queries(0, initial), data.wrist[initial],
                     data.score_scales[0], config)


def assert_observation(observation):
    visited = observation["flags"][:, 0].astype(bool)
    if not all(np.isfinite(value).all() for value in observation.values()):
        raise ValueError("Nonfinite observation")
    if observation["visual"][~visited].any() or observation["scores"][~visited].any():
        raise ValueError("Unvisited visual/score leakage")


def compare_nested(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        # Optimizer counters may move from CPU to CUDA during checkpoint load.
        # Compare exact values/dtypes independently of storage placement.
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(compare_nested(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(compare_nested(a, b) for a, b in zip(left, right))
    return left == right


def checkpoint_admission(model, path, data, settings):
    """Check policy, critic, optimizer and a deterministic trajectory after load."""
    import torch
    from sb3_contrib import MaskablePPO
    restored = MaskablePPO.load(path, device="cpu")
    if not compare_nested(model.policy.state_dict(), restored.policy.state_dict()):
        raise ValueError("Checkpoint policy/critic mismatch")
    if not compare_nested(model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()):
        raise ValueError("Checkpoint optimizer mismatch")
    envs = [gymnasium_env(make_core(data, data.development[0], 8, settings["search"]), [1, 2])
            for _ in range(2)]
    paths = []
    for policy, env in zip((model, restored), envs):
        obs, _ = env.reset(seed=91, options={"start": 1})
        actions, values = [], []
        while True:
            assert_observation(obs)
            tensor, _ = policy.policy.obs_to_tensor(obs)
            with torch.no_grad():
                value = policy.policy.predict_values(tensor).cpu().numpy()
            action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
            actions.append(int(action)); values.append(value.tolist())
            obs, _, done, truncated, info = env.step(int(action))
            if done or truncated:
                break
        paths.append(dict(actions=actions, values=values, visited=info["visited"], selected=info["selected"]))
    if paths[0] != paths[1]:
        raise ValueError("Checkpoint trajectory/critic replay mismatch")
    return dict(policy_exact=True, critic_exact=True, optimizer_exact=True,
                deterministic_trajectory_exact=True, trajectory=paths[0],
                resume_scope="weights and optimizer restored; no claim of mid-episode RNG resume")


def train_seed(data, settings, output, seed, steps):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ..selectors.search_ppo import build_model, SearchAuditCallback
    output.mkdir(exist_ok=False)
    factories = [
        (lambda initial=i: gymnasium_env(make_core(data, initial, 8, settings["search"]), range(1, 65)))
        for i in data.development
    ]
    env = DummyVecEnv(factories)  # One CPU process, 16 sequential logical environments.
    model = build_model(env, seed, settings["ppo"], output)
    before_actor = model.policy.action_net.weight.detach().clone()
    before_critic = model.policy.value_net.weight.detach().clone()
    callback = SearchAuditCallback(output, data.development)
    begin = time.monotonic()
    model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
    if model.num_timesteps != steps or callback.checked_actions != steps:
        raise ValueError("Training decision budget mismatch")
    actor_changed = not torch.equal(before_actor, model.policy.action_net.weight)
    critic_changed = not torch.equal(before_critic, model.policy.value_net.weight)
    if not actor_changed or not critic_changed:
        raise ValueError("Actor or critic did not update")
    path = output / "final.zip"
    model.save(path)
    admission = checkpoint_admission(model, path, data, settings)
    result = dict(seed=seed, steps=steps, checked_actions=callback.checked_actions,
                  actor_updated=actor_changed, critic_updated=critic_changed,
                  parameters=sum(p.numel() for p in model.policy.parameters()),
                  elapsed_seconds=time.monotonic() - begin, checkpoint_sha256=sha(path),
                  checkpoint_admission=admission)
    write(output / "receipt.json", result)
    env.close()
    return result


def evaluate(data, outcomes, settings, output):
    from sb3_contrib import MaskablePPO
    rows = []
    models = {seed: MaskablePPO.load(output / f"seed-{seed}" / "final.zip", device="cpu")
              for seed in settings["training_seeds"]}
    for budget in settings["evaluation_budgets"]:
        for initial in data.evaluation:
            initial = int(initial)
            for start in range(65, 97):
                methods = [("stay", 0), ("scan", 0), ("local_accel", 0)]
                methods += [("random", seed) for seed in settings["training_seeds"]]
                methods += [("ppo", seed) for seed in settings["training_seeds"]]
                for method, seed in methods:
                    core = make_core(data, initial, budget, settings["search"])
                    begin = time.monotonic()
                    if method == "ppo":
                        env = gymnasium_env(core, [start])
                        obs, _ = env.reset(seed=10000 + initial * 100 + start, options={"start": start})
                        while True:
                            assert_observation(obs)
                            action, _ = models[seed].predict(obs, deterministic=True, action_masks=env.action_masks())
                            obs, _, done, truncated, info = env.step(int(action))
                            if done or truncated:
                                break
                    else:
                        info = rollout_baseline(core, start, method, seed=seed + initial * 100000 + start * 100)
                    paired = outcomes.paired(0, initial, start, info["selected"])
                    if paired["paired_repeats"] != 32:
                        raise ValueError("Missing paired evaluation labels")
                    full_min = float(data.accel_members[0, initial].mean(-1).min())
                    row = dict(
                        method=method, seed=seed, initial=initial, start=start, budget=budget,
                        task=data.states[initial]["task_id"], elapsed_seconds=time.monotonic() - begin,
                        accel_gap=(info["best_accel"] - full_min) / data.score_scales[0],
                        **info,
                    )
                    row.update(paired)
                    # Preserve selected candidate identity separately from measured success.
                    row["selected_candidate"] = info["selected"]
                    row["success"] = paired["selected"]
                    rows.append(row)
            print(f"evaluation budget={budget} initial={initial} complete", flush=True)
    with (output / "evaluation.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = []
    for budget in settings["evaluation_budgets"]:
        for method in ("stay", "scan", "random", "local_accel", "ppo"):
            selected = [r for r in rows if r["budget"] == budget and r["method"] == method]
            metrics = {}
            for field in ("success", "stay", "gain", "rescued_fraction", "harmed_fraction",
                          "queries", "motion_units", "total_reward", "best_accel", "accel_gap"):
                values = np.array([np.mean([r[field] for r in selected if r["initial"] == i])
                                   for i in data.evaluation])
                metrics[field] = {**paired_interval(values, np.arange(16)),
                                  "per_initial": values.tolist()}
            per_seed = {}
            for seed in sorted({r["seed"] for r in selected}):
                group = [r for r in selected if r["seed"] == seed]
                per_seed[str(seed)] = {k: float(np.mean([r[k] for r in group]))
                                      for k in ("success", "gain", "queries", "total_reward", "accel_gap")}
            summary.append(dict(method=method, budget=budget, metrics=metrics, per_seed=per_seed))
    comparisons = []
    for budget in settings["evaluation_budgets"]:
        ppo = next(r for r in summary if r["budget"] == budget and r["method"] == "ppo")
        for method in ("stay", "scan", "random", "local_accel"):
            baseline = next(r for r in summary if r["budget"] == budget and r["method"] == method)
            differences = {}
            for metric in ("success", "queries", "total_reward", "accel_gap"):
                delta = np.array(ppo["metrics"][metric]["per_initial"]) - np.array(
                    baseline["metrics"][metric]["per_initial"])
                differences[metric] = paired_interval(delta, np.arange(16))
            comparisons.append(dict(budget=budget, baseline=method, ppo_minus_baseline=differences))
    result = dict(rows=summary, comparisons=comparisons, evaluation_episodes=len(rows),
                  ppo_episodes=sum(r["method"] == "ppo" for r in rows),
                  evaluation_scope="Known tasks, previously inspected initials, cached static endpoints")
    write(output / "summary.json", result)
    return result


def run(source, output, config_path, execute=False):
    if not execute:
        raise ValueError("Explicit --execute required; no work started")
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = read(config_path)
    if (settings["training_seeds"] != [0, 1, 2] or settings["smoke_steps"] != 2048
            or settings["steps_per_seed"] != 32768):
        raise ValueError("This admission is bounded to the reviewed first-round budget")
    torch.set_num_threads(settings["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, outcomes = load_initial32(source)
    package = Path(__file__).resolve().parents[1]
    code = {p: sha(package / p) for p in (
        "acquisition/search_env.py", "acquisition/discrete_search.py", "data/search_replay.py",
        "selectors/search_ppo.py", "analysis/search_training.py", "analysis/statistics.py")}
    for relative in code:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, destination)
    write(output / "environment.json", dict(
        python=sys.version, prefix=sys.prefix, device="cpu", torch_cuda=torch.version.cuda,
        packages={name: importlib.metadata.version(name) for name in (
            "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "numpy")},
    ))
    write(output / "frozen-plan.json", dict(
        settings=settings, source_hashes=data.source_hashes, code_hashes=code,
        config_sha256=sha(config_path), model_index=0, model_name="canonical",
        manipulator="Historical LIBERO AlphaBrain PaliGemmaPi05 canonical checkpoint; no VLA executed",
        training_initials=list(map(int, data.development)), evaluation_initials=list(map(int, data.evaluation)),
        trained_on_success_labels=False, fixed_final_checkpoint=True,
        resource_contract="Single CPU process, 16 sequential DummyVecEnv instances; no workers/old queue",
    ))
    try:
        smoke = train_seed(data, settings, output / "smoke", 17, settings["smoke_steps"])
        write(output / "smoke-admission.json", dict(status="PASS", **smoke))
        trained = []
        for seed in settings["training_seeds"]:
            trained.append(train_seed(data, settings, output / f"seed-{seed}", seed,
                                      settings["steps_per_seed"]))
        result = evaluate(data, outcomes, settings, output)
        for path, digest in data.source_hashes.items():
            if sha(path) != digest:
                raise ValueError("Historical source changed")
        for relative, digest in code.items():
            if sha(package / relative) != digest:
                raise ValueError("Active scientific code changed: " + relative)
        receipt = dict(status="COMPLETE", smoke_steps=smoke["steps"],
                       formal_steps=sum(t["steps"] for t in trained), total_steps=100352,
                       trained_seeds=trained, new_vla_calls=0, new_simulator_rollouts=0,
                       source_files_unchanged=True, ppo_evaluation_episodes=result["ppo_episodes"])
        write(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        write(output / "failure.json", dict(status="FAILED_STOPPED", error=str(error),
                                           error_type=type(error).__name__))
        raise

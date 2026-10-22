"""Prepare/search W&B exports from existing logs only; no model or sim imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from AlphaBrain.research.dsol.execution.experiment_tracking import digest, numeric_history, publish

UPDATE_FIELDS = (
    "completed_episodes", "train/learning_rate", "train/entropy_loss",
    "train/policy_gradient_loss", "train/value_loss", "train/approx_kl",
    "train/clip_fraction", "train/loss", "train/explained_variance",
    "train/n_updates", "train/clip_range", "episode_mean_total_reward",
    "episode_mean_queries", "episode_mean_initial_accel", "episode_mean_best_accel",
    "budget4_queries", "budget8_queries", "budget16_queries",
)
TABLE_FIELDS = (
    "cohort", "task", "method", "n", "successes", "success_rate", "queries",
    "harm", "rescue", "harm_rate", "rescue_rate", "replay_n",
    "replay_success_rate", "replay_agreement", "replay_pass_live_fail", "replay_fail_live_pass",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aggregate(rows):
    """Equal task means, preserving counts; harm denominator is all trials."""
    output = []
    for cohort, method in sorted({(r["cohort"], r["method"]) for r in rows}):
        subset = [r for r in rows if (r["cohort"], r["method"]) == (cohort, method)]
        item = dict(cohort=cohort, method=method, tasks=len(subset))
        for key in ("n", "successes", "harm", "rescue"):
            item[key] = sum(r[key] for r in subset)
        for key in ("success_rate", "queries", "harm_rate", "rescue_rate"):
            item[key] = sum(r[key] for r in subset) / len(subset)
        output.append(item)
    return output


def prepare(training, audit, formal_config, output):
    training, audit, output = map(Path, (training, audit, output))
    plan = read(training / "frozen-plan.json")["plan"]
    receipt = read(training / "receipt.json")
    summary = read(audit / "summary.json")
    common = read(audit / "common-horizon.json")
    protocol = read(formal_config)["protocol"]
    if receipt["status"] != "COMPLETE" or summary["status"] != "COMPLETE":
        raise ValueError("Import requires completed source receipts")
    output.mkdir(parents=True, exist_ok=False)
    runs, sources = [], []

    def source(path):
        sources.append(dict(local_path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def figure(path):
        # Fixed reporting figure allowlist; never discover arbitrary images.
        target = output / path.name
        shutil.copyfile(path, target)
        source(path)
        return dict(file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest())

    def add(name, group, config, *, history=None, tables=None, figures=None, final=None):
        data = dict(name=name, group=group, config=config, history=history or [],
                    tables=tables or {}, figures=figures or [], summary=final or {})
        filename = name + ".json"
        write(output / filename, data)
        runs.append(dict(file=filename, sha256=digest(data)))

    for path in (training / "frozen-plan.json", training / "receipt.json",
                 training / "summary.json", audit / "summary.json", audit / "common-horizon.json", Path(formal_config)):
        source(path)
    for seed in plan["training"]["seeds"]:
        path = training / f"seed-{seed}" / "updates.jsonl"
        source(path)
        history = numeric_history([json.loads(line) for line in path.read_text().splitlines()], UPDATE_FIELDS)
        seed_receipt = read(path.parent / "receipt.json")
        source(path.parent / "receipt.json")
        if not history or history[-1]["decision_steps"] != seed_receipt["steps"]:
            raise ValueError("Incomplete training history")
        # Explicit scalar keys, never bulk upload original config/environment.
        config = {k: plan["training"][k] for k in (
            "steps_per_seed", "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "checkpoint_selection")}
        config.update(seed=seed, budgets=plan["search"]["training_budgets"],
                      query_cost=plan["search"]["query_cost"], motion_cost=plan["search"]["motion_cost"],
                      noise_count=plan["search"]["primary_noise_count"],
                      feature_source=plan["feature"]["source"], critic=plan["network"]["critic"],
                      width=plan["network"]["width"], heads=plan["network"]["heads"],
                      history_layers=plan["network"]["history_layers"],
                      scope="Cached static search environment; reward is NOT task success",
                      source_log_sha256=sources[-2]["sha256"])
        add(f"attention-seed-{seed}", "historical-attention-training", config, history=history,
            final={"training_decisions": seed_receipt["steps"], "elapsed_seconds": seed_receipt["elapsed_seconds"]})
    replay = read(training / "summary.json")
    replay_rows = []
    for row in replay["rows"]:
        item = dict(method=row["method"], budget=row["budget"])
        for key in ("success", "queries", "harmed_fraction", "rescued_fraction"):
            metric = row["metrics"][key]
            item.update({key: metric["mean"], key + "_ci95_low": metric["ci95"][0], key + "_ci95_high": metric["ci95"][1]})
        replay_rows.append(item)
    add("cached-label-replay", "historical-cached-replay",
        {"scope": "Final checkpoints replayed against historical task labels; NOT newly executed tasks",
         "training_seeds": plan["training"]["seeds"], "budgets": plan["search"]["training_budgets"]},
        tables={"cached_replay": replay_rows},
        figures=[figure(training.parent / "figures" / "training.png"),
                 figure(training.parent / "figures" / "cached-evaluation.png")])
    table = [{k: row[k] for k in TABLE_FIELDS} for row in summary["table"]]
    if sum(r["n"] for r in table) + len(summary["nominal"]) != protocol["formal_tasks"]:
        raise ValueError("A/B trial counts disagree with protocol")
    config = {k: protocol[k] for k in ("query_budget", "scoring_noises", "batch_size", "replan_steps", "task_horizon", "execution_seeds", "source_commit")}
    config.update(actor_seed=protocol["actor"]["seed"], actor_sha256=protocol["actor"]["sha256"],
                  scope=summary["limitation"], A="New initials of RL-training tasks",
                  B="Tasks unseen by search RL, not necessarily unseen by VLA",
                  camera="Fixed external-camera catalog97; not physical head motion",
                  harm_denominator="All paired trials; not only initially successful trials")
    overview = aggregate(table)
    add("fresh-simulator-ab", "historical-fresh-simulator-evaluation", config,
        tables={"AB_overview": overview, "AB_by_task": table},
        figures=[figure(audit / "A-tasks.png"), figure(audit / "B-tasks.png")],
        final={**{f"{r['cohort']}/{r['method']}/{k}": r[k] for r in overview
                  for k in ("success_rate", "queries", "harm_rate", "n", "successes")},
               "canonical_B_trials": len(summary["nominal"]),
               "canonical_B_successes": sum(r["success"] for r in summary["nominal"]),
               "source_total_tasks": protocol["formal_tasks"]})
    add("matched-horizon-audit", "historical-replay-audit",
        {"scope": common["alignment_scope"], "note": "Residual mismatches preserved; not an extra experiment"},
        tables={"common_horizon_by_task": [{k: row[k] for k in TABLE_FIELDS} for row in common["table"]]},
        figures=[figure(audit / "A-replay-common-horizon.png")],
        final={"paired_records": common["records"], "remaining_disagreements": common["remaining_disagreements"],
               "selection_mismatch": summary["alignment"]["selected_mismatch"]})
    write(output / "bundle.json", dict(schema=1, runs=runs, contents="metrics, curated config and reporting figures only"))
    # This provenance file is local-only and is NEVER consumed by publish().
    write(output / "local-provenance.json", dict(sources=sources, new_training=False, new_model_calls=0))
    return dict(runs=len(runs), training_updates=sum(len(read(output / r["file"])["history"]) for r in runs),
                formal_tasks=protocol["formal_tasks"], output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("training", "audit", "formal-config", "output"):
        prep.add_argument("--" + key, required=True)
    export = sub.add_parser("publish")
    for key in ("bundle", "output", "entity"):
        export.add_argument("--" + key, required=True)
    export.add_argument("--project", default="dsol-paper1")
    export.add_argument("--mode", choices=("offline", "online"), default="offline")
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(json.dumps((prepare if command == "prepare" else publish)(**args), indent=2))


if __name__ == "__main__":
    main()

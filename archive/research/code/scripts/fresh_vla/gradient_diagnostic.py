from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from toy_branching_flow import (
    METHODS,
    ToyConfig,
    TinyBranchingFlow,
    flow_loss,
    make_generator,
    sample_batch,
    sample_flow_time,
    supervision_horizon,
)


def fit_model(method: str, seed: int, cfg: ToyConfig, args: argparse.Namespace) -> TinyBranchingFlow:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = TinyBranchingFlow(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    data_generator = make_generator(device, seed + 10_000)
    flow_generator = make_generator(device, seed + 20_000)
    mask_generator = make_generator(device, seed + 30_000)
    method_cfg = METHODS[method]
    for _ in range(args.train_steps):
        context, actions, _, oracle = sample_batch(args.batch_size, cfg, device, generator=data_generator)
        mask_mode = str(method_cfg["mask"])
        horizon = supervision_horizon(mask_mode, oracle, cfg, generator=mask_generator)
        loss = flow_loss(
            model,
            context,
            actions,
            horizon,
            mask_mode=mask_mode,
            tail_weight=float(method_cfg["tail_weight"]),
            generator=flow_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _flatten_gradients(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([gradient.reshape(-1) for gradient in gradients if gradient is not None])


def diagnose(
    model: TinyBranchingFlow,
    cfg: ToyConfig,
    *,
    seed: int,
    boundary: str,
    batches: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    device = next(model.parameters()).device
    data_generator = make_generator(device, seed + 120_000)
    flow_generator = make_generator(device, seed + 130_000)
    groups = {
        "action_output": [parameter for name, parameter in model.named_parameters() if name.startswith("output")],
        "last_action_layer": [
            parameter for name, parameter in model.named_parameters() if name.startswith("transformer.layers.1")
        ],
    }
    values = {name: {"cosine": [], "prefix_norm": [], "suffix_norm": []} for name in groups}
    for _ in range(batches):
        sample_cfg = ToyConfig(branch_strength=0.0) if boundary == "deterministic_midpoint" else cfg
        context, actions, _, oracle = sample_batch(batch_size, sample_cfg, device, generator=data_generator)
        if boundary == "oracle":
            horizons = oracle
        elif boundary == "gripper":
            horizons = torch.full_like(oracle, cfg.gripper_event_horizon)
        elif boundary == "deterministic_midpoint":
            horizons = torch.full_like(oracle, cfg.horizon // 2)
        else:
            raise ValueError(f"unknown boundary: {boundary}")

        noise = torch.randn(actions.shape, device=device, generator=flow_generator)
        flow_time = sample_flow_time(batch_size, device, generator=flow_generator)
        noisy = flow_time[:, None, None] * noise + (1.0 - flow_time[:, None, None]) * actions
        per_step = (model(context, noisy, flow_time) - (noise - actions)).square().mean(dim=-1)
        steps = torch.arange(cfg.horizon, device=device)[None, :]
        prefix_mask = steps < horizons[:, None]
        suffix_mask = ~prefix_mask
        prefix_loss = (per_step * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)
        suffix_loss = (per_step * suffix_mask).sum() / suffix_mask.sum().clamp_min(1)
        for group_index, (name, parameters) in enumerate(groups.items()):
            prefix_gradient = _flatten_gradients(prefix_loss, parameters, retain_graph=True)
            suffix_gradient = _flatten_gradients(
                suffix_loss,
                parameters,
                retain_graph=group_index < len(groups) - 1,
            )
            cosine = torch.nn.functional.cosine_similarity(prefix_gradient, suffix_gradient, dim=0)
            values[name]["cosine"].append(float(cosine.detach().cpu()))
            values[name]["prefix_norm"].append(float(prefix_gradient.norm().detach().cpu()))
            values[name]["suffix_norm"].append(float(suffix_gradient.norm().detach().cpu()))
    return {
        name: {metric: statistics.mean(samples) for metric, samples in metrics.items()}
        for name, metrics in values.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FRESH prefix/suffix gradient conflict diagnostic")
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--diagnostic-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("/share/longjunyu/fresh-vla/toy/gradient-diagnostic.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ToyConfig(branch_strength=1.0)
    rows = []
    for method in ("full_h", "oracle_soft010"):
        for seed in args.seeds:
            model = fit_model(method, seed, cfg, args)
            for boundary in ("oracle", "gripper", "deterministic_midpoint"):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "boundary": boundary,
                        "groups": diagnose(
                            model,
                            cfg,
                            seed=seed,
                            boundary=boundary,
                            batches=args.diagnostic_batches,
                            batch_size=args.batch_size,
                        ),
                    }
                )
            del model

    summary = {}
    for method in ("full_h", "oracle_soft010"):
        summary[method] = {}
        for boundary in ("oracle", "gripper", "deterministic_midpoint"):
            summary[method][boundary] = {}
            selected = [row for row in rows if row["method"] == method and row["boundary"] == boundary]
            for group in ("action_output", "last_action_layer"):
                summary[method][boundary][group] = {
                    metric: statistics.mean(row["groups"][group][metric] for row in selected)
                    for metric in ("cosine", "prefix_norm", "suffix_norm")
                }
    payload = {"config": asdict(cfg), "args": {**vars(args), "output": str(args.output)}, "rows": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

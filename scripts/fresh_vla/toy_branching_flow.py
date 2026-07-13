from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from AlphaBrain.model.modules.action_model.fresh_loss import feedback_weighted_flow_loss


@dataclass(frozen=True)
class ToyConfig:
    horizon: int = 12
    action_dim: int = 2
    context_dim: int = 4
    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    fixed_execution_horizon: int = 3
    branch_strength: float = 1.0


METHODS = {
    "full": {"mask": "oracle", "tail_weight": 1.0},
    "short": {"mask": "short", "tail_weight": 0.0},
    "random": {"mask": "random", "tail_weight": 0.0},
    "event": {"mask": "event", "tail_weight": 0.0},
    "oracle_hard": {"mask": "oracle", "tail_weight": 0.0},
    "oracle_soft_005": {"mask": "oracle", "tail_weight": 0.05},
    "oracle_soft": {"mask": "oracle", "tail_weight": 0.1},
    "oracle_soft_025": {"mask": "oracle", "tail_weight": 0.25},
}


class TinyBranchingFlow(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_proj = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(3, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.position = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.hidden_dim,
            cfg.heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.layers)
        self.output = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, context: Tensor, noisy_actions: Tensor, flow_time: Tensor) -> Tensor:
        time_features = torch.stack(
            [flow_time, torch.sin(math.pi * flow_time), torch.cos(math.pi * flow_time)],
            dim=-1,
        )
        tokens = self.action_proj(noisy_actions)
        tokens = tokens + self.context_proj(context)[:, None, :]
        tokens = tokens + self.time_proj(time_features)[:, None, :]
        tokens = tokens + self.position[None, :, :]
        return self.output(self.transformer(tokens))


def sample_batch(batch_size: int, cfg: ToyConfig, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    context = torch.randn(batch_size, cfg.context_dim, device=device)
    phase = torch.sigmoid(context[:, 2])
    feedback_horizon = 3 + torch.floor(5 * phase).long()
    feedback_horizon = feedback_horizon.clamp(max=cfg.horizon - 2)
    branch = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.long) * 2 - 1

    steps = torch.arange(cfg.horizon, device=device, dtype=torch.float32)[None, :]
    progress = steps / (cfg.horizon - 1)
    common_x = 0.45 * torch.tanh(context[:, 0:1]) + 0.18 * progress
    common_y = 0.45 * torch.tanh(context[:, 1:2]) - 0.12 * progress + 0.04 * torch.sin(math.pi * progress)
    common = torch.stack([common_x, common_y], dim=-1)

    suffix_step = (steps - feedback_horizon[:, None] + 1).clamp_min(0.0)
    suffix_scale = suffix_step / (cfg.horizon - feedback_horizon[:, None]).clamp_min(1)
    direction = torch.tensor([0.85, -0.65], device=device)
    branch_effect = cfg.branch_strength * branch[:, None, None].float() * suffix_scale[:, :, None] * direction
    actions = common + branch_effect
    return context, actions, common, feedback_horizon


def supervision_horizon(mode: str, oracle: Tensor, cfg: ToyConfig) -> Tensor:
    if mode == "oracle":
        return oracle
    if mode == "random":
        return oracle[torch.randperm(oracle.shape[0], device=oracle.device)]
    if mode == "event":
        return torch.full_like(oracle, cfg.horizon // 2)
    if mode == "short":
        return torch.full_like(oracle, cfg.fixed_execution_horizon)
    raise ValueError(f"unknown mask mode: {mode}")


def sample_flow_time(batch_size: int, device: torch.device) -> Tensor:
    # Inverse CDF for Beta(1.5, 1), matching the OpenPI sampling family.
    return torch.rand(batch_size, device=device).pow(1.0 / 1.5) * 0.999 + 0.001


def flow_loss(model: TinyBranchingFlow, context: Tensor, actions: Tensor, horizons: Tensor, tail_weight: float) -> Tensor:
    noise = torch.randn_like(actions)
    flow_time = sample_flow_time(actions.shape[0], actions.device)
    time_expanded = flow_time[:, None, None]
    noisy_actions = time_expanded * noise + (1.0 - time_expanded) * actions
    target_velocity = noise - actions
    predicted_velocity = model(context, noisy_actions, flow_time)
    per_dim = (predicted_velocity - target_velocity).square()
    loss, _ = feedback_weighted_flow_loss(per_dim, horizons, tail_weight)
    return loss


@torch.no_grad()
def sample_actions(model: TinyBranchingFlow, context: Tensor, cfg: ToyConfig, steps: int = 16) -> Tensor:
    actions = torch.randn(context.shape[0], cfg.horizon, cfg.action_dim, device=context.device)
    dt = -1.0 / steps
    for index in range(steps):
        flow_time = torch.full((context.shape[0],), 1.0 + index * dt, device=context.device)
        actions = actions + dt * model(context, actions, flow_time)
    return actions


def gradient_cosine(model: TinyBranchingFlow, cfg: ToyConfig, batches: int = 6) -> float:
    values = []
    parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(("transformer", "output"))]
    for _ in range(batches):
        context, actions, _, horizons = sample_batch(192, cfg, next(model.parameters()).device)
        noise = torch.randn_like(actions)
        flow_time = sample_flow_time(actions.shape[0], actions.device)
        noisy = flow_time[:, None, None] * noise + (1.0 - flow_time[:, None, None]) * actions
        per_step = (model(context, noisy, flow_time) - (noise - actions)).square().mean(dim=-1)
        step_ids = torch.arange(cfg.horizon, device=actions.device)[None, :]
        prefix_mask = step_ids < horizons[:, None]
        prefix = (per_step * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)
        suffix_mask = ~prefix_mask
        suffix = (per_step * suffix_mask).sum() / suffix_mask.sum().clamp_min(1)
        prefix_grad = torch.autograd.grad(prefix, parameters, retain_graph=True, allow_unused=True)
        suffix_grad = torch.autograd.grad(suffix, parameters, allow_unused=True)
        prefix_flat = torch.cat([grad.reshape(-1) for grad in prefix_grad if grad is not None])
        suffix_flat = torch.cat([grad.reshape(-1) for grad in suffix_grad if grad is not None])
        cosine = torch.nn.functional.cosine_similarity(prefix_flat, suffix_flat, dim=0)
        values.append(float(cosine.detach().cpu()))
    return statistics.mean(values)


@torch.no_grad()
def evaluate(model: TinyBranchingFlow, cfg: ToyConfig, evaluation_size: int = 4096) -> dict[str, float]:
    device = next(model.parameters()).device
    context, _, common, horizons = sample_batch(evaluation_size, cfg, device)
    predicted = sample_actions(model, context, cfg)
    fixed_k = cfg.fixed_execution_horizon

    step_ids = torch.arange(cfg.horizon, device=device)[None, :]
    prefix_mask = step_ids < horizons[:, None]
    prefix_error = (predicted - common).square().mean(dim=-1)
    fixed_k_mse = prefix_error[:, :fixed_k].mean()
    safe_prefix_mse = (prefix_error * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)

    suffix_step = (step_ids.float() - horizons[:, None] + 1).clamp_min(0.0)
    suffix_scale = suffix_step / (cfg.horizon - horizons[:, None]).clamp_min(1)
    direction = torch.tensor([0.85, -0.65], device=device)
    plus = common + cfg.branch_strength * suffix_scale[:, :, None] * direction
    minus = common - cfg.branch_strength * suffix_scale[:, :, None] * direction
    suffix_mask = ~prefix_mask
    plus_error = ((predicted - plus).square().mean(dim=-1) * suffix_mask).sum(dim=1) / suffix_mask.sum(dim=1)
    minus_error = ((predicted - minus).square().mean(dim=-1) * suffix_mask).sum(dim=1) / suffix_mask.sum(dim=1)
    branch_min_mse = torch.minimum(plus_error, minus_error).mean()

    residual = predicted - common
    direction_unit = direction / direction.norm()
    premature = (residual * direction_unit).sum(dim=-1).abs()
    premature_commitment = (premature * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)
    return {
        "fixed_k_prefix_mse": float(fixed_k_mse.cpu()),
        "safe_prefix_mse": float(safe_prefix_mse.cpu()),
        "premature_commitment": float(premature_commitment.cpu()),
        "branch_min_suffix_mse": float(branch_min_mse.cpu()),
    }


def train_method(method: str, seed: int, args: argparse.Namespace, cfg: ToyConfig) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = TinyBranchingFlow(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    method_cfg = METHODS[method]
    started = time.time()

    model.train()
    for step in range(args.train_steps):
        context, actions, _, oracle_horizon = sample_batch(args.batch_size, cfg, device)
        horizon = supervision_horizon(str(method_cfg["mask"]), oracle_horizon, cfg)
        loss = flow_loss(model, context, actions, horizon, float(method_cfg["tail_weight"]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.log_every and (step + 1) % args.log_every == 0:
            print(f"method={method} seed={seed} step={step + 1} loss={loss.item():.6f}", flush=True)

    model.eval()
    metrics = evaluate(model, cfg, args.evaluation_size)
    model.train()
    metrics["gradient_cosine"] = gradient_cosine(model, cfg)
    metrics.update(
        {
            "method": method,
            "seed": seed,
            "train_steps": args.train_steps,
            "elapsed_seconds": time.time() - started,
        }
    )
    return metrics


def aggregate(results: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    summary = {}
    metric_names = ("fixed_k_prefix_mse", "safe_prefix_mse", "premature_commitment", "branch_min_suffix_mse", "gradient_cosine")
    for method in sorted({str(result["method"]) for result in results}):
        rows = [result for result in results if result["method"] == method]
        summary[method] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary[method][f"{metric}_mean"] = statistics.mean(values)
            summary[method][f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-download FRESH-VLA branching flow-matching toy benchmark")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--branch-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("/share/longjunyu/fresh-vla/toy/results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ToyConfig(branch_strength=args.branch_strength)
    results = []
    for method in args.methods:
        for seed in args.seeds:
            result = train_method(method, seed, args, cfg)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    payload = {"config": asdict(cfg), "args": {**vars(args), "output": str(args.output)}, "results": results, "summary": aggregate(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

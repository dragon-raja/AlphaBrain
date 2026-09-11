from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from AlphaBrain.model.modules.action_model.fresh_loss import feedback_weighted_flow_loss
from paired_evaluation import EvaluationIdentity, bootstrap_summary, paired_delta_summary, per_sample_flow_metrics


@dataclass(frozen=True)
class ToyConfig:
    horizon: int = 12
    action_dim: int = 2
    context_dim: int = 4
    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    fixed_execution_horizon: int = 3
    short_training_horizon: int = 5
    gripper_event_horizon: int = 4
    branch_strength: float = 1.0


METHODS = {
    "full_h": {"mask": "oracle", "tail_weight": 1.0},
    "random_soft010": {"mask": "random", "tail_weight": 0.1},
    "gripper_soft010": {"mask": "gripper", "tail_weight": 0.1},
    "oracle_soft010": {"mask": "oracle", "tail_weight": 0.1},
    "oracle_hard000": {"mask": "oracle", "tail_weight": 0.0},
    "short_h": {"mask": "short", "tail_weight": 0.0},
    "remac_prefix_mask_control": {"mask": "remac_prefix", "tail_weight": 0.1},
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


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def sample_batch(
    batch_size: int,
    cfg: ToyConfig,
    device: torch.device,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    context = torch.randn(batch_size, cfg.context_dim, device=device, generator=generator)
    phase = torch.sigmoid(context[:, 2])
    feedback_horizon = 3 + torch.floor(5 * phase).long()
    feedback_horizon = feedback_horizon.clamp(max=cfg.horizon - 2)
    if cfg.branch_strength == 0.0:
        feedback_horizon = torch.full_like(feedback_horizon, cfg.horizon)
    branch = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.long, generator=generator) * 2 - 1
    # Hidden physical outcome is deliberately absent from policy conditioning.
    context[:, 3] = 0.0

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


def supervision_horizon(
    mode: str,
    oracle: Tensor,
    cfg: ToyConfig,
    *,
    generator: torch.Generator,
) -> Tensor:
    if mode == "oracle":
        return oracle
    if mode == "random":
        return oracle[torch.randperm(oracle.shape[0], device=oracle.device, generator=generator)]
    if mode == "gripper":
        return torch.full_like(oracle, cfg.gripper_event_horizon)
    if mode == "short":
        return torch.full_like(oracle, cfg.short_training_horizon)
    if mode == "remac_prefix":
        return oracle
    raise ValueError(f"unknown mask mode: {mode}")


def sample_flow_time(batch_size: int, device: torch.device, *, generator: torch.Generator) -> Tensor:
    # Inverse CDF for Beta(1.5, 1), matching the OpenPI sampling family.
    return torch.rand(batch_size, device=device, generator=generator).pow(1.0 / 1.5) * 0.999 + 0.001


def flow_loss(
    model: TinyBranchingFlow,
    context: Tensor,
    actions: Tensor,
    horizons: Tensor,
    *,
    mask_mode: str,
    tail_weight: float,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    flow_time = sample_flow_time(actions.shape[0], actions.device, generator=generator)
    time_expanded = flow_time[:, None, None]
    noisy_actions = time_expanded * noise + (1.0 - time_expanded) * actions
    target_velocity = noise - actions
    predicted_velocity = model(context, noisy_actions, flow_time)
    per_dim = (predicted_velocity - target_velocity).square()
    weighting_mode = "prefix_control" if mask_mode == "remac_prefix" else "suffix"
    loss, _ = feedback_weighted_flow_loss(
        per_dim,
        horizons,
        tail_weight,
        weighting_mode=weighting_mode,
    )
    return loss


@torch.no_grad()
def sample_actions(model: TinyBranchingFlow, context: Tensor, cfg: ToyConfig, initial_actions: Tensor, steps: int = 16) -> Tensor:
    actions = initial_actions.clone()
    dt = -1.0 / steps
    for index in range(steps):
        flow_time = torch.full((context.shape[0],), 1.0 + index * dt, device=context.device)
        actions = actions + dt * model(context, actions, flow_time)
    return actions


def gradient_cosine(model: TinyBranchingFlow, cfg: ToyConfig, seed: int, batches: int = 6) -> float:
    values = []
    device = next(model.parameters()).device
    data_generator = make_generator(device, seed + 50_000)
    flow_generator = make_generator(device, seed + 60_000)
    parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(("transformer", "output"))]
    for _ in range(batches):
        context, actions, _, horizons = sample_batch(192, cfg, device, generator=data_generator)
        noise = torch.randn(actions.shape, device=device, generator=flow_generator)
        flow_time = sample_flow_time(actions.shape[0], device, generator=flow_generator)
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
def evaluate(
    model: TinyBranchingFlow,
    cfg: ToyConfig,
    *,
    seed: int,
    evaluation_size: int,
    multimodal_contexts: int,
    samples_per_context: int,
) -> tuple[dict[str, float], dict[str, list[float]], list[str], str]:
    device = next(model.parameters()).device
    data_generator = make_generator(device, seed + 70_000)
    inference_generator = make_generator(device, seed + 80_000)
    flow_generator = make_generator(device, seed + 90_000)
    context, actions, common, horizons = sample_batch(evaluation_size, cfg, device, generator=data_generator)
    initial_actions = torch.randn(actions.shape, device=device, generator=inference_generator)
    predicted = sample_actions(model, context, cfg, initial_actions)
    fixed_k = cfg.fixed_execution_horizon

    step_ids = torch.arange(cfg.horizon, device=device)[None, :]
    prefix_mask = step_ids < horizons[:, None]
    prefix_error = (predicted - common).square().mean(dim=-1)
    fixed_k_mse = prefix_error[:, :fixed_k].mean(dim=1)
    safe_prefix_mse = (prefix_error * prefix_mask).sum(dim=1) / prefix_mask.sum(dim=1).clamp_min(1)

    suffix_step = (step_ids.float() - horizons[:, None] + 1).clamp_min(0.0)
    suffix_scale = suffix_step / (cfg.horizon - horizons[:, None]).clamp_min(1)
    direction = torch.tensor([0.85, -0.65], device=device)
    plus = common + cfg.branch_strength * suffix_scale[:, :, None] * direction
    minus = common - cfg.branch_strength * suffix_scale[:, :, None] * direction
    suffix_mask = ~prefix_mask
    suffix_count = suffix_mask.sum(dim=1)
    plus_error = ((predicted - plus).square().mean(dim=-1) * suffix_mask).sum(dim=1) / suffix_count.clamp_min(1)
    minus_error = ((predicted - minus).square().mean(dim=-1) * suffix_mask).sum(dim=1) / suffix_count.clamp_min(1)
    branch_min_mse = torch.minimum(plus_error, minus_error)

    residual = predicted - common
    direction_unit = direction / direction.norm()
    premature = (residual * direction_unit).sum(dim=-1).abs()
    premature_commitment = (premature * prefix_mask).sum(dim=1) / prefix_mask.sum(dim=1).clamp_min(1)

    flow_noise = torch.randn(actions.shape, device=device, generator=flow_generator)
    flow_time = sample_flow_time(evaluation_size, device, generator=flow_generator)
    noisy = flow_time[:, None, None] * flow_noise + (1.0 - flow_time[:, None, None]) * actions
    per_step_flow = (model(context, noisy, flow_time) - (flow_noise - actions)).square().mean(dim=-1)
    flow_rows = per_sample_flow_metrics(
        per_step_flow.float().cpu().numpy(),
        horizons.cpu().numpy(),
        fixed_k=fixed_k,
    )
    suffix_values = [row["suffix"] for row in flow_rows]
    has_suffix = any(value is not None for value in suffix_values)

    sample_ids = [f"seed-{seed}-sample-{index}" for index in range(evaluation_size)]
    identity = EvaluationIdentity(
        sample_ids=tuple(sample_ids),
        flow_times=flow_time.float().cpu().numpy(),
        noise=flow_noise.float().cpu().numpy(),
        action_normalization={"mean": [0.0] * cfg.action_dim, "std": [1.0] * cfg.action_dim},
        sample_content=torch.cat([context, actions.flatten(1)], dim=1).float().cpu().numpy(),
    )

    per_sample = {
        "fixed_k_prefix_mse": fixed_k_mse.float().cpu().tolist(),
        "safe_prefix_mse": safe_prefix_mse.float().cpu().tolist(),
        "premature_commitment": premature_commitment.float().cpu().tolist(),
        "branch_min_suffix_mse": branch_min_mse.float().cpu().tolist(),
        "flow_fixed_k3_mse": [float(row["fixed_k"]) for row in flow_rows],
        "flow_oracle_prefix_mse": [float(row["oracle_prefix"]) for row in flow_rows],
        "flow_suffix_mse": [0.0 if value is None else float(value) for value in suffix_values],
        "flow_full_mse": [float(row["full"]) for row in flow_rows],
    }
    metrics = {name: statistics.mean(values) for name, values in per_sample.items()}
    metrics["flow_suffix_available"] = float(has_suffix)
    metrics.update(
        multimodal_evaluate(
            model,
            cfg,
            seed=seed,
            context_count=multimodal_contexts,
            samples_per_context=samples_per_context,
        )
    )
    return metrics, per_sample, sample_ids, identity.fingerprint()


@torch.no_grad()
def multimodal_evaluate(
    model: TinyBranchingFlow,
    cfg: ToyConfig,
    *,
    seed: int,
    context_count: int,
    samples_per_context: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    data_generator = make_generator(device, seed + 100_000)
    inference_generator = make_generator(device, seed + 110_000)
    context, _, common, horizons = sample_batch(context_count, cfg, device, generator=data_generator)
    repeated_context = context.repeat_interleave(samples_per_context, dim=0)
    initial = torch.randn(
        context_count * samples_per_context,
        cfg.horizon,
        cfg.action_dim,
        device=device,
        generator=inference_generator,
    )
    predicted = sample_actions(model, repeated_context, cfg, initial).reshape(
        context_count,
        samples_per_context,
        cfg.horizon,
        cfg.action_dim,
    )
    residual = predicted - common[:, None, :, :]
    direction = torch.tensor([0.85, -0.65], device=device)
    direction = direction / direction.norm()
    projection = (residual * direction).sum(dim=-1)
    step_ids = torch.arange(cfg.horizon, device=device)[None, :]
    prefix_mask = step_ids < horizons[:, None]

    sampling_variance = predicted.var(dim=1, unbiased=False).mean(dim=-1)
    prefix_variance = (sampling_variance * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)
    premature = (projection.abs() * prefix_mask[:, None, :]).sum() / (
        prefix_mask.sum() * samples_per_context
    ).clamp_min(1)
    terminal = projection[:, :, -1]
    margin = max(0.05, 0.2 * cfg.branch_strength)
    positive = (terminal > margin).float().mean(dim=1)
    negative = (terminal < -margin).float().mean(dim=1)
    mode_coverage = ((positive >= 0.1) & (negative >= 0.1)).float().mean()
    mode_balance = (1.0 - (positive - negative).abs()).clamp_min(0.0).mean()
    return {
        "multimodal_prefix_variance": float(prefix_variance.cpu()),
        "multimodal_premature_commitment": float(premature.cpu()),
        "suffix_mode_coverage": float(mode_coverage.cpu()),
        "suffix_mode_balance": float(mode_balance.cpu()),
    }


def train_method(method: str, seed: int, args: argparse.Namespace, cfg: ToyConfig) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = TinyBranchingFlow(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    method_cfg = METHODS[method]
    data_generator = make_generator(device, seed + 10_000)
    flow_generator = make_generator(device, seed + 20_000)
    mask_generator = make_generator(device, seed + 30_000)
    started = time.time()

    model.train()
    for step in range(args.train_steps):
        context, actions, _, oracle_horizon = sample_batch(
            args.batch_size,
            cfg,
            device,
            generator=data_generator,
        )
        mask_mode = str(method_cfg["mask"])
        horizon = supervision_horizon(mask_mode, oracle_horizon, cfg, generator=mask_generator)
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
        if args.log_every and (step + 1) % args.log_every == 0:
            print(f"method={method} seed={seed} step={step + 1} loss={loss.item():.6f}", flush=True)

    model.eval()
    metrics, per_sample, sample_ids, fingerprint = evaluate(
        model,
        cfg,
        seed=seed,
        evaluation_size=args.evaluation_size,
        multimodal_contexts=args.multimodal_contexts,
        samples_per_context=args.samples_per_context,
    )
    model.train()
    metrics["gradient_cosine"] = gradient_cosine(model, cfg, seed)
    metrics.update(
        {
            "method": method,
            "seed": seed,
            "train_steps": args.train_steps,
            "elapsed_seconds": time.time() - started,
            "evaluation_fingerprint": fingerprint,
            "sample_ids": sample_ids,
            "per_sample": per_sample,
        }
    )
    return metrics


def aggregate(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    metric_names = (
        "fixed_k_prefix_mse",
        "safe_prefix_mse",
        "premature_commitment",
        "branch_min_suffix_mse",
        "flow_fixed_k3_mse",
        "flow_oracle_prefix_mse",
        "flow_suffix_mse",
        "flow_full_mse",
        "multimodal_prefix_variance",
        "multimodal_premature_commitment",
        "suffix_mode_coverage",
        "suffix_mode_balance",
        "gradient_cosine",
    )
    for method in sorted({str(result["method"]) for result in results}):
        rows = [result for result in results if result["method"] == method]
        summary[method] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary[method][f"{metric}_mean"] = statistics.mean(values)
            summary[method][f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def paired_aggregate(results: list[dict[str, object]], bootstrap_samples: int) -> dict[str, object]:
    baseline_by_seed = {int(row["seed"]): row for row in results if row["method"] == "full_h"}
    paired = {}
    for method in sorted({str(row["method"]) for row in results} - {"full_h"}):
        candidates = {int(row["seed"]): row for row in results if row["method"] == method}
        paired[method] = {}
        for metric in (
            "fixed_k_prefix_mse",
            "safe_prefix_mse",
            "premature_commitment",
            "flow_fixed_k3_mse",
            "flow_oracle_prefix_mse",
            "flow_suffix_mse",
            "flow_full_mse",
        ):
            combined_baseline = []
            combined_candidate = []
            fingerprints = []
            seed_mean_deltas = []
            per_seed = {}
            for seed in sorted(baseline_by_seed):
                baseline = baseline_by_seed[seed]
                candidate = candidates[seed]
                fingerprint = str(baseline["evaluation_fingerprint"])
                if fingerprint != candidate["evaluation_fingerprint"]:
                    raise ValueError(f"evaluation identity mismatch for method={method} seed={seed}")
                fingerprints.append(fingerprint)
                baseline_values = baseline["per_sample"][metric]
                candidate_values = candidate["per_sample"][metric]
                sample_ids = baseline["sample_ids"]
                baseline_rows = [
                    {
                        "sample_id": sample_id,
                        "evaluation_fingerprint": fingerprint,
                        metric: value,
                    }
                    for sample_id, value in zip(sample_ids, baseline_values, strict=True)
                ]
                candidate_rows = [
                    {
                        "sample_id": sample_id,
                        "evaluation_fingerprint": fingerprint,
                        metric: value,
                    }
                    for sample_id, value in zip(sample_ids, candidate_values, strict=True)
                ]
                per_seed[str(seed)] = paired_delta_summary(
                    baseline_rows,
                    candidate_rows,
                    metric=metric,
                    expected_fingerprint=fingerprint,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
                per_seed[str(seed)].pop("paired_deltas")
                seed_mean_deltas.append(per_seed[str(seed)]["candidate_minus_baseline"]["mean"])
                combined_baseline.extend(baseline_rows)
                combined_candidate.extend(candidate_rows)

            combined_fingerprint = hashlib.sha256("".join(fingerprints).encode()).hexdigest()
            for row in combined_baseline + combined_candidate:
                row["sample_id"] = f"{row['evaluation_fingerprint'][:8]}:{row['sample_id']}"
                row["evaluation_fingerprint"] = combined_fingerprint
            pooled = paired_delta_summary(
                combined_baseline,
                combined_candidate,
                metric=metric,
                expected_fingerprint=combined_fingerprint,
                bootstrap_samples=bootstrap_samples,
                seed=1234,
            )
            pooled.pop("paired_deltas")
            paired[method][metric] = {
                "per_seed": per_seed,
                "across_seeds": {
                    "seed_mean_delta": bootstrap_summary(
                        seed_mean_deltas,
                        bootstrap_samples=bootstrap_samples,
                        seed=1234,
                    ),
                    "pooled_sample_delta": pooled,
                },
            }
    return paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-download FRESH-VLA branching flow-matching toy benchmark")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-size", type=int, default=4096)
    parser.add_argument("--multimodal-contexts", type=int, default=64)
    parser.add_argument("--samples-per-context", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--branch-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/toy/counterfactual-paired-results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ToyConfig(branch_strength=args.branch_strength)
    results = []
    for method in args.methods:
        for seed in args.seeds:
            result = train_method(method, seed, args, cfg)
            results.append(result)
            compact = {key: value for key, value in result.items() if key not in {"sample_ids", "per_sample"}}
            print(json.dumps(compact, sort_keys=True), flush=True)

    payload = {
        "config": asdict(cfg),
        "args": {**vars(args), "output": str(args.output)},
        "results": results,
        "summary": aggregate(results),
        "paired_vs_full_h": paired_aggregate(results, args.bootstrap_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

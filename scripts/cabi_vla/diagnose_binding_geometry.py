from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.action_model.cabi_binding import select_binding_state
from evaluate_libero_bind_offline import CORNER_ORDER, corner_example


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    value = 1.0 - F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-6)
    return float(value.mean().item())


def cosine_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    value = F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-6)
    return float(value.mean().item())


def normalized_entropy(attention: torch.Tensor, mask: torch.Tensor) -> float:
    probabilities = attention.float().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    support = mask.sum(dim=-1).clamp_min(2).float().log()[:, None]
    return float((entropy / support).mean().item())


def relative_norm(delta: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = delta.float().flatten(1).norm(dim=-1)
    denominator = reference.float().flatten(1).norm(dim=-1).clamp_min(1e-8)
    return float((numerator / denominator).mean().item())


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def diagnose_tetrad(model, examples: list[dict]) -> dict[str, float]:
    prefix, pad_mask, _ = model._prepare_prefix(examples)
    language_token_count = int(
        getattr(getattr(model.config.framework, "paligemma", None), "max_token_len", 48)
    )
    adapted, state = model._apply_cabi_prefix(
        prefix,
        pad_mask,
        language_token_count=language_token_count,
    )
    if state is None:
        raise ValueError("checkpoint does not enable CABI")

    base = select_binding_state(state, [0])
    source = select_binding_state(state, [1])
    target = select_binding_state(state, [2])
    fourth = select_binding_state(state, [3])
    source_swap = model.cabi_adapter.transport(base, source, [0])
    target_swap = model.cabi_adapter.transport(base, target, [1])
    source_then_target = model.cabi_adapter.transport(source_swap, fourth, [1])
    target_then_source = model.cabi_adapter.transport(target_swap, fourth, [0])

    source_change = cosine_distance(base.role_states[:, 0], source.role_states[:, 0])
    source_leakage = cosine_distance(base.role_states[:, 1], source.role_states[:, 1])
    target_change = cosine_distance(base.role_states[:, 1], target.role_states[:, 1])
    target_leakage = cosine_distance(base.role_states[:, 0], target.role_states[:, 0])
    return {
        "source_role_change": source_change,
        "source_specificity_leakage": source_leakage,
        "source_specificity_margin": source_change - source_leakage,
        "target_role_change": target_change,
        "target_specificity_leakage": target_leakage,
        "target_specificity_margin": target_change - target_leakage,
        "source_target_role_distance": cosine_distance(
            state.role_states[:, 0], state.role_states[:, 1]
        ),
        "visual_attention_overlap": cosine_overlap(
            state.visual_attention[:, 0], state.visual_attention[:, 1]
        ),
        "language_attention_overlap": cosine_overlap(
            state.language_attention[:, 0], state.language_attention[:, 1]
        ),
        "write_attention_overlap": cosine_overlap(
            state.write_attention[:, 0], state.write_attention[:, 1]
        ),
        "visual_attention_entropy": normalized_entropy(
            state.visual_attention, state.vision_mask
        ),
        "language_attention_entropy": normalized_entropy(
            state.language_attention, state.language_mask
        ),
        "write_attention_entropy": normalized_entropy(
            state.write_attention, state.language_mask
        ),
        "normal_adapter_relative_norm": relative_norm(adapted - prefix, prefix),
        "source_transport_relative_norm": relative_norm(
            source_swap.tokens - base.tokens, base.tokens
        ),
        "target_transport_relative_norm": relative_norm(
            target_swap.tokens - base.tokens, base.tokens
        ),
        "source_swap_role_error": cosine_distance(
            source_swap.role_states[:, 0], source.role_states[:, 0]
        ),
        "target_swap_role_error": cosine_distance(
            target_swap.role_states[:, 1], target.role_states[:, 1]
        ),
        "fourth_anchor_role_error": 0.5
        * (
            cosine_distance(source_then_target.role_states, fourth.role_states)
            + cosine_distance(target_then_source.role_states, fourth.role_states)
        ),
        "commutator_role_error": cosine_distance(
            source_then_target.role_states,
            target_then_source.role_states,
        ),
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit learned CABI role geometry")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tetrads", type=int)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    manifest = json.loads((args.training_view / "manifest.json").read_text())
    tetrads = list(manifest["tetrads"])
    if args.max_tetrads is not None:
        tetrads = tetrads[: args.max_tetrads]
    if not tetrads:
        raise ValueError("no tetrads selected")

    model = BaseFramework.from_pretrained(str(args.checkpoint))
    model = model.to(torch.bfloat16).to(args.device).eval()
    model.gripper_remap = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    with np.load(args.training_view / "anchors.npz", allow_pickle=False) as anchors:
        with torch.inference_mode():
            for index, tetrad in enumerate(tetrads, start=1):
                state_index = int(tetrad["canonical_state_index"])
                corners = tetrad["corners"]
                examples = [
                    corner_example(
                        anchors,
                        manifest["edge_instructions"],
                        corners[name],
                        state_index,
                    )
                    for name in CORNER_ORDER
                ]
                rows.append(
                    {
                        "tetrad_id": tetrad["tetrad_id"],
                        "canonical_state_index": state_index,
                        "withheld_edge": str(corners["fourth_anchor"]["instruction_edge"]),
                        **diagnose_tetrad(model, examples),
                    }
                )
                if index % 8 == 0 or index == len(tetrads):
                    print(f"binding_geometry {index}/{len(tetrads)}", flush=True)

    metric_names = [name for name in rows[0] if name not in {
        "tetrad_id", "canonical_state_index", "withheld_edge"
    }]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "training_view": str(args.training_view),
        "tetrad_count": len(rows),
        "action_labels_loaded": False,
        "summary": {
            name: float(np.mean([row[name] for row in rows])) for name in metric_names
        },
        "rows": rows,
    }
    atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), **payload["summary"]}))


if __name__ == "__main__":
    main()

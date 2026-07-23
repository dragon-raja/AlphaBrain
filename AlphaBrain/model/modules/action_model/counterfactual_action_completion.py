"""Action-label-free completion for phase-aligned compositional tetrads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


CORNER_NAMES = ("base", "source_anchor", "target_anchor", "fourth_anchor")


def _corner_indices(
    grouped: Mapping[str, Sequence[int]],
    corner: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if corner not in grouped:
        raise ValueError(f"missing counterfactual corner: {corner}")
    return torch.as_tensor(tuple(grouped[corner]), dtype=torch.long, device=device)


def counterfactual_fourth_actions(
    actions: torch.Tensor,
    grouped: Mapping[str, Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete the fourth action with the observed three-corner parallelogram.

    The action tensor may already be normalized. Any affine normalization
    commutes with ``source + target - base``, so the completion remains valid in
    either raw or normalized action coordinates.
    """

    if actions.ndim != 3:
        raise ValueError(f"actions must be [B, H, D], got {tuple(actions.shape)}")
    indices = {
        corner: _corner_indices(grouped, corner, device=actions.device)
        for corner in CORNER_NAMES
    }
    counts = {corner: len(value) for corner, value in indices.items()}
    if len(set(counts.values())) != 1 or not counts["base"]:
        raise ValueError(f"counterfactual corners must be non-empty and aligned: {counts}")
    for corner, value in indices.items():
        if torch.any(value < 0) or torch.any(value >= len(actions)):
            raise IndexError(f"{corner} indices are outside the action batch")

    base = actions.index_select(0, indices["base"])
    source = actions.index_select(0, indices["source_anchor"])
    target = actions.index_select(0, indices["target_anchor"])
    pseudo_fourth = source + target - base
    return pseudo_fourth, indices["fourth_anchor"]

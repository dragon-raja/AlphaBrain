"""Strict same-state pair handling for flow-matching consistency training."""

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def dsol_pair_groups(
    examples: Sequence[Mapping],
    *,
    marker_key: str = "dsol_pair_objective",
) -> list[tuple[int, int]]:
    """Return validated index pairs explicitly marked for the DSOL objective."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if bool(example.get(marker_key, False)):
            pair_id = example.get("dsol_pair_id")
            if not pair_id:
                raise ValueError("DSOL pair objective sample is missing dsol_pair_id")
            grouped[str(pair_id)].append(index)

    pairs = []
    for pair_id, indices in grouped.items():
        if len(indices) != 2:
            raise ValueError(
                f"DSOL pair {pair_id!r} must occur exactly twice, got {len(indices)}"
            )
        roles = {str(examples[index].get("dsol_pair_role", "")) for index in indices}
        if len(roles) != 2:
            raise ValueError(
                f"DSOL pair {pair_id!r} must have two distinct roles, got {sorted(roles)}"
            )
        pairs.append((indices[0], indices[1]))
    return pairs


def validate_paired_actions(
    actions: torch.Tensor,
    groups: Iterable[tuple[int, int]],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Fail closed if a purported same-state pair has different action targets."""

    for left, right in groups:
        if not torch.allclose(actions[left], actions[right], atol=atol, rtol=rtol):
            max_error = (actions[left] - actions[right]).abs().max().item()
            raise ValueError(
                "DSOL same-state pair has mismatched normalized actions: "
                f"indices=({left}, {right}), max_abs_error={max_error:.8g}"
            )


def share_pair_noise_time(
    noise: torch.Tensor,
    time: torch.Tensor,
    groups: Iterable[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use identical flow noise and time for both observations in every pair."""

    shared_noise = noise.clone()
    shared_time = time.clone()
    for left, right in groups:
        shared_noise[right] = shared_noise[left]
        shared_time[right] = shared_time[left]
    return shared_noise, shared_time


def paired_prediction_consistency(
    prediction: torch.Tensor,
    groups: Sequence[tuple[int, int]],
    *,
    action_dim_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean squared and root-mean-squared paired velocity disagreement."""

    if not groups:
        raise ValueError("paired prediction consistency requires at least one pair")
    values = prediction
    if action_dim_mask is not None:
        values = values[:, :, action_dim_mask]
    losses = [F.mse_loss(values[left], values[right]) for left, right in groups]
    mse = torch.stack(losses).mean()
    return mse, torch.sqrt(mse.detach())

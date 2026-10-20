"""Candidate-shared search networks; no trainer, simulator or model loading.

Actor input is deployable observation only. Privileged Accel is an explicit
separate critic argument, never a Gym observation key. Networks share no weights.
The SB3 integration and live-policy admission are deliberately not implied by
these architecture components.
"""

import torch
from torch import nn


class SearchHistoryEncoder(nn.Module):
    """Two-layer set attention over observed candidates, with explicit order."""

    def __init__(self, visual_dim, width=128, heads=4, layers=2):
        super().__init__()
        if visual_dim < 1 or width < 1 or heads < 1 or width % heads or layers < 1:
            raise ValueError("Positive dimensions and divisible attention width required")
        self.visual_dim = visual_dim
        self.visual = nn.Sequential(nn.Linear(visual_dim, width), nn.LayerNorm(width), nn.GELU())
        self.record = nn.Linear(width + 12 + 2 + 4, width)
        self.context = nn.Linear(width + 6, width)
        self.history = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, heads, dim_feedforward=width * 2, dropout=0.,
                activation="gelu", batch_first=True, norm_first=False),
            num_layers=layers, enable_nested_tensor=False)

    def forward(self, observation):
        allowed = {"geometry", "visual", "scores", "flags", "wrist", "global", "action_mask"}
        if set(observation) != allowed:
            raise ValueError("Actor observation schema must contain no privileged keys")
        geom, visual = observation["geometry"], observation["visual"]
        if geom.ndim != 3 or geom.shape[-1] != 12:
            raise ValueError("Geometry must be [batch, candidates, 12]")
        batch, count = geom.shape[:2]
        shapes = {"visual": (batch, count, self.visual_dim), "scores": (batch, count, 2),
                  "flags": (batch, count, 4), "wrist": (batch, self.visual_dim),
                  "global": (batch, 6), "action_mask": (batch, count + 1)}
        if any(tuple(observation[k].shape) != v for k, v in shapes.items()):
            raise ValueError("Search observation shape mismatch")
        if any(not torch.isfinite(x).all() for x in observation.values()):
            raise ValueError("Search observation must be finite")
        visited = observation["flags"][..., 0] > .5
        if not visited.any(1).all():
            raise ValueError("At least the initial observation must be visible")
        # Defense in depth: unknown scores/images cannot affect actor or critic.
        seen_visual = torch.where(visited[..., None], visual, torch.zeros_like(visual))
        scores = torch.where(visited[..., None], observation["scores"],
                             torch.zeros_like(observation["scores"]))
        record = self.record(torch.cat(
            [self.visual(seen_visual), geom, scores, observation["flags"]], -1))
        context = self.context(torch.cat(
            [self.visual(observation["wrist"]), observation["global"]], -1))
        record = record + context[:, None, :]
        memory = self.history(record, src_key_padding_mask=~visited)
        pooled = (memory * visited[..., None]).sum(1) / visited.sum(1, keepdim=True)
        return memory, pooled, visited


class SharedSearchActor(nn.Module):
    """Order-equivariant next-candidate logits and order-invariant STOP logit."""

    def __init__(self, visual_dim, width=128, heads=4, layers=2):
        super().__init__()
        self.encoder = SearchHistoryEncoder(visual_dim, width, heads, layers)
        self.geometry = nn.Linear(12, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.candidate = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        self.stop = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, observation):
        memory, context, visited = self.encoder(observation)
        query = self.geometry(observation["geometry"]) + context[:, None, :]
        response, _ = self.attention(query, memory, memory, key_padding_mask=~visited,
                                     need_weights=False)
        candidate = self.candidate(torch.cat([query, response], -1)).squeeze(-1)
        logits = torch.cat([self.stop(context), candidate], -1)
        mask = observation["action_mask"].bool()
        if not mask.any(1).all() or (mask[:, 1:] & visited).any():
            raise ValueError("Legal actions must exist and cannot reacquire visited candidates")
        return logits.masked_fill(~mask, -torch.inf)


class PrivilegedSearchCritic(nn.Module):
    """Train-only value estimate with full, same-noise Accel for this scene."""

    def __init__(self, visual_dim, width=128, heads=4, layers=2):
        super().__init__()
        self.encoder = SearchHistoryEncoder(visual_dim, width, heads, layers)
        self.field = nn.Sequential(nn.Linear(13, width), nn.GELU(), nn.Linear(width, width))
        self.value = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, observation, full_accel):
        if (full_accel.shape != observation["geometry"].shape[:2]
                or not torch.isfinite(full_accel).all() or (full_accel < 0).any()):
            raise ValueError("Full Accel must be finite, nonnegative [batch, candidates]")
        _, context, visited = self.encoder(observation)
        # Mean and remaining-field mean expose search opportunity without a
        # canonical ID, task-success label, or privileged actor activation.
        field = self.field(torch.cat([observation["geometry"], full_accel[..., None]], -1))
        remaining = ~visited
        remaining_mean = (field * remaining[..., None]).sum(1) / remaining.sum(
            1, keepdim=True).clamp_min(1)
        return self.value(torch.cat([context, field.mean(1), remaining_mean], -1)).squeeze(-1)

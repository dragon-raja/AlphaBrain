from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_NAMES = ("source", "target")
CABI_CORNER_NAMES = ("base", "source_anchor", "target_anchor", "fourth_anchor")


@dataclass
class BindingState:
    """A policy prefix together with its learned multimodal role variables."""

    tokens: torch.Tensor
    input_tokens: torch.Tensor
    role_states: torch.Tensor
    language_attention: torch.Tensor
    write_attention: torch.Tensor
    visual_attention: torch.Tensor
    vision_mask: torch.Tensor
    language_mask: torch.Tensor


def prefix_modality_masks(
    prefix_pad_mask: torch.Tensor,
    *,
    language_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a PaliGemma prefix mask into visual and valid language tokens."""

    if prefix_pad_mask.ndim != 2:
        raise ValueError("prefix pad mask must be [B, N]")
    if language_token_count <= 0 or language_token_count >= prefix_pad_mask.shape[1]:
        raise ValueError("language token count must lie inside the prefix")
    valid = prefix_pad_mask.to(dtype=torch.bool)
    language_region = torch.zeros_like(valid)
    language_region[:, -language_token_count:] = True
    language_mask = valid & language_region
    vision_mask = valid & ~language_region
    if torch.any(vision_mask.sum(dim=1) == 0):
        raise ValueError("every prefix requires a valid visual token")
    if torch.any(language_mask.sum(dim=1) == 0):
        raise ValueError("every prefix requires a valid language token")
    return vision_mask, language_mask


def select_binding_state(
    state: BindingState,
    indices: Sequence[int] | torch.Tensor,
) -> BindingState:
    index = torch.as_tensor(indices, device=state.tokens.device, dtype=torch.long)
    if index.ndim != 1:
        raise ValueError("binding-state indices must be one-dimensional")
    return BindingState(
        tokens=state.tokens.index_select(0, index),
        input_tokens=state.input_tokens.index_select(0, index),
        role_states=state.role_states.index_select(0, index),
        language_attention=state.language_attention.index_select(0, index),
        write_attention=state.write_attention.index_select(0, index),
        visual_attention=state.visual_attention.index_select(0, index),
        vision_mask=state.vision_mask.index_select(0, index),
        language_mask=state.language_mask.index_select(0, index),
    )


def group_cabi_tetrads(
    examples: Sequence[Mapping[str, object]],
) -> Mapping[str, tuple[int, ...]]:
    """Return corner-aligned indices while keeping role metadata training-only."""

    groups: dict[str, dict[str, int]] = {}
    for index, example in enumerate(examples):
        has_group = "cabi_tetrad_id" in example
        has_corner = "cabi_corner" in example
        if not has_group and not has_corner:
            continue
        if has_group != has_corner:
            raise ValueError("CABI examples require both tetrad id and corner")
        group_id = str(example["cabi_tetrad_id"])
        corner = str(example["cabi_corner"])
        if corner not in CABI_CORNER_NAMES:
            raise ValueError(f"unknown CABI corner: {corner}")
        if corner in groups.setdefault(group_id, {}):
            raise ValueError(f"duplicate {corner} in CABI tetrad {group_id}")
        groups[group_id][corner] = index

    if not groups:
        return {}

    aligned = {corner: [] for corner in CABI_CORNER_NAMES}
    for group_id in sorted(groups):
        missing = set(CABI_CORNER_NAMES) - set(groups[group_id])
        if missing:
            raise ValueError(f"incomplete CABI tetrad {group_id}: missing {sorted(missing)}")
        for corner in CABI_CORNER_NAMES:
            aligned[corner].append(groups[group_id][corner])
    return {corner: tuple(indices) for corner, indices in aligned.items()}


def _validate_token_masks(
    tokens: torch.Tensor,
    vision_mask: torch.Tensor,
    language_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B, N, D], got {tuple(tokens.shape)}")
    expected = tokens.shape[:2]
    if tuple(vision_mask.shape) != expected or tuple(language_mask.shape) != expected:
        raise ValueError(
            f"token masks must have shape {expected}, got "
            f"{tuple(vision_mask.shape)} and {tuple(language_mask.shape)}"
        )
    vision_mask = vision_mask.to(device=tokens.device, dtype=torch.bool)
    language_mask = language_mask.to(device=tokens.device, dtype=torch.bool)
    if torch.any(vision_mask & language_mask):
        raise ValueError("vision and language masks must be disjoint")
    if torch.any(vision_mask.sum(dim=1) == 0):
        raise ValueError("every sample requires at least one visual token")
    if torch.any(language_mask.sum(dim=1) == 0):
        raise ValueError("every sample requires at least one language token")
    return vision_mask, language_mask


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, :].expand_as(logits)
    minimum = torch.finfo(logits.dtype).min
    return torch.softmax(logits.masked_fill(~mask, minimum), dim=-1)


class CausalBindingAdapter(nn.Module):
    """Learn and transport source/target bindings without changing inference shape.

    Role queries first attend to language tokens, then use the resulting semantic
    query to bind visual tokens. A role-specific low-rank basis writes the fused
    binding back into language positions. No visual patch or hidden coordinate is
    assigned to a role by hand.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        binding_dim: int = 128,
        transport_rank: int = 32,
        num_roles: int = 2,
        temperature: float = 1.0,
        residual_scale: float = 0.1,
        tie_read_write_attention: bool = False,
        role_state_residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or binding_dim <= 0 or transport_rank <= 0:
            raise ValueError("CABI dimensions must be positive")
        if num_roles != len(ROLE_NAMES):
            raise ValueError(f"CABI v0 requires {len(ROLE_NAMES)} roles")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_dim = hidden_dim
        self.binding_dim = binding_dim
        self.transport_rank = transport_rank
        self.num_roles = num_roles
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.tie_read_write_attention = bool(tie_read_write_attention)
        self.role_state_residual = bool(role_state_residual)

        self.role_queries = nn.Parameter(torch.empty(num_roles, binding_dim))
        self.language_key = nn.Linear(hidden_dim, binding_dim, bias=False)
        self.language_value = nn.Linear(hidden_dim, binding_dim, bias=False)
        self.visual_key = nn.Linear(hidden_dim, binding_dim, bias=False)
        self.visual_value = nn.Linear(hidden_dim, binding_dim, bias=False)
        self.language_to_visual = nn.Linear(binding_dim, binding_dim, bias=False)
        self.binding_fusion = nn.Sequential(
            nn.Linear(2 * binding_dim, binding_dim),
            nn.SiLU(),
            nn.Linear(binding_dim, binding_dim),
        )
        self.transport_coefficients = nn.ModuleList(
            nn.Linear(binding_dim, transport_rank, bias=False)
            for _ in range(num_roles)
        )
        self.transport_bases = nn.Parameter(
            torch.empty(num_roles, transport_rank, hidden_dim)
        )
        self.write_key = nn.Linear(hidden_dim, binding_dim, bias=False)
        self.write_query = nn.Linear(binding_dim, binding_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.role_queries, std=self.binding_dim**-0.5)
        nn.init.normal_(self.transport_bases, std=self.hidden_dim**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _summarize(
        self,
        tokens: torch.Tensor,
        vision_mask: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.binding_dim**-0.5 / self.temperature
        language_key = self.language_key(tokens)
        language_logits = torch.einsum(
            "rd,bnd->brn", self.role_queries, language_key
        ) * scale
        language_attention = _masked_softmax(language_logits, language_mask)
        language_state = torch.einsum(
            "brn,bnd->brd", language_attention, self.language_value(tokens)
        )

        visual_query = self.role_queries[None] + self.language_to_visual(language_state)
        visual_logits = torch.einsum(
            "brd,bnd->brn", visual_query, self.visual_key(tokens)
        ) * scale
        visual_attention = _masked_softmax(visual_logits, vision_mask)
        visual_state = torch.einsum(
            "brn,bnd->brd", visual_attention, self.visual_value(tokens)
        )
        fused_state = self.binding_fusion(
            torch.cat([language_state, visual_state], dim=-1)
        )
        if self.role_state_residual:
            role_states = F.layer_norm(
                language_state + visual_state + 0.1 * fused_state,
                (self.binding_dim,),
            )
        else:
            role_states = fused_state

        if self.tie_read_write_attention:
            write_attention = language_attention
        else:
            write_logits = torch.einsum(
                "brd,bnd->brn",
                self.write_query(role_states),
                self.write_key(tokens),
            ) * scale
            write_attention = _masked_softmax(write_logits, language_mask)
        return role_states, language_attention, write_attention, visual_attention

    def _role_updates(self, role_states: torch.Tensor) -> torch.Tensor:
        if role_states.shape[-2:] != (self.num_roles, self.binding_dim):
            raise ValueError(
                "role states must be [B, R, binding_dim], got "
                f"{tuple(role_states.shape)}"
            )
        updates = []
        for role_index, projection in enumerate(self.transport_coefficients):
            coefficients = projection(role_states[:, role_index])
            updates.append(coefficients @ self.transport_bases[role_index])
        return torch.stack(updates, dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        vision_mask: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> BindingState:
        vision_mask, language_mask = _validate_token_masks(
            tokens, vision_mask, language_mask
        )
        role_states, language_attention, write_attention, visual_attention = self._summarize(
            tokens, vision_mask, language_mask
        )
        role_updates = self._role_updates(role_states)
        residual = torch.einsum(
            "brn,brd->bnd", write_attention, role_updates
        )
        adapted_tokens = tokens + self.residual_scale * residual
        return BindingState(
            tokens=adapted_tokens,
            input_tokens=tokens,
            role_states=role_states,
            language_attention=language_attention,
            write_attention=write_attention,
            visual_attention=visual_attention,
            vision_mask=vision_mask,
            language_mask=language_mask,
        )

    def transport(
        self,
        base: BindingState,
        donor: BindingState,
        roles: Sequence[int] | torch.Tensor,
    ) -> BindingState:
        if base.tokens.shape != donor.tokens.shape:
            raise ValueError("base and donor token shapes must match")
        if isinstance(roles, torch.Tensor):
            role_mask = roles.to(device=base.tokens.device, dtype=base.tokens.dtype)
            if role_mask.ndim == 1:
                role_mask = role_mask[None].expand(base.tokens.shape[0], -1)
        else:
            role_mask = torch.zeros(
                base.tokens.shape[0],
                self.num_roles,
                device=base.tokens.device,
                dtype=base.tokens.dtype,
            )
            for role_index in roles:
                if role_index < 0 or role_index >= self.num_roles:
                    raise IndexError(f"invalid role index: {role_index}")
                role_mask[:, role_index] = 1
        if tuple(role_mask.shape) != (base.tokens.shape[0], self.num_roles):
            raise ValueError(
                f"role mask must be [B, {self.num_roles}], got {tuple(role_mask.shape)}"
            )

        delta_states = (donor.role_states - base.role_states) * role_mask[..., None]
        delta_updates = self._role_updates(delta_states)
        residual = torch.einsum(
            "brn,brd->bnd", base.write_attention, delta_updates
        )
        transported_tokens = base.tokens + self.residual_scale * residual
        role_states, language_attention, write_attention, visual_attention = self._summarize(
            transported_tokens,
            base.vision_mask,
            base.language_mask,
        )
        return BindingState(
            tokens=transported_tokens,
            input_tokens=transported_tokens,
            role_states=role_states,
            language_attention=language_attention,
            write_attention=write_attention,
            visual_attention=visual_attention,
            vision_mask=base.vision_mask,
            language_mask=base.language_mask,
        )

    def orthogonality_loss(self) -> torch.Tensor:
        flattened = self.transport_bases.flatten(1)
        normalized = F.normalize(flattened, dim=-1)
        gram = normalized @ normalized.transpose(0, 1)
        identity = torch.eye(
            self.num_roles, device=gram.device, dtype=gram.dtype
        )
        return (gram - identity).square().mean()


def cabi_closure_losses(
    adapter: CausalBindingAdapter,
    *,
    base: BindingState,
    source_anchor: BindingState,
    target_anchor: BindingState,
    fourth_anchor: BindingState,
    intervention_margin: float = 0.1,
    contrastive_temperature: float = 0.1,
) -> Mapping[str, torch.Tensor]:
    """Compute three-corner supervision and action-free fourth-corner closure."""

    if intervention_margin <= 0:
        raise ValueError("intervention_margin must be positive")
    if contrastive_temperature <= 0:
        raise ValueError("contrastive_temperature must be positive")

    def alignment(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (
            1.0
            - F.cosine_similarity(
                left.float(),
                right.float(),
                dim=-1,
                eps=1e-6,
            )
        ).mean()

    source_swap = adapter.transport(base, source_anchor, [0])
    target_swap = adapter.transport(base, target_anchor, [1])
    source_then_target = adapter.transport(source_swap, fourth_anchor, [1])
    target_then_source = adapter.transport(target_swap, fourth_anchor, [0])
    source_cycle = adapter.transport(source_swap, base, [0])
    target_cycle = adapter.transport(target_swap, base, [1])

    def sample_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(
            left.float(), right.float(), dim=-1, eps=1e-6
        )

    source_intended = sample_distance(
        base.role_states[:, 0], source_anchor.role_states[:, 0]
    )
    source_nuisance = sample_distance(
        base.role_states[:, 1], source_anchor.role_states[:, 1]
    )
    target_intended = sample_distance(
        base.role_states[:, 1], target_anchor.role_states[:, 1]
    )
    target_nuisance = sample_distance(
        base.role_states[:, 0], target_anchor.role_states[:, 0]
    )
    intervention_identifiability = 0.5 * (
        F.relu(intervention_margin + source_nuisance - source_intended).mean()
        + F.relu(intervention_margin + target_nuisance - target_intended).mean()
    )

    def attention_overlap(state: BindingState) -> torch.Tensor:
        overlaps = [
            F.cosine_similarity(
                state.language_attention[:, 0].float(),
                state.language_attention[:, 1].float(),
                dim=-1,
                eps=1e-6,
            ),
            F.cosine_similarity(
                state.visual_attention[:, 0].float(),
                state.visual_attention[:, 1].float(),
                dim=-1,
                eps=1e-6,
            ),
            F.cosine_similarity(
                state.write_attention[:, 0].float(),
                state.write_attention[:, 1].float(),
                dim=-1,
                eps=1e-6,
            ),
        ]
        return torch.stack(overlaps, dim=-1).mean()

    attention_separation = torch.stack(
        [
            attention_overlap(base),
            attention_overlap(source_anchor),
            attention_overlap(target_anchor),
            attention_overlap(fourth_anchor),
        ]
    ).mean()

    def factor_contrastive(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        fourth_negative: torch.Tensor,
    ) -> torch.Tensor:
        anchor = F.normalize(anchor.float(), dim=-1)
        candidates = torch.stack(
            [positive, negative, fourth_negative], dim=1
        ).float()
        candidates = F.normalize(candidates, dim=-1)
        logits = torch.einsum("bd,bkd->bk", anchor, candidates)
        logits = logits / contrastive_temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    causal_factor_contrastive = 0.5 * (
        factor_contrastive(
            base.role_states[:, 0],
            target_anchor.role_states[:, 0],
            source_anchor.role_states[:, 0],
            fourth_anchor.role_states[:, 0],
        )
        + factor_contrastive(
            base.role_states[:, 1],
            source_anchor.role_states[:, 1],
            target_anchor.role_states[:, 1],
            fourth_anchor.role_states[:, 1],
        )
    )

    def intervention_attention_grounding(
        left: BindingState,
        right: BindingState,
        role_index: int,
    ) -> torch.Tensor:
        valid = left.language_mask | right.language_mask
        token_change = (
            left.input_tokens.float() - right.input_tokens.float()
        ).square().mean(dim=-1).sqrt()
        changed = (token_change > 1e-5) & valid
        attention = 0.5 * (
            left.language_attention[:, role_index]
            + right.language_attention[:, role_index]
        )
        changed_mass = (attention * changed.float()).sum(dim=-1)
        has_change = changed.any(dim=-1)
        losses = -torch.log(changed_mass.clamp_min(1e-6))
        return torch.where(has_change, losses, torch.zeros_like(losses)).mean()

    counterfactual_attention_grounding = 0.5 * (
        intervention_attention_grounding(base, source_anchor, 0)
        + intervention_attention_grounding(base, target_anchor, 1)
    )

    return {
        "single_source": alignment(
            source_swap.role_states[:, 0], source_anchor.role_states[:, 0]
        ),
        "single_target": alignment(
            target_swap.role_states[:, 1], target_anchor.role_states[:, 1]
        ),
        "fourth_anchor": 0.5
        * (
            alignment(source_then_target.role_states, fourth_anchor.role_states)
            + alignment(target_then_source.role_states, fourth_anchor.role_states)
        ),
        "commutator": alignment(
            source_then_target.role_states,
            target_then_source.role_states,
        ),
        "cycle": 0.5
        * (
            alignment(source_cycle.role_states, base.role_states)
            + alignment(target_cycle.role_states, base.role_states)
        ),
        "specificity": 0.5
        * (
            alignment(source_swap.role_states[:, 1], base.role_states[:, 1])
            + alignment(target_swap.role_states[:, 0], base.role_states[:, 0])
        ),
        "orthogonality": adapter.orthogonality_loss(),
        "intervention_identifiability": intervention_identifiability,
        "attention_separation": attention_separation,
        "causal_factor_contrastive": causal_factor_contrastive,
        "counterfactual_attention_grounding": counterfactual_attention_grounding,
    }

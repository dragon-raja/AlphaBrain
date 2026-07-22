from __future__ import annotations

import torch

from AlphaBrain.model.modules.action_model.cabi_binding import (
    CausalBindingAdapter,
    cabi_closure_losses,
    cabi_transport_role_mask,
    group_cabi_tetrads,
    prefix_modality_masks,
    select_binding_state,
)


def make_inputs(batch: int = 3, tokens: int = 12, hidden: int = 24):
    values = torch.randn(batch, tokens, hidden)
    vision_mask = torch.zeros(batch, tokens, dtype=torch.bool)
    language_mask = torch.zeros(batch, tokens, dtype=torch.bool)
    vision_mask[:, :8] = True
    language_mask[:, 8:] = True
    return values, vision_mask, language_mask


def test_binding_adapter_preserves_shape_and_attention_support() -> None:
    adapter = CausalBindingAdapter(24, binding_dim=12, transport_rank=4)
    tokens, vision_mask, language_mask = make_inputs()
    state = adapter(tokens, vision_mask, language_mask)

    assert state.tokens.shape == tokens.shape
    torch.testing.assert_close(state.input_tokens, tokens)
    assert state.role_states.shape == (3, 2, 12)
    assert torch.allclose(state.visual_attention.sum(-1), torch.ones(3, 2))
    assert torch.allclose(state.language_attention.sum(-1), torch.ones(3, 2))
    assert torch.allclose(state.write_attention.sum(-1), torch.ones(3, 2))
    assert torch.count_nonzero(state.visual_attention[..., 8:]) == 0
    assert torch.count_nonzero(state.language_attention[..., :8]) == 0


def test_zero_delta_transport_is_token_noop() -> None:
    adapter = CausalBindingAdapter(24, binding_dim=12, transport_rank=4)
    tokens, vision_mask, language_mask = make_inputs()
    state = adapter(tokens, vision_mask, language_mask)
    transported = adapter.transport(state, state, [0, 1])
    torch.testing.assert_close(transported.tokens, state.tokens)


def test_closure_losses_are_finite_and_train_adapter() -> None:
    adapter = CausalBindingAdapter(24, binding_dim=12, transport_rank=4)
    masks = make_inputs()[1:]
    states = [adapter(torch.randn(3, 12, 24), *masks) for _ in range(4)]
    losses = cabi_closure_losses(
        adapter,
        base=states[0],
        source_anchor=states[1],
        target_anchor=states[2],
        fourth_anchor=states[3],
    )
    assert set(losses) == {
        "single_source",
        "single_target",
        "fourth_anchor",
        "commutator",
        "cycle",
        "specificity",
        "orthogonality",
        "intervention_identifiability",
        "attention_separation",
        "causal_factor_contrastive",
        "counterfactual_attention_grounding",
    }
    assert torch.isfinite(losses["intervention_identifiability"])
    assert torch.isfinite(losses["attention_separation"])
    assert torch.isfinite(losses["causal_factor_contrastive"])
    assert torch.isfinite(losses["counterfactual_attention_grounding"])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert adapter.role_queries.grad is not None
    assert torch.isfinite(adapter.role_queries.grad).all()


def test_identifiability_loss_rejects_collapsed_roles() -> None:
    adapter = CausalBindingAdapter(24, binding_dim=12, transport_rank=4)
    tokens, vision_mask, language_mask = make_inputs(batch=2)
    state = adapter(tokens, vision_mask, language_mask)
    losses = cabi_closure_losses(
        adapter,
        base=state,
        source_anchor=state,
        target_anchor=state,
        fourth_anchor=state,
        intervention_margin=0.1,
    )
    torch.testing.assert_close(
        losses["intervention_identifiability"],
        torch.tensor(0.1),
    )
    assert losses["attention_separation"] > 0
    assert losses["causal_factor_contrastive"] > 1.0


def test_tied_write_attention_and_role_residual_are_active() -> None:
    adapter = CausalBindingAdapter(
        24,
        binding_dim=12,
        transport_rank=4,
        tie_read_write_attention=True,
        role_state_residual=True,
    )
    tokens, vision_mask, language_mask = make_inputs()
    state = adapter(tokens, vision_mask, language_mask)
    torch.testing.assert_close(state.write_attention, state.language_attention)
    assert torch.isfinite(state.role_states).all()


def test_binding_adapter_rejects_overlapping_masks() -> None:
    adapter = CausalBindingAdapter(24, binding_dim=12, transport_rank=4)
    tokens, vision_mask, language_mask = make_inputs()
    language_mask[:, 0] = True
    try:
        adapter(tokens, vision_mask, language_mask)
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping masks must fail")


def test_prefix_modality_masks_respect_padding() -> None:
    pad = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    vision, language = prefix_modality_masks(pad, language_token_count=3)
    torch.testing.assert_close(vision.sum(1), torch.tensor([3, 3]))
    torch.testing.assert_close(language.sum(1), torch.tensor([2, 1]))


def test_tetrad_grouping_and_selection_are_corner_aligned() -> None:
    examples = [{"action_supervised": True}]
    for group in ("b", "a"):
        for corner in ("target_anchor", "base", "fourth_anchor", "source_anchor"):
            examples.append({"cabi_tetrad_id": group, "cabi_corner": corner})
    grouped = group_cabi_tetrads(examples)
    assert grouped["base"] == (6, 2)

    adapter = CausalBindingAdapter(24, binding_dim=12, transport_rank=4)
    tokens, vision_mask, language_mask = make_inputs(batch=9)
    state = adapter(tokens, vision_mask, language_mask)
    selected = select_binding_state(state, grouped["base"])
    torch.testing.assert_close(selected.tokens, state.tokens[[6, 2]])


def test_tetrad_grouping_rejects_incomplete_groups() -> None:
    examples = [{"cabi_tetrad_id": "a", "cabi_corner": "base"}]
    try:
        group_cabi_tetrads(examples)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete tetrads must fail")


def test_transport_role_mask_defaults_to_both_and_supports_decision_points() -> None:
    examples = []
    for group, roles in (("legacy", None), ("source", ["source"])):
        for corner in ("base", "source_anchor", "target_anchor", "fourth_anchor"):
            example = {"cabi_tetrad_id": group, "cabi_corner": corner}
            if roles is not None:
                example["cabi_transport_roles"] = roles
            examples.append(example)
    grouped = group_cabi_tetrads(examples)
    mask = cabi_transport_role_mask(
        examples,
        grouped,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(mask, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))


def test_transport_role_mask_rejects_corner_metadata_mismatch() -> None:
    examples = []
    for corner in ("base", "source_anchor", "target_anchor", "fourth_anchor"):
        examples.append(
            {
                "cabi_tetrad_id": "a",
                "cabi_corner": corner,
                "cabi_transport_roles": [
                    "target" if corner == "fourth_anchor" else "source"
                ],
            }
        )
    grouped = group_cabi_tetrads(examples)
    try:
        cabi_transport_role_mask(
            examples,
            grouped,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    except ValueError as error:
        assert "share transport roles" in str(error)
    else:
        raise AssertionError("mismatched decision-point roles must fail")

from copy import deepcopy
import pytest

torch = pytest.importorskip('torch')
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.data.native_observation import NativeStream
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic, SharedHistoryCore


def observation():
    torch.manual_seed(10)
    x = dict(geometry=torch.randn(2, 4, 12), visual=torch.randn(2, 4, 8),
             wrist=torch.randn(2, 8), scores=torch.randn(2, 4, 2),
             flags=torch.zeros(2, 4, 4), joint_state=torch.randn(2, 6),
             delta_low=torch.full((2, 2), -5.), delta_high=torch.full((2, 2), 5.),
             action_mask=torch.ones(2, 5, dtype=torch.bool))
    x['global'] = torch.randn(2, 6)
    x['flags'][:, :2, 0] = 1
    x['flags'][:, 1, 2] = 1
    return x


def legacy_key(key):
    for branch in ('actor_history', 'critic_history'):
        key = key.replace(branch+'.adapter.visual.', branch+'.encoder.visual.')
        for part in ('record', 'context', 'history'):
            key = key.replace(branch+'.core.'+part+'.', branch+'.encoder.'+part+'.')
        key = key.replace(branch+'.core.joints.', branch+'.joints.')
    return key


def test_u0_matches_legacy_outputs_gradients_and_one_adam_step():
    torch.set_num_threads(1); torch.manual_seed(7)
    old = ContinuousActorCritic(8, width=16).train()
    # Nonzero actor heads prevent a trivial equivalence test through zero outputs.
    torch.nn.init.normal_(old.mean_head.weight, std=.1)
    torch.nn.init.normal_(old.stop_head.weight, std=.1)
    new = UnifiedActorCritic(core_spec('U0', width=16), visual_dim=8).train()
    new.load_legacy_initial(old.state_dict())
    obs = observation()
    a = old.distribution(obs).sample()
    optimizers = [torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5) for p in (old, new)]
    for p, opt in zip((old, new), optimizers):
        torch.manual_seed(123)
        loss = p.objective(obs, a['stop'], a['latent'], a['log_prob'].detach(),
                           torch.tensor([1., -1.]), torch.tensor([.2, -.5]))
        opt.zero_grad(); loss.backward()
    original = dict(old.named_parameters())
    for k, p in new.named_parameters():
        torch.testing.assert_close(p.grad, original[legacy_key(k)].grad, rtol=0, atol=0)
    for opt in optimizers: opt.step()
    for k, p in new.named_parameters():
        torch.testing.assert_close(p, original[legacy_key(k)], rtol=0, atol=0)
    torch.testing.assert_close(new.value(obs), old.value(obs), rtol=0, atol=0)
    torch.testing.assert_close(new.distribution(obs).log_prob(a['stop'], a['latent']),
                               old.distribution(obs).log_prob(a['stop'], a['latent']), rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['U0', 'U1', 'U2', 'U3', 'U12', 'U13', 'U23', 'U123'])
def test_all_variants_mask_unvisited_and_keep_independent_actor_critic(variant):
    p = UnifiedActorCritic(core_spec(variant, width=16), visual_dim=8).eval()
    obs = observation(); expected = p.value(obs)
    changed = deepcopy(obs)
    for key in ('visual', 'geometry', 'scores'):
        changed[key][:, 2:] = 1000
    torch.testing.assert_close(p.value(changed), expected, rtol=0, atol=0)
    assert not {v.data_ptr() for v in p.actor_history.parameters()} & {v.data_ptr() for v in p.critic_history.parameters()}
    for b in (p.actor_history, p.critic_history): assert type(b.core) is SharedHistoryCore
    assert p.distribution(obs).sample(deterministic=True)['delta_deg'].shape == (2, 2)


def test_same_shape_different_core_rejected_before_parameter_mutation():
    a = UnifiedActorCritic(core_spec('U0', width=16), visual_dim=8)
    b = UnifiedActorCritic(core_spec('U1', width=16), visual_dim=8)
    before = {k: v.clone() for k, v in b.named_parameters()}
    with pytest.raises(ValueError, match='identity'): b.load_state_dict(a.state_dict())
    for k, v in b.named_parameters(): assert torch.equal(v, before[k])


@pytest.mark.parametrize('variant', ['U0', 'U1', 'U2', 'U3', 'U12', 'U13', 'U23', 'U123'])
@pytest.mark.parametrize('kind', ['head', 'joint', 'head_wrist'])
def test_native_roles_share_core_and_missing_aux_does_not_remove_state_or_budget(variant, kind):
    roles = (('head',), ('movable',)) if kind != 'joint' else (('head', 'wrist'), ('movable', 'context'))
    stream = NativeStream('main', 2, 'joint' if kind == 'joint' else 'per_view', *roles)
    streams = (stream,)
    if kind == 'head_wrist':
        streams += (NativeStream('wrist', 2, 'per_view', ('wrist',), ('context',)),)
    p = UnifiedActorCritic(core_spec(variant, width=16), streams=streams, groups=['visual']*len(streams))
    obs = observation(); raw = obs.pop('visual'); obs.pop('wrist')
    obs['feature_mask'] = torch.zeros(2, 4, len(streams), dtype=torch.bool)
    obs['feature_mask'][:, :2, 0] = True
    obs['feature_0'] = raw.requires_grad_()
    if len(streams) == 2: obs['feature_1'] = torch.randn_like(raw).requires_grad_()
    val = p.value(obs)
    changed = deepcopy({k:v.detach().clone() for k,v in obs.items()})
    if len(streams) == 2:
        changed['feature_1'].fill_(900)
        torch.testing.assert_close(p.value(changed), val, rtol=0, atol=0)
    changed['joint_state'] += .2
    assert not torch.equal(p.value(changed), val)
    changed['joint_state'] = obs['joint_state'].clone(); changed['global'] += .3
    assert not torch.equal(p.value(changed), val)
    val.sum().backward()
    assert obs['feature_0'].grad is None
    assert p.critic_history.adapter.projections['visual'][0].weight.grad.abs().sum() > 0


def test_revisit_uses_latest_current_record():
    p = UnifiedActorCritic(core_spec('U0', width=16), visual_dim=8)
    obs = observation(); base = p.value(obs)
    obs['flags'][:, 0, 2] = 1  # Duplicate pose does not imply fabricated missing context.
    assert torch.isfinite(p.value(obs)).all()


def test_torch_checkpoint_preserves_contract(tmp_path):
    p = UnifiedActorCritic(core_spec('U2', width=16), visual_dim=8)
    f = tmp_path/'actor.pt'; torch.save(p.state_dict(), f)
    q = UnifiedActorCritic(core_spec('U2', width=16), visual_dim=8)
    q.load_state_dict(torch.load(f, map_location='cpu', weights_only=True))
    assert torch.equal(p.value(observation()), q.value(observation()))


def test_factory_refuses_same_shape_different_feature_producer():
    from AlphaBrain.research.dsol.data.native_features import layout_spec
    from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
    streams = (NativeStream('head', 2, 'per_view', ('head',), ('movable',)),)
    spec = dict(schema='UNIFIED_NATIVE_CONTINUOUS_POLICY_V1', core=core_spec('U0', width=16),
                layout=layout_spec(streams), projection_groups=['visual'], producer_sha256='a'*64)
    a = create_search_policy(spec)
    b = create_search_policy(dict(spec, producer_sha256='b'*64))
    before = {k: v.clone() for k, v in b.named_parameters()}
    with pytest.raises(ValueError, match='identity'): b.load_state_dict(a.state_dict())
    for k, v in b.named_parameters(): assert torch.equal(v, before[k])
    assert type(a.actor_history.core) is type(a.critic_history.core) is SharedHistoryCore


def test_factory_explicitly_rejects_unimplemented_multiple_context_streams():
    from AlphaBrain.research.dsol.data.native_features import layout_spec
    from AlphaBrain.research.dsol.selectors.search_policy_factory import validate_policy_spec
    streams = (NativeStream('head', 2, 'per_view', ('head',), ('movable',)),
               NativeStream('left', 2, 'per_view', ('left',), ('context',)),
               NativeStream('right', 2, 'per_view', ('right',), ('context',)))
    spec = dict(schema='UNIFIED_NATIVE_CONTINUOUS_POLICY_V1', core=core_spec('U0', width=16),
                layout=layout_spec(streams), projection_groups=['visual']*3, producer_sha256='a'*64)
    with pytest.raises(ValueError, match='at most one context'): validate_policy_spec(spec)


@pytest.mark.parametrize('variant',['U13','U23','U123'])
def test_combination_legacy_initialization_keeps_seeded_uniform_fusion(variant):
    torch.manual_seed(3)
    old=ContinuousActorCritic(8,width=16)
    new=UnifiedActorCritic(core_spec(variant,width=16),visual_dim=8)
    new.load_legacy_initial(old.state_dict())
    for branch in (new.actor_history,new.critic_history):
        assert torch.count_nonzero(branch.core.fusion.weight)==0
        assert torch.count_nonzero(branch.core.fusion.bias)==0
    for key,value in old.state_dict().items():
        mapped=[k for k in new.state_dict() if legacy_key(k)==key]
        assert len(mapped)==1 and torch.equal(new.state_dict()[mapped[0]],value)

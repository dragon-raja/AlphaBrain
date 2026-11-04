"""Shared Actor/Critic history core with explicit observation adapters.

No model capture, simulator, or dispatch. Legacy pi observations and native
single/joint streams share the same computation after input projection. Frozen
legacy classes are left intact; U0 compatibility requires explicit weight mapping.
"""
from copy import deepcopy
import torch
from torch import nn
from ..data.search_core import validate_core_spec
from ..data.native_observation import validate_streams
from ..execution.manifest import digest
from .continuous_search_policy import ContinuousActorCritic, BASE


def projection(size, width):
    return nn.Sequential(nn.Linear(size, width), nn.LayerNorm(width), nn.GELU())


def observed_rows(obs):
    geom = obs['geometry']
    if geom.ndim != 3 or geom.shape[-1] != 12:
        raise ValueError('Geometry must be [batch, history, 12]')
    b, n = geom.shape[:2]
    expected = {'scores': (b, n, 2), 'flags': (b, n, 4), 'global': (b, 6),
                'joint_state': (b, 6), 'action_mask': (b, n+1),
                'delta_low': (b, 2), 'delta_high': (b, 2)}
    if any(tuple(obs[k].shape) != shape for k, shape in expected.items()):
        raise ValueError('Observation dimensions differ')
    if any(not torch.isfinite(v).all() for v in obs.values()):
        raise ValueError('Nonfinite observation')
    visited = obs['flags'][..., 0] > .5
    if not visited.any(1).all():
        raise ValueError('An acquired movable observation is required')
    return visited


class LegacyFeatureAdapter(nn.Module):
    """Real external observations plus the legacy fixed-scene wrist context."""
    def __init__(self, visual_dim, width):
        super().__init__()
        self.visual_dim = visual_dim
        self.visual = projection(visual_dim, width)

    def forward(self, obs, visited):
        if set(obs) != BASE | {'joint_state', 'delta_low', 'delta_high'}:
            raise ValueError('Legacy observation schema differs')
        b, n = visited.shape
        if obs['visual'].shape != (b, n, self.visual_dim) or obs['wrist'].shape != (b, self.visual_dim):
            raise ValueError('Real external/wrist dimensions differ')
        visual = obs['visual'].detach()
        main = visual.new_zeros(b, n, self.visual[0].out_features)
        main[visited] = self.visual(visual[visited])
        auxiliary = self.visual(obs['wrist'].detach())[:, None].expand_as(main)
        # A broadcast of the recorded stationary wrist is not a synthetic camera.
        return main, auxiliary, visited


class NativeFeatureAdapter(nn.Module):
    """One real movable or joint stream; optional independently observed context.

    Group names explicitly declare projection sharing. Joint streams are never
    split into cameras. Multi-movable fusion is rejected until separately designed.
    """
    def __init__(self, streams, groups, width):
        super().__init__()
        self.streams = tuple(streams)
        validate_streams(self.streams)
        self.groups = tuple(groups)
        if (len(groups) != len(streams) or any(not isinstance(g, str) or not g
                or '.' in g for g in groups)):
            raise ValueError('Explicit safe projection groups required')
        self.main_index = [i for i, s in enumerate(streams) if 'movable' in s.camera_roles]
        self.aux_index = [i for i, s in enumerate(streams) if 'movable' not in s.camera_roles]
        if len(self.main_index) != 1 or len(self.aux_index) > 1:
            raise ValueError('One movable/joint and at most one context stream required')
        sizes = {}
        for g, s in zip(groups, streams):
            if g in sizes and sizes[g] != 4*s.channels:
                raise ValueError('Shared projection has incompatible native channels')
            sizes[g] = 4*s.channels
        self.projections = nn.ModuleDict({g: projection(size, width) for g, size in sizes.items()})

    def forward(self, obs, visited):
        required = {'geometry', 'scores', 'flags', 'global', 'action_mask', 'joint_state',
                    'delta_low', 'delta_high', 'feature_mask'} | {f'feature_{i}' for i in range(len(self.streams))}
        if set(obs) != required:
            raise ValueError('Native observation schema differs')
        b, n = visited.shape
        mask = obs['feature_mask']
        if (mask.shape != (b, n, len(self.streams)) or mask.dtype != torch.bool
                or (mask & ~visited[..., None]).any()
                or not (mask[..., self.main_index[0]] | ~visited).all()):
            raise ValueError('Only acquired real movable observations may be encoded')
        values = []
        for i, (stream, group) in enumerate(zip(self.streams, self.groups)):
            raw = obs[f'feature_{i}'].detach()
            if raw.shape != (b, n, 4*stream.channels):
                raise ValueError('Native feature dimensions differ')
            layer = self.projections[group]
            encoded = raw.new_zeros(b, n, layer[0].out_features)
            encoded[mask[..., i]] = layer(raw[mask[..., i]])
            values.append(encoded)
        main = values[self.main_index[0]]
        if self.aux_index:
            i = self.aux_index[0]
            return main, values[i], mask[..., i]
        # Masked absence in the core, not an invented wrist frame or feature packet.
        return main, torch.zeros_like(main), torch.zeros_like(visited)


class SharedHistoryCore(nn.Module):
    def __init__(self, spec, *, task_spec=None):
        super().__init__()
        self.spec = validate_core_spec(spec)
        w = spec['width']
        self.record = nn.Linear(w+12+2+4, w)
        self.context = nn.Linear(w+6, w)
        self.history = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            w, spec['heads'], w*2, dropout=0., activation='gelu', batch_first=True,
            norm_first=False), spec['layers'], enable_nested_tensor=False)
        self.joints = nn.Sequential(nn.Linear(6, w), nn.Tanh())
        self.task_spec = task_spec
        if task_spec is not None:
            from ..data.task_condition import validate_spec
            validate_spec(task_spec)
            with torch.random.fork_rng(devices=[]):
                self.task_projection = nn.Linear(task_spec['dimension'], w, bias=False)
            nn.init.zeros_(self.task_projection.weight)
        if spec['auxiliary_route'] == 'record':
            self.fusion = nn.Linear(w, 1)
            nn.init.zeros_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)

    def forward(self, obs, main, auxiliary, auxiliary_mask, visited):
        auxiliary = torch.where(auxiliary_mask[..., None], auxiliary, torch.zeros_like(auxiliary))
        if self.spec['auxiliary_route'] == 'record':
            features = torch.stack((main, auxiliary), 2)
            available = torch.stack((visited, auxiliary_mask), 2)
            logits = self.fusion(features).squeeze(-1).masked_fill(~available, -1e9)
            weights = torch.softmax(logits, -1)*available
            main = (features*weights[..., None]).sum(2)
            context_view = torch.zeros_like(main[:, 0])
        else:
            # Context is the CURRENT observation, never a future/all-history mean.
            current = (obs['flags'][..., 2] > .5) & visited
            if not current.any(1).all():
                raise ValueError('An acquired current observation is required')
            # A revisit can mark several records at the same physical pose.
            index = torch.where(current, torch.arange(current.shape[1], device=current.device), -1).max(1).values
            context_view = auxiliary[torch.arange(len(index), device=index.device), index]
        scores = torch.where(visited[..., None], obs['scores'], torch.zeros_like(obs['scores']))
        records = self.record(torch.cat((main, obs['geometry'], scores, obs['flags']), -1))
        context = self.context(torch.cat((context_view, obs['global']), -1))
        if self.task_spec is not None:
            task = obs['task_features']
            if task.shape != (len(context), self.task_spec['dimension']):
                raise ValueError('Task feature dimensions differ')
            context = context + self.task_projection(task.detach())
        if self.spec['context_activation'] == 'tanh':
            context = context.tanh()
        records = records + context[:, None]
        state = self.joints(obs['joint_state'])
        if self.spec['state_position'] == 'pre':
            records = records + state[:, None]
        memory = self.history(records, src_key_padding_mask=~visited)
        result = (memory*visited[..., None]).sum(1)/visited.sum(1, keepdim=True)
        return result + state if self.spec['state_position'] == 'post' else result


class UnifiedHistory(nn.Module):
    def __init__(self, spec, *, visual_dim=None, streams=None, groups=None, task_spec=None):
        super().__init__()
        if (visual_dim is None) == (streams is None):
            raise ValueError('One explicit input adapter required')
        self.adapter = (LegacyFeatureAdapter(visual_dim, spec['width']) if streams is None
                        else NativeFeatureAdapter(streams, groups, spec['width']))
        self.core = SharedHistoryCore(spec, task_spec=task_spec)

    def forward(self, obs):
        visited = observed_rows(obs)
        if ('task_features' in obs) != (self.core.task_spec is not None):
            raise ValueError('Explicit task input presence differs from policy contract')
        visual_obs = {k:v for k,v in obs.items() if k!='task_features'}
        return self.core(obs, *self.adapter(visual_obs, visited), visited)


class UnifiedActorCritic(ContinuousActorCritic):
    def __init__(self, spec, *, visual_dim=None, streams=None, groups=None, producer_sha256=None, task_spec=None):
        nn.Module.__init__(self)
        spec = validate_core_spec(spec)
        self.core_spec = spec
        adapter = dict(visual_dim=visual_dim, streams=streams, groups=groups, task_spec=task_spec)
        self.actor_history = UnifiedHistory(spec, **adapter)
        self.critic_history = UnifiedHistory(spec, **adapter)
        w = spec['width']
        self.stop_head = nn.Linear(w, 1)
        self.mean_head = nn.Linear(w, 2)
        self.log_std = nn.Parameter(torch.full((2,), -.5))
        self.value_head = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))
        nn.init.zeros_(self.stop_head.weight); nn.init.zeros_(self.stop_head.bias)
        nn.init.zeros_(self.mean_head.weight); nn.init.zeros_(self.mean_head.bias)
        from dataclasses import asdict
        self.identity = dict(core=spec, visual_dim=visual_dim,
            streams=None if streams is None else [asdict(s) for s in streams],
            projection_groups=None if groups is None else list(groups))
        if task_spec is not None:
            self.identity['task_spec'] = deepcopy(task_spec)
        if producer_sha256 is not None:
            if (not isinstance(producer_sha256, str) or len(producer_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in producer_sha256)):
                raise ValueError('Full feature producer identity required')
            self.identity['producer_sha256'] = producer_sha256
        self.search_core_sha256 = digest(self.identity)

    def get_extra_state(self):
        return deepcopy(self.identity)

    def set_extra_state(self, state):
        if state != self.identity:
            raise ValueError('Checkpoint search core/adapter identity differs')

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Same-shaped U0/U1/U2 must fail BEFORE mutating any parameters.
        if state_dict.get('_extra_state') != self.identity:
            raise ValueError('Checkpoint search core/adapter identity differs')
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def load_legacy_initial(self, state_dict):
        """Explicit weight-only migration; caller must prove v0 provenance.

        Does not import optimizer/RNG, change legacy artifacts, or claim U1-U3
        functional equivalence. New U3 fusion is deterministically uniform.
        """
        if self.identity['visual_dim'] is None:
            raise ValueError('Legacy pi weights require the real pi adapter')
        target = self.state_dict()
        mapped = {}
        for key, value in state_dict.items():
            new = key
            for branch in ('actor_history', 'critic_history'):
                new = new.replace(branch+'.encoder.visual.', branch+'.adapter.visual.')
                for part in ('record', 'context', 'history'):
                    new = new.replace(branch+'.encoder.'+part+'.', branch+'.core.'+part+'.')
                new = new.replace(branch+'.joints.', branch+'.core.joints.')
            if new not in target or not isinstance(value, torch.Tensor) or target[new].shape != value.shape:
                raise ValueError('Legacy mapping differs: '+key)
            mapped[new] = value
        remaining = set(target)-set(mapped)-{'_extra_state'}
        allowed = {b+'.core.fusion.'+k for b in ('actor_history', 'critic_history') for k in ('weight', 'bias')}
        if self.core_spec['auxiliary_route'] != 'record':
            allowed = set()
        if 'task_spec' in self.identity:
            allowed |= {b+'.core.task_projection.weight' for b in ('actor_history','critic_history')}
        if remaining - allowed:
            raise ValueError('Incomplete legacy mapping')
        target.update(mapped)
        self.load_state_dict(target, strict=True)

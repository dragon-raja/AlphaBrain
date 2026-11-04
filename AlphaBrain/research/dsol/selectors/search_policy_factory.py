"""Closed, declarative policy construction; no dynamic import or model loading."""
from copy import deepcopy
from ..data.native_features import read_layout
from ..execution.manifest import digest


def validate_policy_spec(spec):
    if spec.get('schema') == 'UNIFIED_NATIVE_CONTINUOUS_POLICY_V1':
        from ..data.search_core import validate_core_spec
        required = {'schema', 'layout', 'core', 'projection_groups', 'producer_sha256'}
        if set(spec) not in (required, required | {'task_spec'}):
            raise ValueError('Complete shared core/input specification required')
        if 'task_spec' in spec:
            from ..data.task_condition import validate_spec
            validate_spec(spec['task_spec'])
        validate_core_spec(spec['core'])
        streams = read_layout(spec['layout'])
        groups = spec['projection_groups']
        stamp = spec['producer_sha256']
        if (not isinstance(groups, list) or len(groups) != len(streams)
                or any(not isinstance(g, str) or not g or '.' in g for g in groups)
                or not isinstance(stamp, str) or len(stamp) != 64
                or any(c not in '0123456789abcdef' for c in stamp)):
            raise ValueError('Explicit projection sharing and producer hash required')
        sizes = {}
        for group, stream in zip(groups, streams):
            if group in sizes and sizes[group] != stream.channels:
                raise ValueError('Shared projections require equal input channels')
            sizes[group] = stream.channels
        if (sum('movable' in s.camera_roles for s in streams) != 1
                or sum('movable' not in s.camera_roles for s in streams) > 1):
            raise ValueError('Shared core currently supports one main and at most one context stream')
        return streams
    if (set(spec)!={'schema','layout','width','heads','layers','producer_sha256'} or spec['schema']!='NATIVE_CONTINUOUS_POLICY_V1'
            or any(type(spec[k]) is not int or spec[k]<1 for k in ('width','heads','layers'))
            or spec['width']%spec['heads'] or not isinstance(spec['producer_sha256'],str)
            or len(spec['producer_sha256'])!=64 or any(c not in '0123456789abcdef' for c in spec['producer_sha256'])):
        raise ValueError('Explicit compatible native policy/producer specification required')
    return read_layout(spec['layout'])


def create_search_policy(spec):
    streams=validate_policy_spec(spec)
    if spec['schema'] == 'UNIFIED_NATIVE_CONTINUOUS_POLICY_V1':
        from .unified_search_policy import UnifiedActorCritic
        policy = UnifiedActorCritic(spec['core'], streams=streams,
            groups=spec['projection_groups'], producer_sha256=spec['producer_sha256'],
            task_spec=spec.get('task_spec'))
    else:
        from .native_search_policy import NativeContinuousActorCritic
        policy=NativeContinuousActorCritic(streams,width=spec['width'],heads=spec['heads'],layers=spec['layers'])
    policy.native_spec=deepcopy(spec)
    policy.feature_contract_sha256=digest(spec)
    return policy


def validate_publication(policy,payload,*,version,expected_origin):
    from ..data.native_training import validate_origin
    validate_origin(expected_origin)
    if (payload['version']!=version or payload.get('feature_contract_sha256')!=policy.feature_contract_sha256):
        raise ValueError('Stale policy or native feature contract differs')
    if (not isinstance(expected_origin,dict) or expected_origin.get('schema')!='NATIVE_ACTOR_ORIGIN_V1'
            or expected_origin.get('feature_contract_sha256')!=policy.feature_contract_sha256
            or payload.get('origin')!=expected_origin
            or payload.get('training_protocol_sha256')!=expected_origin.get('training_protocol_sha256')):
        raise ValueError('Published actor origin/model/preprocessing/training protocol differs')
    policy.load_state_dict(payload['state_dict'],strict=True)

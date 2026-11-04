"""Native-feature target training identity contract, no model/optimizer imports."""
from .artifacts import read
from copy import deepcopy
from ..execution.manifest import file_digest,digest

COLLECTION_SCHEMA='CONTROLLED_NATIVE_COLLECTION_V1'
LEARNER_SCHEMA='CONTROLLED_NATIVE_LEARNER_V1'
PROTOCOL_SCHEMA='TARGET_NATIVE_PPO_V1'


def from_campaign(c):
    path=c['protocol']
    if c['input_hashes'].get(path)!=file_digest(path):raise ValueError('Unbound native training protocol')
    p=read(path)
    if (p['schema']!=PROTOCOL_SCHEMA or p['arms']!=['G'] or p['seeds']!=[0,1,2]
            or p['updates']!=64 or p['episodes_per_update']!=32 or p['query_cap']!=8 or p['query_cost']!=.01
            or p['observation_scale']!=.16535300100378683
            or not isinstance(p['family'],str) or not p['family'].strip()):
        raise ValueError('Explicit aligned native G training protocol required')
    if c.get('native_policy')!=p['native_policy']:raise ValueError('Native policy contract differs')
    if (p['native_policy']['width'],p['native_policy']['heads'],p['native_policy']['layers'])!=(128,4,2):
        raise ValueError('Frozen native mainline search architecture differs')
    if set(p['schedules'])!={'0','1','2'}:raise ValueError('Three native seed schedules required')
    if set(p['model_hashes'])!={'libero_spatial','libero_object','libero_goal','libero_10'}:
        raise ValueError('Pinned per-suite operation model identities required')
    if any(not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h) for h in p['model_hashes'].values()):
        raise ValueError('Native operation model SHA256 required')
    if set(p['runtime_contract_hashes'])!=set(p['model_hashes']) or any(not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h) for h in p['runtime_contract_hashes'].values()):
        raise ValueError('Pinned per-suite native preprocessing/runtime contracts required')
    return p


def reward_config():
    from .reward_pilot import reward_config as existing
    return existing('G')


def publication_origin(protocol, protocol_sha256, seed):
    """Mandatory semantic origin, separate from shape and file integrity.

    The caller has validated the frozen protocol. A transfer must publish a new
    explicitly reviewed target initialization; matching tensor shapes is not a
    transfer contract. No operation model or tensor is loaded here.
    """
    if type(seed) is not int or seed not in protocol['seeds']:
        raise ValueError('Explicit native training seed required')
    if not isinstance(protocol_sha256,str) or len(protocol_sha256)!=64 or any(c not in '0123456789abcdef' for c in protocol_sha256):
        raise ValueError('Frozen training protocol SHA256 required')
    origin=dict(schema='NATIVE_ACTOR_ORIGIN_V1',family=protocol['family'],
        model_hashes=deepcopy(protocol['model_hashes']),
        runtime_contract_hashes=deepcopy(protocol['runtime_contract_hashes']),
        producer_sha256=protocol['native_policy']['producer_sha256'],
        feature_contract_sha256=digest(protocol['native_policy']),
        training_protocol_sha256=protocol_sha256,
        initial_checkpoint_sha256=file_digest(protocol['initial_checkpoint']),
        run=f'G-seed-{seed}',seed=seed,
        scoring=dict(metric='accel',prefix=3,selection='minimum_visited',reward='G',
                     query_cap=protocol['query_cap'],query_cost=protocol['query_cost']))
    validate_origin(origin)
    return origin


def validate_origin(origin):
    required={'schema','family','model_hashes','runtime_contract_hashes','producer_sha256',
              'feature_contract_sha256','training_protocol_sha256','initial_checkpoint_sha256','run','seed','scoring'}
    if not isinstance(origin,dict) or set(origin)!=required or origin['schema']!='NATIVE_ACTOR_ORIGIN_V1':
        raise ValueError('Complete native actor origin required')
    suites={'libero_spatial','libero_object','libero_goal','libero_10'}
    if (not isinstance(origin['family'],str) or not origin['family'].strip()
            or type(origin['seed']) is not int or origin['seed'] not in (0,1,2)
            or origin['run']!=f"G-seed-{origin['seed']}"
            or set(origin['model_hashes'])!=suites or set(origin['runtime_contract_hashes'])!=suites
            or origin['scoring']!=dict(metric='accel',prefix=3,selection='minimum_visited',reward='G',query_cap=8,query_cost=.01)):
        raise ValueError('Native actor origin family/run/scoring contract differs')
    hashes=[origin[k] for k in ('producer_sha256','feature_contract_sha256','training_protocol_sha256','initial_checkpoint_sha256')]
    hashes.extend(origin['model_hashes'].values());hashes.extend(origin['runtime_contract_hashes'].values())
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in hashes):
        raise ValueError('Native actor origin requires full SHA256 identities')


def admission_runtime_sha256(sim):
    """Admit the runtime on train states, not the held-out data itself.

    Only the population and anchor-noise paths vary. Every other field, including
    weights, preprocessing, cameras, operation limits and feature tap, is pinned.
    Each campaign must separately bind and validate its own inputs and noises.
    """
    if any(not isinstance(sim.get(k),str) or not sim[k] for k in ('inputs','anchors')):
        raise ValueError('Explicit population and anchor noise assets required')
    return digest({k:v for k,v in sim.items() if k not in ('inputs','anchors')})


def validate_admission(c,sim,p):
    path=c['native_admission']
    if c['input_hashes'].get(path)!=file_digest(path):raise ValueError('Unbound native feature GPU admission')
    proof=read(path)
    if (proof.get('status')!='PASS_NATIVE_FEATURE_TRAJECTORY' or proof.get('scope')!='REAL_NATIVE_MODEL'
            or proof.get('native_policy_sha256')!=digest(p['native_policy'])
            or proof.get('family')!=p['family']
            or proof.get('source_identity')!=p.get('native_source_identity') or p.get('native_source_identity') is None
            or any(proof.get(k) is not True for k in ('actions_exact','trace_exact','rng_exact','fresh_features','return_exact'))):
        raise ValueError('CPU simulation cannot admit a native feature production runtime')
    admitted_path=proof.get('admitted_sim_config')
    if (not isinstance(admitted_path,str) or not proof.get('sim_config_sha256')
            or c['input_hashes'].get(admitted_path)!=proof['sim_config_sha256']
            or file_digest(admitted_path)!=proof['sim_config_sha256']):
        raise ValueError('Unbound original native admission simulator configuration')
    expected=admission_runtime_sha256(sim)
    if (proof.get('runtime_identity_sha256')!=expected
            or admission_runtime_sha256(read(admitted_path))!=expected):
        raise ValueError('Native runtime differs from admitted training-state runtime')
    if sim.get('native_features',{}).get('layout')!=p['native_policy']['layout']:
        raise ValueError('Native simulator/actor feature mapping differs')
    if digest(sim['native_features'])!=p['native_policy']['producer_sha256']:
        raise ValueError('Native feature layer/grid/producer differs from trained policy')

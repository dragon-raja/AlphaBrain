"""Explicit fixed-prefix training amendments; no sampling or model imports."""
import json
from pathlib import Path
from .artifacts import sha


def load_stage(path, expected_sha256):
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError('Training stage amendment hash changed')
    stage = json.loads(path.read_text())
    if (stage.get('schema') != 'PPO_FIXED_STAGE_V1'
            or stage.get('selection') != 'USER_FIXED_BEFORE_ENDPOINT_NOT_OUTCOME_SELECTED'
            or stage.get('continuation') != 'EXPLICIT_LATER_AUTHORIZATION_REQUIRED'
            or stage.get('evaluation_panel_changed') is not False
            or stage.get('optimizer_schedule_or_reward_changed') is not False
            or stage.get('preserve_full_checkpoint') is not True):
        raise ValueError('Unsupported training stage amendment')
    if sha(stage['original_plan']) != stage['original_plan_sha256']:
        raise ValueError('Original training plan changed')
    original = json.loads(Path(stage['original_plan']).read_text())['training']
    updates, episodes, size = (stage[k] for k in ('updates', 'episodes_per_run', 'episodes_per_update'))
    if (any(type(x) is not int or x < 1 for x in (updates, episodes, size))
            or episodes != updates * size or size != original['episodes_per_update']
            or updates > original['updates_per_run'] or episodes > original['episodes_per_run']):
        raise ValueError('Stage must end at a complete prefix of the original schedule')
    names = {f'{a}-seed-{s}' for a in original['arms'] for s in original['seeds']}
    if set(stage['runs']) != names:
        raise ValueError('Stage must preserve all original arms and seeds')
    for name, spec in stage['runs'].items():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Original learner config changed')
        cfg = json.loads(Path(spec['path']).read_text())
        if (cfg['run'] != name or cfg['protocol'] != stage['original_plan']
                or cfg['updates'] != original['updates_per_run'] or cfg['episodes_per_update'] != size):
            raise ValueError('Stage config does not preserve the original learner')
    expected_cap = episodes * len(names) * original['query_cap'] * 2
    if stage['training_calls_cap'] != expected_cap:
        raise ValueError('Stage query cap differs from the original per-episode allowance')
    return stage


def evaluation_endpoint(campaign):
    """Old campaigns retain the original endpoint; amendments must be bound."""
    from .coordinate_probe import SCHEMA as COORDINATE_SCHEMA
    from .coordinate_validation import SCHEMA as FULL_COORDINATE_SCHEMA
    if campaign.get('schema') in (COORDINATE_SCHEMA,FULL_COORDINATE_SCHEMA):
        path=campaign['mechanism_protocol']
        if sha(path)!=campaign['input_hashes'].get(path):raise ValueError('Unbound mechanism actor endpoint')
        p=json.loads(Path(path).read_text())
        if p['actor_updates']!=64 or p['actor_episodes']!=2048:raise ValueError('Changed mechanism endpoint')
        return 64,2048
    if 'adaptation_protocol' in campaign:
        from .reward_pilot import from_campaign
        p=from_campaign(campaign)
        return p['updates'], p['updates']*p['episodes_per_update']
    if 'training_stage' not in campaign:
        return 256, 8192
    path = campaign['training_stage']
    stamp = campaign.get('input_hashes', {}).get(path)
    if not stamp:
        raise ValueError('Unbound training stage amendment')
    stage = load_stage(path, stamp)
    return stage['updates'], stage['episodes_per_run']

"""Separate pi0.5 random-acquisition ablation; never rewrites frozen rows."""
from .artifacts import read
from .flow_noise import stable_uint64
from ..execution.manifest import file_digest

SCHEMA='INDEPENDENT_RANDOM_RECOVERY_V1'
PROTOCOL_SCHEMA='UNIFORM_LEGAL_INCREMENT_MINIMUM_V1'


def rows(original):
    base=[r for r in original if r['method']=='B-seed-0']
    if len(base)!=3200 or len({(r['state_key'],r['view_id']) for r in base})!=3200:
        raise ValueError('Full frozen 40-task/10-initial/8-start paired population required')
    return [dict(r,method=f'random_acquisition-seed-{seed}',selector='random_acquisition',
        acquisition_seed=stable_uint64(f"{r['state_key']}::{r['view_id']}::{seed}",root_seed=20261030))
        for r in base for seed in (0,1,2)]


def validate_contract(c):
    path=c['independent_random_protocol']
    if c['input_hashes'].get(path)!=file_digest(path):raise ValueError('Unbound independent random protocol')
    p=read(path)
    expected=dict(schema=PROTOCOL_SCHEMA,seeds=[0,1,2],root_seed=20261030,
        proposal='uniform_legal_increment',stop='query_cap',selection='minimum_visited',
        query_cap=8,selection_members=[0,1],validation_members=[2,3],execution_repeats=[0,1],
        diagnostics='selected_and_start_not_used_for_selection',training_updates=0,
        baseline_campaign_sha256=file_digest(c['baseline_campaign']))
    if p!=expected or c['actors']:
        raise ValueError('Independent random acquisition must not load actors or alter its frozen rule')
    return p

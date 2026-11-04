"""Predeclared held-out-task test of recovery versus a mechanical-home rule.

Offsets alter installation reference, not the existing camera FK. The common
operational yaw window is +/-15; physical yaw remains inside +/-30. Endpoints
are compared within each condition, never treated as a pure input intervention.
"""
from .artifacts import read, sha
from .flow_noise import stable_uint64
from .native_recovery import limits

SCHEMA='G_COORDINATE_VALIDATION_V1'
FAMILY='heldout_installation_v1'
OFFSETS=(-15.,0.,15.)
STARTS=((-10.,-10.),(-10.,10.),(10.,-10.),(10.,10.))
METHODS=('G-seed-0','G-seed-1','G-seed-2','local','stay')
CRITERIA=dict(primary_offsets=[-15.,15.],minimum_net_recovery_pp=5.,
    minimum_home_advantage_pp=5.,paired_task_interval_lower_must_exceed=0.,
    maximum_harm_vs_stay_pp=10.,minimum_positive_seeds_vs_stay=3,
    minimum_nonnegative_seeds_vs_home=2,local_noninferiority_margin_pp=5.,
    bootstrap_draws=10000,bootstrap_seed=20261028,task_weighting='equal',
    insufficient_evidence='HOLD_NOT_AUTOMATICALLY_EXPAND',
    normative_reference='reported_oracle_not_policy_input')


def planned_rows(baseline,records):
    states=[s for s in baseline['states'] if records[s['initial']]['state']['split']=='test_unseen_task'
        and records[s['initial']]['state']['initialization']['init_state_index'] in (40,44,48)]
    if len(states)!=36 or len({s['task'] for s in states})!=12:
        raise ValueError('All twelve unseen tasks and three fixed initials required')
    result=[]
    for state in sorted(states,key=lambda s:(s['suite'],s['task'],s['initial'])):
        for oi,offset in enumerate(OFFSETS):
            for si,start in enumerate(STARTS):
                condition=100+oi*4+si
                for method in METHODS+(('home',) if si==0 else ()):
                    selector=('continuous_ppo' if method.startswith('G-') else
                              'local_near_best_accel' if method=='local' else
                              'fixed_home' if method=='home' else 'stay')
                    result.append(dict(state,split='unseen_test',start=si,view_id=condition,
                        angles_deg=[offset+start[0],start[1]],method=f'{method}-mount-{int(offset):+d}',
                        method_family=method,actor_key=method if method.startswith('G-') else None,
                        selector=selector,coordinate_family=FAMILY,mount_yaw_offset_deg=offset,
                        search_yaw_limit_deg=15.,query_cap=8,execution_repeats=[0,1],
                        proposal_seed=stable_uint64(f"{state['state_key']}::{condition}",root_seed=20261028),
                        random_seed=0))
    if len(result)!=2268:raise ValueError('Complete validation panel changed')
    return result


def reference_artifacts(refs,rows):
    result={}
    for row in rows:
        for path,h in refs['views'][f"{row['state_key']}::0"]['artifacts'].items():
            if path in result and result[path]!=h:raise ValueError('Conflicting nominal reference')
            result[path]=h
    return result


def validate_contract(c,planned):
    path=c['mechanism_protocol']
    if sha(path)!=c['input_hashes'].get(path):raise ValueError('Unbound full mechanism protocol')
    p=read(path)
    if (p['schema']!=SCHEMA or p['rows']!=planned or p['criteria']!=CRITERIA
            or p['call_caps']!=limits(planned) or p['actor_updates']!=64
            or p['actor_episodes']!=2048 or p['training_updates']!=0
            or p['automatic_retry'] is not False):raise ValueError('Full mechanism protocol changed')
    for path,h in p['input_hashes'].items():
        if sha(path)!=h:raise ValueError('Changed full mechanism input')
    a=read(p['authorization'])
    if (a['scope']!='USER_REQUESTED_COMPLETE_MECHANISM_VALIDATION'
            or a['call_caps']!=p['call_caps'] or a['main_experiment'] is not False):
        raise ValueError('Complete bounded validation authorization required')
    for row in planned:
        if row['actor_key'] and c['actors'][row['method']]!=p['actors'][row['actor_key']]:
            raise ValueError('Frozen G actor substitution')
    return p

"""Small matched G evaluation with displaced mounting references.

Existing virtual-camera kinematics and physical bounds are unchanged. Search
uses a common +/-25 yaw operational envelope so +/-5 installation offsets stay
within the existing physical +/-30 envelope. No task outcomes select the panel.
"""
from .artifacts import read, sha
from .native_view_panel import views
from .flow_noise import stable_uint64

SCHEMA = 'G_MOUNTING_MECHANISM_RECOVERY_V1'


def planned_rows(baseline, records):
    tasks = {}
    for s in baseline['states']:
        if records[s['initial']]['state']['split'] == 'validation':
            tasks.setdefault(s['suite'], set()).add(s['task'])
    if len(tasks) != 4: raise ValueError('Four-suite development panel required')
    chosen = {suite: sorted(names)[0] for suite, names in tasks.items()}
    selected=[]
    for suite, task in sorted(chosen.items()):
        states=sorted([s for s in baseline['states'] if s['suite']==suite and s['task']==task],
                      key=lambda s: records[s['initial']]['state']['initialization']['init_state_index'])[:2]
        if len(states)!=2: raise ValueError('Two distinct development initial states required')
        selected.extend(states)
    rows=[]
    for state in selected:
        for view in (1, 2):
            for offset in (-5., 0., 5.):
                for seed in (0,1,2):
                    name=f'G-seed-{seed}-mount-{int(offset):+d}'
                    rows.append(dict(state,split='validation',start=0,view_id=view,angles_deg=views()[view],
                        method=name,actor_key=f'G-seed-{seed}',mount_yaw_offset_deg=offset,
                        search_yaw_limit_deg=25.,query_cap=8,execution_repeats=[0,1],
                        proposal_seed=stable_uint64(f"{state['state_key']}::{view}",root_seed=20261028),
                        random_seed=0))
    if len(rows)!=144: raise ValueError('Small mechanism panel must contain 144 searches')
    return rows


def validate_contract(c, planned):
    protocol=read(c['mechanism_protocol'])
    if sha(c['mechanism_protocol'])!=c['input_hashes'].get(c['mechanism_protocol']):
        raise ValueError('Unbound mechanism protocol')
    if (protocol['schema']!=SCHEMA or protocol['rows']!=planned
            or protocol['actor_updates']!=64 or protocol['actor_episodes']!=2048
            or protocol['training_updates']!=0 or protocol['automatic_retry'] is not False
            or protocol['mount_offsets_deg']!=[-5.,0.,5.] or protocol['search_yaw_limit_deg']!=25.):
        raise ValueError('Mechanism panel, frozen actor or mounting convention changed')
    from .native_recovery import limits
    if protocol['call_caps']!=limits(planned): raise ValueError('Mechanism budget changed')
    for path,stamp in protocol['input_hashes'].items():
        if sha(path)!=stamp:raise ValueError('Mechanism scientific input changed')
    authorization=read(protocol['authorization'])
    if (authorization['scope']!='USER_REQUESTED_SMALL_COORDINATE_MECHANISM_CHECK'
            or authorization['call_caps']!=protocol['call_caps'] or authorization['main_experiment'] is not False):
        raise ValueError('Small mechanism authorization required')
    for row in planned:
        spec=c['actors'][row['method']]
        if spec != protocol['actors'][row['actor_key']]:raise ValueError('Actor substitution')
    return protocol

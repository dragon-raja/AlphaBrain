"""CPU preparation for matched LIBERO / Plus protocols, not new evaluation."""
from pathlib import Path
import numpy as np

from ..data.artifacts import read,sha,write,path_mapping
from .initial_study import MODELS


def describe_task(success,accel,extra):
    success,accel=np.asarray(success),np.asarray(accel)
    if success.ndim!=3 or accel.shape!=success.shape[:2] or not np.isin(success,[0,1]).all():
        raise ValueError('Expected initial/view/repeat outcomes and matching scores')
    q=success.mean(-1)
    canonical=float(q[:,0].mean());stay=float(q[:,extra].mean())
    choice=accel.argmin(1)
    return dict(initials=len(q),canonical=canonical,perturbed_stay=stay,view_loss=canonical-stay,
        initial_accel_selected=float(q[np.arange(len(q)),choice].mean()),
        descriptive_group=('capable_view_sensitive' if canonical>=.8 and canonical-stay>=.1
                           else 'capable_low_view_loss' if canonical>=.8 else 'lower_reference_capacity'))


def prepare(matrix_pointer,camera_inventory,output,*,path_mappings=()):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    with path_mapping(path_mappings):
        pointer=read(matrix_pointer);root=Path(pointer['archive']);states=read(root/'states.json')
        for name in ('matrix.npz','states.json','summary.json'):
            if sha(root/name)!=pointer['artifacts'][name]:raise ValueError('Historical analysis artifact changed')
    inventory=read(Path(camera_inventory)/'summary.json')
    cameras_file=Path(camera_inventory)/'camera-entries.json'
    if sha(cameras_file)!=inventory['camera_entries_sha256']:raise ValueError('Camera inventory changed')
    cameras=read(cameras_file)
    with np.load(root/'matrix.npz',allow_pickle=False) as z:
        y=z['success'].copy();a=z['accel_members'].mean(-1);views=z['candidate_ids'].tolist()
    if y.shape!=(2,32,97,32) or len(states)!=32 or views[0]!='canonical':
        raise ValueError('Expected original complete LIBERO matrix')
    extra=np.array([i for i,v in enumerate(views) if v.startswith('broad_heldout_')])
    if len(extra)!=32:raise ValueError('Keep the original 32-view disturbance baseline')
    tasks=np.array([s['task_id'] for s in states]);splits=np.array([s['split'] for s in states])
    rows=[]
    for m,model in enumerate(MODELS):
        for task in sorted(set(tasks)):
            for scope in ('development','heldout_initial','all'):
                mask=(tasks==task)&(np.ones(32,bool) if scope=='all' else splits==scope)
                if not mask.any():raise ValueError('Missing historical initial split '+scope)
                rows.append(dict(model=model,task=task,scope=scope,**describe_task(y[m,mask],a[m,mask],extra)))
    profiles=[dict(model=r['model'],task=r['task'],group=r['descriptive_group'],
                   capacity=r['canonical'],view_loss=r['view_loss'],initials=r['initials'])
              for r in rows if r['scope']=='development']
    mappings=[]
    for camera in cameras:
        matches=[s for s in states if s['suite']==camera['suite']
                 and Path(s['bddl_file']).stem==camera['base_task']]
        mappings.append(dict(suite=camera['suite'],camera_task=camera['name'],
            base_task=camera['base_task'],camera_parameters={k:camera[k] for k in
                ('horizon','vertical','distance_scale','endpoint_yaw','endpoint_pitch')},
            robot_initial_variant=camera['robot_initial_variant'],
            historical_initials=[dict(task_id=s['task_id'],split=s['split'],
                index=s['initialization']['init_state_index'],
                array_path=s['initialization']['array_path'],
                array_sha256=s['initialization']['array_sha256']) for s in matches],
            inference_status='NOT_COLLECTED_BY_THIS_PREPARATION',
            reuse_rule='Do not reuse a custom-catalog outcome without identical initial, pose, policy and noise identities'))
    result=dict(status='CPU_PROTOCOL_PREPARATION_COMPLETE_GPU_NOT_AUTHORIZED_BY_THIS_FILE',
        sources={str(matrix_pointer):sha(matrix_pointer),str(root/'matrix.npz'):sha(root/'matrix.npz'),
                 str(root/'states.json'):sha(root/'states.json'),str(cameras_file):sha(cameras_file),
                 str(Path(__file__)):sha(__file__)},
        scope='Keep all original eight tasks; descriptive groups are not an eligibility filter.',
        task_removal=False,training_steps=0,new_policy_inferences=0,new_closed_loops=0,
        group_definition='Same descriptive C>=0.80, D>=0.10 thresholds as RoboCasa; use old development initials only.',
        primary_claim='No new cross-benchmark result. Original data are exploration, not fresh confirmation.',
        task_profiles=rows,frozen_development_groups=profiles,
        plus_mapping=dict(camera_entries=len(mappings),entries_with_original_initials=sum(bool(r['historical_initials']) for r in mappings),
            possible_initial_camera_pairs=sum(len(r['historical_initials']) for r in mappings),
            meaning='Address mapping only, not an approved GPU workload or newly generated data.'),
        reusable=['Original matched one-window scores and 32-repeat full outcomes in the custom97 catalog.',
                  'Original eight task definitions, 32 initial arrays and policy provenance.'],
        missing=['LIBERO candidate-specific short probes for windows2-5; cannot derive them from initial scores.',
                 'Plus-native pose/RGB/runtime admission; task-name matching alone is insufficient.',
                 'Plus matched scores and full outcomes; no outcome relabeling from custom97.',
                 'Fresh confirmation initials/noise if independent LIBERO or Plus claims are required.'],
        deployment_caveat='Camera installation perturbations are not yet executable active-head action trajectories.')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'plus-initial-mapping.json',mappings)
    write(output/'summary.json',result)
    return result

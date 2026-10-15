"""Freeze and launch the user-approved bounded XR-1 diagnostic, not a full matrix."""
import os
from pathlib import Path
import shutil
import subprocess
import time

from .artifacts import read, sha, write


def phase_views(original, base, phase):
    """Partition the original 97 catalog; never choose views from outcomes."""
    small = base['panel_views']
    if len(small) != 5 or len(set(small)) != 5 or small[0] != 'canonical':
        raise ValueError('Expected original five-view diagnostic')
    if phase == 'canonical':
        return small[:1]
    if phase == 'perturbed':
        return small[1:]
    if phase == 'remaining':
        full = original['all_views']
        if len(full) != 97 or len(set(full)) != 97 or full[0] != 'canonical' or not set(small) <= set(full):
            raise ValueError('Expected original complete 97-view catalog')
        return [v for v in full if v not in small]
    raise ValueError('Unknown phase')


def prepare_and_launch(panel,checkpoint,upstream,model_python,sim_python,source,output,pilot,phase='canonical'):
    panel,checkpoint,upstream,source,output=map(Path,(panel,checkpoint,upstream,source,output))
    passed=read(pilot)
    if passed['status']!='COMPLETE' or passed.get('completed_rollouts')!=1 or not passed['pilot']:
        raise ValueError('One completed full-task engineering pilot is required')
    imported=read(checkpoint/'IMPORT_RECEIPT.json')
    if imported['status']!='BYTE_VERIFIED':
        raise ValueError('Checkpoint is not byte verified')
    original=read(panel/'plan.json');base=read(original['baseline_plan'])
    initials=original['initial_ids'];views=base['panel_views']
    selection=read(panel/'selection.json')
    if len(initials)!=40 or len(set(initials))!=40 or len(views)!=5 or views[0]!='canonical':
        raise ValueError('Expected frozen 40-initial, five-view panel')
    views=phase_views(original,base,phase)
    total=40*len(views)*4
    output.mkdir(parents=True,exist_ok=False)
    frozen=output/'source'
    for directory in ('AlphaBrain','benchmarks','scripts','configs'):
        shutil.copytree(source/directory,frozen/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    write(output/'source-manifest.json',files)
    workers=[dict(shard=i,shards=8,gpu=i,initials=initials[i::8]) for i in range(8)]
    plan=dict(format='robocasa_xr1_bounded_panel_v1',created_unix=time.time(),
        authorization='User approved four-step validation and stronger public model on 2026-09-29',
        panel=str(panel),panel_sha256=sha(panel/'plan.json'),jobs=selection['jobs'],views=views,
        initial_ids=initials,execution_repeats=4,rollouts=total,score_repeats=8,phase=phase,
        total_bounded_budget=15520 if phase=='remaining' else 800,
        native_replan_steps=16,native_observation_history=4,native_observation_interval=2,
        native_flow_steps=5,noise_version='xr1_paired_v1',renderer='native_4x_msaa',
        checkpoint=str(checkpoint),checkpoint_receipt_sha256=sha(checkpoint/'IMPORT_RECEIPT.json'),
        upstream=str(upstream),upstream_revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
        model_python=str(model_python),sim_python=str(sim_python),source=str(frozen),
        source_manifest_sha256=sha(output/'source-manifest.json'),workers=workers,
        physical_camera_motion=False,selector_confirmation_not_launched=True,
        caveat='Cross-model native inference settings differ; not a pure causal intervention on model competence',
        pilot=str(pilot),pilot_sha256=sha(pilot))
    write(output/'plan.json',plan)
    launches=[]
    for worker in workers:
        destination=output/'workers'/f'fresh-gpu{worker["gpu"]}'
        env=dict(CUDA_VISIBLE_DEVICES=str(worker['gpu']),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_MODULES_CACHE=str(Path(model_python).parents[1]/'hf-modules'),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(frozen))
        cmd=[str(sim_python),str(frozen/'scripts/dsol_paper1/runtime/robocasa_xr1_worker.py'),
            '--panel',str(panel),'--checkpoint',str(checkpoint),'--upstream',str(upstream),
            '--model-python',str(model_python),'--output',str(destination),'--formal',
            '--shard',str(worker['shard']),'--shards','8','--phase',phase,'--execute']
        with (output/f'worker-{worker["gpu"]}.log').open('xb') as log:
            process=subprocess.Popen(cmd,cwd=frozen,env={**os.environ,**env},
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launches.append(dict(**worker,pid=process.pid,command=cmd,environment=env,output=str(destination)))
        write(output/'launch.json',dict(status='LAUNCHING',workers=launches))
    write(output/'launch.json',dict(status='LAUNCHED',workers=launches,planned_rollouts=total,phase=phase))
    return dict(status='LAUNCHED',workers=len(launches),planned_rollouts=total,phase=phase,root=str(output))

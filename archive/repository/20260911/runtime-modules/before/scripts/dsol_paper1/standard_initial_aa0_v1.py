#!/usr/bin/env python3
"""32 official initial states, fixed-camera whole-task evaluation, AA0."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import dualhost_aa0_v1 as core
from standard_initialization_v1 import KIND, load_initial, initialize

PREDECESSOR = core.ROOT
ROOT = Path('/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908')
REPO = ROOT / 'repo'
core.ROOT = ROOT
core.REPO = REPO
core.__file__ = str(Path(__file__).resolve())
old_assets = core.assets
read, write, sha, require = core.read, core.write, core.sha, core.require


def build():
    import re
    import torch
    from explicit_flow_noise import materialize_bank, stable_uint64
    require(not (ROOT/'release.json').exists(), 'Never overwrite frozen release')
    old = read(PREDECESSOR/'release.json')
    catalog_path = REPO/'configs/dsol_paper1/libero_view_catalog_v2_m1.json'
    require(sha(catalog_path) == old['catalog']['sha256'], 'Candidate catalog changed')
    candidates = [{k: c[k] for k in ['selected_candidate_id', 'pose']}
                  for c in old['protocol']['state_blocks'][0]['candidates']]
    require(len(candidates) == 97 and candidates[0]['selected_candidate_id'] == 'canonical', 'Full candidate list required')
    templates = {s['task_id']: s for s in old['states']}
    states, sources = [], {}
    for ti, task_id in enumerate(sorted(templates)):
        prior = templates[task_id]
        stem = Path(prior['hdf5']).stem
        require(stem.endswith('_demo'), 'Unexpected task-name metadata')
        bddl_name = stem[:-5] + '.bddl'
        # Only the historical task filename is used to locate the official BDDL.
        # Neither HDF5 states nor HDF5 model XML nor HDF5 metadata are loaded.
        bddl = core.RUNTIME/'libero/libero/bddl_files'/prior['suite']/bddl_name
        languages = re.findall(r'\(:language\s+([^)]*)\)', bddl.read_text())
        require(len(languages)==1, 'Ambiguous BDDL language')
        prompt = ' '.join(languages[0].split())
        official = core.RUNTIME/'libero/libero/init_files'/prior['suite']/(Path(bddl_name).stem+'.pruned_init')
        initial = torch.load(official, weights_only=False)
        require(len(initial) >= 4, 'Fewer than four official initial states')
        sources[str(official)] = sha(official); sources[str(bddl)] = sha(bddl)
        seen = set()
        for ii in range(4):
            state = np.asarray(initial[ii])
            identity = core.array_identity(state)
            require(identity['sha256'] not in seen, 'Duplicate official initial state; do not silently replace')
            seen.add(identity['sha256'])
            path = ROOT/'initial-states'/task_id/f'{ii:02d}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            require(not path.exists(), 'Initial state already materialized')
            np.save(path, state, allow_pickle=False)
            key = f'official-initial-v1::{task_id}::{ii}'
            index = len(states)
            states.append(dict(pair_key=key,asset_source_pair_key=key,task_id=task_id,
                task_ordinal=ti,suite=prior['suite'],hdf5=prior['hdf5'],
                hdf5_usage='filename reference only to locate BDDL; file not opened',
                bddl_file=str(bddl),bddl_sha256=sha(bddl),prompt=prompt,
                split='development' if ii<2 else 'heldout_initial',source_group=key,
                environment_seed=int(stable_uint64(key,root_seed=20260908)%(2**31-1)),
                initial_asset_record=str(ROOT/'assets/aa0-a'/f'{index:02d}'/'render.json'),
                initialization=dict(kind=KIND,official_file=str(official),official_file_sha256=sources[str(official)],
                    init_state_index=ii,array_path=str(path),array_sha256=sha(path),
                    array_identity=identity,settle_steps=10)))
            sources[str(path)] = sha(path)
    bank = materialize_bank(output_dir=ROOT/'noise-banks',bank_id='INITIAL32',
                state_keys=[s['pair_key'] for s in states],repeat_count=32,max_replans=104,
                action_horizon=10,action_dim=7,root_seed=2026090801)
    protocol = dict(schema='dsol_compact_view_matrix_protocol_v1',state_blocks=[
        dict(state=s,candidates=candidates,scene_construction=None) for s in states],
        policy_repeat_ids=list(range(32)),episode_identity_prefix='standard-initial-aa0-v1',
        diagnostic_role='official_initial_whole_task_fixed_camera',noise_bank_id='INITIAL32',
        sensor_control='both',catalog=str(catalog_path))
    gate=[]
    for ti in range(8):
        si=ti*4+ti%4
        for ci,c in enumerate(candidates):
            if c['selected_candidate_id'] in ['canonical','broad_train_000','broad_heldout_000']:
                gate.extend(si*3104+ci*32+n for n in [0,1])
    paths=[p for folder in ['AlphaBrain','scripts','configs'] for p in (REPO/folder).rglob('*')
           if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py','.sh','.yaml','.json']]
    # Rotate host assignment by task so each initial-index wave covers all tasks across two hosts.
    hosts={h:dict(state_indices=[i for i,s in enumerate(states)
                if (s['task_ordinal']+s['initialization']['init_state_index'])%2==parity])
           for h,parity in [('fresh',0),('gnu',1)]}
    r=dict(schema='standard_initial_aa0_32_v1',protocol=protocol,states=states,models=old['models'],
        catalog=dict(path=str(catalog_path),sha256=sha(catalog_path)),
        noise_bank=dict(path=bank['manifest_path'],manifest_sha256=bank['manifest_sha256'],
                        file_sha256=bank['noise_file_sha256'],bank_id='INITIAL32'),
        hosts=hosts,gate_indices=gate,source_hashes={str(p):sha(p) for p in paths},initial_source_hashes=sources,
        candidate_count=97,noise_repeats=32,total_episodes=198656,offsamples=0,replan_steps=5,
        denoising_steps=10,workers=32,copies=2,base_port=30800,asset_cells=['aa0-a'],score_ensemble_size=8,
        settle_steps=10,max_steps_by_suite={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520},
        initialization='official reset + official pruned_init indices 0,1,2,3 + 10 common no-op steps BEFORE camera install',
        selection_estimands=['canonical','empirical_global_fixed','empirical_task_fixed','empirical_initial_oracle'],
        empirical_rule='average all 32 noises before max; include canonical; no per-noise winner selection',
        holdout='init 0,1 development; init 2,3 held out by initial source, NOT held-out tasks or proven training-disjoint states',
        stage_policy='four complete waves, one official init index across all eight tasks and both models per wave',
        automatic_expansion=False,predecessor_release_sha256=sha(PREDECESSOR/'release.json'),
        requirements=['no demo state or demo XML','no constructed occluder/background change','fixed external pose entire rollout',
                      'first actual model input must exactly match metric render','cross-host repeated full trace gate',
                      'all candidate poses and 32 noises retained','no keeper signals','no historical label union'])
    require(core.protocol_spec_count(protocol)==99328 and len(gate)==48, 'Wrong budget')
    write(ROOT/'release.json',r)
    print('FROZEN',ROOT,flush=True)


def render(args,r):
    import tempfile
    import shutil
    from audit_libero_hdf5_restore import _configure_runtime
    from evaluate_dsol_libero_hdf5_views import masked_policy_observation
    s=r['states'][args.index];out=ROOT/'assets/aa0-a'/f'{args.index:02d}'
    require(not out.exists(),'Asset output exists; inspect instead of overwrite')
    out.mkdir(parents=True)
    config=Path(tempfile.mkdtemp(prefix='dsol-initial-render-'))
    _configure_runtime(core.RUNTIME,Path(s['hdf5']).parent.parent,config)
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference,install_camera_pose
    from scan_libero_hdf5_views import _restore_reference
    from libero_visibility import task_entity_visibility
    from evaluate_pi05_libero_plus_views import agentview_camera_calibration,physics_state_sha256
    candidates=r['protocol']['state_blocks'][args.index]['candidates']
    mapping=[('external_images','observation/image',np.uint8),('wrist_images','observation/wrist_image',np.uint8),
             ('robot_states','observation/state',np.float32),('camera_intrinsics','camera_intrinsics',np.float64),
             ('camera_to_world_opencv','camera_to_world_opencv',np.float64)]
    values={d:[] for d,_,_ in mapping};visibility=[];identities=[];env=None
    try:
        with core.render_protocol(0) as contexts:
            env=OffScreenRenderEnv(bddl_file_name=s['bddl_file'],camera_names=('agentview','robot0_eye_in_hand'),
                    camera_heights=256,camera_widths=256,render_gpu_device_id=args.gpu)
            _,receipt=initialize(env,s,load_initial(s));before,size=physics_state_sha256(env)
            reference=capture_camera_reference(env,camera_name='agentview',table_plane_z=read(r['catalog']['path'])['table_plane_z'])
            def install(c):
                _restore_reference(env,reference)
                if c['pose'] is not None:install_camera_pose(env,reference,c['pose'])
                env.env._update_observables(force=True)
            for c in candidates:
                install(c)
                example,_,_=masked_policy_observation(env.env._get_observations(),prompt=s['prompt'],resize_size=224,
                        eval_seed=s['environment_seed'],camera_calibration=agentview_camera_calibration(env),sensor_control='both')
                identities.append(dict(candidate_id=c['selected_candidate_id'],inputs={k:core.array_identity(example[k]) for _,k,_ in mapping}))
                for d,k,dtype in mapping:values[d].append(np.asarray(example[k],dtype=dtype))
            for c in candidates:
                install(c);visibility.append(dict(candidate_id=c['selected_candidate_id'],**task_entity_visibility(env,
                        entity_names=list(env.env.obj_of_interest),camera_names=('agentview','robot0_eye_in_hand'),height=224,width=224)))
            require(physics_state_sha256(env)==(before,size),'Rendering changed physical state')
            require(contexts and set(contexts)=={0},'AA0 framebuffer missing')
            artifact=out/'policy_inputs.npz'
            np.savez_compressed(artifact,candidate_ids=np.asarray([c['selected_candidate_id'] for c in candidates]),
                               **{k:np.stack(v) for k,v in values.items()})
            write(out/'render.json',dict(status='PASS',pair_key=s['pair_key'],split=s['split'],task_id=s['task_id'],
                source_group=s['source_group'],language=s['prompt'],candidate_count=97,physics_state_sha256=before,
                physics_state_size=size,artifact=str(artifact),artifact_sha256=sha(artifact),render_samples=contexts,
                release_sha256=sha(ROOT/'release.json'),visibility=visibility,input_identities=identities,initialization_receipt=receipt))
    finally:
        if env is not None:env.close()
        shutil.rmtree(config)


def assets(args,r):
    if args.mode=='render':render(args,r)
    else:old_assets(args,r)


def run(args,r):
    import fcntl
    host=args.host; (ROOT/'hosts'/host).mkdir(parents=True,exist_ok=True)
    with (ROOT/'hosts'/host/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.status(host,'VERIFYING_STANDARD_INITIAL')
        for p,d in r['initial_source_hashes'].items():require(sha(p)==d,'Initial/BDDL source changed: '+p)
        for m in r['models'].values():require(sha(Path(m['path'])/'model.safetensors')==m['weights_sha256'],'Model changed')
        bank=read(r['noise_bank']['path']);require(sha(bank['noise_file'])==r['noise_bank']['file_sha256'],'Noise changed')
        write(ROOT/'hosts'/host/'hardware.json',dict(hostname=socket.gethostname(),keepalive_policy='untouched',
             gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv'],text=True)))
        owned=r['hosts'][host]['state_indices'];core.status(host,'RENDERING_OFFICIAL_INITIALS',initial_count=len(owned))
        def render_one(i):
            p=ROOT/'assets/aa0-a'/f'{i:02d}'/'render.json'
            if p.exists():require(read(p)['release_sha256']==sha(ROOT/'release.json'),'Asset release mismatch');return
            core.finish(core.spawn([core.python(True),Path(__file__),'render','--index',i,'--gpu',(i//2)%8],
                    ROOT/'hosts'/host/'logs'/f'render-{i}.log',core.sim_env()),900)
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(render_one,owned))
        core.status(host,'WAITING_ALL_INITIAL_ASSETS');start=time.time()
        while not all((ROOT/'assets/aa0-a'/f'{i:02d}'/'render.json').exists() for i in range(32)):
            require(time.time()-start<3600,'Peer asset preparation timeout');time.sleep(5)
        for phase in ['gate-a','gate-b']:
            core.status(host,phase);core.matrix(host,r,phase,r['gate_indices'])
        core.status(host,'WAITING_CROSS_HOST_GATE');start=time.time()
        while not core.check_gate(r):
            require(time.time()-start<3600,'Peer gate timeout');time.sleep(5)
        write(ROOT/'hosts'/host/'gate-pass.json',dict(status='PASS_STANDARD_INITIAL_FULL_TRACE_AND_INPUT_ASSETS',
            episodes_per_host=192,release_sha256=sha(ROOT/'release.json'),utc=time.time()))
        for model in ['canonical','broad']:
            core.status(host,'SCORING_STANDARD_INITIALS',model=model)
            ps=[core.spawn([core.python(),Path(__file__),'score','--host',host,'--model',model,'--index',g],
                 ROOT/'hosts'/host/'logs'/f'score-{model}-{g}.log',core.env(gpu=g)) for g in range(8)]
            for p in ps:core.finish(p,7200)
        for wave in range(4):
            subset=[i for i in owned if r['states'][i]['initialization']['init_state_index']==wave]
            indices=[i for offset in range(3104) for s in subset for i in [s*3104+offset]]
            phase=f'dense-init-{wave}'
            core.status(host,phase);core.matrix(host,r,phase,indices)
        core.status(host,'COMPLETE',dense_episodes=len(owned)*3104*2)
        write(ROOT/'hosts'/host/'dense-complete.json',dict(episodes=len(owned)*3104*2,utc=time.time(),
              release_sha256=sha(ROOT/'release.json')))


core.build=build
core.assets=assets
core.run=run
if __name__=='__main__':core.main()

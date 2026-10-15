"""Explicit CPU MuJoCo feasibility diagnostics, no renderer, policy or rollout."""
from pathlib import Path
import gzip
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.acquisition.kinematic_audit import camera_ancestry, inside_primitive, bounded_joint_path


def head_probe(xml_path):
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(xml_path)); data=mujoco.MjData(model)
    cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'robotview')
    joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ('head_1_joint','head_2_joint')]
    addresses=model.jnt_qposadr[joint_ids]
    original_positions=model.cam_pos.copy(); original_quats=model.cam_quat.copy()
    mujoco.mj_forward(model,data); p0=data.cam_xpos[cid].copy(); r0=data.cam_xmat[cid].reshape(3,3).copy()
    all_head_bodies=[]
    head_body=int(model.jnt_bodyid[joint_ids[0]])
    for body in range(model.nbody):
        ancestor=body
        while ancestor and ancestor != head_body: ancestor=int(model.body_parentid[ancestor])
        if ancestor == head_body: all_head_bodies.append(body)
    unaffected=[i for i in range(model.nbody) if i not in all_head_bodies]
    x0=data.xpos[unaffected].copy(); mat0=data.xmat[unaffected].copy(); samples=[]; max_error=0.; max_rotation_error=0.
    for pan in np.linspace(-.5,.5,7):
        for tilt in np.linspace(-.5,.5,7):
            data.qpos[:]=0; data.qpos[addresses]=[pan,tilt]; mujoco.mj_forward(model,data)
            max_error=max(max_error,float(np.max(np.abs(data.xpos[unaffected]-x0))))
            max_rotation_error=max(max_rotation_error,float(np.max(np.abs(data.xmat[unaffected]-mat0))))
            direction=-data.cam_xmat[cid].reshape(3,3)[:,2]
            samples.append(dict(pan=float(pan),tilt=float(tilt),position=data.cam_xpos[cid].tolist(),forward=direction.tolist()))
    path=bounded_joint_path([0,0],[.3,-.25],model.jnt_range[joint_ids],.05)
    for q in path:
        data.qpos[:]=0;data.qpos[addresses]=q;mujoco.mj_forward(model,data)
    unchanged=np.array_equal(original_positions,model.cam_pos) and np.array_equal(original_quats,model.cam_quat)
    if max_error>1e-12 or not unchanged: raise ValueError('Head FK altered unrelated body/mount')
    pos=np.array([s['position'] for s in samples]); directions=np.array([s['forward'] for s in samples])
    # Independent, bounded native-motor test. No direct qpos changes after start.
    dyn=mujoco.MjData(model);mujoco.mj_forward(model,dyn)
    target=dyn.qpos.copy();target[addresses]=[.3,-.25];initial_contacts=int(dyn.ncon)
    for _ in range(1500):
        tau=dyn.qfrc_bias+100*(target-dyn.qpos)-20*dyn.qvel
        for actuator in range(model.nu):
            joint=int(model.actuator_trnid[actuator,0]);command=tau[model.jnt_dofadr[joint]]/model.actuator_gear[actuator,0]
            dyn.ctrl[actuator]=np.clip(command,*model.actuator_ctrlrange[actuator]) if model.actuator_ctrllimited[actuator] else command
        mujoco.mj_step(model,dyn)
    dynamic_test=dict(steps=1500,seconds=float(dyn.time),controller='Diagnostic PD (kp100,kd20) plus bias compensation; all native motors, control clipping',
        target_rad=target[addresses].tolist(),achieved_rad=dyn.qpos[addresses].tolist(),
        max_head_error_rad=float(np.max(np.abs(dyn.qpos[addresses]-target[addresses]))),
        max_other_joint_drift=float(np.max(np.abs(np.delete(dyn.qpos-target,addresses)))),
        initial_contacts=initial_contacts,final_contacts=int(dyn.ncon),
        camera_mount_arrays_unchanged=bool(np.array_equal(original_positions,model.cam_pos) and np.array_equal(original_quats,model.cam_quat)),
        scope='Standalone robot native-motor smoke, not kitchen controller certification or collision-free task execution')
    if not np.isfinite(dyn.qpos).all() or dynamic_test['max_head_error_rad']>.005: raise ValueError(f'Native head motor smoke failed: {dynamic_test}')
    return dict(xml_sha256=sha(xml_path), camera_ancestry=camera_ancestry(Path(xml_path).read_text(),'robotview'),
        physical_joint_limits_rad=model.jnt_range[joint_ids].tolist(), sampled_joint_range_rad=[-.5,.5],
        sample_count=len(samples), non_head_body_max_position_change_m=max_error,
        non_head_body_max_rotation_component_change=max_rotation_error,dynamic_test=dynamic_test,
        camera_mount_arrays_unchanged=unchanged,
        max_camera_translation_m=float(np.linalg.norm(pos-p0,axis=1).max()),
        max_optical_angle_deg=float(np.rad2deg(np.arccos(np.clip(directions@(-r0[:,2]),-1,1))).max()),
        bounded_path_waypoints=len(path), samples=samples,
        scope='Actual MJCF forward kinematics plus isolated native-motor smoke; no image acquisition or task success')


def point_audit(job, record):
    import mujoco
    from scipy.spatial import ConvexHull, QhullError
    from .snapshot_bundle import load_snapshot_bundle
    bundle=load_snapshot_bundle(Path(job['path']))
    xml_file=Path(job['path'])/'compiled_model.xml.gz'
    model=mujoco.MjModel.from_xml_string(gzip.decompress(xml_file.read_bytes()).decode())
    data=mujoco.MjData(model); snap=bundle.snapshot
    # Restore geometry-relevant state only, never advance physics.
    for name,value in snap.model_runtime.items(): getattr(model,name)[:]=value
    state=snap.sim_state
    if len(state) != 1+model.nq+model.nv: raise ValueError('Unexpected flattened state format')
    data.time=state[0]; data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:]
    if snap.mocap_positions is not None: data.mocap_pos[:]=snap.mocap_positions
    if snap.mocap_quaternions is not None: data.mocap_quat[:]=snap.mocap_quaternions
    model.cam_pos[:]=snap.camera_positions;model.cam_quat[:]=snap.camera_quaternions
    mujoco.mj_forward(model,data)
    root=Path(record['candidate_root']); summary=read(root/'summary.json'); geometry=root/'geometry.npz'
    if sha(root/'summary.json') != record['candidate_summary_sha256'] or sha(geometry) != summary['files']['geometry.npz']['sha256']:
        raise ValueError('Candidate source mismatch')
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,n) for n in ('robot0_agentview_left','robot0_agentview_right')]
    with np.load(geometry) as z:
        local=z['camera_positions'][:,ids]; view_ids=z['view_ids'].tolist()
    points=np.stack([local[:,i]@data.xmat[model.cam_bodyid[c]].reshape(3,3).T + data.xpos[model.cam_bodyid[c]] for i,c in enumerate(ids)],axis=1).reshape(-1,3)
    flags=[]; unsupported=[]; meshes={}; checked=0
    kinds={int(getattr(mujoco.mjtGeom,'mjGEOM_'+k.upper())):k for k in ('box','sphere','capsule','cylinder','ellipsoid')}
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]): continue
        gp=(points-data.geom_xpos[g])@data.geom_xmat[g].reshape(3,3)
        typ=int(model.geom_type[g]); kind=kinds.get(typ)
        hit=inside_primitive(gp,kind,model.geom_size[g])
        if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(model.geom_dataid[g]); adr=int(model.mesh_vertadr[mid]); nv=int(model.mesh_vertnum[mid]); vertices=model.mesh_vert[adr:adr+nv].astype(float)
            relevant=np.all((gp > vertices.min(0)+1e-5)&(gp < vertices.max(0)-1e-5),axis=1)
            hit=np.zeros(len(points),bool)
            if relevant.any():
                try:
                    if mid not in meshes: meshes[mid]=ConvexHull(vertices).equations
                    eq=meshes[mid];hit[relevant]=np.all(gp[relevant]@eq[:,:3].T+eq[:,3] < -1e-5,axis=1)
                except QhullError:
                    unsupported.append(dict(geom=g,reason='degenerate mesh hull'));continue
        if hit is None:
            unsupported.append(dict(geom=g,type=typ,reason='unsupported geometry'));continue
        checked+=1
        for idx in np.flatnonzero(hit):
            flags.append(dict(view_id=view_ids[idx//2],camera=('left','right')[idx%2],geom=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),
                              body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])),type=kind or 'convex_collision_mesh',
                              camera_world_position=points[idx].tolist(),camera_geom_local_position=gp[idx].tolist(),geom_size=model.geom_size[g].tolist()))
    return dict(initial_id=job['initial_id'],task=job['task'],split=job['split'],
        checked_collision_geoms=checked,unsupported=unsupported,flags=flags,
        flagged_views=len({f['view_id'] for f in flags}),canonical_flagged=any(f['view_id']=='canonical' for f in flags),
        scope='Zero-size camera center inside collision primitive/convex hull at initial state only; not full mount/path clearance')


def run(config):
    cfg=read(config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    plan=read(cfg['matrix_plan']); ref=read(plan['reference_plan']);release=read(plan['reference_release'])
    if sha(plan['reference_plan'])!=plan['reference_plan_sha256'] or sha(plan['reference_release'])!=plan['reference_release_sha256']:
        raise ValueError('Frozen reference mismatch')
    records={r['initial_id']:r for r in release['records']};inputs={str(config):sha(config),cfg['matrix_plan']:sha(cfg['matrix_plan']),cfg['tiago_xml']:sha(cfg['tiago_xml'])}
    inputs.update({plan[k]:sha(plan[k]) for k in ('reference_plan','reference_release')})
    chains=[]; selected={}
    for j in ref['jobs']:
        xml=Path(j['path'])/'compiled_model.xml.gz'; manifest=Path(j['path'])/'manifest.json'
        if sha(manifest)!=j['snapshot_manifest_sha256'] or sha(xml)!=read(manifest)['files']['compiled_model.xml.gz']['sha256']:
            raise ValueError('Snapshot XML identity changed')
        inputs[str(xml)]=sha(xml);inputs[str(manifest)]=sha(manifest)
        chain=camera_ancestry(gzip.decompress(xml.read_bytes()).decode(),'robot0_agentview_left')
        chains.append(dict(initial_id=j['initial_id'],**chain))
        if j['split']=='pretrain' and (j['task'] not in selected or j['scene_seed']<selected[j['task']]['scene_seed']): selected[j['task']]=j
    result=dict(scope='CPU engineering feasibility only; no rendering/policy/closed-loop task',
        native_ancestries=chains,tiago=head_probe(cfg['tiago_xml']),initial_point_audits=[],
        selection='Lowest-seed seen initial per task, independent of outcomes')
    write(out/'audit.json',result)
    for task,j in sorted(selected.items()):
        try: result['initial_point_audits'].append(point_audit(j,records[j['initial_id']]))
        except Exception as error: result['initial_point_audits'].append(dict(initial_id=j['initial_id'],task=task,error=f'{type(error).__name__}: {error}'))
        write(out/'audit.json',result)
        print(task,{k:v for k,v in result['initial_point_audits'][-1].items() if k not in ('flags','unsupported')},flush=True)
        root=Path(records[j['initial_id']]['candidate_root'])
        for f in [root/'summary.json',root/'geometry.npz',Path(j['path'])/'snapshot.pkl.gz']:inputs[str(f)]=sha(f)
    from AlphaBrain.research.dsol.acquisition import kinematic_audit
    write(out/'provenance.json',dict(inputs=inputs,sources={p:sha(p) for p in (__file__,kinematic_audit.__file__)},outputs={'audit.json':sha(out/'audit.json')}))

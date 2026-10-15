"""Bounded, renderer-free TIAGo kitchen integration probe, not a policy eval."""
import traceback
import signal
import re
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write, sha


def contact_summary(model, data, mujoco):
    """Read-only collision evidence, not an automatic failure diagnosis."""
    contacts=[]
    for c in data.contact[:data.ncon]:
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
        bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in (c.geom1,c.geom2)]
        contacts.append(dict(distance=float(c.dist),geoms=names,bodies=bodies))
    return dict(count=len(contacts),deepest=sorted(contacts,key=lambda x:x['distance'])[:20],
                penetrating_1cm=sum(c['distance'] < -.01 for c in contacts))


def allowed_support_contact(name1, name2, distance, normal_z):
    """Narrow diagnostic rule, not a general collision tolerance.

    Only native wheels/casters against named room floor, near-horizontal
    contact plane and <=1 micrometre penetration. Cabinets never qualify.
    """
    wheels={'robot0_wheel_left_collision','robot0_wheel_right_collision'}
    wheels.update(f'robot0_{end}_{side}_caster_2' for end in ('front','back') for side in ('left','right'))
    pair=any(wheel in wheels and re.fullmatch(r'floor_\d+_room_g\d+',floor or '')
             for wheel,floor in ((name1,name2),(name2,name1)))
    return bool(pair and np.isfinite(distance) and np.isfinite(normal_z)
                and distance >= -1e-6 and abs(normal_z) >= .999)


def forbidden_placement_contacts(env, mujoco):
    m=env.sim.model._model;d=env.sim.data._data
    forbidden=[];support=0
    for c in d.contact[:d.ncon]:
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in (c.geom1,c.geom2)]
        robot=[name.startswith(('robot0_','gripper0_','mobilebase0_')) for name in names]
        if robot[0] == robot[1]: continue
        if allowed_support_contact(*names,float(c.dist),float(c.frame[2])):
            support+=1
        else: forbidden.append(dict(geoms=names,distance=float(c.dist)))
    return forbidden,support


def prescreen_robot_base(env, anchor_pos, anchor_ori, rot_dev, pos_dev_x, pos_dev_y,
                         *, utils, mujoco, receipt, max_attempts=500, allow_support=False):
    """Diagnostic-only native proposal sequence, collision check before solve.

    Does not disable contacts for accepted placement or change the solver.
    Bounded unlike the upstream unbounded placement loop. Not a P0 adapter.
    """
    with utils.no_collision(env.sim):
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr('mobilebase0_joint_mobile_yaw')] = env.rng.uniform(-rot_dev,rot_dev)
        env.sim.forward()
    state=env.sim.get_state()
    m=env.sim.model._model;d=env.sim.data._data
    for attempt in range(max_attempts):
        expansion=attempt//50
        pos=utils.generate_random_robot_pos(env,anchor_pos,anchor_ori,
                                           pos_dev_x+.10*expansion,pos_dev_y+.05*expansion)
        utils.set_robot_to_position(env,pos)
        mujoco.mj_kinematics(m,d)
        mujoco.mj_comPos(m,d)
        mujoco.mj_collision(m,d)
        forbidden,support=forbidden_placement_contacts(env,mujoco) if allow_support else ([],0)
        collision=bool(forbidden) if allow_support else utils.detect_robot_collision(env)
        receipt.append(dict(attempt=attempt+1,position=pos.tolist(),rejected=bool(collision),contacts=int(d.ncon),
                            allowed_support=support,forbidden_sample=forbidden[:3]))
        if not collision:
            env.sim.forward()
            after=bool(forbidden_placement_contacts(env,mujoco)[0]) if allow_support else utils.detect_robot_collision(env)
            if after:
                raise RuntimeError('Collision pre-screen disagrees with full forward')
            return pos
        env.sim.set_state(state)
    raise RuntimeError(f'No collision-free placement in {max_attempts} native proposals')


def probe(task, layout, style, seed, *, dense_diagnostic=False, steps=40, inspect_contacts=False,
          placement_prescreen=False, allow_support=False, monitor_hold=False):
    import robosuite
    import robocasa
    import mujoco
    from robocasa.utils import env_utils
    from robosuite.controllers import load_composite_controller_config
    result=dict(task=task,layout=layout,style=style,seed=seed,dense_diagnostic=dense_diagnostic,
                placement_prescreen=placement_prescreen,allow_support=allow_support,
                scope='No renderer, no VLA, no task-success estimate',stage='create',status='RUNNING')
    env=None;original_placement=env_utils.set_robot_base
    try:
        if placement_prescreen:
            result['placement_proposals']=[]
            def checked_placement(*args,**kwargs):
                return prescreen_robot_base(*args,**kwargs,utils=env_utils,mujoco=mujoco,
                                            receipt=result['placement_proposals'],allow_support=allow_support)
            # Only this single-process diagnostic; restored even on failure.
            env_utils.set_robot_base=checked_placement
        controller=load_composite_controller_config(controller=None,robot='Tiago')
        env=robosuite.make(env_name=task,robots='Tiago',controller_configs=controller,
            has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,use_object_obs=True,
            camera_names=['robot0_robotview'],seed=seed,layout_and_style_ids=[(layout,style)],obj_instance_split='pretrain')
        if dense_diagnostic:
            # Local instance only; retain native failure and never promote this
            # numerical diagnostic to the frozen P0 evaluation configuration.
            original_initialize=env._initialize_sim
            def initialize(*args,**kwargs):
                value=original_initialize(*args,**kwargs)
                env.sim.model._model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
                return value
            env._initialize_sim=initialize
        result['stage']='reset';obs=env.reset()
        m=env.sim.model._model;d=env.sim.data._data;robot=env.robots[0]
        if inspect_contacts: result['reset_contacts']=contact_summary(m,d,mujoco)
        result.update(action_dim=env.action_dim,action_slices={k:list(v) for k,v in robot._action_split_indexes.items()},
            observation_keys=list(obs),camera_names=list(env.sim.model.camera_names),
            native_head_camera_present='robot0_robotview' in env.sim.model.camera_names,
            requested_renderers=False)
        result['camera_parents']={name:mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.cam_bodyid[i])) for i,name in enumerate(env.sim.model.camera_names)}
        cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'robot0_robotview')
        joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ('robot0_head_1_joint','robot0_head_2_joint')]
        if cid<0 or min(joints)<0:raise ValueError('Missing native head camera/joints')
        addresses=m.jnt_qposadr[joints];before_q=d.qpos[addresses].copy();before_r=d.cam_xmat[cid].reshape(3,3).copy()
        cam_pos=m.cam_pos.copy();cam_quat=m.cam_quat.copy();before_time=float(d.time)
        head_controller=robot.composite_controller.part_controllers['head']
        result['head_controller']=dict(type=type(head_controller).__name__,input_type=getattr(head_controller,'input_type',None))
        actions={'head':np.array([.4,-.3])}
        monitored={};site_initial={};max_joint={};max_site={}
        if monitor_hold:
            torso=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot0_torso_lift_joint')
            if torso<0: raise ValueError('Missing torso hold joint')
            actions['torso']=np.array([d.qpos[m.jnt_qposadr[torso]]])
            for j in range(m.njnt):
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
                if name.startswith(('robot0_','mobilebase0_')) and j not in joints and int(m.jnt_type[j]) in (2,3):
                    monitored[name]=(int(m.jnt_qposadr[j]),float(d.qpos[m.jnt_qposadr[j]]))
                    max_joint[name]=0.
            for name in ('gripper0_right_grip_site','gripper0_left_grip_site'):
                sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name)
                if sid<0: raise ValueError('Missing end effector hold site')
                site_initial[name]=(sid,d.site_xpos[sid].copy());max_site[name]=0.
            result['hold_diagnostic']=dict(torso_target=actions['torso'].tolist(),
                other_commands='Native zero delta arms, zero base velocity, zero gripper command; not joint locking',
                max_joint_displacement=max_joint,max_eef_translation_m=max_site,
                external_contact_steps=0,external_contact_examples=[])
        command=robot.create_action_vector(actions)
        result['stage']='head_action';trace=[]
        for i in range(steps):
            _,_,_,_=env.step(command)
            trace.append(dict(step=i+1,time=float(d.time),head_qpos=d.qpos[addresses].tolist()))
            if not np.isfinite(d.qpos).all():raise ValueError('Nonfinite robot state')
            if monitor_hold:
                for name,(address,start) in monitored.items():max_joint[name]=max(max_joint[name],abs(float(d.qpos[address])-start))
                for name,(sid,start) in site_initial.items():max_site[name]=max(max_site[name],float(np.linalg.norm(d.site_xpos[sid]-start)))
                forbidden,_=forbidden_placement_contacts(env,mujoco)
                if forbidden:
                    result['hold_diagnostic']['external_contact_steps']+=1
                    if not result['hold_diagnostic']['external_contact_examples']:
                        result['hold_diagnostic']['external_contact_examples']=forbidden[:5]
        result.update(status='PASS',stage='finished',control_steps=steps,physics_seconds=float(d.time)-before_time,
            initial_head_qpos=before_q.tolist(),final_head_qpos=d.qpos[addresses].tolist(),
            optical_angle_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(-before_r[:,2],-d.cam_xmat[cid].reshape(3,3)[:,2]),-1,1)))),
            camera_mount_arrays_unchanged=bool(np.array_equal(cam_pos,m.cam_pos) and np.array_equal(cam_quat,m.cam_quat)),trace=trace)
    except Exception as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        if env is not None and getattr(env,'sim',None) is not None:
            m=env.sim.model._model;d=env.sim.data._data
            result['failure_state']=dict(nq=m.nq,nv=m.nv,nu=m.nu,ncon=int(d.ncon),nefc=int(d.nefc),
                qpos_finite=bool(np.isfinite(d.qpos).all()),qvel_finite=bool(np.isfinite(d.qvel).all()),
                solver=int(m.opt.solver),jacobian=int(m.opt.jacobian),cone=int(m.opt.cone))
            if inspect_contacts: result['failure_contacts']=contact_summary(m,d,mujoco)
    finally:
        env_utils.set_robot_base=original_placement
        if env is not None:
            try:env.close()
            except Exception as error:result['close_error']=str(error)
    return result


def run(config):
    cfg=read(config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    results=[]
    # Record identity before environment creation; external interruption must
    # leave an explicit RUNNING receipt instead of disappearing from the log.
    source_hash=sha(__file__)
    for case in cfg['cases']:
        write(out/'progress.json',dict(status='RUNNING',case=case,completed_cases=len(results),source_sha256=source_hash))
        previous=signal.getsignal(signal.SIGALRM)
        def deadline(signum,frame): raise TimeoutError('Bounded CPU integration probe deadline exceeded')
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,cfg.get('case_timeout_seconds',60))
        try: result=probe(**case)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        results.append(result);write(out/'results.json',results)
        print({k:v for k,v in result.items() if k not in ('traceback','observation_keys','trace',
              'placement_proposals','reset_contacts','failure_contacts')},flush=True)
    import robocasa,robosuite
    inputs={str(config):sha(config)}
    for root,rel in [(Path(robocasa.__file__).parent,'environments/kitchen/kitchen.py'),
                     (Path(robocasa.__file__).parent,'utils/env_utils.py'),
                     (Path(robosuite.__file__).parent,'controllers/config/robots/default_tiago.json'),
                     (Path(robosuite.__file__).parent,'models/assets/robots/tiago/robot.xml')]:
        path=root/rel;inputs[str(path)]=sha(path)
    write(out/'provenance.json',dict(inputs=inputs,source_sha256=source_hash,results_sha256=sha(out/'results.json')))
    write(out/'progress.json',dict(status='COMPLETE',completed_cases=len(results),passed=sum(r['status']=='PASS' for r in results)))

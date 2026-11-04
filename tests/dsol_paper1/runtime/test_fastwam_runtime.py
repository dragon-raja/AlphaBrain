"""CPU adapter/entry contracts. Real models, GPU and simulator never constructed."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.LIBERO.legacy import fastwam_runtime as runtime
from benchmarks.LIBERO.search_runtime import create_policy_runtime, selected_checkpoint
from AlphaBrain.research.dsol.execution import dispatch_policy


def example():
    rng=np.random.default_rng(45)
    return dict(lang='put the cup on the plate',image=[rng.integers(0,256,(224,224,3),dtype=np.uint8) for _ in range(2)],
        fastwam_raw=runtime.raw_observation(dict(
            agentview_image=rng.integers(0,256,(256,256,3),dtype=np.uint8),
            robot0_eye_in_hand_image=rng.integers(0,256,(256,256,3),dtype=np.uint8),
            robot0_eef_pos=np.array([.1,.2,.3]),robot0_eef_quat=np.array([.3,.4,0.,np.sqrt(.75)]),
            robot0_gripper_qpos=np.array([.01,-.01]))))


def processor():
    class Field:
        def backward(self,x):return x*.5+.5
    class Normalizer:
        normalizers={'action':{'default':Field()}}
        def forward(self,b):
            b['state']['default']=b['state']['default']*2-.1
            return b
    return SimpleNamespace(normalizer=Normalizer(),action_state_transform=lambda x:x,
        shape_meta={'state':[{'key':'default'}],'action':[{'key':'default'}],
                    'images':[{'shape':[3,224,224]}]*2},num_output_cameras=2)


def test_actual_native_feature_factory_skips_source_tower_without_action_or_rng_change(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from AlphaBrain.research.dsol.data.native_features import layout_spec,from_wire
    from AlphaBrain.research.dsol.data.native_observation import NativeStream
    from benchmarks.LIBERO.local_search_rollout import input_identity
    model=fake_model(torch);original=model.infer_action
    class Expert:
        def prepare(self,*,x,timestep,action):
            return (torch.arange(24,dtype=torch.float32).reshape(1,8,3)+x.mean(),)+(None,)*9
    model.video_expert=Expert()
    def infer(**kw):
        model.video_expert.prepare(x=torch.ones(1,1,1,2,4)*kw['input_image'].float().mean(),timestep=torch.zeros(1),action=None)
        return original(**kw)
    model.infer_action=infer;loads=[]
    def load(c,*,load_visual=True):
        loads.append(load_visual)
        if load_visual:raise AssertionError('Native mode must not load a source visual tower')
        return model,processor(),None
    monkeypatch.setattr(runtime,'load_components',load)
    monkeypatch.setattr(runtime,'validate_contract',lambda *args:dict(checkpoint='weights',score_definition='accel_prefix3_native_velocity_all32x7_v1',input_hashes={'weights':'e'*64}))
    layout=layout_spec((NativeStream('joint',3,'joint',('external','wrist'),('movable','context')),))
    spec=dict(schema='LIBERO_NATIVE_FEATURE_TAP_V1',family='fastwam',tap='video_expert.prepare.first_frame',shape=[1,8,3],grid=[1,2,4],layout=layout)
    config=dict(operation_model='fastwam',native_features=spec,models={'fastwam':dict(path='weights',weights_sha256='e'*64,
        implementation='fastwam_native',runtime_contract='contract',runtime_contract_sha256='contract_sha')})
    grant=dict(operation='LIBERO_BLOCK',permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    e=example();noise=np.zeros((10,7),np.float32);request=dict(query_id='q',input_sha256=input_identity(e),history_sha256='b'*64)
    with dispatch_policy._live_model_scope(grant,lambda:None):
        obj=create_policy_runtime(config,tmp_path,3);rng=torch.get_rng_state()
        control=obj.infer(e,noise,dict(case='control'),capture=False,instrument=False)
        observed=obj.infer(e,noise,dict(case='native'),native_request=request)
        assert torch.equal(rng,torch.get_rng_state())
        np.testing.assert_array_equal(control['full_actions'],observed['full_actions'])
        assert loads==[False] and 'external' not in observed and 'wrist' not in observed
        assert from_wire(observed['native_features']).model_sha256=='e'*64
        with pytest.raises(ValueError,match='provenance'):
            obj.infer(e,noise,{},native_request=dict(request,input_sha256='0'*64))
        assert obj.calls==2 and model.forward_calls==2


def test_noise_is_addressed_and_not_a_global_rng_side_effect():
    before=np.random.get_state()
    x=np.arange(70,dtype=np.float32).reshape(10,7)
    assert runtime.noise_seed(x)==runtime.noise_seed(x.astype('>f4'))
    assert runtime.noise_seed(x)!=runtime.noise_seed(x+1)
    assert 0<=runtime.noise_seed(x)<2**63
    after=np.random.get_state()
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    for bad in (np.zeros((32,7)),np.full((10,7),np.nan)):
        with pytest.raises(ValueError):runtime.noise_seed(bad)


def test_raw_pixels_are_required_and_copied():
    e=example();raw=e['fastwam_raw'];copy=runtime.raw_observation(raw)
    raw['agentview_image'][:]=0
    assert copy['agentview_image'].any()
    copy['robot0_eef_pos'][0]=np.nan
    with pytest.raises(ValueError):runtime.raw_observation(copy)
    raw['agentview_image']=np.zeros((224,224,3),dtype=np.uint8)
    with pytest.raises(ValueError):runtime.raw_observation(raw)


def test_visual_pixels_match_native_values_and_batch_strides():
    torch=pytest.importorskip('torch');e=example()
    actual=runtime.source_pixels(e,device='cpu')
    for name,image in zip(('external','wrist'),e['image']):
        native=torch.from_numpy(np.array(image)).to('cpu')[None,...]
        native=native.to(torch.float32).permute(0,3,1,2)/255.*2.-1.
        assert torch.equal(actual[name],native)
        assert actual[name].stride()==native.stride()==(150528,1,672,3)


def fake_model(torch):
    class Scheduler:
        def build_inference_schedule(self,num_inference_steps,device,dtype,shift_override):
            u=torch.linspace(1,0,num_inference_steps+1,device=device,dtype=dtype)
            s=shift_override*u/(1+(shift_override-1)*u)
            return s[:-1]*1000,s[1:]-s[:-1]
        def step(self,model_output,delta,sample):return sample+model_output*delta
    class Model:
        device='cpu';torch_dtype=torch.bfloat16
        def __init__(self):self.infer_action_scheduler=Scheduler();self.forward_calls=0;self.fail=False
        def infer_action(self,**kw):
            self.forward_calls+=1;self.kw=kw
            if self.fail:raise RuntimeError('unknown forward outcome')
            assert kw['action_horizon']==32 and kw['sigma_shift']==5. and not kw['compile_action_infer']
            x=torch.randn((1,32,7),generator=torch.Generator().manual_seed(kw['seed'])).to(self.torch_dtype)
            times,deltas=self.infer_action_scheduler.build_inference_schedule(
                kw['num_inference_steps'],'cpu',self.torch_dtype,kw['sigma_shift'])
            for t,d in zip(times,deltas):x=self.infer_action_scheduler.step(x*.2+t/1000,d,x)
            return {'action':x[0].float()}
    return Model()


def bound_runtime(tmp_path,monkeypatch,limit=3):
    torch=pytest.importorskip('torch')
    model=fake_model(torch);calls=[]
    def visual(pixels):
        calls.append(tuple(pixels))
        return {k:torch.ones(1,4608)*v.mean() for k,v in pixels.items()}
    monkeypatch.setattr(runtime,'validate_contract',lambda *args:dict(checkpoint='weights',score_definition='accel_prefix3_native_velocity_all32x7_v1',input_hashes={'weights':'sha'}))
    monkeypatch.setattr(runtime,'load_components',lambda c:(model,processor(),visual))
    config=dict(operation_model='fastwam',models={'fastwam':dict(path='weights',weights_sha256='sha',
        implementation='fastwam_native',runtime_contract='contract',runtime_contract_sha256='contract_sha')})
    grant=dict(operation='LIBERO_BLOCK',permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    guards=[]
    with dispatch_policy._live_model_scope(grant,lambda:guards.append(1)):
        obj=create_policy_runtime(config,tmp_path,limit)
    return obj,model,calls,grant,guards


def test_direct_constructor_and_inference_remain_closed(tmp_path):
    with pytest.raises(dispatch_policy.ProductionDispatchDisabled):runtime.FastWAMRuntime({},tmp_path,1)
    with pytest.raises(dispatch_policy.ProductionDispatchDisabled):runtime.load_components({})
    with pytest.raises(dispatch_policy.ProductionDispatchDisabled):runtime.FastWAMRuntime.__new__(runtime.FastWAMRuntime).infer({},None,{})
    c=dict(operation_model='f',models={'f':dict(path='w',weights_sha256='h',implementation='fastwam_native')})
    with pytest.raises(ValueError):selected_checkpoint(c,allow_native=True)


@pytest.mark.parametrize('definition', list(runtime.SCORE_WINDOWS))
def test_actual_factory_native_trace_preserves_actions_and_rng(tmp_path,monkeypatch,definition):
    torch=pytest.importorskip('torch')
    obj,model,visual,grant,guards=bound_runtime(tmp_path,monkeypatch)
    obj.contract['score_definition']=definition
    e=example();noise=np.zeros((10,7),np.float32);before=torch.get_rng_state()
    with dispatch_policy._live_model_scope(grant,lambda:guards.append(1)):
        control=obj.infer(e,noise,{'case':'control'},capture=False,instrument=False)
        observed=obj.infer(e,noise,{'case':'observed'})
    assert torch.equal(before,torch.get_rng_state())
    np.testing.assert_array_equal(control['full_actions'],observed['full_actions'])
    assert observed['trace'].shape==(10,32,7) and observed['actions'].shape==(10,7)
    assert observed['score_definition']==definition
    assert observed['score']==runtime.score_trace(observed['trace'],definition)
    assert len(visual)==1 and model.forward_calls==2 and len(guards)>=7
    assert not np.allclose(np.diff(observed['timesteps']),np.diff(observed['timesteps'])[0])
    rows=[json.loads(x) for x in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert [r['index'] for r in rows]==[1,2] and rows[0]['seed']==rows[1]['seed']
    assert 'step' not in model.infer_action_scheduler.__dict__


def test_action_window_excludes_only_future_actions_and_preserves_old_score():
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    trace=np.random.default_rng(42).normal(size=(10,32,7)).astype(np.float32)
    before=trace.copy()
    old='accel_prefix3_native_velocity_all32x7_v1'
    new='accel_prefix3_native_velocity_first10x7_v2'
    assert runtime.score_trace(trace,old)==float(compute_accel_scores(trace[None],prefixes=[3])[3]['score'][0])
    assert runtime.score_trace(trace,new)==float(compute_accel_scores(trace[None,:,:10],prefixes=[3])[3]['score'][0])
    changed=trace.copy();changed[:,10:]*=np.arange(1,11)[:,None,None]
    assert runtime.score_trace(changed,new)==runtime.score_trace(trace,new)
    assert runtime.score_trace(changed,old)!=runtime.score_trace(trace,old)
    np.testing.assert_array_equal(trace,before)
    for bad,definition in ((trace,'auto'),(trace[:3],new),(trace[:,:10],new),(trace*np.nan,new)):
        with pytest.raises(ValueError):runtime.score_trace(bad,definition)


def test_unknown_forward_remains_charged_and_cannot_be_replayed(tmp_path,monkeypatch):
    obj,model,_,grant,_=bound_runtime(tmp_path,monkeypatch);model.fail=True
    with dispatch_policy._live_model_scope(grant,lambda:None):
        with pytest.raises(RuntimeError,match='unknown'):obj.infer(example(),np.zeros((10,7)),{})
        model.fail=False
        with pytest.raises(ValueError,match='Failed/unknown'):obj.infer(example(),np.zeros((10,7)),{})
    assert model.forward_calls==1 and obj.calls==1 and obj.failed
    assert len((tmp_path/'calls.jsonl').read_text().splitlines())==1
    assert 'step' not in model.infer_action_scheduler.__dict__


def test_lost_grant_after_local_reservation_blocks_forward(tmp_path,monkeypatch):
    obj,model,_,grant,_=bound_runtime(tmp_path,monkeypatch)
    count=0
    def guard():
        nonlocal count
        count+=1
        if count==3:raise RuntimeError('center disconnected')
    with dispatch_policy._live_model_scope(grant,guard):
        with pytest.raises(RuntimeError,match='disconnected'):obj.infer(example(),np.zeros((10,7)),{})
    assert model.forward_calls==0 and obj.calls==1 and obj.failed


def test_exhausted_budget_refuses_additional_forward(tmp_path,monkeypatch):
    obj,model,_,grant,_=bound_runtime(tmp_path,monkeypatch,limit=1)
    with dispatch_policy._live_model_scope(grant,lambda:None):
        obj.infer(example(),np.zeros((10,7)),{},capture=False)
        with pytest.raises(ValueError,match='exhausted'):obj.infer(example(),np.zeros((10,7)),{})
    assert model.forward_calls==1


def reference_functions(source,filename,names,namespace):
    tree=ast.parse((source/filename).read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert len(selected)==len(names)
    # Read pinned function bodies only; never import the simulator/eval main.
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),filename,'exec'),namespace)


@pytest.mark.parametrize('dtype_name',['float32','bfloat16'])
def test_exact_pinned_official_input_and_action_processing(dtype_name):
    torch=pytest.importorskip('torch')
    source=os.environ.get('FASTWAM_REFERENCE_SOURCE')
    if not source:pytest.skip('Explicit pinned official source required for differential check')
    import math
    from PIL import Image
    ns=dict(np=np,torch=torch,math=math,Image=Image)
    source=Path(source)
    reference_functions(source,'experiments/libero/libero_utils.py',{'get_libero_image','quat2axisangle','invert_gripper_action'},ns)
    reference_functions(source,'experiments/libero/eval_libero_single.py',
        {'_center_crop_resize','_normalize_proprio','_extract_sim_state','_obs_to_model_input','_denormalize_action'},ns)
    e=example();p=processor();dtype=getattr(torch,dtype_name)
    cfg=SimpleNamespace(data=SimpleNamespace(train={'concat_multi_camera':'horizontal'}))
    pixels,state,_=ns['_obs_to_model_input'](e['fastwam_raw'],cfg,p,448,224,'cpu',dtype)
    actual,proprio=runtime.model_inputs(e,p,device='cpu',dtype=dtype)
    assert torch.equal(actual,pixels) and torch.equal(proprio,state)
    actions=torch.linspace(-1,1,224).reshape(32,7)
    expected=ns['_denormalize_action'](actions,p)[0]
    expected[:,-1]=expected[:,-1]*2-1
    expected=ns['invert_gripper_action'](expected);expected[:,-1]=np.sign(expected[:,-1])
    np.testing.assert_array_equal(runtime.decode_actions(actions,p),expected)


@pytest.mark.parametrize('fault',[None,'action_mismatch','unknown_forward'])
def test_actual_admission_executor_with_cpu_model(tmp_path,monkeypatch,fault):
    from contextlib import contextmanager
    from benchmarks.LIBERO import controlled_admission,search_live
    from benchmarks.LIBERO.continuous_authority import execute_block
    torch=pytest.importorskip('torch')
    model=fake_model(torch)
    infer=model.infer_action
    def forward(**kw):
        if fault=='unknown_forward' and model.forward_calls==1:
            model.fail=True
        pred=infer(**kw)
        if fault=='action_mismatch' and model.forward_calls==2:pred['action'][:,:6]+=1
        return pred
    model.infer_action=forward
    monkeypatch.setattr(runtime,'validate_contract',lambda *args:dict(checkpoint='weights',score_definition='accel_prefix3_native_velocity_all32x7_v1',input_hashes={'weights':'sha'}))
    monkeypatch.setattr(runtime,'load_components',lambda c:(model,processor(),
        lambda p:{k:torch.ones(1,4608) for k in p}))
    noise=tmp_path/'noise.npz';np.savez(noise,first_score_noise=np.zeros((1,2,10,7),np.float32))
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(anchors=str(noise),operation_model='fastwam',
        models={'fastwam':dict(path='weights',weights_sha256='sha',implementation='fastwam_native',
            runtime_contract='unused',runtime_contract_sha256='unused')})))
    limits=dict(episodes=0,score=4,paired_execution=0,nominal_execution=0)
    c=dict(kind='fastwam_trace',sim_config=str(sim),limits=limits,
        blocks=[dict(index=0,host='fresh',rows=[dict(initial=0,state_key='s')],limits=limits)])
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps(c))
    monkeypatch.setattr(controlled_admission,'validate_campaign',lambda p:c)
    @contextmanager
    def simulator(*args):
        def query(r):
            assert r==dict(kind='query_pose') # No action/search request allowed.
            return {'example':example()}
        yield query
    monkeypatch.setattr(search_live,'simulator',simulator)
    req=controlled_admission.request_for(campaign,0);out=tmp_path/'result'
    binding=dict(mode='libero',scientific_adapter='controlled_admission_v1',
        tasks={'key':dict(request=req,output=str(out))})
    launch=dict(key='key',slot={'gpu_id':'0'},operation='LIBERO_BLOCK',
        permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    guards=[]
    if fault:
        with pytest.raises((ValueError,RuntimeError)):
            execute_block(binding,launch,lambda:guards.append(1))
        assert not (out/'complete.json').exists()
        assert len((out/'calls.jsonl').read_text().splitlines())==2
    else:
        counts,_=execute_block(binding,launch,lambda:guards.append(1))
        assert counts==limits and model.forward_calls==4
        assert controlled_admission.validate_block(req,out)['calls']==dict(score=4,execution=0)
        (out/'admission.json').write_text('{}')
        with pytest.raises(ValueError,match='Changed admission result'):
            controlled_admission.validate_block(req,out)
    assert guards


def test_admission_requires_guard_and_explicit_small_budget(tmp_path):
    from benchmarks.LIBERO import controlled_admission
    with pytest.raises(dispatch_policy.ProductionDispatchDisabled):
        controlled_admission.run({},tmp_path/'out',guard=lambda:None)
    c=dict(schema=controlled_admission.SCHEMA,kind='fastwam_trace',
        blocks=[dict(index=0,host='fresh',rows=[dict(initial=0)],limits={})],limits={})
    p=tmp_path/'c.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='budget'):controlled_admission.validate_campaign(p)


def test_admission_reuses_only_hashes_verified_in_this_validation(tmp_path,monkeypatch):
    from benchmarks.LIBERO import controlled_admission as a
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    weights=tmp_path/'weights';weights.write_bytes(b'bounded CPU asset')
    cp=save('runtime.json',{'placeholder':True});noise=save('noise.json',{})
    states=save('states.json',dict(records=[dict(state=dict(split='train',pair_key='s'))]))
    hashes={str(weights):file_digest(weights)};calls=[]
    def validated(path,stamp):
        assert file_digest(path)==stamp
        for name,h in hashes.items():
            calls.append(name)
            if file_digest(name)!=h:raise ValueError('Changed contract asset')
        return dict(input_hashes=hashes)
    monkeypatch.setattr(runtime,'validate_contract',validated)
    original=a.file_digest
    def counted(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(a,'file_digest',counted)
    sim=save('sim.json',dict(inputs=states,anchors=noise,continuous_camera=True,
        raw_observation=runtime.RAW_OBSERVATION,operation_model='fastwam',models=dict(fastwam=dict(
            path=str(weights),weights_sha256=hashes[str(weights)],implementation='fastwam_native',
            runtime_contract=cp,runtime_contract_sha256=file_digest(cp)))))
    limits=dict(episodes=0,score=4,paired_execution=0,nominal_execution=0)
    c=dict(schema=a.SCHEMA,kind='fastwam_trace',sim_config=sim,limits=limits,
        input_hashes={p:file_digest(p) for p in (sim,states,noise,cp,str(weights))},
        blocks=[dict(index=0,host='fresh',rows=[dict(initial=0,state_key='s')],limits=limits)])
    path=save('campaign.json',c)
    a.validate_campaign(path);assert calls.count(str(weights))==1
    a.validate_campaign(path);assert calls.count(str(weights))==2
    weights.write_bytes(b'changed after prior validation')
    with pytest.raises(ValueError,match='Changed contract asset'):a.validate_campaign(path)


def test_admission_addresses_enter_the_actual_standard_queue_plan(tmp_path):
    from benchmarks.LIBERO.controlled_admission import request_for
    from benchmarks.LIBERO.continuous_authority_plan import plans
    from AlphaBrain.research.dsol.execution.manifest import file_digest,task_key
    c=tmp_path/'campaign.json';c.write_text(json.dumps(dict(kind='fastwam_trace',
        blocks=[dict(index=0,host='fresh',rows=[dict(initial=0,state_key='train-task-state')])])) )
    req=request_for(c,0);addresses=req['block']['addresses'];experiment='cpu-admission'
    key=task_key(experiment,file_digest(c),addresses)
    binding=dict(campaign=str(c),budget=dict(stages=['formal'],campaign_sha256=file_digest(c)),
        tasks={key:dict(stage='formal',host='fresh',addresses=addresses,output=str(tmp_path/'result'),
            identity_experiment_id=experiment,request=req,eligible_slots=[dict(host='fresh',id='0',gpu_id='0')])})
    path=tmp_path/'binding.json';path.write_text(json.dumps(binding))
    assert len(plans(binding,path)['formal']['tasks'])==1
    addresses[0]['stream']='unknown_stream'
    with pytest.raises(ValueError,match='unknown noise stream'):plans(binding,path)


@pytest.mark.parametrize('steps',[10,30])
def test_initialization_receipt_reports_the_executed_bound_count(monkeypatch,steps):
    import sys
    from benchmarks.LIBERO.initialization import initialize
    env=SimpleNamespace(seed=lambda x:None,reset=lambda:None,set_init_state=lambda x:{},
        check_success=lambda:False)
    actions=[]
    def step(a):actions.append(a);return {},0,False,{}
    env.step=step
    from benchmarks.LIBERO import simulation_inputs
    monkeypatch.setattr(simulation_inputs, 'physics_state_sha256', lambda e: ('hash', 3))
    spec=dict(environment_seed=0,initialization=dict(settle_steps=steps,official_file='file',
        official_file_sha256='sha',init_state_index=0))
    _,receipt=initialize(env,spec,None)
    assert len(actions)==receipt['settle_steps']==steps

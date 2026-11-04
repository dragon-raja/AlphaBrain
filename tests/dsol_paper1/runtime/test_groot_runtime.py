"""CPU-only admission barriers and native noise identity tests."""
import numpy as np
import pytest
from benchmarks.LIBERO import groot_runtime
from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled


def test_native_seed_is_pairable_and_model_domain_separated():
    from benchmarks.LIBERO.legacy.fastwam_runtime import noise_seed as fastwam_seed
    noise=np.arange(70,dtype=np.float32).reshape(10,7)/100
    assert groot_runtime.noise_seed(noise)==groot_runtime.noise_seed(noise.astype(np.float64))
    assert groot_runtime.noise_seed(noise)!=fastwam_seed(noise)
    changed=noise.copy();changed[0,0]+=.01
    assert groot_runtime.noise_seed(noise)!=groot_runtime.noise_seed(changed)


@pytest.mark.parametrize('noise',[np.zeros((4,10,7)),np.full((10,7),np.nan),np.zeros((10,32))])
def test_bad_noise_rejected(noise):
    with pytest.raises(ValueError):groot_runtime.noise_seed(noise)


def test_uncontrolled_constructor_and_processor_do_not_import_target_model(tmp_path):
    import sys
    before={k for k in sys.modules if k.startswith('fluxvla')}
    with pytest.raises(ProductionDispatchDisabled):
        groot_runtime.GrootRuntime('missing-contract','bad',tmp_path,1,instrument=True)
    with pytest.raises(ProductionDispatchDisabled):
        groot_runtime.NativeProcessor({},None)
    assert before=={k for k in sys.modules if k.startswith('fluxvla')}


def test_uncontrolled_infer_rejected_before_processing():
    runtime=object.__new__(groot_runtime.GrootRuntime)
    with pytest.raises(ProductionDispatchDisabled):runtime.infer({},None,{})
    processor=object.__new__(groot_runtime.NativeProcessor)
    with pytest.raises(ProductionDispatchDisabled):processor.inputs({})


@pytest.mark.parametrize('fault',[None,'action_mismatch','unknown_forward','disconnect'])
def test_actual_groot_entry_with_simulated_model(tmp_path,monkeypatch,fault):
    import json,sys
    from types import SimpleNamespace
    from contextlib import contextmanager
    from benchmarks.LIBERO import controlled_groot_admission as admission,search_live
    from benchmarks.LIBERO.continuous_authority import execute_block
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    state={'calls':0}
    class SimulatedRuntime:
        def __init__(self,path,stamp,output,limit,*,instrument):
            reject_production('benchmarks/LIBERO/groot_runtime.py:GrootRuntime.__init__')
            self.output,self.limit,self.instrument,self.calls=output,limit,instrument,0
        def infer(self,example,noise,identity,*,capture=False,warmup=False):
            reject_production('benchmarks/LIBERO/groot_runtime.py:GrootRuntime.infer')
            self.calls+=1;state['calls']+=1
            assert self.calls<=self.limit
            with (self.output/'calls.jsonl').open('a') as f:
                f.write(json.dumps(dict(identity,index=self.calls,warmup=warmup))+'\n')
            if fault=='unknown_forward' and self.instrument and capture:
                raise RuntimeError('unknown native call result')
            actions=np.ones((10,7))
            if fault=='action_mismatch' and self.instrument:actions+=1
            return dict(actions=actions,normalized_actions=actions.copy(),scores={'3':.1,'4':.2},seconds=.01,peak_allocated_mib=0)
    monkeypatch.setattr(groot_runtime,'GrootRuntime',SimulatedRuntime)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    noise=tmp_path/'noise.npz';np.savez(noise,first_score_noise=np.zeros((1,2,10,7),np.float32))
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(anchors=str(noise))))
    limits=dict(episodes=0,score=5,paired_execution=0,nominal_execution=0)
    c=dict(sim_config=str(sim),runtime_contract='cpu-fixture',runtime_contract_sha256='fixture',limits=limits,
        blocks=[dict(index=0,host='fresh',rows=[dict(initial=0,state_key='s')],limits=limits)])
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps(c))
    monkeypatch.setattr(admission,'validate_campaign',lambda path:c)
    @contextmanager
    def simulator(*args):
        def query(request):
            assert request==dict(kind='query_pose')
            return dict(example=dict(lang='fixture',fastwam_raw={'schema':'fixture','agentview_image':np.zeros((4,4,3),np.uint8)}))
        yield query
    monkeypatch.setattr(search_live,'simulator',simulator)
    request=admission.request_for(campaign,0);output=tmp_path/'result'
    binding=dict(mode='libero',scientific_adapter='controlled_groot_admission_v1',tasks={'key':dict(request=request,output=str(output))})
    launch=dict(key='key',slot={'gpu_id':'0'},operation='LIBERO_BLOCK',permit={'authorization':'EXPLICIT_MODEL_STAGE_V1'})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def guard():
        if fault=='disconnect' and state['calls']>=3:raise RuntimeError('authority disconnected')
    if fault:
        with pytest.raises((ValueError,RuntimeError)):execute_block(binding,launch,guard)
        assert not (output/'complete.json').exists()
        assert (output/'native/calls.jsonl').exists()
        assert state['calls'] <= 4
    else:
        counts,_=execute_block(binding,launch,guard)
        assert counts==limits and state['calls']==5
        assert admission.validate_block(request,output)['calls']=={'score':5,'execution':0}
        (output/'observed/calls.jsonl').write_text('tampered')
        with pytest.raises(ValueError):admission.validate_block(request,output)


def test_contract_accepts_actual_five_file_bpe_and_rejects_asset_drift(tmp_path,monkeypatch):
    import json,hashlib
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    source=tmp_path/'upstream';source.mkdir();hashes={}
    def asset(path,text='fixture'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        hashes[str(path)]=file_digest(path)
        return str(path)
    head=source/'fluxvla/models/heads/flow_matching_inference_head.py'
    asset(head);monkeypatch.setattr(groot_runtime,'UPSTREAM_SHA256',file_digest(head))
    config=asset(source/'configs/gr00tn15/gr00tn15_eagle_3b_libero_spatial_full_finetune.py')
    asset(source/'fluxvla/models/third_party_models/eagle2_hg_model/config.json','{}')
    tokenizer=tmp_path/'tokenizer'
    for name in ('vocab.json','merges.txt','added_tokens.json','special_tokens_map.json','tokenizer_config.json'):
        asset(tokenizer/name)
    manifest=asset(source/'dsol-import-surface.json',json.dumps(dict(preserved_python={str(head.relative_to(source)):file_digest(head)},changed_initializers={})))
    kernels={}
    for i in range(3):
        name=f'kernel-{i}.so';asset(source/name);kernels[name]=file_digest(source/name)
    c=dict(schema=groot_runtime.SCHEMA,sampler=groot_runtime.SAMPLER,noise_rule=groot_runtime.NOISE_RULE,suite='libero_spatial',source=str(source),checkpoint=asset(tmp_path/'checkpoint'),model_config=config,statistics=asset(tmp_path/'stats'),tokenizer=str(tokenizer),import_manifest=manifest,kernel_hashes=kernels)
    c['input_hashes']=hashes
    path=tmp_path/'contract.json';path.write_text(json.dumps(c))
    assert groot_runtime.validate_contract(path,file_digest(path))['sampler']['flow_steps']==4
    (tokenizer/'vocab.json').write_text('changed')
    with pytest.raises(ValueError,match='Changed GR00T asset'):groot_runtime.validate_contract(path,file_digest(path))


def test_native_operation_contract_and_single_step_reference_match():
    from AlphaBrain.research.dsol.data.operation_contract import groot_contract,simulator_action_contract
    from benchmarks.LIBERO.local_search_rollout import rollout_groot_episode
    c=groot_contract('a'*64)
    config=dict(operation_contract=c,continuous_execution='PAIRED_EVALUATION_ONLY',
        raw_observation='LIBERO_RAW_RGB256_PROPRIO_V1',operation_model='groot',
        models={'groot':dict(implementation='groot_native',runtime_contract_sha256='a'*64)})
    assert simulator_action_contract(config)==(10,'float64')
    def run(single):
        step=0;events=[]
        def example():return dict(camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4),state=np.array([step]))
        def request(value):
            nonlocal step
            assert value['kind']=='step'
            actions=np.asarray(value['actions']);assert actions.dtype==np.float64
            count=min(len(actions),11-step);step+=count
            return dict(example=example(),success=step==11,executed=count)
        class Bank:
            def get(self,*args):return dict(noise=np.zeros((10,7),np.float32),noise_sha256='b'*64)
        def infer(*args,**kwargs):return dict(actions=np.full((10,7),1.25,np.float64))
        result,arrays=rollout_groot_episode(request,infer,example(),Bank(),'fixture',0,{},
            contract=c,suite='libero_spatial',record=events.append,reference_single_step=single)
        assert result['steps']==11 and result['replans']==2 and result['success']
        assert np.all(arrays['actions']==1.25) # No extra clip in the transport.
        return result,arrays
    a,x=run(False);b,y=run(True)
    assert a==b
    for key in x:np.testing.assert_array_equal(x[key],y[key])
    config['models']['groot']['implementation']='fastwam_native'
    with pytest.raises(ValueError):simulator_action_contract(config)

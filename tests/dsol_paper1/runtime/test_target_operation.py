from copy import deepcopy
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.operation_contract import fastwam_contract,simulator_action_contract
from benchmarks.LIBERO.local_search_rollout import rollout_fastwam_episode


def test_borrowed_host_still_requires_explicit_node_binding():
    from benchmarks.LIBERO.controlled_target_evaluation import execution_hosts
    from benchmarks.LIBERO.controlled_campaign import campaign_nodes
    assert execution_hosts({'execution_hosts':['cy']})==['cy']
    assert execution_hosts({})==['fresh']
    for hosts in ([],['cy','cy'],['../cy'],['cy;false'],[None]):
        with pytest.raises(ValueError):execution_hosts({'execution_hosts':hosts})
    for nodes,aliases,excluded in (({}, {'cy':'cy'},()), ({'cy':{}},{},()),
                                   ({'cy':{}},{'cy':'cy'},('cy',))):
        with pytest.raises(ValueError):
            campaign_nodes(nodes,[{'host':'cy'}],aliases,excluded)
    assert campaign_nodes({'cy':{}},[{'host':'cy'}],{'cy':'cy'})=={'cy':{}}


def test_explicit_simulator_contract():
    assert simulator_action_contract({})==(5,'float32')
    config=dict(operation_contract=fastwam_contract('a'*64),continuous_execution='PAIRED_EVALUATION_ONLY',
        raw_observation='LIBERO_RAW_RGB256_PROPRIO_V1',operation_model='target',
        models={'target':dict(implementation='fastwam_native',runtime_contract_sha256='a'*64)})
    assert simulator_action_contract(config)==(10,'float64')
    for field in ('runtime_contract_sha256','settle_steps','action_dtype'):
        bad=deepcopy(config);bad['operation_contract'][field]='bad'
        with pytest.raises(ValueError):simulator_action_contract(bad)
    config['raw_observation']='wrong'
    with pytest.raises(ValueError):simulator_action_contract(config)


@pytest.mark.parametrize('stop', [1,9,10,11,399,None])
def test_full_trajectory_reference_and_chunk_equivalence(stop):
    def run(reference):
        steps=0;events=[]
        def example():return dict(state=np.full(8,steps),camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))
        class Bank:
            def get(self,*a):return dict(noise=np.zeros((10,7),np.float32),noise_sha256=str(a[-1]))
        def infer(*a,**kw):return dict(full_actions=np.full((32,7),1.00000000001,np.float64))
        def request(command):
            nonlocal steps
            actions=command['actions'];assert actions.dtype==np.float64 and np.all(actions>1)
            n=min(len(actions),stop-steps) if stop else len(actions);steps+=n
            return dict(executed=n,success=steps==stop,example=example(),physics=str(steps))
        result,arrays=rollout_fastwam_episode(request,infer,example(),Bank(),'state',0,{},
            contract=fastwam_contract('a'*64),suite='libero_spatial',record=events.append,
            reference_single_step=reference)
        assert result['steps']==(stop or 400)
        assert result['replans']==((stop or 400)+9)//10
        assert 'full_decoded_actions' in events[1] and 'full_clipped_actions' not in events[1]
        return result,arrays
    a,x=run(True);b,y=run(False)
    assert a==b
    for k in x:np.testing.assert_array_equal(x[k],y[k])


@pytest.mark.parametrize('bad', ['partial','nonbool','unknown'])
@pytest.mark.parametrize('reference', [False,True])
def test_target_execution_fail_closed(bad,reference):
    example=dict(state=np.zeros(8),camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4));events=[]
    class Bank:
        def get(self,*a):return dict(noise=np.zeros((10,7)),noise_sha256='n')
    def query(c):
        if bad=='unknown':raise ConnectionError('execution outcome unknown')
        return dict(executed=0 if bad=='partial' else len(c['actions']),
                    success=1 if bad=='nonbool' else False,example=example)
    with pytest.raises((ConnectionError,ValueError)):
        rollout_fastwam_episode(query,lambda *a,**kw:dict(full_actions=np.zeros((32,7))),
            example,Bank(),'state',0,{},contract=fastwam_contract('a'*64),suite='libero_spatial',
            record=events.append,reference_single_step=reference)
    assert len([e for e in events if e['kind']=='inference_attempt'])==1


def test_manipulation_camera_read_does_not_advance_solver():
    from benchmarks.LIBERO.continuous_execution import FixedOperationCamera
    calls=[];camera=FixedOperationCamera()
    def reader():calls.append('sim.forward');return {'matrix':np.eye(4)}
    with pytest.raises(ValueError):camera.read(reader,refresh=False)
    first=camera.read(reader,refresh=True);first['matrix'][:]=0
    for _ in range(10):np.testing.assert_array_equal(camera.read(reader,refresh=False)['matrix'],np.eye(4))
    assert calls==['sim.forward']
    camera.read(reader,refresh=True)
    assert len(calls)==2

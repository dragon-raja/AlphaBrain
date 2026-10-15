import numpy as np
import pytest
from AlphaBrain.research.dsol.data.render_score_panel import compare_requests,IMAGE_KEYS
from AlphaBrain.research.dsol.data.artifacts import write,sha,read
from AlphaBrain.research.dsol.data.rollout_replay import array_identity
from AlphaBrain.research.dsol.analysis.render_score_replay import run
from AlphaBrain.research.dsol.analysis.render_score_results import summarize


def fixture():
    return dict({k:np.zeros((4,4,3),dtype=np.uint8) for k in IMAGE_KEYS},prompt='task',**{'observation/state':np.zeros(16)})


def test_micro_pixels_not_camera_motion():
    a,b=fixture(),fixture();b[IMAGE_KEYS[0]][0,0,0]=1
    result=compare_requests(a,b)
    assert result[IMAGE_KEYS[0]]['max_abs']==1
    assert result[IMAGE_KEYS[0]]['changed_channels']==1
    assert result[IMAGE_KEYS[1]]['max_abs']==0


def test_reject_proprioceptive_confound():
    a,b=fixture(),fixture();b['observation/state'][0]=1
    with pytest.raises(ValueError):compare_requests(a,b)


def test_crossed_replay_preserves_sources_and_checks_native_control(tmp_path):
    times=np.arange(10,dtype=np.float32)
    cells={};hashes={};arrays={}
    for name,value in [('canonical',0),('broad',1)]:
        cell=tmp_path/name;cell.mkdir();request=fixture();request[IMAGE_KEYS[0]][0,0,0]=value
        np.savez(cell/'scores.npz',**{k:v for k,v in request.items() if k!='prompt'},actions=np.full((8,50,12),value*.001))
        write(cell/'scores.json',dict(prompt='task',input_arrays={k:array_identity(v) for k,v in request.items() if k!='prompt'},
            rows=[dict(times=times.tolist(),metrics={'3':{'score':.25+value*.001}}) for _ in range(8)]))
        cells[name]=str(cell);hashes[name]=sha(cell/'scores.json');arrays[name]=sha(cell/'scores.npz')
    plan=tmp_path/'plan.json'
    write(plan,dict(workers=[dict(worker_id='test',arm='canonical',entry_indices=[0])],
        entries=[dict(cells=cells,metadata_sha256=hashes,arrays_sha256=arrays,different=True,initial_id='test',view_id='canonical')],
        runtime_replay_gate=dict(score_atol=1e-6,action_atol=1e-5,action_rtol=1e-5)))
    def predict(request,noise):
        value=request[IMAGE_KEYS[0]][0,0,0]
        return dict(steps=10,times=times,actions=np.full((50,12),value*.001),scores={'3':{'score':.25+value*.001}})
    run(plan,'test',tmp_path/'output',predict)
    assert read(tmp_path/'output/status.json')['status']=='COMPLETE'
    with np.load(tmp_path/'output/cell-0000.npz') as z:
        np.testing.assert_allclose(z['scores'][1]-z['scores'][0],.001)
    assert sha(tmp_path/'canonical/scores.npz')==arrays['canonical']
    def wrong_runtime(request,noise):
        result=predict(request,noise);result['scores']['3']['score']+=.1;return result
    with pytest.raises(ValueError,match='numerical gate'):run(plan,'test',tmp_path/'failed',wrong_runtime)
    assert read(tmp_path/'failed/status.json')['status']=='ERROR'


def test_summary_keeps_model_image_and_initial_axes_separate(tmp_path):
    root=tmp_path/'replay';matrix=tmp_path/'matrix';root.mkdir();matrix.mkdir()
    jobs=[dict(initial_id=f'i{i}',task='task') for i in range(2)];views=['canonical','a','b']
    old=np.broadcast_to(np.array([[[1,2,4],[2,1,4]],[[2,1,4],[1,2,4]]])[...,None],(2,2,3,8)).copy()
    y=np.array([[[1,0,0],[0,1,0]],[[1,0,0],[0,1,0]]],float)
    np.savez(matrix/'matrix.npz',accel_prefixes=np.repeat(old[...,None],9,axis=-1),success=np.repeat(y[...,None],4,axis=-1))
    write(matrix/'population.json',dict(jobs=jobs,views=[dict(view_id=v,role='test') for v in views]))
    entries=[dict(task='task',initial_id=j['initial_id'],view_id=v,different=i==0 and k==0,replay=i==0 and k==0,
        pixel_difference={'image':dict(max_abs=int(i==0 and k==0),mean_abs=.01 if i==0 and k==0 else 0)})
        for i,j in enumerate(jobs) for k,v in enumerate(views)]
    workers=[dict(worker_id=arm,arm=arm,entry_indices=[0]) for arm in ('canonical','broad')]
    write(root/'plan.json',dict(jobs=jobs,views=views,tasks=['task'],entries=entries,workers=workers,counts={}))
    for arm,scores in [('canonical',[[1]*8,[3]*8]),('broad',[[2.1]*8,[2]*8])]:
        out=root/'workers'/arm;out.mkdir(parents=True)
        np.savez(out/'cell-0000.npz',scores=np.array(scores),actions=np.zeros((2,8,50,12)))
        write(out/'cell-0000.json',dict(arrays_sha256=sha(out/'cell-0000.npz'),control=dict(score_max_abs=0,action_max_abs=0)))
        write(out/'status.json',dict(status='COMPLETE',plan_sha256=sha(root/'plan.json')))
    result=summarize(root,matrix);r=result['tasks']['task']
    assert r['models']['canonical']['winner_changes']==1
    assert r['models']['broad']['winner_changes']==0
    assert r['models']['canonical']['chosen_success_by_image_source']==[1.,.5]
    assert r['fully_identical_initials']==1
    np.testing.assert_allclose(r['crossover_by_common_image_source']['canonical_archive'],[[1,1],[0,0]])

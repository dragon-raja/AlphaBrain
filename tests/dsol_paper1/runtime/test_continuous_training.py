"""CPU-only live-adapter contract tests; no simulator/model construction."""
import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO.continuous_training import DurableScoring, run
from benchmarks.LIBERO.continuous_training_queue import ContinuousTrainingBackend


class Runtime:
    def __init__(self,limit=4):self.calls=0;self.limit=limit;self.fail=False
    def infer(self,example,noise,identity,capture=True):
        self.calls+=1
        if self.fail:raise RuntimeError('model failed')
        return {'identity':identity}


def test_named_operation_model_cannot_fall_back_to_canonical():
    from benchmarks.LIBERO.search_runtime import selected_checkpoint
    models={name:dict(path='/'+name,weights_sha256=name) for name in ('canonical','broad','step_zero')}
    assert selected_checkpoint(dict(models=models)) == models['canonical']
    for name in ('broad','step_zero'):
        assert selected_checkpoint(dict(models=models,operation_model=name)) == models[name]
    with pytest.raises(ValueError,match='Unknown'):
        selected_checkpoint(dict(models=models,operation_model='missing'))
    with pytest.raises(ValueError,match='hash'):
        selected_checkpoint(dict(models={'broad':dict(path='/broad')},operation_model='broad'))


def test_attempt_ledger_survives_resume_and_counts_failure(tmp_path):
    runtime=Runtime();s=DurableScoring(runtime,tmp_path,0,4);s.episode=2
    example={'state':np.zeros((1,7)),'image':[np.zeros((2,2,3),dtype=np.uint8)]}
    noise=np.zeros((10,7),dtype=np.float32)
    for i in range(2):s.infer(example,noise,dict(query=1,member=i))
    rows=[json.loads(l) for l in s.path.read_text().splitlines()]
    assert rows[0]['identity']['stage']=='continuous_ppo_training' and rows[0]['identity']['episode']==2
    assert len(rows[0]['input_sha256'])==len(rows[0]['noise_sha256'])==64
    resumed=DurableScoring(Runtime(),tmp_path,2,4)
    resumed.infer(example,noise,dict(query=2,member=0))
    resumed.runtime.fail=True
    with pytest.raises(RuntimeError):resumed.infer(example,noise,dict(query=2,member=1))
    assert len(s.path.read_text().splitlines())==4
    with pytest.raises(ValueError,match='exhausted'):resumed.infer(example,noise,{})
    assert len(s.path.read_text().splitlines())==4
    with pytest.raises(ValueError,match='no replay'):DurableScoring(Runtime(),tmp_path,2,4)


def test_draft_rejected_before_model_or_asset_load(tmp_path):
    config=tmp_path/'draft.json';config.write_text(json.dumps({'authorization':'NOT_AUTHORIZED'}))
    with pytest.raises(ValueError,match='not authorization'):run(config,tmp_path/'output',execute=True)
    assert not (tmp_path/'output').exists()
    backend=ContinuousTrainingBackend(dict(runtime=dict(mode='continuous_ppo_training',workers_per_gpu=1,
        config=str(config)),slots=[{}],tasks=[{}]),tmp_path)
    with pytest.raises(ValueError,match='not authorized'):backend.worker_command({'output':'unused'})


def test_continuous_queue_rejects_discrete_mode(tmp_path):
    backend=ContinuousTrainingBackend(dict(runtime=dict(mode='full_tasks'),slots=[{}],tasks=[{}]),tmp_path)
    with pytest.raises(ValueError,match='Single persistent'):backend.worker_command({})


def test_old_launch_cannot_silently_select_new_training_math(tmp_path):
    from benchmarks.LIBERO.continuous_training import validate_inputs
    path=tmp_path/'old.json';path.write_text(json.dumps({'authorization':'EXPLICIT_TRAINING_ONLY'}))
    with pytest.raises(ValueError,match='corrected training revision'):validate_inputs(path)
    with pytest.raises(ValueError,match='corrected training revision'):run(path,tmp_path/'output',execute=True)
    assert not (tmp_path/'output').exists()


def test_live_episode_adapter_with_4608_model_on_synthetic_renderer(tmp_path,monkeypatch):
    """Exercise actual query/prefetch/return/log wiring, using fake VLA + renderer."""
    torch=pytest.importorskip('torch');pytest.importorskip('gymnasium')
    from contextlib import contextmanager
    from benchmarks.LIBERO import search_runtime,search_live
    from benchmarks.LIBERO.continuous_training import LiveEpisodes
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.data.artifacts import read
    torch.set_num_threads(2);torch.manual_seed(0)
    requests=[]
    class FakeVLA:
        def __init__(self,config,output,limit):self.calls=0;self.limit=limit
        def infer(self,example,noise,identity,capture=True):
            self.calls+=1;torch.rand(3) # Simulate a backend consuming global CPU randomness.
            pose=example['camera_to_world_opencv']
            feature=np.full(4608,pose[0,0],dtype=np.float32)
            return dict(actions=np.zeros((10,7)),trace=np.zeros((10,10,7)),score=1.+abs(pose[0,1]),
                external=feature,wrist=np.ones(4608,dtype=np.float32))
    @contextmanager
    def fake_simulator(path,config,initial,output):
        current=np.eye(4)
        def receipt():return dict(camera_to_world=current.copy(),physical_state_id='synthetic',settled=True)
        def request(message):
            nonlocal current
            requests.append(message['kind'])
            if message['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt())
            if message['kind']=='move_pose':
                current=np.array(message['camera_to_world'],copy=True);return dict(receipt=receipt())
            assert message['kind']=='query_pose' # Manipulation is never sent.
            return dict(receipt=receipt(),example=dict(image=[current.copy(),np.ones((2,2))],state=np.zeros(7),
                camera_intrinsics=np.eye(3),camera_to_world_opencv=current.copy(),instruction='synthetic',lang='synthetic'))
        yield request
    monkeypatch.setattr(search_runtime,'PolicyRuntime',FakeVLA)
    monkeypatch.setattr(search_live,'simulator',fake_simulator)
    anchors=tmp_path/'anchors.npz';np.savez(anchors,first_score_noise=np.zeros((16,2,10,7),dtype=np.float32))
    sim=tmp_path/'sim.json';sim.write_text(json.dumps({'anchors':str(anchors)}))
    cfg=dict(sim_config=str(sim),trajectory_cap=512)
    plan=read(Path(__file__).resolve().parents[3]/'configs/dsol_paper1/search/continuous_search_v1.json')
    output=tmp_path/'training';live=LiveEpisodes(cfg,plan,output)
    schedule=[dict(episode=i,initial=i%16,angles_deg=[-17.5,9.5],proposal_seed=i) for i in range(32)]
    policy=ContinuousActorCritic(4608)
    with ContinuousPPO(policy,ContinuousPPOConfig(updates=1),schedule,live,output,
            binding={'source_kind':'SYNTHETIC_CPU_FAKE_RENDERER_AND_VLA'}) as trainer:
        status=trainer.run()
        assert status['episodes']==32 and status['updates']==1
        assert live.scoring.calls==status['queries']*2
        assert len(list(output.glob('episodes/*-queries/search.json')))==32
        assert len(list(output.glob('episodes/*-queries/query-*.npz')))==status['queries']
        assert len((output/'model-attempts.jsonl').read_text().splitlines())==status['queries']*2
        assert 'step' not in requests
        from benchmarks.LIBERO.continuous_training_gate import audit_first_batch
        from reports.paper1.continuous_training_tracking import update_rows
        gate=audit_first_batch(trainer)
        assert gate['queries']==status['queries'] and gate['extra_admission_calls']==0
        assert len(update_rows(output))==1
        (output/'episodes/0000-queries/query-01.npz').write_bytes(b'changed')
        with pytest.raises(ValueError,match='evidence changed'):audit_first_batch(trainer)

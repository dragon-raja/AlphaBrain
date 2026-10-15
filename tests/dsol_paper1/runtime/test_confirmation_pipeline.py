import pytest

from AlphaBrain.research.dsol.data.artifacts import write,read
from benchmarks.Robocasa365.confirmation_pipeline import after_preparation, after_publication
from AlphaBrain.research.dsol.data.artifacts import sha


def test_failed_preparation_never_starts_gpu(tmp_path,monkeypatch):
    write(tmp_path/'previous.json',dict(status='FAILED'))
    write(tmp_path/'cfg.json',dict(status_file=str(tmp_path/'status.json'),
          preparation_status=str(tmp_path/'previous.json'),timeout_seconds=10))
    monkeypatch.setattr('benchmarks.Robocasa365.confirmation_pipeline.run_program',lambda *_:pytest.fail('GPU launch'))
    with pytest.raises(RuntimeError,match='CPU preparation failed'):after_preparation(tmp_path/'cfg.json')
    assert read(tmp_path/'status.json')['status']=='FAILED'


def test_only_configured_model_is_published_and_started(tmp_path,monkeypatch):
    write(tmp_path/'previous.json',dict(status='READY_FOR_ENDPOINT_RUNTIME_ADMISSION'))
    write(tmp_path/'runtime.json',dict(model='official',plan='new-plan'))
    write(tmp_path/'cfg.json',dict(status_file=str(tmp_path/'status.json'),preparation_status=str(tmp_path/'previous.json'),
          timeout_seconds=10,runtime=str(tmp_path/'runtime.json'),selection_source='source',selection_destination='dest',
          release='release',probe_config='probe',control_root='control'))
    calls=[]
    monkeypatch.setattr('benchmarks.Robocasa365.confirmation_pipeline.publish_selections',lambda *_:'dest')
    def prepare(*args,**kwargs):
        assert args[2]=={'official':'dest'} and kwargs['model_partition']=='official'
        return {'rollout_count':22000}
    monkeypatch.setattr('benchmarks.Robocasa365.confirmation_pipeline.prepare_execution_plan',prepare)
    monkeypatch.setattr('benchmarks.Robocasa365.confirmation_pipeline.run_program',lambda *args:calls.append(args))
    after_preparation(tmp_path/'cfg.json')
    assert len(calls)==1 and read(tmp_path/'status.json')['status']=='COMPLETE'


def test_follower_starts_only_exact_published_partition(tmp_path, monkeypatch):
    write(tmp_path/'plan.json', {'model': 'official'})
    write(tmp_path/'partition.json', {'hosts': {'video': ['i']}})
    write(tmp_path/'runtime.json', dict(plan=str(tmp_path/'plan.json'), host_partition=str(tmp_path/'partition.json')))
    write(tmp_path/'ready.json', dict(status='READY', plan_sha256=sha(tmp_path/'plan.json'),
                                    partition_sha256=sha(tmp_path/'partition.json')))
    write(tmp_path/'cfg.json', dict(status_file=str(tmp_path/'status.json'), publication_ready=str(tmp_path/'ready.json'),
          runtime=str(tmp_path/'runtime.json'), control_root='control', timeout_seconds=10))
    calls=[]
    monkeypatch.setattr('benchmarks.Robocasa365.confirmation_pipeline.run_program', lambda *a: calls.append(a))
    after_publication(tmp_path/'cfg.json')
    assert len(calls)==1 and read(tmp_path/'status.json')['status']=='COMPLETE'

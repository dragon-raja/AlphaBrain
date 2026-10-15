import pytest

from AlphaBrain.research.dsol.data.artifacts import read, write
from benchmarks.Robocasa365 import capacity_benchmark as module


@pytest.mark.parametrize('fail', [False, True])
def test_capacity_isolated_and_resumes_without_promotion(tmp_path, monkeypatch, fail):
    control = tmp_path/'control'; control.mkdir()
    config = dict(queue=str(control), output=str(tmp_path/'result'), host='video', gpu='0',
                  workers=[1,2,3], abort_free_mib=512, required_headroom_mib=2048,
                  drain_timeout_seconds=1,anchor_config='anchor.json')
    path=tmp_path/'config.json';write(path,config)
    monkeypatch.setattr(module,'load',lambda _: (dict(runtime=dict(model='official',config='config.json',source='source')),{}))
    monkeypatch.setattr(module,'sha',lambda _:'f'*64)
    monkeypatch.setattr(module,'resource_status',lambda _: dict(devices=[dict(host='video',gpu='0',drained=True,quiescent=True)]))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'testgpu')
    resumed=[];monkeypatch.setattr(module,'set_drain',lambda *a,**k:resumed.append(k))
    def stage(cfg,plan,root,kind,workers):
        if fail:raise RuntimeError('injected error')
        return dict(status='PASS',kind=kind,workers=workers,quality_pass=True,
                    throughput=workers*10,headroom_mib=4000 if workers<3 else 1000,
                    required_headroom_mib=2048)
    monkeypatch.setattr(module,'run_stage',stage)
    result=module.run(path)
    assert resumed==[dict(reason='Capacity benchmark finished or stopped',enabled=False)]
    assert result['production_resumed']
    if fail:assert result['status']=='HOLD'
    else:
        assert result['recommended_probe_workers']==2
        assert not result['production_promotion']
        assert len(result['stages'])==6
    assert read(tmp_path/'result/status.json')==result

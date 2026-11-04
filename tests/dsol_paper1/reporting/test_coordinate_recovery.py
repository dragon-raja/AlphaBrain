import json
import pytest
from reports.paper1 import coordinate_recovery


def fixture_config(tmp_path,monkeypatch):
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)
    rows=[];searches=[];tasks=[];refs={}
    for state in range(8):
        for view in (1,2):
            for offset in (-5.,0.,5.):
                for seed in range(3):
                    method=f'G-seed-{seed}-mount-{int(offset):+d}'
                    identity=dict(state_key=str(state),view_id=view,method=method)
                    rows.append(dict(identity,actor_key=f'G-seed-{seed}',mount_yaw_offset_deg=offset))
                    searches.append(dict(identity,queries=4))
                    tasks.extend(dict(identity,repeat=r,success=bool(state%2)) for r in (0,1))
            for v in (0,view):
                refs[f'{state}::{v}']=dict(episode_results=[save(tmp_path/f'base-{state}-{v}-{repeat}.json',
                    dict(success=(v==0))) for repeat in (0,1)])
    protocol=save(tmp_path/'protocol.json',dict(rows=rows))
    references=save(tmp_path/'references.json',dict(views=refs))
    root=tmp_path/'run';phase=root/'execution/probe-00-real';block=phase/'block'
    save(phase/'binding.json',dict(tasks={'block':dict(request={},output=str(block))}))
    save(phase/'completion.json',dict(status='COMPLETE',dispatch_enabled=False))
    result=block/'results.json';save(result,dict(searches=searches,tasks=tasks))
    config=save(tmp_path/'config.json',dict(output=str(root),protocol=protocol,campaigns=['one'],baseline_references=references))
    monkeypatch.setattr(coordinate_recovery,'validate_block',lambda *a:dict(status='SYNTHETIC_TEST_ONLY'))
    return config,result


def test_complete_pairing_and_same_offset_contrast(tmp_path,monkeypatch):
    config,_=fixture_config(tmp_path,monkeypatch)
    report=coordinate_recovery.build(config,tmp_path/'report')
    assert report['episodes']==288 and report['searches']==144
    assert report['stay_success_percent']==0 and report['nominal_success_percent']==100
    for row in report['offsets']:
        assert row['success_percent']==50 and row['rescue_percent']==50
        assert row['harm_percent']==0 and row['paired_change_vs_zero_pp']==0


@pytest.mark.parametrize('damage',['missing','duplicate'])
def test_missing_or_duplicate_results_never_form_complete_report(tmp_path,monkeypatch,damage):
    config,result=fixture_config(tmp_path,monkeypatch)
    data=json.loads(result.read_text())
    if damage=='missing':data['tasks'].pop()
    else:data['tasks'].append(data['tasks'][0])
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incomplete|Duplicate'):
        coordinate_recovery.build(config,tmp_path/'report')

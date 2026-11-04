import json
from pathlib import Path
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.coordinate_validation import planned_rows,CRITERIA
from AlphaBrain.research.dsol.execution.manifest import file_digest
from reports.paper1 import coordinate_validation as report


def build_fixture(tmp_path,monkeypatch):
    states=[];records=[]
    for task in range(12):
        for initial in (40,44,48):
            i=len(records);states.append(dict(initial=i,task=str(task),state_key=f'{task}-{initial}',suite='libero_goal'))
            records.append(dict(state=dict(split='test_unseen_task',initialization=dict(init_state_index=initial))))
    rows=planned_rows(dict(states=states),records)
    def save(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return str(path)
    root=tmp_path/'run';phase=root/'execution/probe-00-real';block=phase/'block';refs={};searches=[];tasks=[]
    for state in states:
        paths=[]
        for repeat in (0,1):
            paths.append(save(tmp_path/f"reference-{state['state_key']}-{repeat}.json",dict(success=True,input_sha256='same',steps=5,replans=1,execution_noise_sha256=['noise'])))
        refs[state['state_key']+'::0']=dict(episode_results=paths,artifacts={p:file_digest(p) for p in paths})
    for j,row in enumerate(rows):
        key={k:row[k] for k in ('state_key','view_id','method')}
        searches.append(dict(key,queries=4))
        for repeat in (0,1):
            success=row['method_family'].startswith('G-') or row['method_family']=='local' or (row['method_family']=='home' and row['mount_yaw_offset_deg']==0.)
            tasks.append(dict(key,repeat=repeat,success=success,replans=1,steps=5,input_sha256='same',execution_noise_sha256=['noise']))
        folder=block/f'search-{j:03d}';folder.mkdir(parents=True)
        np.savez(folder/'query-01.npz',shared_input=[row['initial'],row['view_id']])
    results=block/'results.json';save(results,dict(searches=searches,tasks=tasks))
    save(phase/'binding.json',dict(tasks={'one':dict(request={},output=str(block))}))
    save(phase/'completion.json',dict(status='COMPLETE',dispatch_enabled=False))
    protocol=save(tmp_path/'protocol.json',dict(rows=rows,criteria=CRITERIA))
    references=save(tmp_path/'references.json',dict(views=refs))
    config=save(tmp_path/'config.json',dict(output=str(root),protocol=protocol,baseline_references=references,campaigns=['one']))
    monkeypatch.setattr(report,'validate_block',lambda *a:dict(status='SYNTHETIC_TEST_ONLY'))
    return config,results,block


def test_complete_report_uses_full_panel_and_predeclared_gate(tmp_path,monkeypatch):
    config,_,_=build_fixture(tmp_path,monkeypatch)
    result=report.build(config,tmp_path/'report')
    assert result['decision']=='READY_G_MAIN_WITHIN_VALIDATED_SCOPE'


@pytest.mark.parametrize('damage',['missing','input'])
def test_partial_or_mismatched_input_cannot_pass(tmp_path,monkeypatch,damage):
    config,path,block=build_fixture(tmp_path,monkeypatch)
    if damage=='missing':
        result=json.loads(path.read_text());result['tasks'].pop();path.write_text(json.dumps(result))
    else:np.savez(block/'search-001/query-01.npz',shared_input=[999])
    with pytest.raises(ValueError,match='Incomplete|identical'):
        report.build(config,tmp_path/'report')

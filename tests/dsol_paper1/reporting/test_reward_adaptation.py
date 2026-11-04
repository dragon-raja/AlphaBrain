"""Actual full-panel CPU report on synthetic artifacts; no simulator/models."""
import json
from pathlib import Path
import pytest
pytest.importorskip('matplotlib')
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from reports.paper1.reward_adaptation import build


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


@pytest.mark.parametrize('recovered',[False,True])
def test_complete_report_keeps_failures_and_strict_coverage(tmp_path,recovered):
    root=tmp_path/'run';refs={};rows=[];episodes=[];searches=[]
    for state in range(16):
        key=f'state-{state}';task=f'task-{state//2}'
        for view in range(5):
            paths=[]
            for repeat in (0,1):
                path=save(root/'baseline'/f'{state}-{view}-{repeat}.json',dict(state_key=key,task=task,view_id=view,
                    repeat=repeat,success=view==0,replans=1,execution_noise_sha256=['same']))
                paths.append(path)
            refs[f'{key}::{view}']=dict(episode_results=paths,artifacts={p:file_digest(p) for p in paths})
        for view in range(1,5):
            for arm in ('B','G','L'):
                for seed in range(3):
                    common=dict(state_key=key,task=task,view_id=view,method=f'{arm}-seed-{seed}')
                    rows.append(common)
                    searches.append(dict(common,queries=2,reason='stop',initial_accel=1.,best_accel=.5,
                        validation_scores=[.7,.8],reference_validation_scores=[1.,1.]))
                    for repeat in (0,1):
                        episodes.append(dict(common,repeat=repeat,success=arm=='G',replans=1,
                            execution_noise_sha256=['same'],queries=2,selected_angles=[1.,2.]))
    refs_path=save(root/'references.json',dict(views=refs))
    save(root/'execution-config.json',dict(baseline_references=refs_path,
        input_hashes={refs_path:file_digest(refs_path)},source_manifest='synthetic-source'))
    save(root/'protocol/adaptation.json',dict(synthetic=True))
    campaign=save(root/'evaluation/campaign.json',dict(synthetic=True))
    save(root/'evaluation/allocation.json',dict(phases=[dict(phase='eval-000',path=campaign,sha256=file_digest(campaign))]))
    output=root/'execution/eval-000-real/results/block-0';request=dict(block=dict(rows=rows))
    result=save(output/'results.json',dict(tasks=episodes,searches=searches))
    save(output/'complete.json',dict(status='COMPLETE',request_sha256=digest(request),artifacts={'results.json':file_digest(result)}))
    save(root/'execution/eval-000-real/binding.json',dict(tasks={'block':dict(output=str(output),request=request)}))
    save(root/'execution/eval-000-real/completion.json',dict(status='COMPLETE',dispatch_enabled=False,actual_model_calls=5760))
    for arm in ('B','G','L'):
        for seed in range(3):
            path=root/'training/runs'/f'{arm}-seed-{seed}'
            for i in range(256):save(path/'episodes'/f'{i:04d}.json',dict(metrics=dict(queries=2,reason='stop')))
            for i in range(1,9):
                save(path/'updates'/f'{i:03d}.json',dict(completed_episodes=i*32,means={},
                    reward_settlement=dict(kind=arm,count=i*32,update=i,scale=.01,tau=.001,disagreement_q75=.0001,rewards=[.1]*32)))
    status=dict(status='ALL_SCIENTIFIC_WORK_COMPLETE',dispatch_enabled=False,
        actual_model_calls=dict(training=9216,evaluation=5760,engineering=0))
    if recovered:
        proof=save(root/'recovery.json',dict(interrupted_attempts=271,reused_calls=222,
            records={str(i):{} for i in range(26)},held_failed_phase_cap=4608))
        config=json.loads((root/'execution-config.json').read_text())
        config['continuation_recovery']=proof;config['input_hashes'][proof]=file_digest(proof)
        allocation=json.loads((root/'evaluation/allocation.json').read_text())
        save(root/'evaluation-new/allocation.json',allocation)
        (root/'evaluation/allocation.json').unlink()
        config['evaluation_root']=str(root/'evaluation-new')
        save(root/'execution-config.json',config);status['actual_model_calls']['training']+=49
    save(root/'current.json',status)
    summary=build(root,tmp_path/'report')
    assert summary['arms']['G']['success_rate']==1 and summary['arms']['L']['success_rate']==0
    assert summary['recovery_accounting'].get('discarded_attempts',0)==(49 if recovered else 0)
    assert (tmp_path/'report/results.pdf').is_file()
    assert json.loads((tmp_path/'report/completion.json').read_text())['operation_episodes']==1152
    with pytest.raises(ValueError,match='new'):build(root,tmp_path/'report')
    # Missing one failure must not produce an apparently improved complete report.
    episodes.pop();save(result,dict(tasks=episodes,searches=searches))
    save(output/'complete.json',dict(status='COMPLETE',request_sha256=digest(request),artifacts={'results.json':file_digest(result)}))
    with pytest.raises(ValueError,match='coverage'):build(root,tmp_path/'incomplete')

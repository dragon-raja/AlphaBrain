"""Synthetic complete G report, including the deliberately unequal B seed coverage."""
import json
from pathlib import Path
import pytest
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from AlphaBrain.research.dsol.analysis.recovery_summary import paired_method_contrast
from reports.paper1 import native_online_reward as report


def test_matching_never_fills_missing_B_seed():
    row=dict(task='t',state_key='s',view_id=1,repeat=0,method='G-seed-0',success=True)
    ref=dict(row,method='B-seed-0',success=False)
    assert paired_method_contrast([row],[ref])['difference_pp']==100
    with pytest.raises(ValueError):paired_method_contrast([row,dict(row,method='G-seed-1')],[ref])


def test_actual_full_G_report(tmp_path,monkeypatch):
    root=tmp_path/'run';old=tmp_path/'old'
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)
    refs={};grows=[];brows=[];searches=[];planned=[]
    splits=['seen_new_initials']*20+['validation']*8+['unseen_test']*12
    for task,split in enumerate(splits):
        for initial in range(10):
            state=f'{task}:{initial}'
            for view in range(9):
                paths=[];artifacts={}
                for repeat in (0,1):
                    value=dict(state_key=state,view_id=view,repeat=repeat,success=view==0,replans=1,execution_noise_sha256=['paired'])
                    p=save(tmp_path/f'base/{state}/{view}-{repeat}.json',value);paths.append(p);artifacts[p]=file_digest(p)
                refs[f'{state}::{view}']=dict(episode_results=paths,artifacts=artifacts)
                if not view:continue
                for seed in (0,1,2):
                    row=dict(task=f't{task}',state_key=state,view_id=view,method=f'G-seed-{seed}',split=split)
                    planned.append(row);searches.append(dict(row,queries=2))
                    for repeat in (0,1):
                        value=dict(row,repeat=repeat,success=True,replans=1,execution_noise_sha256=['paired'],queries=2,selected_angles=[1,2])
                        grows.append(value)
                        if split=='unseen_test' or seed==0:brows.append(dict(value,method=f'B-seed-{seed}',success=False))
    def phase(path,rows,sr):
        addresses={(r['state_key'],r['view_id'],r['method']) for r in rows}
        selected=[dict(state_key=a,view_id=b,method=c) for a,b,c in sorted(addresses)]
        request=dict(block=dict(rows=selected));folder=path/'results';p=save(folder/'results.json',dict(tasks=rows,searches=sr))
        save(folder/'complete.json',dict(status='COMPLETE',request_sha256=digest(request),artifacts={'results.json':file_digest(p)}))
        save(path/'binding.json',dict(tasks={'block':dict(output=str(folder),request=request)}))
        save(path/'completion.json',dict(status='COMPLETE',dispatch_enabled=False,actual_model_calls=100))
    phase(root/'execution/eval-000-real',grows,searches);phase(old/'execution-recovery/old-000-real',brows,[])
    campaign=save(tmp_path/'campaign.json',{})
    save(root/'evaluation/allocation.json',dict(phases=[dict(phase='eval-000',path=campaign,sha256=file_digest(campaign))]))
    save(old/'protocol/recovery/allocation.json',dict(phases=[dict(phase='old-000')]))
    for seed in (0,1,2):
        for i in range(2048):save(root/f'training/runs/G-seed-{seed}/episodes/{i:04d}.json',dict(metrics=dict(queries=1)))
        for i in range(1,65):save(root/f'training/runs/G-seed-{seed}/updates/{i:03d}.json',dict(completed_episodes=i*32,reward_settlement=dict(scale=.01)))
    save(root/'current.json',dict(status='ALL_SCIENTIFIC_WORK_COMPLETE',dispatch_enabled=False,actual_model_calls=dict(training=12288,evaluation=100,engineering=0)))
    reference=save(tmp_path/'refs.json',dict(views=refs));protocol=save(tmp_path/'protocol.json',{});original=save(tmp_path/'original.json',{})
    config=save(tmp_path/'config.json',dict(protocol=protocol,baseline_references=reference,original_recovery_schedule=original,input_hashes={reference:file_digest(reference),protocol:file_digest(protocol)}))
    monkeypatch.setattr(report.reward_pilot,'load_protocol',lambda *args:dict(initial_update=0))
    monkeypatch.setattr(report.reward_pilot,'evaluation_rows',lambda *args:planned)
    result=report.build(root,tmp_path/'report',config_path=config,historical_root=old)
    assert result['episodes']==19200 and result['searches']==9600
    assert result['matched_G_minus_B']['unseen_test']['policy_seeds']==[0,1,2]
    assert result['matched_G_minus_B']['validation']['policy_seeds']==[0]
    assert result['matched_G_minus_B']['seen_new_initials']['difference_pp']==100
    with pytest.raises(ValueError):report.build(root,tmp_path/'report',config_path=config,historical_root=old)

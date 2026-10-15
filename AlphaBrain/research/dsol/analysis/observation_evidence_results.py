"""Summarize completed fixed-panel probes without turning errors into success."""
from pathlib import Path
import argparse
import numpy as np
import yaml
from ..data.artifacts import read,sha,write
from ..metrics.observation_response import response_metrics
from ..metrics import observation_response


def summarize(panel_file):
    root=Path(panel_file).parent;panel=read(panel_file);launch=read(root/'launch.json')
    result=dict(scope='Outcome-blind, in-sample diagnostic; one source episode and two states per task. Not heldout accuracy, not closed-loop success, not a modality-isolated causal test.',
        primary_channels='First6 end-effector action coordinates; errors standardized by fixed training action std; compare models within benchmark, not absolute errors across embodiments',
        state_response='Canonical camera only; paired same scoring noise across the two real states; changed whole observation includes robot state in RoboCasa',
        benchmarks={},provenance={str(panel_file):sha(panel_file),str(root/'visual_review.json'):sha(root/'visual_review.json'),str(Path(__file__)):sha(__file__),
            str(Path(observation_response.__file__)):sha(observation_response.__file__)})
    for benchmark,spec in panel['benchmarks'].items():
        with np.load(spec['payload']) as z:target=z['actions']
        records=spec['records'];tasks=[r['task'] for r in records[::2]]
        assert all(records[i]['task']==records[i+1]['task'] and records[i]['phase']==0 and records[i+1]['phase']==1 for i in range(0,len(records),2))
        models={};scale=None
        for job in launch['jobs']:
            if not job['name'].startswith(benchmark+'-'):continue
            out=root/'probes'/job['name'];status=read(out/'status.json');assert status['status']=='COMPLETE'
            assert status['panel_sha256']==sha(panel_file)
            path=out/'predictions.npz';assert sha(path)==status['payload_sha256']
            identity=status['identity'];checkpoint=Path(identity['checkpoint'])
            if benchmark=='robocasa':local_scale=np.array(read(checkpoint/'assets/norm_stats.json')['norm_stats']['actions']['std'][:12])
            else:local_scale=np.array(yaml.safe_load((checkpoint/'framework_config.yaml').read_text())['framework']['normalization']['action_std'])
            if scale is None:scale=local_scale
            else:np.testing.assert_array_equal(scale,local_scale)
            with np.load(path) as z:pred=z['actions'];acc=z['accel']
            steps=25 if benchmark=='robocasa' else 5
            metrics=response_metrics(pred,target,scale,execute_steps=steps,dims=list(range(6)))
            metrics['mean_accel']=float(acc.mean());metrics['tasks']=tasks
            if benchmark=='robocasa':metrics['base_motion']=response_metrics(pred,target,scale,execute_steps=steps,dims=[7,8,9])
            models[job['name'].removeprefix(benchmark+'-')]=metrics
            result['provenance'][str(path)]=sha(path)
        result['benchmarks'][benchmark]=dict(tasks=tasks,states=len(records),models=models,action_scale=scale.tolist())
    write(root/'response-summary.json',result)
    print({b:{m:{k:round(v[k],5) for k in ('view_response_rms','canonical_state_response_rms','canonical_demonstration_mse','perturbed_demonstration_mse','mean_accel')}
        for m,v in item['models'].items()} for b,item in result['benchmarks'].items()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);args=p.parse_args();summarize(args.panel)

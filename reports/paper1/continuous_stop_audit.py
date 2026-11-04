"""Read existing training/evaluation artifacts; never start a simulator or VLA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from AlphaBrain.research.dsol.analysis.continuous_stop_audit import training_summary, distribution


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-policy',action='store_true')
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args();cfg=read(args.config);out={}
    binding=read(cfg['evaluation_binding'])
    for model,root in cfg['training'].items():
        root=Path(root);records=[read(p) for p in sorted((root/'episodes').glob('*.json'))]
        assert len(records)==512
        summary=training_summary(records,scale=cfg['scale'],cost=cfg['cost'])
        summary['updates_optimizer']=[read(root/'updates'/f'{i:03d}.json') for i in range(1,17)]
        summary['status']=read(root/'status.json');summary['budget']=read(root/'budget/state.json')
        summary['binding']=read(root/'binding.json')
        summary['inputs_sha256']={str(p):sha(p) for p in [root/'latest.json',root/'status.json',root/'binding.json']}
        methods={};evaluations=[]
        for spec in binding['tasks'].values():
            if spec['model']!=model or spec['request']['block']['kind']!='paired':continue
            p=Path(spec['output']);payload=read(p/'results.json')
            for index,s in enumerate(payload['searches']):
                method=s['method'];rows=methods.setdefault(method,[])
                results=[t for t in payload['tasks'] if t['method']==method]
                row=dict(initial=s['initial'],task=s['task'],start=s['start'],queries=s['queries'],
                    initial_accel=s['initial_accel'],best_accel=s['best_accel'],reason=s['reason'],
                    selected_angles=s['selected_angles'],initial_angles=s['events'][0]['angles'],
                    success=sum(t['success'] for t in results),episodes=len(results),
                    absolute_improvement=s['initial_accel']-s['best_accel'],
                    proxy_return=(s['initial_accel']-s['best_accel'])/cfg['scale']-cfg['cost']*(s['queries']-1),
                    session=str(p/f'session-{index:03d}'))
                rows.append(row)
                if method=='continuous_ppo':evaluations.append(row)
        summary['evaluation_methods']={m:dict(sessions=rows,
            initial_accel=distribution(r['initial_accel'] for r in rows),
            absolute_improvement=distribution(r['absolute_improvement'] for r in rows),
            positive_proxy_return=sum(r['proxy_return']>1e-10 for r in rows),
            mean_proxy_return=distribution(r['proxy_return'] for r in rows),
            reasons=dict(Counter(r['reason'] for r in rows))) for m,rows in methods.items()}
        if args.cpu_policy:
            summary['cpu_policy']=policy_audit(root,records,Path(cfg['frozen'][model]),evaluations,cfg)
        out[model]=summary
        print(model,'complete',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(models=out,config=cfg,scientific_calls=0,new_training=False),indent=2)+'\n')
    if args.figure:write_figure(out,args.figure)


def write_figure(models,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['Canonical SFT','Broad SFT','Native LIBERO'];colors=['#d46a27','#387bb5','#278864']
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for (model,row),label,color in zip(models.items(),labels,colors):
        updates=row['updates_optimizer']
        axes[0,0].plot(range(1,17),[r['means']['mean_stop_probability'] for r in updates],label=label,color=color)
        axes[0,1].plot(range(1,17),[row['updates'][str(i)]['queries']/32 for i in range(16)],color=color)
    axes[0,0].axhline(.5,color='grey',linestyle='--',linewidth=1);axes[0,0].set(title='STOP probability on optimization batches',xlabel='PPO update',ylabel='Mean probability',ylim=(0,1));axes[0,0].legend()
    axes[0,1].set(title='Actual training query use',xlabel='PPO update',ylabel='Queries per episode')
    values=[r['continued']['return_after_initial']['mean'] for r in models.values()]
    axes[1,0].bar(labels,values,color=colors);axes[1,0].axhline(0,color='black',linewidth=.8);axes[1,0].set(title='Observed continuations: net proxy reward',ylabel='Mean return (initial cost excluded)')
    x=np.arange(3)
    for i,(method,label) in enumerate([('stay','Stay'),('local_near_best_accel','Local Accel'),('continuous_ppo','PPO')]):
        vals=[sum(s['success'] for s in r['evaluation_methods'][method]['sessions'])/64 for r in models.values()]
        axes[1,1].bar(x+(i-1)*.24,vals,.24,label=label)
    axes[1,1].set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='Success / 64 paired executions',title='Completed evaluation');axes[1,1].legend()
    fig.suptitle('Frozen training/evaluation audit — descriptive, one RL seed',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


def policy_audit(root,records,frozen,evaluations,cfg):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Explicit CPU-only environment required')
    import numpy as np
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_ppo import batch_observations
    from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
    from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
    from AlphaBrain.research.dsol.data.search_replay import QueryResult
    torch.set_num_threads(1)
    policy=ContinuousActorCritic(4608).eval()
    receipt=read(frozen/'receipt.json');assert sha(frozen/'actor.pt')==receipt['actor_sha256']
    final=torch.load(frozen/'actor.pt',map_location='cpu',weights_only=False)
    latest=read(root/'latest.json');assert sha(root/latest['path'])==latest['sha256']==receipt['checkpoint_sha256']
    ck=torch.load(root/latest['path'],map_location='cpu',weights_only=False)
    assert all(torch.equal(v,ck['policy'][k]) for k,v in final['state_dict'].items())
    initial=[];first_action_stop=[];rewards=[];all_finite=True;by_update={}
    for record in records:
        p=root/record['path'];assert sha(p)==record['sha256']
        ep=torch.load(p,map_location='cpu',weights_only=False)['episode']
        assert ep['policy_update']==record['metrics']['policy_update']
        by_update.setdefault(ep['policy_update'],[]).extend(ep['transitions'])
        assert abs(sum(t['reward'] for t in ep['transitions'])-record['metrics']['total_reward'])<1e-8
        initial.append(ep['transitions'][0]['observation'])
        first_action_stop.append(ep['transitions'][0]['action']['stop'])
        for t in ep['transitions']:
            all_finite &= all(torch.isfinite(torch.as_tensor(t[k])).all().item() for k in ('reward','value','log_prob'))
            rewards.append(dict(stop=bool(t['action']['stop']),reward=float(t['reward'])))
    def probabilities(obs):
        vals=[]
        with torch.inference_mode():
            for i in range(0,len(obs),32):
                vals.extend(policy.distribution(batch_observations(obs[i:i+32])).stop.probs.tolist())
        return vals
    checkpoint_panels={}
    replay=[]
    for update,transitions in sorted(by_update.items()):
        snapshot=root/'batches'/f'update-{update:03d}'/'policy.pt'
        state=torch.load(snapshot,map_location='cpu',weights_only=False)
        policy.load_state_dict(state['state_dict'],strict=True)
        probabilities_fixed=probabilities(initial)
        errors=[];value_errors=[]
        with torch.inference_mode():
            for i in range(0,len(transitions),32):
                batch=transitions[i:i+32];obs=batch_observations([t['observation'] for t in batch])
                stop=torch.tensor([t['action']['stop'] for t in batch],dtype=torch.bool)
                latent=torch.stack([torch.as_tensor(t['action']['latent']) for t in batch])
                logp=policy.distribution(obs).log_prob(stop,latent)
                errors.extend(abs(float(p)-t['log_prob']) for p,t in zip(logp,batch))
                value_errors.extend(abs(float(p)-t['value']) for p,t in zip(policy.value(obs),batch))
        replay.append(dict(update=update,snapshot_sha256=sha(snapshot),transitions=len(transitions),
            max_log_prob_error=max(errors),max_value_error=max(value_errors),
            fixed_initial_stop_probability=distribution(probabilities_fixed),
            fixed_initial_deterministic_stop=sum(p>=.5 for p in probabilities_fixed)))
    if max(r['max_log_prob_error'] for r in replay)>1e-4 or max(r['max_value_error'] for r in replay)>1e-4:
        raise ValueError('Stored rollout likelihood/value differs from sampling snapshot')
    for path in sorted((root/'checkpoints').glob('update-*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False);policy.load_state_dict(state['policy'],strict=True)
        probs=probabilities(initial)
        checkpoint_panels[path.stem]=dict(sha256=sha(path),probability=distribution(probs),deterministic_stop=sum(p>=.5 for p in probs))
    policy.load_state_dict(final['state_dict'],strict=True)
    eval_obs=[]
    for row in evaluations:
        with np.load(Path(row['session'])/'query-01.npz',allow_pickle=False) as z:
            angles=np.asarray(row['initial_angles']);rotation=PanTiltMount(np.eye(4)).pose(angles)
            base=z['camera_to_world_opencv']@np.linalg.inv(rotation)
            mount=PanTiltMount(base)
            scores=np.asarray([z['score_0'],z['score_1']])
            value=QueryResult(z['external_0'].copy(),float(scores.mean()),float(scores.std()),2)
            core=ContinuousSearch(mount,None,initial_angles=angles,physical_state_id='saved-observation-only',
                wrist=z['wrist_0'].copy(),score_scale=cfg['scale'],seed=0)
            core.records=[(angles.copy(),value)];core.status='ACTIVE';core.queries=1
            eval_obs.append(continuous_observation(core.observation()))
    eval_probs=probabilities(eval_obs)
    # Export controls are independent of reconstructed evaluation inputs.
    # Reconstructed inputs use saved measurements and original CPU primitives.
    controls=torch.load(frozen/'controls.pt',map_location='cpu',weights_only=False)
    controls_ok=[]
    with torch.inference_mode():
        for c in controls:
            actual=policy.distribution(c['observation']).sample(deterministic=True)
            controls_ok.append(all(torch.equal(actual[k],c['action'][k]) for k in c['action']))
    if not all(controls_ok):raise ValueError('Exported CPU control differs')
    return dict(export_matches_final=True,controls_exact=len(controls_ok),episode_hashes_verified=len(records),
        finite_recorded_values=all_finite,first_action_stop=sum(bool(x) for x in first_action_stop),
        stored_transition_rewards={str(stop):distribution(r['reward'] for r in rewards if r['stop']==stop) for stop in (False,True)},
        checkpoint_panels=checkpoint_panels,rollout_replay=replay,
        evaluation_initial_probabilities=eval_probs,evaluation_initial_stop=sum(p>=.5 for p in eval_probs),
        reconstructed_evaluation_inputs=True,source_policy=str(sys.modules[ContinuousActorCritic.__module__].__file__),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])


if __name__=='__main__':main()

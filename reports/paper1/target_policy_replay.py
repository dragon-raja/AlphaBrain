"""CPU-only probability replay of saved PPO records; no optimizer or simulator."""
import argparse
import hashlib
import json
from pathlib import Path


def run(root, output):
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU replay must not initialize CUDA')
    root=Path(root);output=Path(output);results=[];manifest={}
    for folder in sorted((root/'training/runs').iterdir()):
        for version in range(8):
            actors=[]
            for v in (version,version+1):
                p=folder/'publications'/f'policy-{v:04d}.pt'
                manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                receipt=(json.loads((root/'training/batches'/f'{folder.name}-{v:02d}.json').read_text())
                         if v<8 else json.loads((folder/'publication.json').read_text()))
                assert manifest[str(p)]==receipt['policy_sha256']
                saved=torch.load(p,map_location='cpu',weights_only=True)
                assert saved['version']==v
                policy=ContinuousActorCritic(4608).eval();policy.load_state_dict(saved['state_dict'],strict=True)
                actors.append(policy)
            errors=[];ratios=[];before=[];after=[];first_before=[];first_after=[]
            for i in range(version*8,(version+1)*8):
                receipt=json.loads((folder/'episodes'/f'{i:04d}.json').read_text())
                p=folder/receipt['path'];stamp=hashlib.sha256(p.read_bytes()).hexdigest()
                assert stamp==receipt['sha256'];manifest[str(p)]=stamp
                # These are our hash-bound local training artifacts, not external pickles.
                e=torch.load(p,map_location='cpu',weights_only=False)['episode']
                assert e['policy_update']==version and e['status']=='COMPLETE'
                for j,t in enumerate(e['transitions']):
                    obs={k:v.unsqueeze(0) for k,v in t['observation'].items()}
                    stop=torch.tensor([t['action']['stop']],dtype=torch.bool)
                    latent=t['action']['latent'].unsqueeze(0)
                    with torch.no_grad():
                        a,b=[m.distribution(obs) for m in actors]
                        lp=a.log_prob(stop,latent).item();n=b.log_prob(stop,latent).item()
                    errors.append(abs(lp-float(t['log_prob'])))
                    ratios.append(n-lp);before.append(a.stop.probs.item());after.append(b.stop.probs.item())
                    if j==0:first_before.append(before[-1]);first_after.append(after[-1])
            delta=torch.tensor(ratios,dtype=torch.float64)
            results.append(dict(run=folder.name,update=version+1,transitions=len(errors),
                old_logprob_max_error=max(errors),mean_stop_before=sum(before)/len(before),
                mean_stop_after=sum(after)/len(after),initial_stop_before=sum(first_before)/8,
                initial_stop_after=sum(first_after)/8,
                sampled_kl_before_after=float((delta.exp()-1-delta).mean()),
                ratio_outside_clip_fraction=float(((delta.exp()-1).abs()>.2).double().mean())))
    assert not torch.cuda.is_initialized()
    output.write_text(json.dumps(dict(status='COMPLETE',optimizer_updates=0,operation_model_calls=0,
        cuda_initialized=False,rows=results,input_sha256=manifest),indent=2)+'\n')
    print(json.dumps(dict(rows=len(results),max_old_logprob_error=max(r['old_logprob_max_error'] for r in results))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)

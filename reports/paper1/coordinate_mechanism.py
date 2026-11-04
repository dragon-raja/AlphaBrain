"""CPU-only paired first-decision audit of already trained search policies.

Consumes frozen tensor observations; never constructs a simulator, operation
model, optimizer, production queue, or remote service. It cannot report task
success for counterfactual actions that were never executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from AlphaBrain.research.dsol.analysis.coordinate_mechanism import mounting_intervention, aggregate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(manifest_path, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CUDA_VISIBLE_DEVICES empty required for this CPU audit')
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest['schema'] != 'G_COORDINATE_FIRST_DECISION_V1'
            or manifest['mount_offsets_deg'] != [-5., 0., 5.]
            or manifest['new_vla_call_cap'] != 0 or manifest['training_updates'] != 0):
        raise ValueError('Frozen CPU mechanism protocol required')
    for filename, expected in manifest['input_hashes'].items():
        if digest(filename) != expected: raise ValueError('Input changed: '+filename)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    (root/'attempt.json').write_text(json.dumps(dict(started=time.time(),manifest_sha256=digest(manifest_path),automatic_retry=False)))
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if str(torch.__version__) != manifest['torch_version']:
        raise ValueError('Unadmitted CPU numerical stack')
    rows, gates = [], []
    with torch.inference_mode():
        observations = []
        for case in manifest['cases']:
            saved = torch.load(case['episode_path'], map_location='cpu', weights_only=True)
            observation = {k: v.numpy().copy() for k,v in saved['episode']['transitions'][0]['observation'].items()}
            observations.append((case, observation))
        for spec in manifest['actors']:
            value = torch.load(spec['path'], map_location='cpu', weights_only=True)
            actor = ContinuousActorCritic(value['visual_dim']).eval().requires_grad_(False)
            actor.load_state_dict(value['state_dict'], strict=True)
            controls = torch.load(spec['controls'], map_location='cpu', weights_only=True)
            for control in controls:
                action = actor.distribution(control['observation']).sample(deterministic=True)
                for key in ('stop','delta_deg','log_prob'):
                    if not torch.equal(action[key], control['action'][key]):
                        raise ValueError('Original deployment control differs: '+key)
            gates.append(dict(seed=spec['seed'], exact_deployment_controls=len(controls)))
            for case, observation in observations:
                for offset in manifest['mount_offsets_deg']:
                    changed = mounting_intervention(observation, case['angles_deg'], offset)
                    for key in observation:
                        if key != 'joint_state' and not np.array_equal(observation[key], changed[key]):
                            raise ValueError('Nonmechanical observation changed')
                    tensor = {k: torch.as_tensor(v[None]) for k,v in changed.items()}
                    dist = actor.distribution(tensor)
                    action = dist.sample(deterministic=True)
                    conditional = dist.transform(dist.normal.loc)[0].numpy()
                    rows.append(dict(seed=spec['seed'],task=case['task'],source_episode=case['source_episode'],
                        state_key=case['state_key'],original_angles_deg=case['angles_deg'],mount_yaw_deg=offset,
                        mechanical_angles_deg=(np.asarray(case['angles_deg'])-[offset,0.]).tolist(),
                        stop_probability=float(dist.stop.probs[0]),stop=bool(action['stop'][0]),
                        conditional_delta_deg=conditional.tolist(),effective_delta_deg=action['delta_deg'][0].numpy().tolist()))
            print(json.dumps(dict(seed=spec['seed'],cases=len(observations),status='CPU_ACTOR_COMPLETE')),flush=True)
    if torch.cuda.is_initialized(): raise ValueError('Unexpected CUDA initialization')
    summary = dict(status='COMPLETE_CPU_MECHANISM_ONLY',cases=len(observations),
        actor_forward_interventions=len(rows),gates=gates,seeds=aggregate(rows),
        new_vla_calls=0,new_manipulation_episodes=0,training_updates=0,
        interpretation='First-decision response to lawful mounting changes with identical images, scores, history and action support. Not a counterfactual success-rate evaluation or proof of repaired G performance.',
        finished=time.time(),torch_version=str(torch.__version__),manifest_sha256=digest(manifest_path))
    (root/'rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(args.manifest,args.output)

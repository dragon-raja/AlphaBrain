"""Bounded continuous-action engineering checks, no optimization/manipulation."""
import argparse
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from .local_view_runtime import RenderedViewSource
from .search_live import simulator, check_simulator_dependencies
from .search_runtime import PolicyRuntime


def run(config_path, output, *, execute=False):
    if not execute: raise ValueError('Explicit engineering authorization required')
    config = read(config_path); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    runtime = None; cases = []
    try:
        if (config['continuous_action_cap'] != 64 or
                [(c['cohort'],c['initial']) for c in config['continuous_cases']] != [('A',2),('A',10),('B',0),('B',4)]):
            raise ValueError('Four fixed cases, cap64 required')
        for path,digest in config['input_hashes'].items():
            if sha(path) != digest: raise ValueError('Bound input changed: '+path)
        check_simulator_dependencies(config)
        runtime = PolicyRuntime(config, output, 64)
        import torch
        from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
        payload = torch.load(config['continuous_actor'], map_location='cpu', weights_only=True)
        actor = ContinuousActorCritic(4608).eval()
        actor.load_state_dict(payload['state_dict'], strict=True); actor.requires_grad_(False)
        for index,spec in enumerate(config['continuous_cases']):
            initial = spec['initial']; directory = output/f"{spec['cohort']}-{initial:02d}"; directory.mkdir()
            case_config = read(spec['config'])
            with np.load(config['anchors'] if spec['cohort']=='A' else config['domain_noise'], allow_pickle=False) as z:
                noises = z['first_score_noise'][initial,:2].copy() if spec['cohort']=='A' else z['score'][initial].copy()
            start = [-17.5,9.5] if index%2==0 else [17.5,-9.5]
            with simulator(spec['config'], case_config, initial, directory) as request:
                source = RenderedViewSource(request, runtime, noises, directory)
                source.move_angles(start)
                # Reuse this one actual acquisition for the core's charged initial observation.
                first = source.query(); original_query = source.query; pending = [first]
                def query(): return pending.pop() if pending else original_query()
                source.query = query
                core = ContinuousSearch(source.mount, source, initial_angles=start,
                    physical_state_id=source.physical, wrist=source.last_values[0]['wrist'],
                    score_scale=config['score_scale'], seed=20261022+index)
                obs = core.start()
                tensor = {k:torch.as_tensor(v[None]) for k,v in obs.items()}
                torch.manual_seed(1)  # Fixed engineering draw; no success-based sampling.
                with torch.inference_mode(): action = actor.distribution(tensor).sample()
                if action['stop'].item(): raise ValueError('Engineering continue draw unexpectedly STOP')
                delta = action['delta_deg'][0].numpy()
                core.step(dict(stop=False, delta_deg=delta))
                visited = core.current.copy()
                if np.array_equal(first.value.visual, core.records[-1][1].visual):
                    raise ValueError('Nonzero continuous action did not change external feature')
                core.step(dict(stop=True, delta_deg=[0.,0.]))
                info = core.info()
                if info['status'] != 'COMPLETE' or info['queries'] != source.queries:
                    raise ValueError('Core/source query accounting mismatch')
                # Independent return controls at the newly sampled pose and initial pose.
                # Charged to engineering budget, not disguised as core episode queries.
                for target in (visited, np.asarray(start)):
                    core._move(target); source.query()
                row = dict(cohort=spec['cohort'], initial=initial, start=start, delta_deg=delta.tolist(),
                    info=info, engineering_queries=source.queries, exact_return=True,
                    stop_log_probability=float(action['log_prob'][0]), latent=action['latent'][0].tolist(),
                    events=core.events, training_steps=0, manipulation_actions=0)
                write(directory/'control.json',row)
                row.update(reference=str(directory/'control.json'),sha256=sha(directory/'control.json'))
                cases.append(row); write(output/'progress.json',dict(calls=runtime.calls,completed=len(cases)))
        write(output/'status.json',dict(status='COMPLETE',config_sha256=sha(config_path),calls=runtime.calls,
            cases=cases,training_steps=0,manipulation_actions=0,automatic_training=False))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),
            calls=runtime.calls if runtime else 0,completed=len(cases),automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    run(a.config,a.output,execute=a.execute)

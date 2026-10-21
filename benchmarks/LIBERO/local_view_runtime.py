"""Actual continuous render/score source and bounded 64-call admission.

Four frozen states, four poses each, two fixed noises at first visit and
after a detour/return. No manipulation actions or actor training/evaluation.
"""
import argparse
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount, PoseReceipt, ViewMeasurement
from AlphaBrain.research.dsol.data.search_replay import QueryResult
from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .search_live import simulator, verify_example, check_simulator_dependencies
from .search_domain import exact_input
from .search_features import load_observations
from .search_runtime import PolicyRuntime, exact_prediction


TARGETS = [[0., 0.], [7.125, -4.375], [-9.625, 6.25], [29.5, 19.5]]


class RenderedViewSource:
    def __init__(self, request, runtime, noises, output):
        self.request, self.runtime = request, runtime
        self.noises = np.array(noises, copy=True)
        if self.noises.shape != (2, 10, 7) or not np.isfinite(self.noises).all():
            raise ValueError('Two explicit scoring-noise tensors required')
        self.output = Path(output)
        info = request(dict(kind='continuous_info'))
        self.mount = PanTiltMount(info['reference'])
        self.physical = info['receipt']['physical_state_id']
        self.current = np.zeros(2)
        self.queries = 0
        self.references = {}
        self.moves = 0
        self.failed = False

    def _receipt(self, value):
        if value['physical_state_id'] != self.physical or value['settled'] is not True:
            raise ValueError('Actual continuous source state/settling mismatch')
        return PoseReceipt(value['camera_to_world'], value['physical_state_id'], value['settled'])

    def move(self, pose):
        if self.failed:
            raise ValueError('Source previously failed; no retry')
        try:
            receipt = self._receipt(self.request(dict(kind='move_pose',camera_to_world=pose))['receipt'])
            if not np.allclose(receipt.camera_to_world, pose, atol=1e-9, rtol=0):
                raise ValueError('Real renderer did not apply requested pose')
            self.moves += 1
            return receipt
        except Exception:
            self.failed = True
            raise

    def move_angles(self, angles):
        for point in self.mount.route(self.current, angles):
            self.move(self.mount.pose(point))
            self.current = point.copy()

    def query(self):
        if self.failed or self.queries >= 8:
            raise ValueError('Source failed or eight-query admission budget exhausted')
        self.queries += 1
        try:
            response = self.request(dict(kind='query_pose'))
            receipt = self._receipt(response['receipt'])
            example = response['example']
            key = tuple(receipt.camera_to_world.ravel())
            # Check input repeats before spending the next model call.
            if key in self.references:
                exact_input(self.references[key][0], example)
            values = [self.runtime.infer(example, noise, dict(stage='continuous_admission',
                case=self.output.name,query=self.queries,member=m),capture=True)
                for m, noise in enumerate(self.noises)]
            if key in self.references:
                for a,b in zip(values,self.references[key][1]):
                    exact_prediction(a,b)
            else:
                self.references[key]=(example,values)
            saved=dict(external_image=example['image'][0],wrist_image=example['image'][1],
                robot_state=example['state'],camera_intrinsics=example['camera_intrinsics'],
                camera_to_world_opencv=example['camera_to_world_opencv'])
            for m,value in enumerate(values):
                for field in ('actions','trace','external','wrist','score'):
                    saved[f'{field}_{m}']=value[field]
            np.savez_compressed(self.output/f'query-{self.queries:02d}.npz',**saved)
            self.last_example,self.last_values=example,values
            scores=np.asarray([v['score'] for v in values])
            return ViewMeasurement(receipt,QueryResult(values[0]['external'],float(scores.mean()),float(scores.std()),2))
        except Exception:
            self.failed=True
            raise


def run(config_path, output, *, execute=False):
    if not execute:
        raise ValueError('Explicit --execute required')
    config=read(config_path); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    runtime=None;cases=[]
    try:
        if (config['continuous_call_cap']!=64 or config['continuous_targets']!=TARGETS
                or [(c['cohort'],c['initial']) for c in config['continuous_cases']] != [('A',2),('A',10),('B',0),('B',4)]):
            raise ValueError('Frozen four-state, sixteen-pose, sixty-four-call admission required')
        for path,digest in config['input_hashes'].items():
            if sha(path)!=digest:raise ValueError('Bound continuous asset changed: '+path)
        check_simulator_dependencies(config)
        runtime=PolicyRuntime(config,output,64)
        for spec in config['continuous_cases']:
            case_config=read(spec['config']); initial=spec['initial']
            directory=output/f"{spec['cohort']}-{initial:02d}";directory.mkdir()
            if spec['cohort']=='A':
                with np.load(config['anchors'],allow_pickle=False) as z:noises=z['first_score_noise'][initial,:2].copy()
                cached=load_observations(read(case_config['inputs'])['records'][initial])
                with np.load(Path(config['native'])/'initials'/f'{initial:02d}.npz',allow_pickle=False) as z:
                    native={k:z[k][0].copy() for k in ('actions','trace')}
                with np.load(Path(config['native'])/'siglip-scores.npz',allow_pickle=False) as z:
                    native.update({k:z[k][initial,0].copy() for k in ('scores','external','wrist')})
            else:
                with np.load(config['domain_noise'],allow_pickle=False) as z:noises=z['score'][initial].copy()
                cached=native=None
            with simulator(spec['config'],case_config,initial,directory) as request:
                source=RenderedViewSource(request,runtime,noises,directory)
                base_example=None
                for pose_index,target in enumerate(TARGETS):
                    source.move_angles(target)
                    source.query(); first=source.last_example; values=source.last_values
                    if pose_index==0:
                        base_example=first
                        if cached is not None:
                            verify_example(first,cached,0)
                            for m,value in enumerate(values):
                                reference={k:native[k][m] for k in ('actions','trace')}
                                reference.update(score=native['scores'][m],external=native['external'],wrist=native['wrist'])
                                exact_prediction(value,reference)
                    else:
                        for key in ('state','camera_intrinsics'):
                            if not np.array_equal(first[key],base_example[key]):raise ValueError('Fixed input changed: '+key)
                        if not np.array_equal(first['image'][1],base_example['image'][1]):raise ValueError('Wrist pixels changed')
                        if np.array_equal(first['image'][0],base_example['image'][0]):raise ValueError('External image did not respond to movement')
                    source.move_angles([-29.,-19.])
                    source.move_angles(target)
                    source.query()
                    exact_input(first,source.last_example)
                    for a,b in zip(values,source.last_values):exact_prediction(a,b)
                    cases.append(dict(cohort=spec['cohort'],initial=initial,angles=target,
                        pose_input_return_exact=True,model_return_exact=True,
                        reference=str(directory/f'query-{source.queries:02d}.npz'),
                        sha256=sha(directory/f'query-{source.queries:02d}.npz')))
                    write(output/'progress.json',dict(calls=runtime.calls,completed=len(cases)))
        write(output/'status.json',dict(status='COMPLETE',calls=runtime.calls,cases=cases,
            config_sha256=sha(config_path),new_manipulation_actions=0,training_steps=0,
            scope='Continuous kinematic camera rendering; not physical head motion or search efficacy'))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),calls=runtime.calls if runtime else 0,
            completed=len(cases),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    run(args.config,args.output,execute=args.execute)

"""Bounded input/reset and feature-hook admission for frozen unseen tasks.

Four calls per initial: two independent, explicitly stored noises, each with
feature capture off/on. No manipulation actions, fitting, or outcome selection.
"""
import argparse
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .search_live import simulator, check_simulator_dependencies
from .search_runtime import PolicyRuntime


def exact_input(first, second):
    for key in ('state', 'camera_intrinsics', 'camera_to_world_opencv'):
        if not np.array_equal(first[key], second[key]):
            raise ValueError('Reset/render input mismatch: '+key)
    if first['lang'] != second['lang'] or len(first['image']) != len(second['image']):
        raise ValueError('Language or view count changed')
    for a, b in zip(first['image'], second['image']):
        if not np.array_equal(a, b):
            raise ValueError('Reset/render pixels changed')


def run(config_path, output, *, execute=False):
    if not execute:
        raise ValueError('Explicit --execute required')
    config=read(config_path); output=Path(output)
    output.mkdir(parents=True,exist_ok=False); runtime=None
    initials=config['domain_initials']
    if initials!=list(range(16)) or config['domain_call_cap']!=64:
        raise ValueError('Frozen 16-initial, 64-call admission required')
    try:
        for path,digest in config['input_hashes'].items():
            if sha(path)!=digest:
                raise ValueError('Bound domain source changed: '+path)
        check_simulator_dependencies(config)
        with np.load(config['domain_noise'],allow_pickle=False) as arrays:
            noises=arrays['score'].copy()
        if noises.shape!=(16,2,10,7) or noises.dtype!=np.float32 or not np.isfinite(noises).all():
            raise ValueError('Explicit domain noise shape/dtype mismatch')
        runtime=PolicyRuntime(config,output,64)
        cases=[]
        for initial in initials:
            directory=output/f'initial-{initial:02d}';directory.mkdir()
            with simulator(config_path,config,initial,directory) as request:
                first=request(dict(kind='render',candidate=65))['example']
                request(dict(kind='reset'))
                second=request(dict(kind='render',candidate=65))['example']
                exact_input(first,second)
                saved=dict(external_image=first['image'][0],wrist_image=first['image'][1],
                    robot_state=first['state'],camera_intrinsics=first['camera_intrinsics'],
                    camera_to_world_opencv=first['camera_to_world_opencv'])
                for member in (0,1):
                    identity=dict(initial=initial,member=member)
                    off=runtime.infer(first,noises[initial,member],dict(identity,hook=False),capture=False)
                    on=runtime.infer(second,noises[initial,member],dict(identity,hook=True),capture=True)
                    for key in ('trace','actions'):
                        if not np.array_equal(off[key],on[key]):
                            raise ValueError('Feature hook changed '+key)
                    if any(on[k].shape!=(4608,) or not np.isfinite(on[k]).all() for k in ('external','wrist')):
                        raise ValueError('Policy-native feature shape/value mismatch')
                    for key in ('trace','actions','external','wrist','score'):
                        saved[f'{key}_{member}']=on[key]
                asset=directory/'reference.npz';np.savez_compressed(asset,**saved)
                cases.append(dict(initial=initial,candidate=65,reset_pixels_exact=True,
                    feature_hook_exact=True,reference=str(asset),sha256=sha(asset)))
            write(output/'progress.json',dict(calls=runtime.calls,completed=len(cases)))
        write(output/'status.json',dict(status='COMPLETE',calls=runtime.calls,cases=cases,
            config_sha256=sha(config_path),scope='Unseen-task input and feature-hook admission; no task-success claim'))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),
            calls=0 if runtime is None else runtime.calls))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    run(args.config,args.output,execute=args.execute)

"""24-call native input/capture/return gate, no actions or optimizer updates."""
import argparse
from pathlib import Path
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from .continuous_training import validate_inputs
from .native_search_runtime import NativePolicyRuntime
from .search_live import simulator
from .local_view_runtime import RenderedViewSource
from .search_runtime import exact_prediction
from .continuous_parallel import CollectorBackend
from AlphaBrain.research.dsol.execution.queue import MissingResult


class NativeAdmissionBackend(CollectorBackend):
    def worker_command(self,task):
        return [self.runtime['python'],'-u','-m','benchmarks.LIBERO.native_search_admission',
                '--config',task['payload']['config'],'--output',task['output'],'--execute']

    def validate(self,task):
        path=Path(task['output'])/'status.json'
        if not path.exists():raise MissingResult(str(path))
        value=read(path)
        if (value['status']!='PASS_NATIVE_CAPTURE_RETURN' or value['calls']!=24
                or value['config_sha256']!=sha(task['payload']['config'])):
            raise ValueError('Native admission receipt mismatch')
        for name,digest in value['assets'].items():
            if sha(Path(task['output'])/name)!=digest:raise ValueError('Native evidence changed')
        return dict(status='PASS_NATIVE_CAPTURE_RETURN',receipt_sha256=sha(path))


def factory(plan,root):return NativeAdmissionBackend(plan,root)


def run(config_path,output,*,execute=False):
    cfg=read(config_path)
    if not execute or cfg.get('authorization')!='EXPLICIT_ADMISSION_ONLY':
        raise ValueError('Native admission not authorized')
    validate_inputs(config_path)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);runtime=None
    try:
        sim=read(cfg['sim_config']);runtime=NativePolicyRuntime(sim,root,24)
        rows=read(cfg['training_schedule']);cases=[]
        for initial in (0,8,16,24):
            row=next(r for r in rows if r['initial']==initial)
            case=root/f'initial-{initial:02d}';case.mkdir()
            with np.load(sim['anchors'],allow_pickle=False) as z:noises=z['first_score_noise'][initial,:2].copy()
            with simulator(cfg['sim_config'],sim,initial,case) as request:
                source=RenderedViewSource(request,runtime,noises,case)
                source.move_angles(row['angles_deg']);source.query()
                first=source.last_values
                for m,noise in enumerate(noises):
                    control=runtime.infer(source.last_example,noise,
                        dict(stage='native_uninstrumented_control',initial=initial,member=m),
                        capture=False,instrument=False)
                    if not np.array_equal(control['actions'],first[m]['actions']):
                        raise ValueError('Native instrumentation changed decoded actions')
                    np.savez_compressed(case/f'control-{m}.npz',**control)
                target=[row['angles_deg'][0]+(5 if row['angles_deg'][0]<=25 else -5),row['angles_deg'][1]]
                source.move_angles(target);source.move_angles(row['angles_deg']);source.query()
                for a,b in zip(first,source.last_values):exact_prediction(a,b)
                cases.append(dict(initial=initial,status='PASS_EXACT',queries=2,control_calls=2))
        result=dict(status='PASS_NATIVE_CAPTURE_RETURN',calls=runtime.calls,limit=24,cases=cases,
            manipulation_actions=0,training_updates=0,config_sha256=sha(config_path),
            source_commit=cfg['source_commit'],assets={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')})
        if result['calls']!=24:raise ValueError('Native admission accounting mismatch')
        write(root/'status.json',result);return result
    except BaseException as error:
        write(root/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),calls=runtime.calls if runtime else 0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();run(a.config,a.output,execute=a.execute)

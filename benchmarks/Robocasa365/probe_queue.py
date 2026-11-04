"""Legacy RoboCasa probe-specific admission, command and result validation.

Process ownership is shared with other adapters via execution.process_backend.
Frozen releases retain their original code; this module is compatibility-only.
"""
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from AlphaBrain.research.dsol.execution.process_backend import (
    LocalProcessBackend, process_identity, group_members,
)


class ProbeBackend(LocalProcessBackend):
    def admit(self,slot):
        if slot['host']!=self.runtime['host']:raise ValueError('Wrong host coordinator')
        probe=read(self.runtime['admission'])
        if probe['status']!='COMPLETE' or probe['model']!=self.runtime['model'] or probe['completed_cells']!=2:
            raise ValueError('Real five-window admission is missing')
        return self.gpu_available(slot)

    def worker_command(self,task):
        args=[self.runtime['python'],'-m','benchmarks.Robocasa365.short_view_probe',
            '--config',self.runtime['config'],'--model',self.runtime['model'],'--output',task['output'],
            '--initials',task['payload']['initial'],'--views',*task['payload']['views'],'--execute']
        if self.runtime['model']=='xr1':args += ['--model-python',self.runtime['model_python'],'--upstream',self.runtime['upstream']]
        return args

    def validate(self,task,*,allow_partial=False):
        from .short_view_probe import prefix_means
        root=Path(task['output']);path=root/'status.json'
        if not path.exists() or (not allow_partial and read(path)['status']!='COMPLETE'):raise MissingResult(str(root))
        status=read(path);identity=read(root/'identity.json');views=task['payload']['views'];initial=task['payload']['initial']
        if (identity['initials']!=[initial] or identity['views']!=views or identity['model']!=self.runtime['model']
                or identity['source_sha256']!=sha(Path(self.runtime['source'])/'benchmarks/Robocasa365/short_view_probe.py')
                or identity['config_sha256']!=sha(self.runtime['config'])
                or (not allow_partial and (status['completed_cells']!=len(views) or status['completed_probes']!=len(views)*8))):
            raise ValueError('Completed probe identity/coverage mismatch')
        present={p.parent.name for p in (root/'cells'/initial).glob('*/complete.json')}
        if not present <= set(views) or (not allow_partial and present!=set(views)):
            raise ValueError('Unexpected or missing committed probe cell')
        if status['status']=='COMPLETE' and (present!=set(views) or status['completed_cells']!=len(views)):
            raise ValueError('Incomplete probe declared complete')
        receipts={}
        for view in views:
            if view not in present:continue
            cell=root/'cells'/initial/view;done=read(cell/'complete.json')
            if not done['window1_exact'] or done['arrays_sha256']!=sha(cell/'probes.npz'):raise ValueError('Probe receipt mismatch')
            with np.load(cell/'probes.npz',allow_pickle=False) as z:
                if not np.array_equal(prefix_means(z['scores']),z['prefix_means']):raise ValueError('Prefix arithmetic mismatch')
                if identity.get('population_mode')=='fresh_confirmation':
                    anchor=read(cell/'initial-anchor.json')
                    if (anchor['initial']!=initial or anchor['view']!=view or anchor['model']!=identity['model']
                            or anchor['action_steps']!=0 or anchor['score_repeats']!=8
                            or anchor['arrays_sha256']!=sha(cell/'initial-anchor.npz')):
                        raise ValueError('Independent initial-only anchor identity mismatch')
                    with np.load(cell/'initial-anchor.npz',allow_pickle=False) as a:
                        if (not np.array_equal(a['scores'],z['scores'][:,0])
                                or not np.array_equal(a['velocity_trace'],z['velocity_trace'][:,0])):
                            raise ValueError('Window1 differs from independent initial anchor')
            receipts[view]=sha(cell/'complete.json')
        return dict(status='PASS' if present==set(views) else 'PASS_PARTIAL',identity_sha256=sha(root/'identity.json'),cells=receipts)


def factory(plan, root):
    return ProbeBackend(plan, root)

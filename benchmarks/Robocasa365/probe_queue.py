"""Host-local short-probe adapter for the standard durable execution queue.

Run one coordinator on each host. No SSH side effects or GPU work on import.
Lost launch receipts fail closed and require manual process audit.
"""
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,sha
from AlphaBrain.research.dsol.execution.queue import MissingResult,_durable_write


def process_identity(pid):
    try:
        fields=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:return None
    return dict(pid=int(pid),start=fields[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                state=fields[0],group=int(fields[2]))


def group_members(group):
    members=[]
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=process_identity(int(path.name))
            if row is not None and row['group']==group and row['state']!='Z':members.append(row)
    return members


class ProbeBackend:
    def __init__(self,plan,root):
        self.plan,self.root=plan,Path(root);self.processes={};self.logs={}
        self.runtime=plan['runtime'];self.handles=self.root/'handles';self.handles.mkdir(exist_ok=True)

    def admit(self,slot):
        if slot['host']!=self.runtime['host']:raise ValueError('Wrong host coordinator')
        probe=read(self.runtime['admission'])
        if probe['status']!='COMPLETE' or probe['model']!=self.runtime['model'] or probe['completed_cells']!=2:
            raise ValueError('Real five-window admission is missing')
        return self.gpu_available(slot)

    def gpu_available(self,slot):
        """Fresh supervisor + /proc + GPU telemetry, including real memory use."""
        from AlphaBrain.research.dsol.execution.gpu_guard import preflight
        config=self.runtime.get('gpu_guard')
        if not isinstance(config,dict):
            raise RuntimeError('Explicit supervisor-backed gpu_guard configuration required')
        if config['profile']!=slot['host'] or slot['host']!=self.runtime['host']:
            raise RuntimeError('GPU admission profile differs from queue host')
        config=dict(config)
        if self.runtime.get('workers_per_gpu',1)>1:
            if self.runtime['workers_per_gpu']!=2:
                raise ValueError('Only admitted dual-worker capacity supported')
            config['minimum_free_mib']=max(13500,config['minimum_free_mib'])
        owned=[]
        state=self.root/'state.json'
        if state.exists():
            for row in read(state)['tasks'].values():
                if row['status']=='RUNNING':
                    handle=row['handle'];device=str(row['slot'].get('gpu_id',row['slot']['id']))
                    devices={str(g['index']):g['uuid'] for g in config['devices']}
                    owned.append(dict(pid=handle['pid'],start=handle['start'],boot=handle['boot'],
                        group=handle['group'],gpu_uuid=devices[device]))
        result=preflight(config,owned=owned)
        self.last_gpu_admission=result
        if result['status']=='MONITOR_ERROR':
            raise RuntimeError('GPU monitoring failed closed: '+result['error'])
        device=str(slot.get('gpu_id',slot['id']))
        rows=[r for r in result['rows'] if str(r['index'])==device]
        if len(rows)!=1:
            raise RuntimeError('Requested GPU missing from monitored inventory')
        return rows[0]['allowed']

    def worker_command(self,task):
        args=[self.runtime['python'],'-m','benchmarks.Robocasa365.short_view_probe',
            '--config',self.runtime['config'],'--model',self.runtime['model'],'--output',task['output'],
            '--initials',task['payload']['initial'],'--views',*task['payload']['views'],'--execute']
        if self.runtime['model']=='xr1':args += ['--model-python',self.runtime['model_python'],'--upstream',self.runtime['upstream']]
        return args

    def start(self,task,slot,attempt):
        receipt=self.handles/f'{task["key"]}-{attempt}.json'
        if receipt.exists() or Path(task['output']).exists():raise ValueError('Existing work requires explicit recovery')
        args=self.worker_command(task)
        try:
            if not self.gpu_available(slot):
                raise RuntimeError('GPU ownership or memory changed before launch')
        except Exception as error:
            # Durable proof that Popen was never reached. Queue cleanup must not
            # mistake an intentionally refused launch for an unresolved gap.
            _durable_write(receipt,dict(key=task['key'],output=task['output'],
                not_started=True,reason=str(error)))
            raise
        env={**os.environ,**self.runtime['environment'],'CUDA_VISIBLE_DEVICES':slot.get('gpu_id',slot['id'])}
        log=(self.root/f'{task["key"]}-{attempt}.log').open('xb')
        process=subprocess.Popen(args,env=env,cwd=self.runtime['source'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_identity(process.pid)
        if identity is None:raise RuntimeError('Worker exited before durable identity; inspect launch log')
        handle=dict(**identity,output=task['output'],command=args,receipt=str(receipt),key=task['key'])
        _durable_write(receipt,handle)
        self.processes[process.pid]=process;self.logs[process.pid]=log
        return handle

    def locate(self,task,slot,attempt):
        receipt=self.handles/f'{task["key"]}-{attempt}.json'
        if not receipt.exists():raise RuntimeError('Unresolved launch gap: manual process audit required, never auto-reissue')
        handle=read(receipt)
        if handle['key']!=task['key'] or handle['output']!=task['output']:raise ValueError('Launch receipt mismatch')
        if handle.get('not_started') is True:return None
        return handle

    def alive(self,handle):
        current=process_identity(handle['pid'])
        if current is not None and any(current[k]!=handle[k] for k in ('start','boot','group')):
            raise RuntimeError('PID identity changed; no signal or recovery allowed')
        return bool(group_members(handle['group']))

    def poll(self,handle):
        process=self.processes.get(handle['pid'])
        if process is not None:
            code=process.poll()
            if code is not None:self.logs[handle['pid']].close()
            return code
        if self.alive(handle):return None
        path=Path(handle['output'])/'status.json'
        return 0 if path.exists() and read(path)['status']=='COMPLETE' else 1

    def cancel(self,handle):
        def reap():
            process=self.processes.get(handle['pid'])
            if process is not None:process.wait(timeout=10)
            log=self.logs.get(handle['pid'])
            if log is not None:log.close()
        if not self.alive(handle):
            reap();return
        current=process_identity(handle['pid'])
        if current is None:raise RuntimeError('Orphan process group: manual audit required')
        os.killpg(handle['group'],signal.SIGTERM)
        for _ in range(30):
            if not self.alive(handle):
                reap();return
            time.sleep(.1)
        os.killpg(handle['group'],signal.SIGKILL)
        reap()

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


def factory(plan,root):return ProbeBackend(plan,root)

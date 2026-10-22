"""Queue-owned resident VLA collectors with synchronous version barriers.

The standard queue owns processes and failure cleanup. Immutable mailboxes carry
bounded batch requests; only the central training driver updates the policy.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest,task_key,verify_artifacts
from AlphaBrain.research.dsol.execution.queue import prepare,run,MissingResult,_durable_write
from AlphaBrain.research.dsol.execution.resident_mailbox import ResidentMailbox
from benchmarks.Robocasa365.probe_queue import process_identity
from .continuous_parallel import CollectorBackend,worker


class ResidentBackend(CollectorBackend):
    def worker_command(self,task):
        return [self.runtime['python'],'-u','-m','benchmarks.LIBERO.continuous_resident',
                '--binding',task['payload']['binding'],'--output',task['output'],'--execute']

    def validate(self,task):
        path=Path(task['output'])/'status.json'
        if not path.exists():raise MissingResult(str(path))
        value=read(path);binding=read(task['payload']['binding'])
        if (value['status']!='COMPLETE' or value['binding_sha256']!=file_digest(task['payload']['binding'])
                or read(Path(binding['mailbox'])/'stop.json')['completed_versions']!=value['completed_versions']):
            raise ValueError('Resident completion mismatch')
        mailbox=ResidentMailbox(binding['mailbox'])
        for v in range(value['completed_versions']):
            if mailbox.result(v) is None:raise ValueError('Missing resident version')
        return dict(status='PASS_RESIDENT_STOPPED',receipt_sha256=file_digest(path))


def serve(binding_path,output,*,execute=False):
    binding=read(binding_path)
    if not execute or binding['authorization']!='EXPLICIT_RESIDENT_COLLECTION_ONLY':
        raise ValueError('Resident collection not authorized')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    mailbox=ResidentMailbox(binding['mailbox']);shared={};version=0
    parent=binding['coordinator']
    def alive():
        now=process_identity(parent['pid'])
        return now is not None and all(now[k]==parent[k] for k in ('start','boot','group')) and now['state']!='Z'
    try:
        while True:
            request=mailbox.next(version,timeout=binding['request_timeout_seconds'],alive=alive)
            if request is None:break
            req=read(request['request'])
            if (req['config']!=binding['config'] or file_digest(req['config'])!=binding['config_sha256']
                    or req['version']!=version or req['owner']!=binding['owner']):
                raise ValueError('Resident config/owner/policy version mismatch')
            worker(request['request'],req['resident_output'],execute=True,shared_runtime=shared)
            mailbox.complete(version,request['sha256'],Path(req['resident_output'])/'status.json')
            version+=1
        _durable_write(root/'status.json',dict(status='COMPLETE',completed_versions=version,
            binding_sha256=file_digest(binding_path)))
    except BaseException as error:
        _durable_write(root/'failure.json',dict(status='FAILED_STOPPED',error=repr(error)))
        raise


class ResidentPool:
    """One local pool per run. Interrupted/in-flight pools are never auto-resumed."""
    def __init__(self,config,runtime,root,control,*,workers=4):
        if workers not in (1,4,8):raise ValueError('Unsupported resident worker count')
        self.root,self.control=Path(root),Path(control)
        self.root.mkdir(parents=True,exist_ok=False)
        self.runtime=runtime;self.workers=workers;self.version=0
        self.stop_event=threading.Event();self.error=None;self.thread=None
        self.mailboxes=[];tasks=[];artifacts=[dict(path=config,sha256=file_digest(config))]
        for i in range(workers):
            owner=f'resident-collector-{i}';directory=self.root/f'mailbox-{i}';directory.mkdir()
            self.mailboxes.append(ResidentMailbox(directory))
            binding=dict(authorization='EXPLICIT_RESIDENT_COLLECTION_ONLY',owner=owner,
                config=config,config_sha256=file_digest(config),mailbox=str(directory),
                coordinator=process_identity(os.getpid()),request_timeout_seconds=runtime.get('batch_timeout_seconds',7200))
            path=self.root/f'binding-{i}.json';_durable_write(path,binding)
            artifacts.append(dict(path=str(path),sha256=file_digest(path)))
            address=[dict(model='continuous-frozen-policy',task='resident-collection',initial='fixed-manifest',
                view='continuous',stream='score',repeat=i,noise='bound-per-request')]
            tasks.append(dict(key=task_key(self.root.name,file_digest(config),address),addresses=address,
                output=str(self.root/f'worker-{i}'),payload=dict(binding=str(path))))
        self.plan=dict(format='dsol_work_queue_v1',experiment_id=self.root.name,
            protocol_sha256=file_digest(config),slots=runtime['slots'][:workers],tasks=tasks,
            artifacts=artifacts+runtime['source_artifacts'],
            backend=dict(module='benchmarks.LIBERO.continuous_resident',factory='factory'),
            runtime={**runtime,'mode':'continuous_parallel_collection','workers_per_gpu':1})
        if len(self.plan['slots'])!=workers or len({s['gpu_id'] for s in self.plan['slots']})!=workers:
            raise ValueError('Distinct admitted GPUs required')
        prepare(self.plan,self.control)

    def start(self):
        def sleep(seconds):
            if self.stop_event.wait(seconds):raise RuntimeError('Resident pool stopped by central learner')
        def target():
            try:run(self.control,ResidentBackend(self.plan,self.control),poll_seconds=.2,
                    timeout_seconds=self.runtime.get('run_timeout_seconds',86400),sleep=sleep)
            except BaseException as error:self.error=error
        self.thread=threading.Thread(target=target,name='resident-queue',daemon=False);self.thread.start()

    def __enter__(self):return self

    def __exit__(self,kind,value,tb):
        self.close(failed=kind is not None)
        return False

    def collect(self,requests):
        if len(requests)!=self.workers:raise ValueError('Resident quota mismatch')
        for box,path in zip(self.mailboxes,requests):box.publish(self.version,path)
        if self.thread is None:self.start()
        deadline=time.monotonic()+self.runtime.get('batch_timeout_seconds',7200)
        while time.monotonic()<deadline:
            if self.error is not None:raise RuntimeError('Resident queue failed') from self.error
            results=[box.result(self.version) for box in self.mailboxes]
            if all(results):
                self.version+=1
                return results
            time.sleep(.1)
        raise TimeoutError('Resident batch deadline exceeded')

    def close(self,*,failed=False):
        if self.thread is None:return
        try:
            if failed:self.stop_event.set()
            else:
                for box in self.mailboxes:box.close(self.version)
        finally:
            self.thread.join(timeout=60)
        if self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=60)
            if self.thread.is_alive():raise RuntimeError('Resident cleanup unresolved; do not restart')
        if self.error is not None and not failed:raise RuntimeError('Resident queue failed') from self.error


def factory(plan,root):return ResidentBackend(plan,root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();serve(args.binding,args.output,execute=args.execute)

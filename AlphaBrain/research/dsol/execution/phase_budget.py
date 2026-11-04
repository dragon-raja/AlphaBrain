"""Single-center cumulative phase reservations above existing task ledgers.

Unknown phases retain their full reservation. No refund, retry or reconstruction
from the shared mirror is automatic; the run's outer owner lock is mandatory.
"""
from pathlib import Path
from threading import RLock
import time
from .authority_wire import require_local
from .manifest import file_digest
from .queue import _durable_write,exclusive
from ..data.artifacts import read


class PhaseBudget:
    def __init__(self,root,mirror,authorization,*,authorization_sha256,allocation=None):
        self.root=Path(root);self.mirror=Path(mirror);self.authorization=Path(authorization)
        self.root.mkdir(parents=True,exist_ok=True);require_local(self.root)
        self.mirror.mkdir(parents=True,exist_ok=True)
        if file_digest(authorization)!=authorization_sha256:raise ValueError('Changed authorization')
        a=read(authorization)
        if a['status']!='EXPLICIT_USER_APPROVAL' or type(a['total_model_call_limit']) is not int or a['total_model_call_limit']<=0:
            raise ValueError('Explicit cumulative call ceiling required')
        self.limit=a['total_model_call_limit'];self.auth_hash=authorization_sha256;self.lock=RLock()
        if allocation is not None:
            if file_digest(allocation['path'])!=allocation['sha256']:raise ValueError('Changed parent allocation')
            parent=read(allocation['path'])
            if (parent['status']!='RESERVED_BEFORE_DISPATCH' or parent['authorization_sha256']!=self.auth_hash
                    or not 0<parent['reserved_model_calls']<=self.limit
                    or file_digest(parent['campaign'])!=parent['campaign_sha256']):raise ValueError('Unreserved or stale delegated budget')
            self.limit=parent['reserved_model_calls']

    def reserve(self,key,campaign):
        if Path(key).name!=key or key in ('','.','..'):raise ValueError('Single phase key required')
        with self.lock,exclusive(self.root):
            if file_digest(self.authorization)!=self.auth_hash:raise ValueError('Changed authorization')
            target=self.root/(key+'.json')
            if target.exists():raise ValueError('Existing phase requires audit, never replay')
            c=read(campaign);cap=sum(c['limits'][k] for k in ('score','paired_execution','nominal_execution'))
            if type(cap) is not int or cap<=0:raise ValueError('Positive integer reservation required')
            if self.status()['reserved']+cap>self.limit:raise ValueError('Cumulative approved hard cap exceeded')
            row=dict(key=key,status='RESERVED_BEFORE_DISPATCH',campaign=str(campaign),campaign_sha256=file_digest(campaign),reserved_model_calls=cap,
                authorization_sha256=self.auth_hash,automatic_retry=False,unknown_refund=False,time_unix=time.time())
            _durable_write(target,row);_durable_write(self.mirror/target.name,row)
            return cap

    def settle(self,key,result):
        with self.lock,exclusive(self.root):
            target=self.root/(key+'.json');row=read(target);calls=result['actual_model_calls']
            if (row['status']!='RESERVED_BEFORE_DISPATCH' or result['status']!='COMPLETE'
                    or type(calls) is not int or not 0<=calls<=row['reserved_model_calls']
                    or file_digest(row['campaign'])!=row['campaign_sha256']):raise ValueError('Phase settlement mismatch')
            row.update(status='COMPLETE',actual_model_calls=calls,settlement=result,settled_unix=time.time())
            _durable_write(target,row);_durable_write(self.mirror/target.name,row)

    def status(self):
        with self.lock:
            rows=[read(p) for p in self.root.glob('*.json')]
            return dict(hard_limit=self.limit,reserved=sum(r['reserved_model_calls'] for r in rows),
                known_consumed=sum(r.get('actual_model_calls',0) for r in rows),
                held_unsettled=sum(r['reserved_model_calls'] for r in rows if r['status']!='COMPLETE'),phases=len(rows))

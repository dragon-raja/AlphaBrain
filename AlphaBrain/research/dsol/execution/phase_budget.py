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

    def delegate_settled_surplus(self,key,campaign,phase_keys):
        """Explicitly allocate unused *settled* reservations to one recovery owner.

        Old owners still count their original reservations, so they cannot reuse
        this credit. Unknown phases are ineligible. Each settled phase can back
        only one allocation; a failed child keeps its entire allocation. This
        works while an immutable older owner is running without changing its
        budget semantics or refunding an interrupted task.
        """
        if (Path(key).name!=key or key in ('','.','..') or not phase_keys
                or len(phase_keys)!=len(set(phase_keys))):
            raise ValueError('Explicit unique surplus allocation required')
        folder=self.root/'surplus-allocations';folder.mkdir(exist_ok=True)
        # COMPLETE rows are immutable. Use a separate allocation lock so an
        # older live parent's nonblocking reservation lock is never obstructed.
        with self.lock,exclusive(folder):
            if file_digest(self.authorization)!=self.auth_hash:raise ValueError('Changed authorization')
            target=folder/(key+'.json')
            if target.exists():raise ValueError('Allocation already exists; never replay')
            used={p['key'] for f in folder.glob('*.json') for p in read(f)['settled_phases']}
            c=read(campaign);cap=sum(c['limits'][k] for k in ('score','paired_execution','nominal_execution'))
            if type(cap) is not int or cap<=0:raise ValueError('Positive integer allocation required')
            phases=[];surplus=0
            for name in phase_keys:
                if Path(name).name!=name or name in ('','.','..') or name in used:
                    raise ValueError('Invalid or already delegated phase')
                path=self.root/(name+'.json');row=read(path);settlement=row.get('settlement',{})
                calls=row.get('actual_model_calls');reserved=row['reserved_model_calls']
                if (row['status']!='COMPLETE' or row['authorization_sha256']!=self.auth_hash
                        or settlement.get('status')!='COMPLETE' or type(calls) is not int
                        or settlement.get('actual_model_calls')!=calls or not 0<=calls<=reserved
                        or file_digest(row['campaign'])!=row['campaign_sha256']):
                    raise ValueError('Only unchanged complete settlements can fund recovery')
                original=read(row['campaign'])
                if sum(original['limits'][k] for k in ('score','paired_execution','nominal_execution'))!=reserved:
                    raise ValueError('Settlement reservation differs from its campaign')
                surplus+=reserved-calls
                phases.append(dict(key=name,path=str(path),sha256=file_digest(path),
                    reserved_model_calls=reserved,actual_model_calls=calls,unused=reserved-calls))
            if surplus<cap:raise ValueError('Insufficient verified settled surplus')
            row=dict(key=key,status='RESERVED_BEFORE_DISPATCH',campaign=str(campaign),
                campaign_sha256=file_digest(campaign),reserved_model_calls=cap,
                authorization_sha256=self.auth_hash,settled_phases=phases,
                settled_unused_credit=surplus,parent_hard_limit=self.limit,
                combined_maximum_consumption=self.limit-surplus+cap,
                accounting='Explicit settled surplus; parent reservations stay unchanged',
                automatic_retry=False,unknown_refund=False,time_unix=time.time())
            _durable_write(target,row)
            (self.mirror/'surplus-allocations').mkdir(exist_ok=True)
            _durable_write(self.mirror/'surplus-allocations'/target.name,row)
            return dict(path=str(target),sha256=file_digest(target))

    def status(self):
        with self.lock:
            rows=[read(p) for p in self.root.glob('*.json')]
            return dict(hard_limit=self.limit,reserved=sum(r['reserved_model_calls'] for r in rows),
                known_consumed=sum(r.get('actual_model_calls',0) for r in rows),
                held_unsettled=sum(r['reserved_model_calls'] for r in rows if r['status']!='COMPLETE'),phases=len(rows))

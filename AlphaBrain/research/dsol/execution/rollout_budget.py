"""Shared, fail-closed reservations for bounded parallel model calls.

One atomic state under flock; a reserved attempt is charged even if inference
fails. No worker can grant itself a fresh experiment-sized allowance.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from .queue import _durable_write


class RolloutBudget:
    def __init__(self, root):
        self.root=Path(root)

    @classmethod
    def create(cls, root, limit):
        if type(limit) is not int or limit<1:raise ValueError('Positive global limit required')
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        _durable_write(root/'state.json',dict(limit=limit,spent=0,reserved=0,failed=None,units={}))
        return cls(root)

    @contextmanager
    def locked(self):
        with (self.root/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=json.loads((self.root/'state.json').read_text())
            units=list(state['units'].values())
            if (any(u['status'] not in ('RESERVED','ACTIVE','COMPLETE') or not 0<=u['calls']<=u['cap'] for u in units)
                    or state['spent']!=sum(u['calls'] for u in units)
                    or state['reserved']!=sum(u['cap']-u['calls'] for u in units if u['status']!='COMPLETE')
                    or not 0<=state['spent']+state['reserved']<=state['limit']):
                raise ValueError('Corrupt global budget counters')
            yield state

    def save(self, state):_durable_write(self.root/'state.json',state)

    def inspect(self):
        with self.locked() as s:return s

    def reserve(self, units):
        with self.locked() as s:
            if s['failed']:raise ValueError('Failed budget; no automatic retry')
            if len({u['key'] for u in units})!=len(units):raise ValueError('Duplicate unit reservation')
            if any(u['key'] in s['units'] or u['cap']<1 for u in units):raise ValueError('Existing unit or invalid cap')
            amount=sum(u['cap'] for u in units)
            if s['spent']+s['reserved']+amount>s['limit']:raise ValueError('Global reservation exceeds budget')
            for u in units:s['units'][u['key']]=dict(u,status='RESERVED',calls=0)
            s['reserved']+=amount;self.save(s)

    @staticmethod
    def unit(s,key,owner,version):
        u=s['units'][key]
        if u['owner']!=owner or u['version']!=version:raise ValueError('Worker owner or policy version mismatch')
        return u

    def claim(self,key,owner,version):
        with self.locked() as s:
            if s['failed']:raise ValueError('Failed budget; no automatic retry')
            u=self.unit(s,key,owner,version)
            if u['status']!='RESERVED':raise ValueError('Unit already claimed; PID audit/recovery required')
            u['status']='ACTIVE';self.save(s)

    def attempt(self,key,owner,version,query,member):
        with self.locked() as s:
            if s['failed']:raise ValueError('Failed budget; no further model dispatch')
            u=self.unit(s,key,owner,version)
            if u['status']!='ACTIVE' or u['calls']>=u['cap']:raise ValueError('Unit inactive or exhausted')
            if (query,member)!=(u['calls']//2+1,u['calls']%2):raise ValueError('Duplicate or missing query/member')
            if s['spent']>=s['limit']:raise ValueError('Global model budget exhausted')
            u['calls']+=1;s['spent']+=1;s['reserved']-=1;self.save(s)
            return s['spent']

    def complete(self,key,owner,version,calls,receipt_sha256):
        with self.locked() as s:
            u=self.unit(s,key,owner,version)
            if u['status']!='ACTIVE' or calls!=u['calls'] or calls<2 or calls%2:
                raise ValueError('Incomplete/mismatched unit ledger')
            s['reserved']-=u['cap']-u['calls'];u.update(status='COMPLETE',receipt_sha256=receipt_sha256)
            self.save(s)

    def fail(self,reason):
        with self.locked() as s:
            if not s['failed']:s['failed']=str(reason)
            self.save(s)

    def recoverable(self, keys):
        """Read-only explicit-recovery gate. Never reissue an incomplete unit."""
        with self.locked() as s:
            if s['failed'] or any(s['units'][k]['status']!='COMPLETE' for k in keys):
                raise ValueError('Incomplete/failed calls require investigation; no automatic replay')
            return {k:s['units'][k] for k in keys}

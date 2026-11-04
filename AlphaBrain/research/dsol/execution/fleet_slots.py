"""Fair pre-dispatch resource leases for independent bounded phase owners.

Actual GPU/process ownership remains in the controlled queue and worker guards.
A failed phase quarantines its cards until explicit stopped-worker reconciliation.
"""
from contextlib import contextmanager
from threading import Condition
import time


class FleetSlots:
    def __init__(self,capacity,*,initial_capacity=None):
        if not capacity or any(not ids or len(set(ids))!=len(ids) for ids in capacity.values()):raise ValueError('Explicit disjoint physical cards required')
        self.total={h:set(ids) for h,ids in capacity.items()}
        self.available={h:set(ids) if initial_capacity is None else set(initial_capacity.get(h,[])) for h,ids in capacity.items()}
        if any(not ids.issubset(self.total[h]) for h,ids in self.available.items()):raise ValueError('Initial cards outside planned capacity')
        self.activated={h:set(ids) for h,ids in self.available.items()}
        self.condition=Condition();self.pending=[];self.held={}

    @contextmanager
    def lease(self,key,hosts,*,count=4,max_count=None,timeout=86400):
        if type(count) is not int or count<1 or not set(hosts).issubset(self.available):raise ValueError('Explicit admitted host/card request required')
        max_count=count if max_count is None else max_count
        if type(max_count) is not int or max_count<count:raise ValueError('Invalid maximum capacity')
        ticket=object();entry=(ticket,list(hosts),count);deadline=time.monotonic()+timeout
        with self.condition:
            self.pending.append(entry)
            try:
                while True:
                    candidates=[h for h in hosts if len(self.available[h])>=count]
                    first=next((v[0] for v in self.pending if any(len(self.available[h])>=v[2] for h in v[1])),None)
                    permanently_lost={h:sum(len(v['gpu_ids']) for v in self.held.values() if v['host']==h) for h in hosts}
                    if all(len(self.total[h])-permanently_lost[h]<count for h in hosts):raise RuntimeError('Admitted capacity quarantined')
                    if first is ticket and candidates:break
                    left=deadline-time.monotonic()
                    if left<=0:raise TimeoutError('No admitted capacity before dispatch deadline')
                    self.condition.wait(min(1,left))
                host=max(candidates,key=lambda h:(len(self.available[h]),-list(hosts).index(h)));ids=sorted(self.available[host])[:max_count]
                self.available[host].difference_update(ids);self.pending.remove(entry);self.condition.notify_all()
            except BaseException:
                self.pending.remove(entry);self.condition.notify_all();raise
        try:yield host,ids
        except BaseException:
            with self.condition:self.held[key]=dict(host=host,gpu_ids=ids);self.condition.notify_all()
            raise
        else:
            with self.condition:self.available[host].update(ids);self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(available={h:sorted(ids) for h,ids in self.available.items()},held={k:dict(v) for k,v in self.held.items()})

    def grant(self,host,ids,*,evidence_sha256):
        """Activate only preplanned, never previously granted cards after audit."""
        with self.condition:
            ids=set(ids)
            if (host not in self.total or not ids or not ids.issubset(self.total[host])
                    or ids & self.activated[host] or not isinstance(evidence_sha256,str) or len(evidence_sha256)!=64):
                raise ValueError('Explicit disjoint audited resource grant required')
            self.available[host].update(ids);self.activated[host].update(ids);self.condition.notify_all()

"""Latched node health and fresh local-center admission, not a GPU supervisor."""
import json
import math
from pathlib import Path
import time


def trusted_keeper_snapshot(config, *, collector=None, verifier=None, sleep=None):
    """Retry only recognized sampling races, not identity or network failures.

    Three fresh observations at most; never reuse a cached PASS or clear HOLD.
    Kept self-contained so the read-only SSH monitor can transmit this function
    without replacing the admitted model runtime on a compute node.
    """
    import time
    from pathlib import Path
    from AlphaBrain.research.dsol.execution.gpu_guard import (
        collect, verify_status, TransientMonitorError, MonitorError)
    collector=collect if collector is None else collector
    verifier=verify_status if verifier is None else verifier
    sleep=time.sleep if sleep is None else sleep
    for attempt in range(1,4):
        try:
            snapshot=collector(config['gpu_guard'],target_gpu_uuid=config['keeper_gpu_uuid'])
            keeper=verifier(config['gpu_guard'],snapshot)
            return dict(status='PASS',hostname=snapshot['hostname'],boot_id=snapshot['boot_id'],
                devices=[g['uuid'] for g in snapshot['gpus']],keeper=keeper,
                shared_root_accessible=Path(config['shared_root']).is_dir(),
                owned_root_accessible=Path(config['owned_root']).is_dir(),
                supervisor_state=keeper['supervisor_state'],attempts=attempt)
        except (TransientMonitorError,FileNotFoundError,ProcessLookupError):
            if attempt==3:return dict(status='FAILED',error='TRUSTED_SNAPSHOT_UNSTABLE',attempts=attempt)
            sleep(1.)
        except MonitorError:
            return dict(status='FAILED',error='TRUSTED_IDENTITY_FAILED',attempts=attempt)


class NodeHealth:
    def __init__(self,expected,observer_id):
        if not isinstance(observer_id,str) or not observer_id:raise ValueError('Explicit monitor generation required')
        self.expected,self.observer_id=expected,observer_id
        self.held=False;self.sequence=0

    def observe(self,snapshot,*,now=None):
        now=time.monotonic() if now is None else now;self.sequence+=1
        reason=None
        if snapshot.get('status')!='PASS':reason='node_probe_failed'
        elif any(snapshot.get(k)!=self.expected[k] for k in ('hostname','boot_id','devices')):reason='node_generation_changed'
        elif any(snapshot.get('keeper',{}).get(k)!=v for k,v in self.expected['keeper'].items()):reason='keeper_identity_changed'
        elif snapshot.get('owned_root_accessible') is not True or snapshot.get('shared_root_accessible') is not True:reason='owned_or_shared_storage_unavailable'
        if reason:self.held=True
        return dict(schema='CENTER_NODE_HEALTH_V1',observer_id=self.observer_id,sequence=self.sequence,
            checked_monotonic=now,checked_unix=time.time(),status='HOLD' if self.held else 'HEALTHY',
            reason=reason or ('explicit_recovery_required' if self.held else None),snapshot=snapshot)


def require_fresh_health(config,*,now=None):
    """Read the center-local heartbeat, never the asynchronously mirrored NFS copy.

    It augments the existing per-forward live peer handshake. Operators must
    configure a local filesystem path; this cooperative check is not a sandbox.
    """
    if set(config)!={'path','observer_id','max_age_seconds'}:raise ValueError('Explicit local monitor binding required')
    limit=config['max_age_seconds']
    if type(limit) not in (int,float) or not 1<=limit<=120:raise ValueError('Bounded monitor freshness required')
    state=json.loads(Path(config['path']).read_text());now=time.monotonic() if now is None else now
    age=now-state['checked_monotonic']
    if (state.get('schema')!='CENTER_NODE_HEALTH_V1' or state.get('observer_id')!=config['observer_id']
            or state.get('status')!='HEALTHY' or not math.isfinite(age) or not 0<=age<=limit):
        raise ConnectionError('Node health unavailable, stale, changed or held; no dispatch/continuation')
    return state

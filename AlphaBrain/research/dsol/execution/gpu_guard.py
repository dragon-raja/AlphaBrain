"""Read-only GPU ownership admission against a trusted live supervisor.

No signals, keepalive controls, CUDA imports, or worker launches. Release trust
comes from explicit deployment configuration, never from status.json itself.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time


class MonitorError(RuntimeError):
    pass


class TransientMonitorError(MonitorError):
    pass


def sha_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def process(pid, *, proc_root=Path('/proc')):
    """Read the actual executable; do not resolve a release's mutable current link."""
    pid=int(pid)
    if pid<=0:
        raise MonitorError('Invalid process PID')
    path=proc_root/str(pid)
    def stat():
        fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),
                    sid=int(fields[3]),start_ticks=int(fields[19]))
    try:
        before=stat()
        executable=os.readlink(path/'exe')
        argv=(path/'cmdline').read_bytes().split(b'\0')
        argv=[s.decode('utf-8') for s in argv if s]
        value=dict(before,executable=executable,executable_sha256=sha_file(path/'exe'),argv=argv)
        after=stat()
        if any(before[k]!=after[k] for k in ('pid','ppid','pgid','sid','start_ticks')):
            raise TransientMonitorError('Process changed while reading identity')
        if after['state'] in ('Z','X'):
            raise TransientMonitorError('Process is no longer running')
        return value
    except (FileNotFoundError,ProcessLookupError) as error:
        raise TransientMonitorError('Process disappeared while reading identity') from error


def holds_supervisor_lock(pid, lock_path, *, proc_root=Path('/proc')):
    """Inspect an existing FLOCK; never acquire or create the supervisor lock."""
    identity=Path(lock_path).stat()
    expected=(os.major(identity.st_dev),os.minor(identity.st_dev),identity.st_ino)
    for fd in (proc_root/str(pid)/'fdinfo').iterdir():
        for line in fd.read_text().splitlines():
            if not line.startswith('lock:'):
                continue
            fields=line.split()
            # lock: <number>: FLOCK ADVISORY WRITE <pid> <dev>:<inode> 0 EOF
            if len(fields)!=9 or fields[2:5]!=['FLOCK','ADVISORY','WRITE'] or fields[5]!=str(pid):
                continue
            major,minor,inode=fields[6].split(':')
            if (int(major,16),int(minor,16),int(inode))==expected:
                return True
    return False


def telemetry():
    def query(*args):
        result=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=5,check=True)
        return [[v.strip() for v in line.split(',')] for line in result.stdout.splitlines() if line.strip()]
    gpus=[]
    for index,uuid,free,used,total in query('--query-gpu=index,uuid,memory.free,memory.used,memory.total'):
        row=dict(index=int(index),uuid=uuid,free_mib=int(free),used_mib=int(used),total_mib=int(total))
        if not uuid.startswith('GPU-') or min(row[k] for k in ('free_mib','used_mib','total_mib'))<0:
            raise MonitorError('Invalid GPU telemetry')
        gpus.append(row)
    if not gpus or len({r['uuid'] for r in gpus})!=len(gpus):
        raise MonitorError('Missing or duplicate GPU telemetry')
    apps=[dict(gpu_uuid=uuid,pid=int(pid),used_mib=int(used))
          for uuid,pid,used in query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]
    if any(a['gpu_uuid'] not in {g['uuid'] for g in gpus} or a['pid']<=0 or a['used_mib']<0 for a in apps):
        raise MonitorError('Process telemetry is outside GPU inventory')
    return gpus,apps


def verify_status(config, snapshot):
    """Pure validation of independent status, /proc and NVML observations."""
    s=snapshot['status'];sup=snapshot['supervisor'];worker=snapshot['worker']
    if snapshot['hostname']!=config['hostname'] or s['profile']!=config['profile'] or snapshot['profile_file']!=config['profile']:
        raise MonitorError('Host/profile mismatch')
    if s['boot_id']!=snapshot['boot_id']:
        raise MonitorError('Supervisor status belongs to a different boot')
    age=snapshot['now']-float(s['checked_at'])
    if not math.isfinite(age) or age<0 or age>config['max_status_age_seconds']:
        raise TransientMonitorError('Supervisor status is stale or in the future')
    release=s['release'];trust=config['trusted_releases'].get(release)
    if trust is None:
        raise MonitorError('Untrusted keepalive release')
    if snapshot['release_hashes']!=trust['files']:
        raise MonitorError('Trusted release files changed')
    ident=s['process_identity']
    if (s['pid']!=sup['pid'] or any(ident[k]!=sup[k] for k in ('pid','start_ticks','pgid','sid'))
            or sup['state'] in ('Z','X') or not snapshot['supervisor_lock_held']):
        raise MonitorError('Supervisor identity or lock ownership mismatch')
    interpreter=config['supervisor_executable']
    if (sup['executable']!=interpreter['path'] or sup['executable_sha256']!=interpreter['sha256']
            or sup['argv'][1:]!=[release+'/supervisor.py','run',config['profile']]):
        raise MonitorError('Supervisor executable/command mismatch')
    # v6 retains its live worker when sampled utilization briefly drops below
    # the retention threshold. That is not a loss of process identity. All
    # executable, parent, birth, UUID and memory checks below still apply.
    low_threshold=(s['state']=='WORKER_THRESHOLD_LOW' and s.get('continuous_worker') is True)
    if s['state'] not in ('PROTECTED_OWN_WORKER','PROTECTED_ASSIST') and not low_threshold:
        raise TransientMonitorError('Supervisor worker is not in an admitted live state: '+str(s['state']))
    if s.get('job_coordination')!='coexist':
        raise MonitorError('Supervisor is not configured for coexistence')
    if (worker['pid']!=s['worker_pid'] or worker['ppid']!=sup['pid']
            or worker['pid'] not in snapshot['supervisor_children']
            or worker['pgid']!=sup['pgid'] or worker['sid']!=sup['sid']):
        raise MonitorError('Worker parent/session identity mismatch')
    if worker['executable']!=release+'/anchor' or worker['executable_sha256']!=trust['files']['anchor']:
        raise MonitorError('Running worker executable is not the trusted release anchor')
    # Older supervisor schemas omit worker start_ticks. Its live birth must
    # precede the fresh supervisor report that names it, and follow its parent.
    worker_epoch=snapshot['now']-snapshot['boottime']+worker['start_ticks']/snapshot['clock_ticks']
    if (worker['state'] in ('Z','X') or worker['start_ticks']<=sup['start_ticks']
            or worker_epoch>float(s['checked_at'])):
        raise TransientMonitorError('Worker birth does not match the supervisor report')
    if s.get('worker_start_ticks',worker['start_ticks'])!=worker['start_ticks']:
        raise MonitorError('Worker PID was reused')
    args=worker['argv']
    if args.count('--device')!=1 or args[args.index('--device')+1]!=str(s['worker_gpu']):
        raise MonitorError('Worker device argument mismatch')
    status_devices={int(g['index']):g['uuid'] for g in s['gpus']}
    devices={g['index']:g['uuid'] for g in snapshot['gpus']}
    if (status_devices!=devices or devices!={g['index']:g['uuid'] for g in config['devices']}
            or s['worker_gpu'] not in devices):
        raise MonitorError('Supervisor GPU UUID inventory mismatch')
    uuid=devices[s['worker_gpu']]
    reported={a['gpu_uuid'] for a in snapshot['apps'] if a['pid']==worker['pid']}
    if reported!={uuid}:
        raise TransientMonitorError('Worker GPU UUID differs from actual process telemetry')
    return dict(pid=worker['pid'],start_ticks=worker['start_ticks'],ppid=worker['ppid'],
        pgid=worker['pgid'],sid=worker['sid'],gpu_uuid=uuid,executable=worker['executable'],
        executable_sha256=worker['executable_sha256'],supervisor_pid=sup['pid'],
        supervisor_start_ticks=sup['start_ticks'],profile=s['profile'],status_age_seconds=age,
        supervisor_state=s['state'])


def collect(config):
    """Double-read live identities and membership; a restart triggers bounded retry."""
    status_path=Path(config['status_path'])
    status=json.loads(status_path.read_text());release=status['release']
    if release not in config['trusted_releases']:
        raise MonitorError('Untrusted keepalive release')
    gpus,apps=telemetry()
    sup=process(status['pid']);worker=process(status['worker_pid'])
    hashes={name:sha_file(Path(release)/name) for name in config['trusted_releases'][release]['files']}
    snapshot=dict(status=status,gpus=gpus,apps=apps,supervisor=sup,worker=worker,
        hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        profile_file=(status_path.parent/'profile').read_text().strip(),release_hashes=hashes,
        supervisor_children=[int(x) for x in Path(f'/proc/{sup["pid"]}/task/{sup["pid"]}/children').read_text().split()],
        supervisor_lock_held=holds_supervisor_lock(sup['pid'],status_path.parent/'supervisor.lock'),
        now=time.time(),boottime=time.clock_gettime(time.CLOCK_BOOTTIME),clock_ticks=os.sysconf('SC_CLK_TCK'))
    verify_status(config,snapshot)
    processes={a['pid']:process(a['pid']) for a in apps}
    gpus_after,apps_after=telemetry()
    after=json.loads(status_path.read_text())
    for key in ('profile','pid','boot_id','release','state','worker_pid','worker_gpu','process_identity'):
        if status[key]!=after[key]:
            raise TransientMonitorError('Supervisor/worker changed during preflight')
    if ({(a['gpu_uuid'],a['pid']) for a in apps}!={(a['gpu_uuid'],a['pid']) for a in apps_after}
            or {(g['index'],g['uuid']) for g in gpus}!={(g['index'],g['uuid']) for g in gpus_after}):
        raise TransientMonitorError('GPU ownership changed during preflight')
    for before in [sup,worker,*processes.values()]:
        live=process(before['pid'])
        if any(before[k]!=live[k] for k in ('pid','ppid','pgid','sid','start_ticks','executable','executable_sha256','argv')):
            raise TransientMonitorError('PID identity changed during preflight')
    snapshot['processes']=processes
    # The most recent real free-memory figure includes the keepalive allocation.
    snapshot['gpus']=gpus_after
    snapshot['apps']=apps_after
    snapshot['now']=time.time()
    snapshot['boottime']=time.clock_gettime(time.CLOCK_BOOTTIME)
    verify_status(config,snapshot)
    return snapshot


def classify(config, snapshot, owned=()):
    known=verify_status(config,snapshot)
    required=max(config['minimum_free_mib'],config['worker_peak_mib']+config['safety_margin_mib'])
    rows=[]
    valid_owners=[]
    for owner in owned:
        leader=snapshot['processes'].get(owner['pid'])
        if (owner['boot']==snapshot['boot_id'] and leader
                and str(leader['start_ticks'])==str(owner['start']) and leader['pgid']==owner['group']):
            valid_owners.append((owner,leader))
    for gpu in snapshot['gpus']:
        occupants=[a['pid'] for a in snapshot['apps'] if a['gpu_uuid']==gpu['uuid']]
        unknown=[];coexisting=[];owned_pids=[]
        for pid in occupants:
            actual=snapshot['processes'][pid]
            if pid==known['pid'] and gpu['uuid']==known['gpu_uuid']:
                coexisting.append(pid);continue
            is_owned=any(owner['gpu_uuid']==gpu['uuid'] and actual['pgid']==leader['pgid']
                and actual['sid']==leader['sid'] and actual['start_ticks']>=leader['start_ticks']
                for owner,leader in valid_owners)
            (owned_pids if is_owned else unknown).append(pid)
        reserve=0
        for owner,leader in valid_owners:
            if owner['gpu_uuid']!=gpu['uuid']:
                continue
            allocated=sum(a['used_mib'] for a in snapshot['apps'] if a['gpu_uuid']==gpu['uuid']
                and a['pid'] in owned_pids and snapshot['processes'][a['pid']]['pgid']==leader['pgid'])
            # A peer still loading its model can have almost no NVML allocation.
            # Keep its remaining peak reservation instead of treating it as free.
            reserve+=max(0,config['worker_peak_mib']-allocated)
        category=('UNKNOWN_OCCUPANCY' if unknown else 'INSUFFICIENT_MEMORY' if gpu['free_mib']<required+reserve
                  else 'KNOWN_KEEPALIVE' if coexisting else 'AVAILABLE')
        rows.append(dict(gpu,category=category,allowed=category in ('AVAILABLE','KNOWN_KEEPALIVE'),
            required_free_mib=required+reserve,peer_peak_reserve_mib=reserve,
            coexisting_pids=coexisting,unknown_pids=unknown,owned_pids=owned_pids))
    return dict(status='PASS' if all(r['allowed'] for r in rows) else 'BLOCKED',rows=rows,
        keepalive=known,checked_at=snapshot['now'],model_calls=0)


def preflight(config, *, owned=(), collector=collect, sleep=time.sleep):
    """At most three fresh snapshots; no cached PASS and no infinite waiting."""
    attempts=config.get('attempts',3);delay=config.get('retry_delay_seconds',1.)
    try:
        if (type(attempts) is not int or not 1<=attempts<=3 or not math.isfinite(delay) or not 0<=delay<=2
                or not 0<config['max_status_age_seconds']<=60
                or any(not math.isfinite(config[k]) or config[k]<0
                       for k in ('worker_peak_mib','safety_margin_mib','minimum_free_mib'))
                or config['worker_peak_mib']==0 or config['safety_margin_mib']<2048):
            raise MonitorError('Invalid bounded admission configuration')
        for attempt in range(1,attempts+1):
            try:
                snapshot=collector(config)
                for owner in owned:
                    if owner['pid'] not in snapshot['processes']:
                        try:
                            snapshot['processes'][owner['pid']]=process(owner['pid'])
                        except TransientMonitorError:
                            pass  # Dead leader confers no ownership on surviving GPU children.
                result=classify(config,snapshot,owned)
                result['attempts']=attempt
                return result
            except TransientMonitorError:
                if attempt==attempts:
                    raise
                sleep(delay)
    except (OSError,ValueError,KeyError,TypeError,IndexError,MonitorError,subprocess.SubprocessError) as error:
        return dict(status='MONITOR_ERROR',rows=[dict(g,category='MONITOR_ERROR',allowed=False)
            for g in config.get('devices',[])],allowed=False,error=str(error),model_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=preflight(json.loads(Path(args.config).read_text()))
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))

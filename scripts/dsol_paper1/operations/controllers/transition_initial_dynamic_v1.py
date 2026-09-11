"""Drain only current in-flight episodes, then replace the owned scheduler."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[4]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from scripts.dsol_paper1.operations.controllers.dynamic_initial_scheduler_v1 import ROOT, EXT, core, verify_extension
from scripts.dsol_paper1.operations.controllers.run_early_landscape_speed_gate_v1 import process_identity, same_process


def alive(p):
    return same_process(p) and process_identity(p['pid'])['state']!='Z'


def children(pid):
    result=[]
    # Popen runs in worker threads: main-thread children alone misses episodes.
    values=set()
    for task in (Path('/proc')/str(pid)/'task').iterdir():
        try:values.update((task/'children').read_text().split())
        except FileNotFoundError:pass
    for value in values:
        try:result.append(process_identity(int(value)))
        except FileNotFoundError:pass
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['fresh','gnu'],required=True);a=p.parse_args()
    manifest=verify_extension()
    core.require(core.sha(Path(__file__))==manifest['transition_sha256'],'Transition source changed')
    out=EXT/a.host/'transition.json';core.require(not out.exists(),'Transition already attempted')
    before=core.read(ROOT/'hosts'/a.host/'status.json');old=process_identity(before['pid'])
    core.require('standard_initial_aa0_v1.py run --host '+a.host in old['cmdline'],'Not the owned static controller')
    core.require(alive(old),'Original controller already stopped')
    record=dict(status='PAUSING_OWNED_DISPATCH_ONLY',utc=time.time(),old_controller=old,previous_status=before,
        keepalive_untouched=True,scientific_release_sha256=core.sha(ROOT/'release.json'),inflight_policy='finish naturally')
    core.write(out,record)
    try:
        inflight=[x for x in children(old['pid']) if 'standard_initial_aa0_v1.py episode' in x['cmdline'] and alive(x)]
        record.update(status='RESTARTING_OWNED_DISPATCH_CHECKPOINTED_RESULTS_PRESERVED',inflight=inflight,
                      inflight_policy='Incomplete episodes rerun using identical frozen identity/noise; no failure labels fabricated',utc=time.time())
        core.write(out,record)
        # The existing handler cleans only its children. Atomic completed files
        # are retained; missing outputs are re-enqueued by the new scheduler.
        os.kill(old['pid'],signal.SIGTERM)
        if process_identity(old['pid'])['state'] in ('T','t'):os.kill(old['pid'],signal.SIGCONT)
        start=time.time()
        while alive(old):
            core.require(time.time()-start<90,'Old controller cleanup did not finish')
            time.sleep(1)
        record.update(status='OLD_STOPPED_DYNAMIC_GATE_STARTING',utc=time.time());core.write(out,record)
        with (EXT/a.host/'controller.log').open('a') as log:
            child=subprocess.Popen(['/alphabrain/.venv/bin/python',str(EXT/'dynamic_initial_scheduler_v1.py'),'--host',a.host],
                 env=core.env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(status='DYNAMIC_SCHEDULER_LAUNCHED',new_pid=child.pid,utc=time.time());core.write(out,record)
        print(record['status'],child.pid,flush=True)
    except BaseException:
        if alive(old) and process_identity(old['pid'])['state'] in ('T','t'):
            os.kill(old['pid'],signal.SIGCONT)
            record.update(status='TRANSITION_ABORTED_OLD_DISPATCH_RESUMED',utc=time.time());core.write(out,record)
        raise


if __name__=='__main__':main()

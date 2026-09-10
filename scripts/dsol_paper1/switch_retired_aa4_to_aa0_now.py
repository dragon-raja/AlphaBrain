#!/usr/bin/env python3
"""One-shot owned AA4 cancellation; preserve data and launch frozen AA0."""
import os
from pathlib import Path
import signal
import subprocess
import time

from dualhost_aa0_v1 import ROOT, REPO, env, write, release
from render_bridge_common_v1 import read
from run_early_landscape_speed_gate_v1 import process_identity, same_process


def live(p):
    return same_process(p) and process_identity(p['pid'])['state'] != 'Z'


def main():
    audit = ROOT / 'old-aa4-cancelled-for-aa0.json'
    assert not audit.exists(), 'Already switched; inspect status before retrying'
    release()  # Verify the frozen experiment, without changing it.
    retirement = read(ROOT / 'old-aa4-retirement.json')
    assert retirement['status'] == 'DRAINING_OWNED_OLD_WAVE'
    child = retirement['child']
    assert live(child) and os.getpgid(child['pid']) == child['pid']
    for p in (retirement['dispatcher'], retirement['extension']):
        assert same_process(p) and process_identity(p['pid'])['state'] in ('T', 't')
    queue = process_identity(567476)
    assert 'python scripts/dsol_paper1/retire_old_aa4_for_dualhost_v1.py' in queue['cmdline']
    assert not (Path('/proc') / str(queue['pid']) / 'task' / str(queue['pid']) / 'children').read_text().strip()
    members = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            pid = int(proc.name)
            if os.getpgid(pid) == child['pid']:
                p = process_identity(pid)
                assert 'keepalive' not in p['cmdline'].lower()
                members.append(p)
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert len(members) > 2
    record = dict(status='CANCELLING_OWNED_AA4', utc=time.time(),
                  reason='AA4 data excluded from AA0 reassessment; no scientific need to finish wave-05',
                  previous_retirement=retirement, queue=queue, owned_group=members,
                  existing_files_preserved=True, keepalive_signalled=False)
    write(audit, record)
    os.kill(queue['pid'], signal.SIGTERM)
    for _ in range(20):
        if not live(queue):
            break
        time.sleep(0.25)
    assert not live(queue), 'Drain launcher did not exit'
    assert same_process(child)
    # The old wrapper's existing trap cleans its own servers and restores
    # any local keepalives it had previously managed. No keeper is signalled.
    os.killpg(child['pid'], signal.SIGTERM)
    for _ in range(45):
        if not any(live(p) for p in members):
            break
        time.sleep(1)
    assert not any(live(p) for p in members), 'Owned old processes remain; inspect before launch'
    record.update(status='OLD_AA4_CANCELLED_PARTIAL_DATA_ARCHIVED', utc=time.time())
    write(audit, record)
    write(ROOT / 'old-aa4-retirement.json', dict(retirement,
          status=record['status'], cancellation_receipt=str(audit), utc=time.time()))
    command = ('cd ' + str(REPO) + ' && exec /alphabrain/.venv/bin/python '
               'scripts/dsol_paper1/dualhost_aa0_v1.py run --host fresh > '
               + str(ROOT / 'fresh-controller.log') + ' 2>&1')
    subprocess.run(['tmux', 'new-session', '-d', '-s',
                    'dsol-aa0-dualhost-fresh-now-v1', command], env=env(), check=True)
    record.update(status='FRESH_AA0_LAUNCHED', utc=time.time())
    write(audit, record)
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()

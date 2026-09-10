"""Stop only the identified snapshot controller, preserving all its files."""
import argparse
import os
import signal
import time
from pathlib import Path
from dualhost_aa0_v1 import ROOT, read, write
from run_early_landscape_speed_gate_v1 import process_identity, same_process

p=argparse.ArgumentParser();p.add_argument('--host',choices=['fresh','gnu'],required=True);a=p.parse_args()
status=read(ROOT/'hosts'/a.host/'status.json');identity=process_identity(status['pid'])
assert 'dualhost_aa0_v1.py run --host '+a.host in identity['cmdline']
audit=ROOT/'hosts'/a.host/'superseded-by-standard-initial.json'
assert not audit.exists()
record=dict(status='STOP_REQUESTED',utc=time.time(),process=identity,previous=status,
            reason='User requested standard official initial-state whole-task main matrix',
            completed_records_preserved=True,keepalive_untouched=True)
write(audit,record);os.kill(identity['pid'],signal.SIGTERM)
for _ in range(90):
    if not same_process(identity) or process_identity(identity['pid'])['state']=='Z':break
    time.sleep(1)
assert not same_process(identity) or process_identity(identity['pid'])['state']=='Z', 'Controller still active'
record.update(status='STOPPED_ARCHIVED_PARTIAL_NOT_A_BALANCED_RESULT',utc=time.time())
write(audit,record);print(record['status'])

#!/usr/bin/env python3
"""Keep the already-isolated GPU window briefly open for the rendering fix probe."""
import os
from pathlib import Path
import signal
import time
import run_full64_view_landscape_v2 as control
from run_early_landscape_speed_gate_v1 import process_identity, same_process, resume

ROOT = control.ROOT/'repeatability-root-cause-v1/render-fix-window-v1'
PARENT = ROOT.parent


def main():
    identity = control.read_json(PARENT/'boundary-entry.json')['gate_process']
    control.require(same_process(identity), 'Boundary controller changed')
    control.require(process_identity(identity['pid'])['state'] not in ('T','t','Z'), 'Boundary already paused/dead')
    control.require(control.read_json(PARENT/'boundary-status.json')['status']=='RUNNING_ISOLATED_DIAGNOSTICS', 'Not in isolated diagnostic window')
    ROOT.mkdir()
    control.write_new_json(ROOT/'entry.json', {'gate': identity, 'holder': process_identity(os.getpid()),
        'deadline_epoch': time.time()+1200, 'reason': 'First divergence located in wrist pixels with identical fixed-action physics; hold boundary controller only while testing rendering fixes.', 'utc':control.stamp()})
    paused=False
    def interrupted(signum, frame): raise InterruptedError('Render fix window interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(identity['pid'], signal.SIGSTOP); paused=True
        deadline=time.time()+1200
        while time.time()<deadline and not (ROOT/'release.json').exists(): time.sleep(2)
    finally:
        process_path=ROOT/'process.json'
        if process_path.exists():
            probe=control.read_json(process_path)
            if same_process(probe):
                try: os.killpg(probe['pid'], signal.SIGTERM)
                except ProcessLookupError: pass
                until=time.time()+15
                while same_process(probe) and process_identity(probe['pid'])['state']!='Z' and time.time()<until: time.sleep(.5)
                if same_process(probe) and process_identity(probe['pid'])['state']!='Z':
                    try: os.killpg(probe['pid'], signal.SIGKILL)
                    except ProcessLookupError: pass
        if paused: control.atomic_json(ROOT/'resumed.json', {'gate_resumed':resume(identity),'utc':control.stamp()})


if __name__=='__main__': main()

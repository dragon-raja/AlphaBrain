#!/usr/bin/env python3
"""Drain one owned segment, run bounded diagnostics, and restore the dispatcher."""
from __future__ import annotations

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
import sys
import time
import scripts.dsol_paper1.run_full64_view_landscape_v2 as control
from scripts.dsol_paper1.operations.controllers.run_early_landscape_speed_gate_v1 import process_identity, same_process, resume
from scripts.dsol_paper1.build_matched_view_landscape_protocol_v1 import source_identity, write_new_json

ROOT = control.ROOT/'repeatability-root-cause-v1'


def stop_probe():
    path = ROOT/'probe-process.json'
    if not path.exists(): return
    identity = control.read_json(path)
    if not same_process(identity): return
    try: os.killpg(identity['pid'], signal.SIGTERM)
    except ProcessLookupError: return
    deadline = time.time()+20
    while same_process(identity) and time.time()<deadline:
        if process_identity(identity['pid'])['state']=='Z': return
        time.sleep(.5)
    if same_process(identity):
        try: os.killpg(identity['pid'], signal.SIGKILL)
        except ProcessLookupError: pass


def watchdog():
    entry = control.read_json(ROOT/'boundary-entry.json')
    while time.time()<entry['watchdog_deadline_epoch']:
        if (ROOT/'resumed.json').exists(): return
        if not same_process(entry['gate_process']): break
        time.sleep(2)
    try: stop_probe()
    finally:
        control.atomic_json(ROOT/'watchdog-restoration.json', {'resumed': resume(entry['dispatcher_process']), 'utc': control.stamp()})


def main():
    release = control.preflight()
    before = control.read_json(control.PARENT/'controller-status.json')
    control.require(before['status']=='RUNNING', 'Original dispatcher is not running')
    dispatcher = process_identity(before['pid'])
    control.require(dispatcher['state'] not in ('T','t','Z'), 'Dispatcher already stopped')
    env = (Path('/proc')/str(dispatcher['pid'])/'environ').read_bytes()
    control.require(('DSOL_LANDSCAPE_REPO_ROOT='+str(control.REPO)).encode() in env, 'Wrong dispatcher ownership')
    children = [process_identity(int(p)) for p in (Path('/proc')/str(dispatcher['pid'])/'task'/str(dispatcher['pid'])/'children').read_text().split()]
    children = [p for p in children if p['cmdline'].startswith('timeout ') and str(control.RUNNER) in p['cmdline']]
    control.require(len(children)==1, 'Expected one active owned segment')
    child = children[0]
    output = control.PARENT/'closed-loop/canonical/O'/before['current_segment']
    manifest = control.read_json(output/'run_manifest.json')
    ROOT.mkdir(exist_ok=True)
    write_new_json(ROOT/'boundary-entry.json', {'gate_process': process_identity(os.getpid()),
        'dispatcher_process': dispatcher, 'segment_process': child, 'segment': before['current_segment'],
        'watchdog_deadline_epoch': time.time()+5700, 'utc': control.stamp(),
        'source': source_identity(Path(__file__)), 'driver': source_identity(control.REPO/'scripts/dsol_paper1/operations/controllers/run_view_repeatability_probe_v1.py')})
    with (ROOT/'watchdog.log').open('a') as stream:
        subprocess.Popen([sys.executable,str(Path(__file__)),'--watchdog'], stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    paused = False
    try:
        os.kill(dispatcher['pid'],signal.SIGSTOP); paused=True
        control.require(control.read_json(control.PARENT/'controller-status.json')==before, 'Segment boundary raced')
        control.atomic_json(ROOT/'boundary-status.json', {'status':'DRAINING_OWNED_SEGMENT', 'segment':before['current_segment'], 'utc':control.stamp()})
        deadline=time.time()+3600
        while same_process(child) and process_identity(child['pid'])['state']!='Z':
            control.require(time.time()<deadline, 'Drain timeout')
            time.sleep(3)
        receipt=control.read_json(output/'run-manifests'/('receipt-'+manifest['run_attempt_id']+'.json'))
        control.require(receipt['status']=='PASS_COMPLETE' and receipt['episode_count']==3104, 'Current segment did not complete')
        control.resources_available(29600,2)
        control.atomic_json(ROOT/'boundary-status.json', {'status':'RUNNING_ISOLATED_DIAGNOSTICS','utc':control.stamp()})
        with (ROOT/'probe.log').open('a') as stream:
            probe=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30s','1800s',sys.executable,
                str(control.REPO/'scripts/dsol_paper1/operations/controllers/run_view_repeatability_probe_v1.py')],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write_new_json(ROOT/'probe-process.json',process_identity(probe.pid))
            code=probe.wait()
            control.require(code==0,'Diagnostic failed; inspect retained traces')
        control.atomic_json(ROOT/'boundary-status.json', {'status':'PASS_DIAGNOSTIC_EXECUTION','utc':control.stamp()})
    except BaseException as error:
        control.atomic_json(ROOT/'boundary-status.json', {'status':'STOPPED_REQUIRES_REVIEW','error':str(error),'utc':control.stamp()})
        raise
    finally:
        try: stop_probe()
        finally:
            if paused:
                control.atomic_json(ROOT/'resumed.json', {'resumed':resume(dispatcher),'utc':control.stamp()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--watchdog',action='store_true'); args=parser.parse_args()
    def interrupted(signum,frame): raise InterruptedError('Boundary controller interrupted')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    if args.watchdog: watchdog()
    else: main()

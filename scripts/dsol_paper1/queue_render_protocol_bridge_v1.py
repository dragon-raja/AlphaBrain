#!/usr/bin/env python3
"""Drain the current owned wave, run a bounded bridge, then always restore it."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from render_bridge_common_v1 import ROOT,REPO,read,sha,write_new,verify_release
import run_full64_view_landscape_v2 as control
from run_early_landscape_speed_gate_v1 import process_identity,same_process,resume

def stop_owned():
    path=ROOT/'process.json'
    if not path.exists():return
    identity=read(path)
    if not same_process(identity):return
    try:os.killpg(identity['pid'],signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.time()+30
    while same_process(identity) and process_identity(identity['pid'])['state']!='Z' and time.time()<deadline:time.sleep(.5)
    if same_process(identity) and process_identity(identity['pid'])['state']!='Z':
        try:os.killpg(identity['pid'],signal.SIGKILL)
        except ProcessLookupError:pass

def watchdog():
    entry=read(ROOT/'queue-entry.json')
    while time.time()<entry['watchdog_deadline_epoch']:
        if (ROOT/'resumed.json').exists():return
        if not same_process(entry['gate']):break
        time.sleep(3)
    try:stop_owned()
    finally:control.atomic_json(ROOT/'watchdog-restoration.json',{'resumed':resume(entry['dispatcher']),'utc':control.stamp()})

def main():
    release=verify_release();control.preflight()
    before=read(control.PARENT/'controller-status.json'); dispatcher=process_identity(before['pid'])
    control.require(before['status']=='RUNNING' and dispatcher['state'] not in ('T','t','Z'),'Expected running owned dispatcher')
    environment=(Path('/proc')/str(dispatcher['pid'])/'environ').read_bytes()
    control.require(('DSOL_LANDSCAPE_REPO_ROOT='+str(REPO)).encode() in environment,'Dispatcher ownership differs')
    children=[process_identity(int(p)) for p in (Path('/proc')/str(dispatcher['pid'])/'task'/str(dispatcher['pid'])/'children').read_text().split()]
    children=[p for p in children if p['cmdline'].startswith('timeout ') and str(control.RUNNER) in p['cmdline']]
    control.require(len(children)==1,'Expected one active wave supervisor')
    child=children[0];output=control.PARENT/'closed-loop/canonical/O'/before['current_segment']; manifest=read(output/'run_manifest.json')
    write_new(ROOT/'queue-entry.json',{'utc':control.stamp(),'gate':process_identity(os.getpid()),'dispatcher':dispatcher,
        'current_segment':before['current_segment'],'child':child,'run_manifest_sha256':sha(output/'run_manifest.json'),
        'watchdog_deadline_epoch':time.time()+release['max_drain_seconds']+release['max_execution_seconds']+300,
        'release_sha256':sha(ROOT/'release.json'),'rollout_children_paused':False})
    with (ROOT/'watchdog.log').open('a') as stream:
        subprocess.Popen([sys.executable,str(Path(__file__)),'--watchdog'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    paused=False
    try:
        os.kill(dispatcher['pid'],signal.SIGSTOP);paused=True
        control.require(read(control.PARENT/'controller-status.json')==before,'Wave boundary raced; restore without takeover')
        control.atomic_json(ROOT/'status.json',{'status':'DRAINING_CURRENT_WAVE','segment':before['current_segment'],'utc':control.stamp()})
        deadline=time.time()+release['max_drain_seconds']
        while same_process(child) and process_identity(child['pid'])['state']!='Z':
            control.require(time.time()<deadline,'Drain exceeded bounded window');time.sleep(3)
        receipt=read(output/'run-manifests'/('receipt-'+manifest['run_attempt_id']+'.json'))
        control.require(receipt['status']=='PASS_COMPLETE' and receipt['episode_count']==3104,'Current wave incomplete')
        control.require(sha(output/'run_manifest.json')==read(ROOT/'queue-entry.json')['run_manifest_sha256'],'Wave manifest changed')
        with (ROOT/'execution.log').open('a') as stream:
            p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=60s',str(release['max_execution_seconds'])+'s',sys.executable,
                str(REPO/'scripts/dsol_paper1/run_render_protocol_bridge_v1.py')],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write_new(ROOT/'process.json',process_identity(p.pid));code=p.wait()
        control.require(code==0,'Bridge failed or timed out; original data unchanged; inspect retained partial outputs')
    except BaseException as error:
        control.atomic_json(ROOT/'queue-error.json',{'error':str(error),'utc':control.stamp()});raise
    finally:
        try:stop_owned()
        finally:
            if paused:control.atomic_json(ROOT/'resumed.json',{'resumed':resume(dispatcher),'utc':control.stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watchdog',action='store_true');args=p.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Bridge boundary interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.watchdog:watchdog()
    else:main()

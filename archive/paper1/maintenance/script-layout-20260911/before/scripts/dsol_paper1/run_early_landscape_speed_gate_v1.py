#!/usr/bin/env python3
"""Temporarily pause only the dispatcher; drain its current segment and benchmark.

No rollout is interrupted, no frozen controller is edited, and a separate
watchdog resumes the exact dispatcher process if this gate exits unexpectedly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_full64_view_landscape_v2 as control
from build_matched_view_landscape_protocol_v1 import read_json, write_new_json, source_identity
from explicit_flow_noise import sha256_file
from audit_statewise_view_oracle_v2_run import atomic_json

ROOT=control.ROOT/'early-speed-gate-v1'


def process_identity(pid):
    base=Path('/proc')/str(pid)
    text=(base/'stat').read_text()
    fields=text[text.rfind(')')+2:].split()
    return {'pid':pid,'start_ticks':int(fields[19]),'state':fields[0],
            'cmdline':(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}


def same_process(identity):
    try: current=process_identity(identity['pid'])
    except FileNotFoundError: return False
    return current['start_ticks']==identity['start_ticks'] and current['cmdline']==identity['cmdline']


def resume(identity):
    if same_process(identity):
        os.kill(identity['pid'],signal.SIGCONT)
        return True
    return False


def watchdog(path):
    receipt=read_json(path); gate=receipt['gate_process']; dispatcher=receipt['dispatcher_process']
    deadline=receipt['watchdog_deadline_epoch']
    while time.time()<deadline:
        if (path.parent/'resumed.json').exists(): return
        if not same_process(gate): break
        time.sleep(2)
    restored=resume(dispatcher)
    atomic_json(path.parent/'watchdog-restoration.json',dict(status='WATCHDOG_RESUME_ATTEMPT',dispatcher_resumed=restored,utc=control.stamp()))


def main_gate():
    control.require(not ROOT.exists(),'Early benchmark gate already exists; no implicit rerun')
    release=control.preflight()
    control.require(not (control.ROOT/'performance-choice.json').exists(),'Concurrency has already been measured')
    predecessor=read_json(control.PARENT/'controller-status.json')
    extension=read_json(control.ROOT/'controller-status.json')
    control.require(predecessor['status']=='RUNNING' and extension['status']=='QUEUED_BEHIND_FIRST_EIGHT','Not the expected active/queued ownership boundary')
    pid=predecessor['pid']; dispatcher=process_identity(pid)
    control.require(dispatcher['state'] not in ('T','t','Z'),'Dispatcher is already stopped/dead; do not change it')
    environment=(Path('/proc')/str(pid)/'environ').read_bytes()
    control.require(('DSOL_LANDSCAPE_REPO_ROOT='+str(control.REPO)).encode() in environment,'Dispatcher ownership differs')
    children=list(map(int,(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()))
    running=[process_identity(child) for child in children]
    running=[p for p in running if str(control.RUNNER) in p['cmdline'] and p['cmdline'].startswith('timeout ')]
    control.require(len(running)==1,'Expected exactly one owned active rollout timeout supervisor')
    child=running[0]; segment=predecessor['current_segment']
    output=control.PARENT/'closed-loop/canonical/O'/segment
    manifest=read_json(output/'run_manifest.json')
    control.require(manifest['checkpoint_sha256']==release['old_release_content']['canonical_checkpoint']['weights_sha256'],'Current segment uses another checkpoint')
    ROOT.mkdir()
    entry=dict(status='FROZEN_EARLY_BENCHMARK_GATE',created_at_utc=control.stamp(),
        script=source_identity(Path(__file__)),release=source_identity(control.ROOT/'release.json'),
        gate_process=process_identity(os.getpid()),dispatcher_process=dispatcher,active_segment_process=child,
        current_segment=segment,current_run_manifest=source_identity(output/'run_manifest.json'),
        current_attempt=manifest['run_attempt_id'],watchdog_deadline_epoch=time.time()+7500,
        scientific_controls_changed=False,rollout_processes_paused=False,
        reason='User requested measuring speed now; drain the current segment before pre-frozen concurrency benchmark, then continue original dispatcher unchanged.')
    write_new_json(ROOT/'entry.json',entry)
    with (ROOT/'watchdog.log').open('a') as stream:
        subprocess.Popen([sys.executable,str(Path(__file__)),'--watchdog',str(ROOT/'entry.json')],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    paused=False
    try:
        control.require(same_process(dispatcher),'Dispatcher changed before pause')
        os.kill(pid,signal.SIGSTOP); paused=True
        after=read_json(control.PARENT/'controller-status.json')
        control.require(after==predecessor,'Dispatcher crossed a segment boundary before pause; abort safely')
        atomic_json(ROOT/'status.json',dict(status='DISPATCHER_PAUSED_CURRENT_ROLLOUT_DRAINING',current_segment=segment,utc=control.stamp()))
        deadline=time.time()+7200
        while same_process(child) and process_identity(child['pid'])['state']!='Z':
            control.require(not control.STOPPED and time.time()<deadline,'Drain interrupted or exceeded two hours')
            time.sleep(5)
        run_receipt=output/'run-manifests'/('receipt-'+manifest['run_attempt_id']+'.json')
        control.require(run_receipt.exists(),'Current rollout did not finish successfully; no benchmark')
        receipt=read_json(run_receipt)
        control.require(receipt['status']=='PASS_COMPLETE' and receipt['episode_count']==3104,'Current segment incomplete')
        control.require(sha256_file(output/'run_manifest.json')==entry['current_run_manifest']['sha256'],'Current run manifest changed')
        control.resources_available(release['base_port'])
        results=[]; baseline=None
        for mode in release['performance_modes']:
            atomic_json(ROOT/'status.json',dict(status='BENCHMARKING',mode=mode,utc=control.stamp()))
            target=control.ROOT/'benchmarks'/mode['label']
            r=control.run_segment(release,release['benchmark_protocol'],target,mode,deadline)
            signatures=control.action_signatures(target)
            if baseline is None: baseline=signatures
            results.append(dict(mode=mode,runner_wall_seconds=r['runner_wall_seconds'],exact_action_equivalence=signatures==baseline,
                receipt=source_identity(target/'segment-receipt.json')))
            atomic_json(ROOT/'progress.json',dict(results=results,utc=control.stamp()))
        chosen=control.select_mode(results)
        write_new_json(control.ROOT/'performance-choice.json',dict(status='PASS_PERFORMANCE_SELECTION',release_sha256=sha256_file(control.ROOT/'release.json'),
            selected_mode=chosen,results=results,selection_uses_success_values=False,
            rule='Exact full action/noise trajectory equality, then >=5% end-to-end speedup; benchmark outcomes excluded from scientific matrix',
            early_execution_gate=source_identity(ROOT/'entry.json')))
        atomic_json(ROOT/'status.json',dict(status='PASS_COMPLETE_SPEED_MEASUREMENT',selected_mode=chosen,results=results,utc=control.stamp()))
    except BaseException as error:
        atomic_json(ROOT/'status.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(error),utc=control.stamp()))
        raise
    finally:
        if paused:
            restored=resume(dispatcher)
            atomic_json(ROOT/'resumed.json',dict(status='ORIGINAL_DISPATCHER_RESUME_ATTEMPT',resumed=restored,utc=control.stamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--watchdog',type=Path); args=parser.parse_args()
    signal.signal(signal.SIGTERM,control.stop); signal.signal(signal.SIGINT,control.stop)
    if args.watchdog: watchdog(args.watchdog)
    else: main_gate()

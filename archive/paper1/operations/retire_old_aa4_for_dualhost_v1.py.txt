#!/usr/bin/env python3
"""Drain owned old wave, preserve its data, then start the new fresh controller."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_early_landscape_speed_gate_v1 import process_identity,same_process
from render_bridge_common_v1 import read,sha
from dualhost_aa0_v1 import ROOT,REPO,write,env

OLD=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')

def main():
    receipt=ROOT/'old-aa4-retirement.json'
    if receipt.exists():raise RuntimeError('Already retired; inspect before retrying')
    before=read(OLD/'controller-status.json');dispatcher=process_identity(before['pid'])
    e=(Path('/proc')/str(dispatcher['pid'])/'environ').read_bytes()
    assert b'DSOL_LANDSCAPE_REPO_ROOT=/workspace/projects/alphabrain-dsol-paper1' in e
    children=[process_identity(int(p)) for p in (Path('/proc')/str(dispatcher['pid'])/'task'/str(dispatcher['pid'])/'children').read_text().split()]
    children=[p for p in children if 'run_dsol_libero_hdf5_closed_loop_eval.sh' in p['cmdline'] and p['cmdline'].startswith('timeout ')]
    assert len(children)==1
    extension=process_identity(read(OLD/'full64-extension-v2/controller-status.json')['pid'])
    assert 'run_full64_view_landscape_v2.py' in extension['cmdline']
    manifest_path=OLD/'closed-loop/canonical/O'/before['current_segment']/'run_manifest.json';manifest=read(manifest_path)
    record=dict(status='DRAINING_OWNED_OLD_WAVE',dispatcher=dispatcher,extension=extension,child=children[0],
                wave=before['current_segment'],manifest_sha256=sha(manifest_path),utc=time.time(),keepalive_untouched=True)
    write(receipt,record)
    # Pause dispatchers only, never the running rollout or any keepalive.
    for p in [extension,dispatcher]:
        assert same_process(p);os.kill(p['pid'],signal.SIGSTOP)
    assert read(OLD/'controller-status.json')==before,'Wave raced during retirement'
    start=time.time()
    while same_process(children[0]) and process_identity(children[0]['pid'])['state']!='Z':
        if time.time()-start>4*3600:raise RuntimeError('Drain timed out; old dispatchers remain deliberately retired')
        time.sleep(5)
    complete=manifest_path.parent/'run-manifests'/('receipt-'+manifest['run_attempt_id']+'.json')
    result=read(complete);assert result['status']=='PASS_COMPLETE' and result['episode_count']==3104
    assert sha(manifest_path)==record['manifest_sha256']
    write(receipt,dict(record,status='OLD_WAVE_COMPLETE_AND_ARCHIVED',completion=str(complete),completion_sha256=sha(complete),utc=time.time()))
    # New controller only owns its own processes. Old dispatchers are not resumed.
    raise SystemExit(subprocess.call([sys.executable,REPO/'scripts/dsol_paper1/dualhost_aa0_v1.py','run','--host','fresh'],env=env()))

if __name__=='__main__':main()

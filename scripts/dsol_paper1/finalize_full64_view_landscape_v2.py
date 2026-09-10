#!/usr/bin/env python3
"""Join immutable first-eight and additional-56 artifacts, then compare 64 vs 64."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from build_matched_view_landscape_protocol_v1 import read_json, write_new_json, source_identity
from run_full64_view_landscape_v2 import ROOT, PARENT, REPO, preflight, verify_first_eight, completed, require


def link_existing(source,destination):
    require(source.exists(),'Missing verified input: '+str(source))
    if destination.is_symlink(): require(destination.resolve()==source.resolve(),'Input link points elsewhere')
    else:
        require(not destination.exists(),'Refusing to replace existing input alias')
        destination.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def main():
    release=preflight(); verify_first_eight()
    status=read_json(ROOT/'controller-status.json')
    require(status['status']=='PASS_COMPLETE_DENSE_FULL64' and status['total_dense_completed']==198656,'Full dense matrix incomplete')
    mode=read_json(ROOT/'performance-choice.json')['selected_mode']
    segments=[]
    for p in release['route']:
        output=ROOT/'closed-loop/O'/p['label']
        require(completed(output,release,p,mode) is not None,'Extension receipt missing')
        segments.append(output)
    first_segments=sorted((PARENT/'closed-loop/canonical/O').iterdir())
    union=ROOT/'verified-union'; (union/'O').mkdir(parents=True,exist_ok=True); (union/'accel').mkdir(exist_ok=True)
    for prefix,items in (('first8',first_segments),('additional56',segments)):
        for source in items: link_existing(source,union/'O'/(prefix+'-'+source.name))
    accel_root=Path(release['accel_manifest']['path']).parent
    require(read_json(accel_root/'completion.json')['status']=='PASS_COMPLETE','Extension scoring incomplete')
    for prefix,source in (('first8',PARENT/'canonical-accel'),('additional56',accel_root)):
        for path in source.glob('rank-shard-*.jsonl'): link_existing(path,union/'accel'/('rank-shard-'+prefix+'-'+path.name))
    receipt=dict(status='PASS_DISJOINT_INPUT_UNION',new_rollouts=0,first_eight_episodes=24832,additional_episodes=173824,
        total_episodes=198656,release=source_identity(ROOT/'release.json'),
        note='Read-only aliases preserve every source run manifest, audit, ledger and score receipt; no scientific data copied or relabeled.',
        segment_receipts=[source_identity(d/'segment-receipt.json') for d in first_segments+segments])
    if not (union/'receipt.json').exists(): write_new_json(union/'receipt.json',receipt)
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    old=release['old_release_content']; report=ROOT/'canonical-report-full64'
    subprocess.run([sys.executable,str(REPO/'scripts/dsol_paper1/build_view_landscape_report_v1.py'),
        '--accel-root',str(union/'accel'),'--dense-dir',str(union/'O'),'--noise-bank-manifest',old['noise_bank']['path'],
        '--selection-manifest',release['selection']['path'],'--checkpoint',old['canonical_checkpoint']['path'],
        '--expected-checkpoint-sha256',old['canonical_checkpoint']['weights_sha256'],'--model-label','Canonical matched / full64 / seed41',
        '--training-support','canonical','--output-dir',str(report)],env=env,check=True)
    subprocess.run([sys.executable,str(REPO/'scripts/dsol_paper1/analyze_view_oracle_comparison_v2.py'),
        '--broad-root',str(PARENT/'broad-existing'),'--canonical-root',str(report),'--output-dir',str(ROOT/'matched64-metrics-with-oracle')],env=env,check=True)


if __name__=='__main__': main()

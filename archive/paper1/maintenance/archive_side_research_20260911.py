"""One-time byte-preserving side-project archival with an explicit allowlist."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
NAMES=['acd_vla','basin_vla','branch_vla','fresh_vla','cora_vla','ccv_vla','policy_response_vla','continuation_distill']


def main():
    dest=ROOT/'archive/research/code/scripts'
    receipt=ROOT/'archive/research/code/manifest.json'
    if receipt.exists():raise RuntimeError('Archival already recorded; do not repeat')
    for name in NAMES:
        if not (ROOT/'scripts'/name).is_dir() or (dest/name).exists():raise RuntimeError('Unexpected source/destination '+name)
    records=[]
    for name in NAMES:
        for p in sorted((ROOT/'scripts'/name).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                records.append({'old':str(p.relative_to(ROOT)),'new':str((dest/name/p.relative_to(ROOT/'scripts'/name)).relative_to(ROOT)),
                                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    shared={'video_io.py':'historical_video_io.py','pi05_policy_server.py':'historical_pi05_policy_server.py'}
    for old,new in shared.items():
        target=ROOT/'scripts/vla_shared'/new
        if target.exists():raise RuntimeError('Shared target exists')
        shutil.copyfile(ROOT/'scripts/fresh_vla'/old,target)
    dest.mkdir(parents=True)
    for name in NAMES:shutil.move(str(ROOT/'scripts'/name),str(dest/name))
    for rec in records:
        if hashlib.sha256((ROOT/rec['new']).read_bytes()).hexdigest()!=rec['sha256']:raise RuntimeError('Archive byte mismatch')
    receipt.write_text(json.dumps({'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'non-mainline side research; no raw results/checkpoints/frozen releases moved',
        'directories':NAMES,'records':records,'shared_exceptions':shared,'recovery':'Copy archived code back to its recorded old path in a separate recovery checkout; never overlay current working code blindly.'},indent=2)+'\n')
    print(json.dumps({'archived_files':len(records),'directories':len(NAMES),'receipt':str(receipt)},indent=2))


if __name__=='__main__':main()

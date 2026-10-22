"""Paper 1 ownership checks and inert run-directory preparation; no job launch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def layout():
    return json.loads((ROOT/'configs/dsol_paper1/project_layout.json').read_text())


def check(artifact_root=None):
    spec = layout(); errors = []
    for relative in [spec[k] for k in ('current_entry', 'current_protocol', 'current_design', 'history_root')] + list(spec['source_owners'].values()):
        if not (ROOT/relative).exists(): errors.append('Missing owned entry: ' + relative)
    if list((ROOT/'docs/dsol_paper1/planning').glob('*.md')) or list((ROOT/'docs/dsol_paper1/execution').glob('*.md')):
        errors.append('Dated plans/execution records belong in history, not competing current folders')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in tracked:
        if any(name == x or name.startswith(x + '/') for x in spec['private_local_entries']):
            errors.append('Private workspace entry tracked: ' + name)
    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        for name in spec['artifact_directories']:
            if not (artifact_root/name).is_dir(): errors.append('Missing artifact category: ' + name)
        for entry in json.loads((artifact_root/'catalog.json').read_text())['entries']:
            if not (artifact_root/entry['entry']).exists(): errors.append('Broken catalog entry: ' + entry['entry'])
    return dict(status='FAIL' if errors else 'PASS', errors=errors,
                scope='Ownership and navigation only; not scientific admission or publication approval')


def init_run(artifact_root, run_id, protocol, purpose):
    """Prepare a new directory; never reopen an existing or archived run."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', run_id):
        raise ValueError('A simple unique lowercase run ID is required')
    if not purpose.strip(): raise ValueError('A concrete purpose is required')
    protocol = Path(protocol); content = protocol.read_bytes()
    json.loads(content)  # Fail before making directories if the protocol is invalid.
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
    target = Path(artifact_root)/'runs'/run_id
    target.mkdir(parents=True, exist_ok=False)
    for name in layout()['run_directories']: (target/name).mkdir()
    shutil.copyfile(protocol, target/'protocol/design.json')
    record = dict(project='dsol-paper1', run_id=run_id, status='PREPARED_NOT_RUNNING',
        purpose=purpose, code_commit=revision, worktree_dirty=dirty,
        protocol_sha256=hashlib.sha256(content).hexdigest(), automatic_execution=False,
        remaining=['explicit scientific authorization', 'frozen source/assets', 'resource preflight', 'runtime admission'])
    (target/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('check'); audit.add_argument('--artifact-root')
    prepare = sub.add_parser('init-run')
    for name in ('artifact-root', 'run-id', 'protocol', 'purpose'): prepare.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        result = check(args.artifact_root); print(json.dumps(result, indent=2)); raise SystemExit(bool(result['errors']))
    print(init_run(args.artifact_root, args.run_id, args.protocol, args.purpose))

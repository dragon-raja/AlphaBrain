"""Runtime relocation checks, without starting a simulator or policy service."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.dsol_paper1.paths import ROOT


def receipt():
    return json.loads((ROOT / 'archive/repository/20260911/runtime-modules/manifest.json').read_text())


def test_all_37_runtime_files_relocated_without_flat_shims():
    records = receipt()['records']
    moved = [r for r in records if r['old'] != r['new']]
    assert len(moved) == 37
    for row in moved:
        assert not (ROOT / row['old']).exists()
        assert hashlib.sha256((ROOT / row['before']).read_bytes()).hexdigest() == row['before_sha256']
        assert (ROOT / row['new']).is_file()
    assert not list((ROOT / 'scripts/dsol_paper1').glob('*.py'))
    assert not list((ROOT / 'scripts/dsol_paper1').glob('*.sh'))
    for row in records:
        assert hashlib.sha256((ROOT / row['before']).read_bytes()).hexdigest() == row['before_sha256']
    ignored = subprocess.run(['git', 'check-ignore', '--stdin'], cwd=ROOT, capture_output=True, text=True,
                             input='\n'.join([r['new'] for r in moved] + [r['before'] for r in records]) + '\n')
    assert not ignored.stdout


def test_all_relocated_cli_bootstraps_point_to_this_checkout():
    for row in receipt()['records']:
        if row['old'] == row['new'] or not row['new'].endswith('.py') or row['new'].startswith('AlphaBrain/'):
            continue
        p = ROOT / row['new']
        tree = ast.parse(p.read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == '_REPOSITORY_ROOT' for t in n.targets))
        index = assignment.value.slice.value
        assert p.resolve().parents[index] == ROOT, row['new']


def test_extracted_accel_visibility_and_noise_bodies_are_exact():
    count = 0
    for row in receipt()['records']:
        if row['old'] == row['new'] or not row['new'].startswith('AlphaBrain/research/dsol/'):
            continue
        before = ast.parse((ROOT / row['before']).read_text())
        after = ast.parse((ROOT / row['new']).read_text())
        # Only import paths changed; all executable definitions and constants match.
        for tree in (before, after):
            tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), row['new']
        count += 1
    assert count == 4


@pytest.mark.parametrize('relative', [
    'scripts/dsol_paper1/analysis/analyze_standard_initial_results_v1.py',
    'scripts/dsol_paper1/analysis/rank_accel_candidates.py',
    'scripts/dsol_paper1/runtime/evaluate_dsol_libero_hdf5_views.py',
])
def test_reviewed_new_cli_help_from_unrelated_directory(relative, tmp_path):
    env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    run = subprocess.run([sys.executable, str(ROOT / relative), '--help'], cwd=tmp_path,
                         env=env, text=True, capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr
    assert 'usage:' in run.stdout
    assert not list(tmp_path.iterdir())

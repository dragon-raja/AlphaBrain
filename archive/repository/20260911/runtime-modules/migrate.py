"""Reviewed one-shot runtime relocation; never runs/imports research modules."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import sys
import os
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[4]
RECEIPT = Path(__file__).parent
BASE = 'scripts/dsol_paper1/'
DESTINATIONS = {
    'accel_core.py': 'AlphaBrain/research/dsol/metrics/accel.py',
    'accel_inference.py': 'AlphaBrain/research/dsol/metrics/accel_inference.py',
    'explicit_flow_noise.py': 'AlphaBrain/research/dsol/data/flow_noise.py',
    'libero_visibility.py': 'AlphaBrain/research/dsol/metrics/visibility.py',
    'analyze_standard_initial_results_v1.py': BASE + 'analysis/analyze_standard_initial_results_v1.py',
    'analyze_view_metric_rules_v1.py': BASE + 'analysis/analyze_view_metric_rules_v1.py',
    'analyze_view_oracle_comparison_v2.py': BASE + 'analysis/analyze_view_oracle_comparison_v2.py',
    'compare_matched_view_landscapes_v1.py': BASE + 'analysis/compare_matched_view_landscapes_v1.py',
    'rank_accel_candidates.py': BASE + 'analysis/rank_accel_candidates.py',
    'summarize_dsol_libero_hdf5_closed_loop.py': BASE + 'analysis/summarize_dsol_libero_hdf5_closed_loop.py',
    'audit_libero_hdf5_restore.py': BASE + 'runtime/audit_libero_hdf5_restore.py',
    'evaluate_dsol_libero_hdf5_views.py': BASE + 'runtime/evaluate_dsol_libero_hdf5_views.py',
    'libero_constructed_view.py': BASE + 'runtime/libero_constructed_view.py',
    'scan_libero_hdf5_views.py': BASE + 'runtime/scan_libero_hdf5_views.py',
    'standard_initialization_v1.py': BASE + 'runtime/standard_initialization_v1.py',
    'render_bridge_common_v1.py': BASE + 'runtime/render_bridge_common_v1.py',
    'shared_runtime_paths.py': BASE + 'runtime/shared_runtime_paths.py',
    'run_view_value_expectation_accel_ensemble.py': BASE + 'runtime/run_view_value_expectation_accel_ensemble.py',
    'run_matched_view_accel_v1.py': BASE + 'runtime/run_matched_view_accel_v1.py',
    'audit_statewise_view_oracle_v2_run.py': BASE + 'diagnostics/audit_statewise_view_oracle_v2_run.py',
    'trace_view_repeatability_v1.py': BASE + 'diagnostics/trace_view_repeatability_v1.py',
    'build_full64_view_landscape_v2.py': BASE + 'protocols/build_full64_view_landscape_v2.py',
    'build_matched_view_landscape_protocol_v1.py': BASE + 'protocols/build_matched_view_landscape_protocol_v1.py',
    'dualhost_aa0_v1.py': BASE + 'operations/controllers/dualhost_aa0_v1.py',
    'dynamic_initial_scheduler_v1.py': BASE + 'operations/controllers/dynamic_initial_scheduler_v1.py',
    'standard_initial_aa0_v1.py': BASE + 'operations/controllers/standard_initial_aa0_v1.py',
    'run_full64_view_landscape_v2.py': BASE + 'operations/controllers/run_full64_view_landscape_v2.py',
    'finalize_full64_view_landscape_v2.py': BASE + 'operations/controllers/finalize_full64_view_landscape_v2.py',
    'run_render_bridge_worker_v1.py': BASE + 'operations/controllers/run_render_bridge_worker_v1.py',
    'run_dsol_libero_hdf5_closed_loop_eval.sh': BASE + 'operations/launchers/run_dsol_libero_hdf5_closed_loop_eval.sh',
    'run_matched_view_landscape_v1.sh': BASE + 'operations/launchers/run_matched_view_landscape_v1.sh',
    'build_view_research_focus_brief.py': BASE + 'reporting/build_view_research_focus_brief.py',
    'report_standard_initial_results_v1.py': BASE + 'reporting/report_standard_initial_results_v1.py',
    'build_unified_research_progress_v1.py': 'reports/paper1/unified.py',
    'build_consolidated_view_analysis_v1.py': 'reports/paper1/historical/build_consolidated_view_analysis_v1.py',
    'build_view_landscape_report_v1.py': 'reports/paper1/historical/build_view_landscape_report_v1.py',
    'build_view_metric_comparison_brief_v1.py': 'reports/paper1/historical/build_view_metric_comparison_brief_v1.py',
}
MOVES = {BASE + name: value for name, value in DESTINATIONS.items()}
MODULES = {Path(old).stem: new[:-3].replace('/', '.') for old, new in MOVES.items() if old.endswith('.py')}
FROZEN_PATHS = {'dynamic_initial_scheduler_v1.py', 'standard_initial_aa0_v1.py', 'dualhost_aa0_v1.py'}


def rewrite_imports(text, source):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    lines = text.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        replacements = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                bare = name in MODULES
                if source.name == 'dynamic_initial_scheduler_v1.py' and name == 'standard_initial_aa0_v1':
                    # Deliberately loaded from the frozen release repo, not this checkout.
                    replacements.append('import ' + name + (' as ' + alias.asname if alias.asname else ''))
                    continue
                if bare:
                    replacements.append('import ' + MODULES[name] + ' as ' + (alias.asname or name))
                else:
                    for old, new in MODULES.items():
                        name = name.replace('scripts.dsol_paper1.' + old, new)
                    replacements.append('import ' + name + (' as ' + alias.asname if alias.asname else ''))
        else:
            module = node.module or ''
            if module == 'scripts.dsol_paper1':
                remaining = []
                for alias in node.names:
                    if alias.name in MODULES:
                        replacements.append('import ' + MODULES[alias.name] + ' as ' + (alias.asname or alias.name))
                    else:
                        remaining.append(alias.name + (' as ' + alias.asname if alias.asname else ''))
                if remaining:
                    replacements.append('from ' + module + ' import ' + ', '.join(remaining))
            else:
                new_module = MODULES.get(module, module)
                for old, new in MODULES.items():
                    new_module = new_module.replace('scripts.dsol_paper1.' + old, new)
                dots = '.' * node.level if new_module == module else ''
                replacements.append('from ' + dots + new_module + ' import ' + ', '.join(
                    a.name + (' as ' + a.asname if a.asname else '') for a in node.names))
        original = ''.join(lines[node.lineno-1:node.end_lineno])
        # Avoid normalizing unrelated imports (and preserve their comments).
        if not any(re.search(r'\b' + re.escape(name) + r'\b', original) for name in MODULES):
            continue
        if node.col_offset and lines[node.lineno-1][:node.col_offset].strip():
            raise ValueError('Inline import needs manual review: ' + str(source) + ':' + str(node.lineno))
        prefix = ' ' * node.col_offset
        lines[node.lineno-1:node.end_lineno] = [prefix + s + '\n' for s in replacements]
    return ''.join(lines)


def rewrite(text, old, new):
    original = text
    if old.suffix == '.py':
        text = rewrite_imports(text, old)
    if old.name not in FROZEN_PATHS:
        for source, dest in MOVES.items():
            # Source snapshots belong to their historical checkout, not this one.
            text = re.sub(r'(?<!repo/)' + re.escape(source), dest, text)
        for name, module in MODULES.items():
            text = text.replace('scripts.dsol_paper1.' + name, module)
    if old != new and old.suffix == '.py':
        if not str(new.relative_to(ROOT)).startswith('AlphaBrain/'):
            for name, dest in DESTINATIONS.items():
                text = re.sub(r'Path\(__file__\)\.with_name\(([\'\"])' + re.escape(name) + r'\1\)',
                              "(_REPOSITORY_ROOT / '" + dest + "')", text)
            text = re.sub(r'Path\(__file__\)(?:\.resolve\(\))?\.parents\[2\]', '_REPOSITORY_ROOT', text)
            text = re.sub(r'Path\(__file__\)\.resolve\(\)\.parents\[1\]', "(_REPOSITORY_ROOT / 'scripts')", text)
            text = text.replace('Path(__file__).resolve().parent', "(_REPOSITORY_ROOT / 'scripts/dsol_paper1')")
            # Fixed root bootstrap: no wildcard subdirectory import search.
            tree = ast.parse(text)
            pos = 0
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                pos = tree.body[0].end_lineno
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                    pos = node.end_lineno
            if pos == 0 and text.startswith('#!'):
                pos = 1
            lines = text.splitlines(keepends=True)
            depth = len(new.relative_to(ROOT).parts) - 1
            lines.insert(pos, f'\nfrom pathlib import Path as _RepoPath\nimport sys as _sys\n_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[{depth}]\nif str(_REPOSITORY_ROOT) not in _sys.path:\n    _sys.path.insert(0, str(_REPOSITORY_ROOT))\n\n')
            text = ''.join(lines)
    if old.suffix == '.sh':
        if old != new:
            climb = '/'.join(['..'] * (len(new.relative_to(ROOT).parts) - 1))
            text = text.replace('$(dirname "$0")/../..', '$(dirname "$0")/' + climb)
        for name, module in MODULES.items():
            text = re.sub(r'\bfrom ' + name + r' import ', 'from ' + module + ' import ', text)
    return text


def main():
    actual = {str(p.relative_to(ROOT)) for p in (ROOT / BASE).iterdir() if p.suffix in {'.py', '.sh'}}
    assert actual == set(MOVES), (actual - set(MOVES), set(MOVES) - actual)
    for target in MOVES.values():
        assert not (ROOT / target).exists(), target
    names = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    changes = []
    for relative in sorted(set(names)):
        old = ROOT / relative
        if not old.is_file() or old.is_symlink() or old.suffix not in {'.py', '.sh'}:
            continue
        if relative.startswith(('archive/', 'configs/', 'schemas/')):
            continue
        new = ROOT / MOVES.get(relative, relative)
        before = old.read_bytes()
        after = rewrite(before.decode(), old, new).encode()
        if before != after or old != new:
            changes.append((old, new, before, after))
    # All transformations and collisions checked before first source mutation.
    records = []
    for old, new, before, after in changes:
        backup = RECEIPT / 'before' / old.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(before)
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_bytes(after)
        new.chmod(old.stat().st_mode)
        if old != new:
            old.unlink()
        records.append(dict(old=str(old.relative_to(ROOT)), new=str(new.relative_to(ROOT)),
                            before=str(backup.relative_to(ROOT)), before_sha256=hashlib.sha256(before).hexdigest(),
                            after_sha256=hashlib.sha256(after).hexdigest()))
    (RECEIPT / 'manifest.json').write_text(json.dumps(dict(
        scope='Runtime relocation after user-authorized retirement of two old controllers; frozen releases unchanged',
        records=records), indent=2) + '\n')
    print(json.dumps({'moved': len(MOVES), 'changed_consumers_and_sources': len(records)}))


def links():
    path = RECEIPT / 'manifest.json'
    manifest = json.loads(path.read_text())
    names = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    test_rows = json.loads((ROOT / 'archive/repository/20260911/test-modules/manifest.json').read_text())['records']
    moves = {**MOVES, **{r['old']: r['new'] for r in test_rows}}
    for name in sorted(set(names)):
        p = ROOT / name
        if p.suffix != '.md' or name.startswith('archive/') or not p.is_file() or p.is_symlink():
            continue
        before = p.read_bytes()
        def replace(match):
            token = match[1]
            if token.startswith(('#', 'http:', 'https:', 'codex:', '<')):
                return match[0]
            location, sep, anchor = token.partition('#')
            target = Path(os.path.normpath(str(p.parent / unquote(location))))
            if not target.is_relative_to(ROOT):
                return match[0]
            relative = str(target.relative_to(ROOT))
            if relative not in moves:
                return match[0]
            return '](' + os.path.relpath(ROOT / moves[relative], p.parent) + (sep + anchor if sep else '') + ')'
        after = re.sub(r'\]\(([^\n)]*)\)', replace, before.decode()).encode()
        if before == after:
            continue
        backup = RECEIPT / 'before' / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(before)
        p.write_bytes(after)
        manifest['records'].append(dict(old=name, new=name, before=str(backup.relative_to(ROOT)),
            before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest()))
    path.write_text(json.dumps(manifest, indent=2) + '\n')


def seal():
    path = RECEIPT / 'manifest.json'
    manifest = json.loads(path.read_text())
    for row in manifest['records']:
        assert hashlib.sha256((ROOT / row['before']).read_bytes()).hexdigest() == row['before_sha256']
        row['after_sha256'] = hashlib.sha256((ROOT / row['new']).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    {'--links': links, '--seal': seal}.get(sys.argv[1] if len(sys.argv) > 1 else '', main)()

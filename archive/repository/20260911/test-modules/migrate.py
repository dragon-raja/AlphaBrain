"""One-shot, reviewed mechanical test migration; never imports experiment code."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'tests/dsol_paper1'
GROUPS = {
    'acquisition': '''test_view_acquisition_executor test_view_acquisition_inputs
        test_view_acquisition_motion test_view_acquisition_protocol''',
    'architecture': '''test_core_architecture test_repository_layout test_script_migration''',
    'data': '''test_schema_contracts test_libero_constructed_view libero_visibility_test
        test_libero_view_scan_geometry test_libero_view_catalog test_libero_eval_sensor_controls
        explicit_flow_noise_test test_standard_initial_aa0_v1''',
    'metrics': '''accel_core_test accel_inference_test accel_cli_test test_matched_view_accel_v1
        join_constructed_accel_m1_test test_view_metric_rules_v1 test_analyze_matched_view_metric_rules_v1''',
    'selectors': '''test_statewise_view_selector_v1 test_dense_test_selector_summary
        select_constructed_m0_candidates_test test_dense_selector_common_failures''',
    'analysis': '''analyze_m_b_multiseed_test test_compare_libero_m1_models
        test_compare_matched_view_landscapes_v1 test_libero_closed_loop_summary
        test_summarize_libero_visibility_scan test_statewise_view_oracle_v2_analysis
        test_view_value_expectation_heldout test_view_repeatability_summary
        view_value_discovery_test test_initial_refactor_golden''',
    'protocols': '''test_protocol test_dense_test_selector_protocol test_dense_test_scan_plan
        build_constructed_view_oracle_protocol_test build_constructed_m1_protocol_test
        build_matched_view_landscape_protocol_v1_test test_freeze_constructed_task_pairs
        test_statewise_view_oracle_v2_builder test_statewise_view_oracle_v2_stage
        test_statewise_view_oracle_v2_reserve test_expand_protocol_noise_repeats
        test_build_strong_information_gate_plan test_build_visibility_selected_strong_gate_protocol
        test_build_prefreeze_bundle test_eval_throughput_benchmark test_plan_precision_power
        view_repeatability_test accel_gate_a97_test test_render_protocol_bridge_v1''',
    'execution': '''test_failure_view_search test_dynamic_initial_scheduler_v1 test_dualhost_aa0_v1
        test_full64_view_landscape_v2 test_matched_view_landscape_controller_v1
        test_statewise_view_oracle_v2_controller test_statewise_view_oracle_v2_tail_controller
        test_view_value_expectation_calibration_controller test_view_value_expectation_post_controller
        test_finalize_canonical_match_v1 finalize_constructed_m0_manual_audit_test
        run_constructed_accel_bank_test test_run_protocol_smoke accel_checkpoint_smoke_test
        test_constructed_blind_reveal test_render_bridge_recovery_v1''',
    'diagnostics': '''test_capture_runtime_identity test_view_repeatability_trace_v1
        test_view_speed_equivalence_v1 test_dsol_training_match_audit
        test_statewise_view_oracle_v2_audit audit_view_value_expectation_protocol_test
        audit_view_value_expectation_heldout_run_test test_view_value_expectation_completion_audit
        test_view_value_expectation_calibration_validation render_constructed_m0_audit_montages_test''',
    'reporting': '''test_consolidated_view_analysis_v1 build_view_landscape_report_v1_test
        test_unified_research_scope_v1 test_view_value_expectation_report''',
}


def main():
    mapping = {}
    for group, names in GROUPS.items():
        for name in names.split():
            filename = 'test_' + name[:-5] if name.endswith('_test') else name
            old, new = BASE / (name + '.py'), BASE / group / (filename + '.py')
            assert old not in mapping and not new.exists(), (old, new)
            mapping[old] = new
    actual = set(BASE.glob('*.py')) - {BASE / 'conftest.py', BASE / 'paths.py'}
    assert actual == set(mapping), (actual - set(mapping), set(mapping) - actual)
    baseline = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    records = []
    for old, new in mapping.items():
        before = old.read_bytes()
        text = before.decode()
        text = re.sub(r'Path\(__file__\)(?:\.resolve\(\))?\.parents\[2\]',
                      'REPOSITORY_ROOT', text)
        if 'REPOSITORY_ROOT' in text:
            lines = text.splitlines(keepends=True)
            # Insert after future imports, before all executable imports/code.
            indices = [i for i, line in enumerate(lines) if line.startswith(('import ', 'from '))
                       and not line.startswith('from __future__')]
            lines.insert(indices[0], 'from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT\n')
            text = ''.join(lines)
        text = text.replace('tests.dsol_paper1.test_compare_matched_view_landscapes_v1',
                            'tests.dsol_paper1.helpers.landscape')
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_text(text)
        new.chmod(old.stat().st_mode)
        old.unlink()
        records.append(dict(old=str(old.relative_to(ROOT)), new=str(new.relative_to(ROOT)),
                            before_sha256=hashlib.sha256(before).hexdigest(),
                            after_sha256=hashlib.sha256(new.read_bytes()).hexdigest()))
    (Path(__file__).parent / 'manifest.json').write_text(json.dumps(dict(
        baseline_commit=baseline, recovery='git show <baseline_commit>:<old>; restore in a separate checkout',
        scope='Test relocation, uniform test_ names and explicit repository-root imports; no experiment changes',
        records=records), indent=2) + '\n')
    print(json.dumps({'moved': len(records), 'groups': {g: len(n.split()) for g, n in GROUPS.items()}}))


def seal():
    """Seal the new (unpublished) receipt after import/path acceptance fixes."""
    path = Path(__file__).parent / 'manifest.json'
    manifest = json.loads(path.read_text())
    for row in manifest['records']:
        before = subprocess.check_output(['git', 'show', manifest['baseline_commit'] + ':' + row['old']], cwd=ROOT)
        assert hashlib.sha256(before).hexdigest() == row['before_sha256']
        source = re.sub(r'Path\(__file__\)(?:\.resolve\(\))?\.parents\[2\]', 'REPOSITORY_ROOT', before.decode())
        row['test_bodies'] = {
            str(index) + ':' + node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for index, node in enumerate(n for n in ast.walk(ast.parse(source))
                                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_'))
        }
        row['after_sha256'] = hashlib.sha256((ROOT / row['new']).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    seal() if sys.argv[1:] == ['--seal'] else main()

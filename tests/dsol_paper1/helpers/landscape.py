"""Synthetic matched-landscape fixtures shared across test modules."""
import copy
import numpy as np
from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
from scripts.dsol_paper1.analysis.compare_matched_view_landscapes_v1 import BANK_FILE_SHA256, BANK_MANIFEST_SHA256, CHECKPOINTS, catalog_bank, read_json

def fixtures():
    root = REPOSITORY_ROOT / 'configs/dsol_paper1'
    catalog = read_json(root / 'libero_view_catalog_v2_m1.json')
    rules = read_json(root / 'libero_view_catalog_v2_m1_rules.json')
    selected = []
    for i in range(8):
        selected.append({'pair_key': f'state-{i}', 'task_id': f'task-{i}', 'source_group': f'source-{i}',
            'asset_source_pair_key': f'asset-{i}', 'source_state_index': i,
            'environment_seed': i+100, 'split': 'development', 'construction_spec_sha256': f'construction-{i}',
            'demo_name': f'demo_{i}', 'static_assets': {'physics_state_sha256': f'physics-{i}',
            'policy_inputs_sha256': f'images-{i}', 'render_receipt_sha256': f'render-{i}', 'visibility_scan_sha256': f'visibility-{i}'}})
    reports = []
    for label in ('canonical', 'broad'):
        poses, _ = catalog_bank(copy.deepcopy(catalog), rules, 'canonical' if label == 'canonical' else 'broad64')
        states = copy.deepcopy(selected)
        for i, state in enumerate(states):
            state['accel_ensemble_seeds'] = [i*10+j for j in range(8)]
            state['accel_action_horizon'] = 10
        success = np.zeros((8, 97, 32), dtype=np.int8)
        for j in range(97):
            success[:, j, :j % 33] = 1
        accel = np.broadcast_to(np.arange(97)[None, :, None], (8, 97, 8)).copy().astype(float)
        reports.append({'states': states, 'poses': poses, 'state_keys': [s['pair_key'] for s in states],
            'candidate_ids': [p['pose_id'] for p in poses], 'success': success, 'accel': accel,
            'visibility': np.zeros((8, 97)), 'provenance': {'checkpoint_sha256': CHECKPOINTS[label],
                'dense': {'bank_manifest_sha256': BANK_MANIFEST_SHA256, 'bank_file_sha256': BANK_FILE_SHA256}}})
    return reports[0], reports[1], {'states': selected, 'outcomes_or_view_metrics_used_in_selection': False}

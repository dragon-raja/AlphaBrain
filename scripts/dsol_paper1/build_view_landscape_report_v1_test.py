from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_view_landscape_report_v1 import catalog_bank, clustered_summary, load_dense, ranks, read_json, sha256, spearman, state_metrics, wilson


class ViewLandscapeTests(unittest.TestCase):
    def test_average_ranks_and_constant_outcome(self):
        np.testing.assert_equal(ranks(np.array([3, 1, 1, 2])), [4, 1.5, 1.5, 3])
        self.assertAlmostEqual(spearman(np.array([1, 2, 3]), np.array([6, 4, 2])), -1)
        self.assertTrue(np.isnan(spearman(np.ones(3), np.arange(3))))

    def test_binomial_interval_at_boundary(self):
        low, high = wilson(0, 32)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, .1)
        low, high = wilson(32, 32)
        self.assertLess(low, .9)
        self.assertAlmostEqual(high, 1)

    def test_frozen_catalog_and_support_distance(self):
        root = Path(__file__).resolve().parents[2] / 'configs/dsol_paper1'
        poses, coords = catalog_bank(read_json(root / 'libero_view_catalog_v2_m1.json'), read_json(root / 'libero_view_catalog_v2_m1_rules.json'))
        self.assertEqual(coords.shape, (97, 3))
        self.assertEqual(poses[0]['orientation_mode'], 'original_camera_pose')
        self.assertEqual(sum(p['catalog_group'] == 'train64' for p in poses), 64)
        self.assertTrue(all(p['distance_train64_parameter_normalized'] == 0 for p in poses if p['catalog_group'] == 'train64'))
        self.assertTrue(all(p['distance_train64_parameter_normalized'] > 0 for p in poses if p['catalog_group'] == 'heldout32'))
        canonical_poses, _ = catalog_bank(read_json(root / 'libero_view_catalog_v2_m1.json'), read_json(root / 'libero_view_catalog_v2_m1_rules.json'), 'canonical')
        self.assertEqual(sum(p['is_model_post_training_support'] for p in canonical_poses), 1)
        self.assertTrue(canonical_poses[0]['is_model_post_training_support'])
        np.testing.assert_allclose([p['distance_model_support_parameter_normalized'] for p in canonical_poses], [p['distance_canonical_parameter_normalized'] for p in canonical_poses], rtol=1e-14)
        self.assertEqual([p['pose_id'] for p in canonical_poses], [p['pose_id'] for p in poses])

    def test_source_equal_task_equal_not_row_equal(self):
        rows = [{'task_id': 'a', 'source_group': 'a1', 'x': 1}] * 9 + [
            {'task_id': 'a', 'source_group': 'a2', 'x': 0}, {'task_id': 'b', 'source_group': 'b1', 'x': 0}]
        result = clustered_summary(rows, 'x', resamples=100)
        self.assertAlmostEqual(result['mean'], .25)
        self.assertEqual(result['finite_sources'], 3)
        self.assertEqual(result['ci95'], [None, None])
        self.assertEqual(result['interval_status'], 'UNSUPPORTED_INSUFFICIENT_SOURCES_PER_TASK')

    def test_state_view_noise_axes_and_negative_accel_sign(self):
        root = Path(__file__).resolve().parents[2] / 'configs/dsol_paper1'
        poses, _ = catalog_bank(read_json(root / 'libero_view_catalog_v2_m1.json'), read_json(root / 'libero_view_catalog_v2_m1_rules.json'))
        state = {'pair_key': 's', 'task_id': 't', 'split': 'development', 'source_group': 'src', 'demo_name': 'demo_1', 'source_state_index': 0}
        accel = np.repeat(np.arange(97)[None, :, None], 8, axis=2).astype(float)
        success = np.zeros((1, 97, 32), dtype=np.int8)
        for j in range(97): success[0, j, :int((96-j)/3)] = 1
        summaries, cells = state_metrics([state], poses, accel, success, np.ones((1, 97)))
        self.assertEqual(len(cells), 97)
        self.assertGreater(summaries[0]['spearman_negative_accel_success'], .99)
        self.assertEqual(summaries[0]['noise_top1_all_agree'], 1)
        self.assertEqual(cells[0]['success_count'], 32)
        self.assertEqual(cells[0]['repeat_count'], 32)
        self.assertEqual(cells[96]['success_count'], 0)
        self.assertEqual(summaries[0]['accel_selected_id'], 'canonical')

    def test_noise_crossfit_does_not_reuse_selection_half(self):
        root = Path(__file__).resolve().parents[2] / 'configs/dsol_paper1'
        poses, _ = catalog_bank(read_json(root / 'libero_view_catalog_v2_m1.json'), read_json(root / 'libero_view_catalog_v2_m1_rules.json'))
        state = {'pair_key': 's', 'task_id': 't', 'split': 'development', 'source_group': 'src', 'demo_name': 'demo_1', 'source_state_index': 0}
        accel = np.repeat(np.arange(97)[None, :, None], 8, axis=2).astype(float)
        success = np.zeros((1, 97, 32), dtype=np.int8)
        success[0, 0, :8] = 1
        success[0, 1, :16] = 1
        success[0, 2, 16:] = 1
        summaries, _ = state_metrics([state], poses, accel, success, np.ones((1, 97)))
        self.assertEqual(summaries[0]['noise_crossfit_search_success'], 0)
        self.assertEqual(summaries[0]['noise_crossfit_gain_vs_canonical_pp'], -25)
        self.assertEqual(summaries[0]['empirical_max_success_diagnostic'], .5)

    def test_partial_dense_includes_smoke_and_remainder_and_rejects_holes(self):
        config = Path(__file__).resolve().parents[2] / 'configs/dsol_paper1'
        poses, _ = catalog_bank(read_json(config / 'libero_view_catalog_v2_m1.json'), read_json(config / 'libero_view_catalog_v2_m1_rules.json'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            noise = root / 'noise.bin'; noise.write_bytes(b'fixture')
            bank_path = root / 'bank.json'
            bank_path.write_text(json.dumps({'bank_id': 'O', 'repeat_count': 32, 'noise_file': str(noise), 'noise_file_sha256': sha256(noise)}))
            bank_hash = sha256(bank_path)
            checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            state = {'pair_key': 's', 'asset_source_pair_key': 'old-s', 'environment_seed': 4,
                     'static_assets': {'physics_state_sha256': 'physical', 'policy_inputs_sha256': 'images'}}
            for name, repeat_ids in [('smoke', range(16)), ('wave-00-remainder-a', range(16, 32))]:
                segment = root / 'O' / name; segment.mkdir(parents=True)
                protocol = segment / 'protocol.json'; protocol.write_text('{}')
                manifest_path = segment / 'run_manifest.json'
                manifest_path.write_text(json.dumps({'checkpoint': str(checkpoint), 'checkpoint_sha256': 'weights',
                    'noise_bank_manifest_sha256': bank_hash, 'replan_steps': 5, 'wait_steps': 0}))
                rows = []
                for j, pose in enumerate(poses):
                    for repeat in repeat_ids:
                        rows.append({'status': 'complete', 'episode_id': f'{j}-{repeat}', 'pair_key': 's',
                            'selected_candidate_id': pose['pose_id'], 'policy_repeat_id': repeat,
                            'asset_source_pair_key': 'old-s', 'noise_bank_id': 'O', 'noise_bank_manifest_sha256': bank_hash,
                            'explicit_flow_noise': True, 'initial_metrics': {'physics_state_sha256': 'physical'},
                            'static_assets': {'policy_inputs_sha256': 'images'}, 'environment_seed': 4,
                            'pose': pose if j else None, 'candidate_features': {'visibility_score': .1}, 'success': repeat % 2 == 0})
                ledger = segment / 'episodes-shard-00.jsonl'
                ledger.write_text('\n'.join(map(json.dumps, rows)) + '\n')
                audit = {'status': 'PASS_COMPLETE', 'every_policy_call_matches_frozen_noise_bank': True,
                    'paired_noise_identity_at_common_replan_indices': True, 'physics_hash_constant_within_state': True,
                    'environment_seed_constant_within_state': True, 'run_manifest_sha256': sha256(manifest_path),
                    'protocol': str(protocol), 'protocol_sha256': sha256(protocol), 'noise_bank_manifest_sha256': bank_hash,
                    'episode_count': len(rows)}
                (segment / 'audit.json').write_text(json.dumps(audit))
            args = SimpleNamespace(rollout_root=root, dense_dir=root/'O', noise_bank_manifest=bank_path, checkpoint=checkpoint, selection_manifest=root/'selection.json')
            success, _, receipt = load_dense(args, [state], poses, 'weights')
            self.assertEqual(success.shape, (1, 97, 32))
            self.assertEqual(success.sum(), 97*16)
            self.assertEqual(receipt['waves'], 2)
            # Deliberately ignore the smoke segment: strict 97x32 completeness must fail.
            (root/'O/smoke/audit.json').rename(root/'O/smoke/audit.removed.json')
            with self.assertRaisesRegex(ValueError, 'Missing dense cells'):
                load_dense(args, [state], poses, 'weights')


if __name__ == '__main__':
    unittest.main()

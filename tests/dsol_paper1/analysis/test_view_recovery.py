"""Input permissions and exact endpoint replay checks (no simulator)."""
import json
from pathlib import Path
import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.view_recovery import outcome_summary, metric_reference_summary
from AlphaBrain.research.dsol.acquisition.discrete_search import camera_graph
from AlphaBrain.research.dsol.selectors.query_ranker import QueryRanker


class RecoveryTests(unittest.TestCase):
    def test_single_noise_is_not_mean_noise_or_success_selected_seed(self):
        scores = np.full((2, 16, 97, 8), 200.)
        scores[:, :, 0, :] = [0, 0, 0, 0, 0, 0, 0, 100]
        scores[:, :, 1, :] = 1.
        success = np.zeros((2, 16, 97))
        success[:, :, 1] = 1.
        visibility = np.broadcast_to(np.arange(97), (16, 97))
        result = metric_reference_summary(success, scores, visibility, np.arange(16))
        rows = result['models']['canonical']
        self.assertEqual(rows['single_accel']['heldout32']['success']['mean'], .125)
        self.assertEqual(rows['min_accel']['heldout32']['success']['mean'], 1.)
        self.assertEqual(rows['stay']['heldout32']['success']['mean'], 0.)

    def test_bootstrap_units_are_initials_not_start_poses(self):
        baseline = np.zeros((16, 32))
        values = np.broadcast_to(np.arange(16)[:, None] / 16., baseline.shape)
        first = outcome_summary(values, baseline, baseline, baseline, slice(None))
        second = outcome_summary(np.repeat(values, 2, axis=1), np.repeat(baseline, 2, axis=1),
                                 np.repeat(baseline, 2, axis=1), np.repeat(baseline, 2, axis=1), slice(None))
        self.assertEqual(first['success'], second['success'])

    def test_local_archived_replay_integrity(self):
        repo = Path(__file__).resolve().parents[3]
        cfg = json.loads((repo / 'configs/dsol_paper1/analysis/view_recovery_v1.json').read_text())
        pointer = Path(cfg['output']) / 'latest.json'
        if not pointer.exists():
            self.skipTest('Private recovery replay unavailable')
        receipt = json.loads(pointer.read_text())
        out, source = Path(receipt['archive']), Path(cfg['audited_analysis'])
        with np.load(source / 'matrix.npz') as z:
            q, a, camera = z['success'].mean(-1), z['accel_members'], z['camera']
        states = json.loads((source / 'states.json').read_text())
        test = np.array([i for i, s in enumerate(states) if s['initialization']['init_state_index'] >= 2])
        with np.load(source / 'embeddings.npz') as z:
            images, wrists = z['external'], z['context'][:, 512:]
        for m, name in enumerate(['canonical', 'broad']):
            with np.load(out / f'{name}-candidate-scores.npz') as z:
                scores = dict(z)
            with np.load(out / f'{name}-rgb_accel-weights.npz') as z:
                ranker = QueryRanker(**dict(z))
            i = int(test[0])
            predicted = ranker.score(images[i, 65], wrists[i], states[i]['task_ordinal'], a[m, i, 65])
            self.assertAlmostEqual(float(predicted), float(scores['rgb_accel'][0, 65]), places=10)
            for method, family in [('local_accel', 'accel'), ('random_accel', 'accel'),
                                   ('local_rgb', 'rgb'), ('local_rgb_accel', 'rgb_accel'),
                                   ('random_rgb_accel', 'rgb_accel')]:
                with np.load(out / f'{name}-{method}-replay.npz') as z:
                    selected, visited, success = z['selected'], z['visited'], z['success']
                expected = q[m, test[:, None, None, None], selected]
                np.testing.assert_array_equal(success, expected)
                np.testing.assert_array_equal(selected[..., 0], np.broadcast_to(np.arange(1, 97)[None, :, None], selected[..., 0].shape))
                for bi, budget in enumerate(cfg['budgets']):
                    prefix = visited[..., :budget]
                    self.assertTrue(np.all(np.any(prefix == selected[..., bi, None], axis=-1)))
                    best = scores[family][np.arange(16)[:, None, None, None], prefix].min(axis=-1)
                    chosen = scores[family][np.arange(16)[:, None, None], selected[..., bi]]
                    np.testing.assert_array_equal(chosen, best)
                self.assertTrue(np.all(np.diff(np.sort(visited, axis=-1), axis=-1) > 0))
        examples = json.loads((out / 'trace-examples.json').read_text())
        for example in examples:
            edges, _, d, r = camera_graph(camera[example['initial']], cfg['neighbors'], cfg['translation_scale_m'], cfg['rotation_scale_deg'])
            for b, trace in example['traces'].items():
                self.assertEqual(len(trace['visited']), int(b))
                self.assertEqual(trace['path'][0], example['start'])
                self.assertEqual(trace['path'][-1], trace['selected'])
                self.assertTrue(all(edges[x, y] for x, y in zip(trace['path'][:-1], trace['path'][1:])))
                self.assertTrue(set(trace['path']).issubset(trace['visited']))


if __name__ == '__main__':
    unittest.main()

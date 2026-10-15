import unittest

from AlphaBrain.research.dsol.data.rig_training import candidate_catalog, render_jobs


class RigTrainingPlanTest(unittest.TestCase):
    def test_catalog_and_episode_stability(self):
        catalog = candidate_catalog()
        self.assertEqual(len(catalog['views']), 97)
        self.assertEqual(len({tuple(r['parameters']) for r in catalog['views']}), 97)
        self.assertEqual(sum(r['role'] == 'train' for r in catalog['views']), 64)
        ledger = [dict(training_pair_index=i, exposure_indices=[2*i, 2*i+1],
                       source_window_id=str(i), action_horizon=50, dataset_identity='task',
                       episode_index=i//2, frame_index=i*10) for i in range(4)]
        jobs = render_jobs(ledger, catalog)
        self.assertEqual(jobs, render_jobs(ledger, candidate_catalog()))
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertEqual(len(job['rows']), 2)
            self.assertEqual(len(set(job['view_ids'])), 2)
            self.assertTrue(all(v.startswith('train_') for v in job['view_ids']))
        ledger[0]['action_horizon'] = 10
        with self.assertRaises(ValueError):
            render_jobs(ledger, catalog)

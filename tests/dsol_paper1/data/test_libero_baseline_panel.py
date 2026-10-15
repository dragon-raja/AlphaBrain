import unittest

from AlphaBrain.research.dsol.data.libero_baseline_panel import select_candidates, panel_indices


class LiberoBaselinePanelTests(unittest.TestCase):
    def candidates(self):
        rows = [dict(selected_candidate_id='canonical', pose=None)]
        for i in range(96):
            rows.append(dict(selected_candidate_id=('broad_train_' if i < 64 else 'broad_heldout_')+str(i),
                             pose=dict(azimuth_deg=i-48, elevation_deg=i%25-12, radius_scale=1)))
        return rows

    def test_distinct_and_heldout(self):
        candidates = self.candidates()
        selected = select_candidates(candidates)
        self.assertEqual(selected[0], 0)
        self.assertEqual(len(set(selected)), 5)
        self.assertTrue(all(i >= 65 for i in selected[1:]))
        self.assertEqual(selected, select_candidates(candidates))

    def test_exact_bounded_indices(self):
        candidates = self.candidates()
        release = dict(states=[dict(task_id=str(i//4)) for i in range(32)], candidate_count=97,
                       noise_repeats=32, protocol=dict(state_blocks=[dict(candidates=candidates)]*32))
        indices = panel_indices(release)
        self.assertEqual(len(indices), 640)
        self.assertEqual(len(set(indices)), 640)
        self.assertEqual({i%32 for i in indices}, {0, 1, 2, 3})
        self.assertEqual({i//3104 for i in indices}, set(range(32)))
        shards = [indices[worker::16] for worker in range(16)]
        self.assertTrue(all(len(shard) == 40 for shard in shards))
        self.assertEqual(sorted(index for shard in shards for index in shard), sorted(indices))

    def test_refuse_reduced_task_population(self):
        with self.assertRaises(ValueError):
            panel_indices(dict(states=[], candidate_count=97, noise_repeats=32))


if __name__ == '__main__':
    unittest.main()

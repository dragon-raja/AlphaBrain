import unittest
import numpy as np
from AlphaBrain.research.dsol.acquisition.discrete_search import search, camera_graph


class DiscreteSearchTests(unittest.TestCase):
    def test_query_budget_path_and_incumbent(self):
        edges = np.abs(np.arange(5)[:, None] - np.arange(5)) == 1
        distance = np.abs(np.arange(5)[:, None] - np.arange(5)).astype(float)
        calls = []
        def query(i):
            calls.append(i)
            return [3, 2, 0, 1, 5][i]
        results = search(edges, distance, 0, [1, 3, 5], query)
        self.assertEqual(calls, [0, 1, 2, 3, 4])
        self.assertEqual(results[1]['selected'], 0)
        self.assertEqual(results[5]['selected'], 2)
        self.assertEqual(results[5]['path'][-1], 2)
        for a, b in zip(results[5]['path'][:-1], results[5]['path'][1:]):
            self.assertTrue(edges[a, b])
        tie = search(edges, distance, 3, [5], lambda i: 0.)
        self.assertEqual(tie[5]['selected'], 3)

    def test_unvisited_scores_not_requested(self):
        edges = ~np.eye(5, dtype=bool)
        calls = []
        def query(i):
            calls.append(i)
            if len(calls) > 2:
                raise AssertionError('Budget leak')
            return float(i)
        r = search(edges, edges.astype(float), 4, [1, 2], query)
        self.assertEqual(len(calls), 2)
        self.assertIn(r[2]['selected'], calls)

    def test_graph_invariant_to_world_translation(self):
        c = np.broadcast_to(np.eye(4), (8, 4, 4)).copy()
        c[:, 0, 3] = np.arange(8) * .03
        g = camera_graph(c, 2)
        c[:, :3, 3] += [7., -3., 2.]
        h = camera_graph(c, 2)
        np.testing.assert_allclose(g[1], h[1], atol=1e-10)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.accel_subspace import trace_subspaces, panel_cell
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


class AccelSubspaceTests(unittest.TestCase):
    def test_internal_padding_does_not_enter_active_scores(self):
        x=np.random.default_rng(4).normal(size=(8,10,50,32))
        a,e=trace_subspaces(x)
        changed=x.copy();changed[...,12:]*=3
        b,_=trace_subspaces(changed)
        np.testing.assert_allclose(a[1:3],b[1:3])
        self.assertFalse(np.allclose(a[0],b[0]))
        self.assertTrue(np.all((e>0)&(e<1)))

    def test_executed_window_and_prefix(self):
        x=np.random.default_rng(8).normal(size=(8,10,50,32))
        a,_=trace_subspaces(x, prefix=8)
        expected=compute_accel_scores(x[:,:,:25,:12],prefixes=(8,))[8]['score']
        np.testing.assert_allclose(a[2],expected)
        x[:,:,25:,:]*=4
        b,_=trace_subspaces(x,prefix=8)
        np.testing.assert_allclose(a[2],b[2])

    def test_reused_official_cell(self):
        plan=dict(reuse_views=['canonical'],baseline_plan='/old/plan.json',
                  workers=[dict(worker_id='new',initials=['i'])])
        old=dict(workers=[dict(worker_id='old',initials=['i'])])
        self.assertEqual(str(panel_cell('/new',plan,old,'i','canonical')),
                         '/old/workers/old/official/cells/i/canonical')
        self.assertEqual(str(panel_cell('/new',plan,old,'i','extra')),
                         '/new/workers/new/official/cells/i/extra')


if __name__=='__main__':
    unittest.main()

import unittest
import numpy as np

from AlphaBrain.research.dsol.analysis.robocasa_statistics import (
    empirical_hierarchy, interval, layout_interval, fit_heldout_ranker, correlation,
)
from AlphaBrain.research.dsol.selectors.query_ranker import fit_ranker


class RoboCasaStatisticsTests(unittest.TestCase):
    def test_oracle_averages_noise_before_selection(self):
        y=np.array([[[1,0],[0,1]], [[0,0],[1,1]]])
        h=empirical_hierarchy(y.mean(-1),np.array([0,1]))
        self.assertEqual(h['initial_oracle'],.75)
        self.assertNotEqual(h['initial_oracle'],y.max(1).mean())

    def test_interval_does_not_count_views_as_initials(self):
        result=interval(np.repeat(.25,8),np.repeat([0,1],4),repeats=100)
        self.assertEqual(result['mean'],.25)
        self.assertEqual(result['ci95'],[.25,.25])
        self.assertEqual(result['initials'],8)

    def test_flat_success_correlation_is_undefined(self):
        self.assertIsNone(correlation(np.arange(5),np.ones(5)))
        self.assertAlmostEqual(correlation(np.arange(5),np.arange(5)),1.)

    def test_rank_correlation_uses_relative_order_not_absolute_scale(self):
        accel=np.array([.1,.2,.3,.4,.5])
        success=np.array([1.,.75,.75,.25,0.])
        expected=correlation(-accel,success)
        for relative in (accel*100+7, accel-accel.mean(), accel-accel[0],
                         (accel-accel.min())/np.ptp(accel)):
            self.assertAlmostEqual(correlation(-relative,success),expected)

    def test_layout_cluster_interval_preserves_shared_groups(self):
        result=layout_interval(np.repeat(.2,8),np.tile([0,1],4),np.repeat([1,2,3,4],2),repeats=100)
        self.assertEqual(result['layouts'],4)
        self.assertEqual(result['valid_draws'],100)
        np.testing.assert_allclose(result['ci95'],[.2,.2])

    def test_dynamic_task_count_and_candidate_wrist(self):
        rng=np.random.default_rng(2)
        image=rng.normal(size=(10,5,7));wrist=rng.normal(size=(10,5,6))
        acc=rng.normal(size=(10,5,8));labels=rng.random((10,5));tasks=np.arange(10)
        model=fit_ranker(image,wrist,tasks,acc,labels,use_accel=True,alpha=10,image_dim=2,wrist_dim=2,num_tasks=10)
        self.assertEqual(model.score(image,wrist,np.broadcast_to(tasks[:,None],labels.shape),acc).shape,(10,5))

    def test_target_labels_cannot_change_fit_or_choices(self):
        rng=np.random.default_rng(7)
        image=rng.normal(size=(8,5,7));wrist=rng.normal(size=(8,5,6));acc=rng.random((8,5,8))
        q=rng.random((8,5));tasks=np.tile([0,1],4);layouts=np.repeat([11,12,1,2],2)
        splits=np.repeat(['pretrain','target'],4)
        cfg=dict(alpha_grid=[1.,10.],image_pca_dim=2,wrist_pca_dim=2)
        a,_,ra=fit_heldout_ranker(image,wrist,tasks,acc,q,splits,layouts,cfg,use_accel=True)
        changed=q.copy();changed[4:]=1-changed[4:]
        b,_,rb=fit_heldout_ranker(image,wrist,tasks,acc,changed,splits,layouts,cfg,use_accel=True)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ra,rb)


if __name__=='__main__':
    unittest.main()

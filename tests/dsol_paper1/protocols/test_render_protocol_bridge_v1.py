from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(REPOSITORY_ROOT/'scripts/dsol_paper1'))
from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol, array_identity, cell_key
from scripts.dsol_paper1.analysis.analyze_render_protocol_bridge_v1 import interval, signatures_equal, rank_comparison

class BridgeTest(unittest.TestCase):
    def test_noaa_is_opt_in_and_context_hook_restored(self):
        class Context:
            def _set_mujoco_context_and_buffers(self):
                self.con=SimpleNamespace(offSamples=self.model.vis.quality.offsamples)
        original=Context._set_mujoco_context_and_buffers
        modules={'robosuite.utils.binding_utils':SimpleNamespace(MjRenderContext=Context)}
        c=Context();c.model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))
        with patch.dict(sys.modules,modules):
            with render_protocol(0) as seen:
                c._set_mujoco_context_and_buffers();self.assertEqual(seen,[0])
            self.assertIs(Context._set_mujoco_context_and_buffers,original)
            with self.assertRaises(ValueError):
                with render_protocol(4):c._set_mujoco_context_and_buffers()
            self.assertIs(Context._set_mujoco_context_and_buffers,original)

    def test_determinism_is_not_common_prefix_or_success_only(self):
        a={'success':False,'completion_steps':2,'bridge':{'calls':[1,2],'steps':[3,4]}}
        b={'success':False,'completion_steps':2,'bridge':{'calls':[1],'steps':[3,4]}}
        self.assertFalse(signatures_equal(a,b));self.assertTrue(signatures_equal(a,a))

    def test_cluster_interval_and_units(self):
        r=interval(np.ones(8)*.02)
        self.assertAlmostEqual(r['mean_pp'],2)
        np.testing.assert_allclose(r['ci95_pp'],[2,2])
        with self.assertRaises(ValueError):interval(np.zeros(192))

    def test_complete_array_identity_includes_shape_dtype_and_signed_zero(self):
        self.assertNotEqual(array_identity(np.array([0.])),array_identity(np.array([-0.])) )
        self.assertNotEqual(array_identity(np.ones(2)),array_identity(np.ones((1,2))))

    def test_rank_comparison_preserves_candidate_mapping(self):
        rows=[{'candidate_id':str(i),'mean_accel_3':float(i),'member_accel_3':[float(i)]*8} for i in range(97)]
        rank={'ranking':rows,'selected_candidate_id':'0'}
        result=rank_comparison(rank,rank)
        self.assertTrue(result['member_scores_exact']);self.assertEqual(result['top10_overlap'],10)
        self.assertAlmostEqual(result['spearman'],1)

    def test_noise_repeat_is_part_of_pairing(self):
        spec={'pair_key':'state','selected_candidate_id':'view','policy_repeat_id':0}
        self.assertNotEqual(cell_key(spec),cell_key(dict(spec,policy_repeat_id=1)))

if __name__=='__main__':unittest.main()

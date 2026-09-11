from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(REPOSITORY_ROOT/'scripts/dsol_paper1'))
from reports.paper1.historical.build_consolidated_view_analysis_v1 import normalized_ranks, empirical_hierarchy
from scripts.dsol_paper1.operations.controllers.run_early_landscape_speed_gate_v1 import process_identity, same_process
from scripts.dsol_paper1.analysis.analyze_view_oracle_comparison_v2 import paired_training_rows, analysis, ORACLES
from tests.dsol_paper1.helpers.landscape import fixtures


class ConsolidatedViewAnalysisTest(unittest.TestCase):
    def test_empirical_hierarchy_uses_one_candidate_per_state_not_per_noise(self):
        y=np.array([[[1,0],[0,1]],[[0,0],[1,1]]])
        h=empirical_hierarchy(y,['a','a'])
        self.assertEqual(h,dict(canonical=.25,global_fixed=.75,task_fixed=.75,statewise=.75))

    def test_empirical_hierarchy_nested_and_task_balanced(self):
        y=np.random.default_rng(41).integers(0,2,(7,5,8))
        h=empirical_hierarchy(y,['a']*5+['b']*2)
        values=list(h.values())
        self.assertTrue(all(a<=b+1e-12 for a,b in zip(values,values[1:])))
        self.assertAlmostEqual(h['canonical'],(y[:5,0].mean()+y[5:,0].mean())/2)

    def test_rank_endpoints_and_direction(self):
        ranks=normalized_ranks(np.arange(97))
        self.assertEqual(ranks[0],0); self.assertEqual(ranks[-1],1)
        self.assertTrue(np.all((ranks>=0)&(ranks<=1)))
        np.testing.assert_allclose(1-ranks,normalized_ranks(-np.arange(97)))

    def test_rank_ties_are_not_artificial_extremes(self):
        np.testing.assert_allclose(normalized_ranks(np.ones(97)),.5)

    def test_invalid_rank_shape_rejected(self):
        with self.assertRaises(ValueError): normalized_ranks(np.ones((2,2)))

    def test_process_identity_is_bound_to_starttime_and_command(self):
        import os
        identity=process_identity(os.getpid()); self.assertTrue(same_process(identity))
        self.assertFalse(same_process({**identity,'start_ticks':identity['start_ticks']+1}))
        self.assertFalse(same_process({**identity,'cmdline':'different'}))

    def full_pair(self):
        from copy import deepcopy
        canonical,broad,_=fixtures()
        for report in (canonical,broad):
            originals=report['states']; report['states']=[]
            for state in originals:
                for j in range(8):
                    row=deepcopy(state)
                    row.update(pair_key=state['pair_key']+f'::source-{j}',source_group=state['source_group']+f'::source-{j}',
                               split='development' if j<6 else 'test')
                    report['states'].append(row)
            for key in ('success','accel','visibility'): report[key]=np.repeat(report[key],8,axis=0)
        return canonical,broad

    def test_full64_pair_and_nine_rule_source_intervals(self):
        canonical,broad=self.full_pair(); paired_training_rows(canonical,broad)
        summary,rows,_=analysis(canonical)
        self.assertEqual(len(rows),64*9)
        self.assertEqual(summary['state_count'],64)
        for rule in ORACLES:
            value=summary['overall']['rules'][rule]['success']
            self.assertEqual(value['finite_sources'],64)
            self.assertTrue(all(n==8 for n in value['sources_per_task'].values()))
            self.assertIsNotNone(value['ci95'][0])
        self.assertEqual(len(summary['by_historical_split_rules']['test']),9)

    def test_full64_pair_rejects_unmatched_noise_or_physics(self):
        canonical,broad=self.full_pair()
        broad['states'][0]['accel_ensemble_seeds'][0]+=1
        with self.assertRaises(ValueError): paired_training_rows(canonical,broad)


if __name__=='__main__': unittest.main()

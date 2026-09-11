from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(REPOSITORY_ROOT/'scripts/dsol_paper1'))
from scripts.dsol_paper1.dualhost_aa0_v1 import ROOT, output_path
from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count


class FrozenDualhostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=json.loads((ROOT/'release.json').read_text())

    def test_complete_population_and_candidates(self):
        r=self.r
        self.assertEqual(len(r['states']),64)
        self.assertEqual(len({s['pair_key'] for s in r['states']}),64)
        self.assertEqual(protocol_spec_count(r['protocol']),198656)
        self.assertEqual(r['protocol']['policy_repeat_ids'],list(range(32)))
        for b in r['protocol']['state_blocks']:
            self.assertEqual(len(b['candidates']),97)
            self.assertEqual(b['candidates'][0]['selected_candidate_id'],'canonical')

    def test_host_partition_preserves_both_models_and_all_tasks(self):
        groups=[set(self.r['hosts'][h]['state_indices']) for h in ['fresh','gnu']]
        self.assertFalse(groups[0]&groups[1]);self.assertEqual(groups[0]|groups[1],set(range(64)))
        self.assertEqual(set(self.r['models']),{'canonical','broad'})
        for g in groups:self.assertEqual(len({self.r['states'][i]['task_id'] for i in g}),8)

    def test_gate_is_outcome_blind_complete_small_cross(self):
        specs=[protocol_spec_at(self.r['protocol'],i) for i in self.r['gate_indices']]
        self.assertEqual(len(specs),48)
        self.assertEqual(len({s['task_id'] for s in specs}),8)
        self.assertEqual({s['policy_repeat_id'] for s in specs},{0,1})
        self.assertEqual({s['selected_candidate_id'] for s in specs},{'canonical','broad_train_000','broad_heldout_000'})

    def test_namespace_separates_gate_hosts_and_models(self):
        paths={output_path(h,phase,m,0) for h in ['fresh','gnu'] for phase in ['gate-a','gate-b','dense'] for m in ['canonical','broad']}
        self.assertEqual(len(paths),12)

    def test_compact_index_preserves_explicit_noise_pairing(self):
        for si in [0,31,63]:
            specs=[protocol_spec_at(self.r['protocol'],si*3104+ci*32+7) for ci in [0,1,96]]
            self.assertEqual({s['pair_key'] for s in specs},{self.r['states'][si]['pair_key']})
            self.assertEqual({s['noise_bank_id'] for s in specs},{'O'})
            self.assertEqual({s['policy_repeat_id'] for s in specs},{7})
            self.assertEqual(len({s['episode_id'] for s in specs}),3)


if __name__=='__main__':unittest.main()

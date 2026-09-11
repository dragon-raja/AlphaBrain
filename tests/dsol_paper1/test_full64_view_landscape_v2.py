from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/dsol_paper1'))
from scripts.dsol_paper1.build_full64_view_landscape_v2 import partition_states, protocol_for
from build_matched_view_landscape_protocol_v1_test import population_fixture, protocol_fixture
from scripts.dsol_paper1.build_matched_view_landscape_protocol_v1 import select_development_states, expanded_specs
from scripts.dsol_paper1.run_full64_view_landscape_v2 import select_mode
from scripts.dsol_paper1.analyze_view_oracle_comparison_v2 import oracle_rows, ORACLES
from scripts.dsol_paper1.finalize_full64_view_landscape_v2 import link_existing


def full_population():
    p=population_fixture()
    test=[]
    for t in range(8):
        for j in (6,7):
            s=deepcopy(p['population']['development']['states'][t*6])
            s.update(split='test',v2_role='test',source_group=f'task-{t}::demo-{j}',pair_key=f'oracle-v2::test::task-{t}::demo-{j}',asset_source_pair_key=f'legacy::{t}::{j}')
            test.append(s)
    p['population']['test']['states']=test
    return p


class Full64ExpansionTest(unittest.TestCase):
    def test_first_eight_and_extension_are_disjoint_complete(self):
        p=full_population(); first,_=select_development_states(p)
        states,remaining,groups=partition_states(p,first)
        self.assertEqual((len(states),len(remaining),len(groups)),(64,56,7))
        self.assertFalse({s['pair_key'] for s in first}&{s['pair_key'] for s in remaining})
        self.assertEqual({s['pair_key'] for _,_,g in groups for s in g},{s['pair_key'] for s in remaining})
        self.assertTrue(all(len({s['task_id'] for s in g})==8 for _,_,g in groups))

    def test_changed_source_or_duplicate_is_rejected(self):
        p=full_population(); first,_=select_development_states(p)
        first[0]['environment_seed']=-1
        with self.assertRaises(ValueError): partition_states(p,first)
        p=full_population(); first,_=select_development_states(p)
        p['population']['test']['states'][1]=p['population']['test']['states'][0]
        with self.assertRaises(ValueError): partition_states(p,first)

    def test_no_scientific_state_or_pose_changes_in_new_protocol(self):
        p=full_population(); template=protocol_fixture(p)[0]
        blocks=template['state_blocks'][:8]
        new=protocol_for(template,blocks,range(4),'example',Path('/tmp/selection.json'))
        specs=expanded_specs(new)
        self.assertEqual(len(specs),8*97*4)
        self.assertEqual(new['state_blocks'],blocks)
        self.assertEqual(new['sensor_control'],'both')
        self.assertFalse(new['camera_motion_within_rollout'])
        self.assertEqual({s['policy_repeat_id'] for s in specs.values()},{0,1,2,3})

    def test_performance_choice_requires_equivalence_and_five_percent(self):
        base={'mode':{'label':'base'},'runner_wall_seconds':100,'exact_action_equivalence':True}
        fast={'mode':{'label':'fast'},'runner_wall_seconds':70,'exact_action_equivalence':False}
        self.assertEqual(select_mode([base,fast]),base['mode'])
        fast['exact_action_equivalence']=True
        self.assertEqual(select_mode([base,fast]),fast['mode'])
        fast['runner_wall_seconds']=98
        self.assertEqual(select_mode([base,fast]),base['mode'])

    def test_reference_links_never_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; source.touch()
            alias=root/'alias'; link_existing(source,alias); link_existing(source,alias)
            self.assertEqual(alias.resolve(),source)
            other=root/'other'; other.touch()
            with self.assertRaises(ValueError): link_existing(other,alias)


class OracleReferenceTest(unittest.TestCase):
    def report(self,y):
        return {'states':[dict(pair_key='s',task_id='t',source_group='d',split='development')],
                'success':y[None], 'candidate_ids':['canonical']+[f'c-{i}' for i in range(96)]}

    def test_crossfit_never_selects_on_evaluation_half(self):
        y=np.zeros((97,32),dtype=np.int8); y[1,:16]=1; y[2,16:]=1
        rows={r['rule']:r for r in oracle_rows(self.report(y))}
        self.assertEqual(rows[ORACLES[0]]['success'],.5)
        self.assertEqual(rows[ORACLES[1]]['success'],0)

    def test_oracle_is_one_view_per_state_not_one_view_per_noise(self):
        y=np.zeros((97,32),dtype=np.int8)
        for r in range(32): y[r+1,r]=1
        rows=oracle_rows(self.report(y))
        self.assertEqual(rows[0]['success'],1/32)

    def test_stable_best_view_and_canonical_tie(self):
        y=np.ones((97,32),dtype=np.int8)
        rows=oracle_rows(self.report(y))
        self.assertEqual(rows[0]['full32_selected_candidate'],'canonical')
        self.assertTrue(all(r['success']==1 for r in rows))


if __name__=='__main__': unittest.main()

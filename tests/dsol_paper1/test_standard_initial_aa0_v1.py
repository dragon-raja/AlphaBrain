import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts/dsol_paper1'
sys.path.insert(0,str(SCRIPTS))
sys.path.insert(0,str(SCRIPTS.parent/'vla_shared'))
import standard_initialization_v1 as init
from evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count


class InitialContractTests(unittest.TestCase):
    def test_materialized_official_state_hash(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'initial.npy';np.save(p,np.arange(4,dtype=np.float64))
            b=Path(t)/'task.bddl';b.write_text('test')
            s=dict(bddl_file=str(b),bddl_sha256=init.sha(b),initialization=dict(
                kind=init.KIND,array_path=str(p),array_sha256=init.sha(p),
                array_identity=init.array_identity(np.arange(4,dtype=np.float64))))
            self.assertEqual(init.load_initial(s).tolist(),[0,1,2,3])
            np.save(p,np.zeros(4))
            with self.assertRaises(AssertionError):init.load_initial(s)

    def test_actual_first_input_must_match_asset(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'render.json';example={'observation/image':np.zeros((2,2,3),np.uint8)}
            p.write_text(json.dumps(dict(input_identities=[dict(candidate_id='canonical',inputs={
                k:init.array_identity(v) for k,v in example.items()})])))
            spec=dict(initial_asset_record=str(p),selected_candidate_id='canonical')
            self.assertEqual(init.verify_policy_input(spec,example),init.sha(p))
            example['observation/image'][0,0,0]=1
            with self.assertRaisesRegex(ValueError,'mismatch'):init.verify_policy_input(spec,example)

    def test_freeze_whole_task_population(self):
        root=Path('/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908')
        if not (root/'release.json').exists():self.skipTest('Release not built')
        r=json.loads((root/'release.json').read_text());p=r['protocol']
        self.assertEqual(protocol_spec_count(p)*2,198656)
        self.assertEqual(len(r['states']),32)
        self.assertEqual(len({s['task_id'] for s in r['states']}),8)
        a=set(r['hosts']['fresh']['state_indices']);b=set(r['hosts']['gnu']['state_indices'])
        self.assertFalse(a&b);self.assertEqual(a|b,set(range(32)))
        for wave in range(4):
            for host in ['fresh','gnu']:
                self.assertEqual(sum(r['states'][i]['initialization']['init_state_index']==wave
                                     for i in r['hosts'][host]['state_indices']),4)
        self.assertEqual(len(r['gate_indices']),48)
        for block in p['state_blocks']:
            self.assertIsNone(block['scene_construction'])
            self.assertEqual(len(block['candidates']),97)
            self.assertNotIn('source_state_index',block['state'])
            self.assertNotIn('demo_name',block['state'])
            self.assertEqual(block['state']['initialization']['settle_steps'],10)
        spec=protocol_spec_at(p,96*32+31)
        self.assertEqual(spec['policy_repeat_id'],31)
        self.assertEqual(spec['initialization']['init_state_index'],0)

    def test_initialization_uses_no_demo_xml_and_ten_settles(self):
        class Env:
            def __init__(self):self.steps=0;self.seed_value=None;self.did_reset=False
            def seed(self,x):self.seed_value=x
            def reset(self):self.did_reset=True
            def set_init_state(self,x):self.x=x;return {}
            def check_success(self):return False
            def step(self,a):self.steps+=1;return {},0,False,{}
        e=Env();s=dict(environment_seed=12,initialization=dict(kind=init.KIND,settle_steps=10,
            official_file='official',official_file_sha256='abc',init_state_index=2))
        with patch('evaluate_pi05_libero_plus_views.physics_state_sha256',return_value=('hash',3)):
            _,receipt=init.initialize(e,s,np.zeros(3))
        self.assertEqual(e.steps,10);self.assertTrue(e.did_reset)
        self.assertEqual(e.seed_value,12);self.assertFalse(receipt['demonstration_state_used'])


if __name__=='__main__':unittest.main()

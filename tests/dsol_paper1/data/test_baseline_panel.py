import unittest
from unittest.mock import patch
from copy import deepcopy
from AlphaBrain.research.dsol.data.baseline_panel import select_initials, select_views, validate_baseline_panel
from AlphaBrain.research.dsol.data.evaluation_matrix import NUMERIC_FLAGS, NOISE_VERSION


class BaselinePanelTests(unittest.TestCase):
    def test_initials_are_balanced_and_order_invariant(self):
        jobs = [dict(task=f't{i}', split=s, scene_seed=n, initial_id=f'{i}-{s}-{n}')
                for i in range(10) for s in ('pretrain','target') for n in range(4)]
        selected = select_initials(jobs)
        self.assertEqual(len(set(selected)),40)
        self.assertEqual(selected, select_initials(jobs[::-1]))
        self.assertTrue(all(x.endswith(('-0','-1')) for x in selected))

    def test_views_are_reference_and_only_heldout(self):
        views = [dict(view_id='canonical',role='reference',parameters=[0,0,1]),
                 dict(view_id='train',role='train',parameters=[-10,-5,1])]
        for n, p in enumerate([[-10,-5,1],[10,5,1],[-30,-15,1.2],[30,15,1.2],[0,0,1]]):
            views.append(dict(view_id=f'extra_{n}',role='extra_eval',parameters=p))
        expected=['canonical','extra_0','extra_1','extra_2','extra_3']
        self.assertEqual(select_views(views),expected)
        self.assertEqual(select_views(views[::-1]),expected)

    def test_missing_groups_fail(self):
        with self.assertRaises(ValueError):
            select_initials([])

    def test_checkpoint_extension_preserves_panel_and_model_identity(self):
        jobs = [dict(task=f't{i}', split=s, scene_seed=n, initial_id=f'{i}-{s}-{n}')
                for i in range(10) for s in ('pretrain','target') for n in range(2)]
        views = [dict(view_id='canonical',role='reference',parameters=[0,0,1])]
        views += [dict(view_id=f'extra_{i}',role='extra_eval',parameters=p)
                  for i,p in enumerate([[-10,-5,1],[10,5,1],[-30,-15,1.2],[30,15,1.2]])]
        initials = select_initials(jobs)
        worker = dict(worker_id='w',host='fresh',gpu=0,lane='checkpoints',
                      phases=['official'],pilot_initials=[],initials=initials)
        base = dict(format='robocasa_baseline_panel_v1',numeric_flags=NUMERIC_FLAGS,
                    noise_version=NOISE_VERSION,checkpoint_repeats=[0,1,2,3],confirmation_repeats=[],
                    models={'official':{}},first_input_source='frozen_p0_canonical_scores',
                    rollout_count=800,p0_plan='p0',p0_plan_sha256='hash',
                    initial_ids=initials,panel_views=select_views(views),workers=[worker])
        models = {n:dict(checkpoint=f'/checkpoints/{n}/4000') for n in ['canonical_4000','broad_4000']}
        new = deepcopy(base)
        new.update(format='robocasa_checkpoint_4000_panel_v1',models=models,rollout_count=1600,
                   baseline_plan='base',baseline_plan_sha256='hash',
                   previous_checkpoint_identity_plan='identities',previous_plan_sha256='hash')
        new['workers'][0]['phases'] = list(models)
        records = {'base':base,'p0':dict(workers=[dict(worker,arm='canonical')]),'identities':dict(models=models)}
        with patch('AlphaBrain.research.dsol.data.baseline_panel.sha',return_value='hash'), \
             patch('AlphaBrain.research.dsol.data.baseline_panel.read',side_effect=lambda p:records[p]), \
             patch('AlphaBrain.research.dsol.data.baseline_panel.validate_plan',return_value=dict(jobs=jobs,catalog=dict(views=views))):
            validate_baseline_panel(base)
            validate_baseline_panel(new)
            for key,value in [('rollout_count',800),('panel_views',new['panel_views'][::-1]),('initial_ids',initials[:-1])]:
                invalid=deepcopy(new);invalid[key]=value
                with self.assertRaises(ValueError):validate_baseline_panel(invalid)
            invalid=deepcopy(new);invalid['models']['broad_4000']['checkpoint']='/wrong/8250'
            with self.assertRaises(ValueError):validate_baseline_panel(invalid)


if __name__ == '__main__':
    unittest.main()

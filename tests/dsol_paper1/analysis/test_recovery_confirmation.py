import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from AlphaBrain.research.dsol.analysis.recovery_confirmation import selected_conditions, freeze_ready_initials
from AlphaBrain.research.dsol.data.artifacts import read, sha, write


class ConfirmationChoiceTests(unittest.TestCase):
    def setUp(self):
        self.views = ['canonical'] + [f'train_{i:03d}' for i in range(64)] + [f'extra_{i:03d}' for i in range(32)]
        self.extra = self.views[-32:]

    def test_catalog_ties_and_shared_baselines(self):
        result = selected_conditions(self.views, self.extra, np.ones((97, 8, 5)))
        self.assertEqual(result['chosen_indices'], [0]*5)
        self.assertEqual(result['evaluation_views'], ['canonical'] + self.extra)
        self.assertEqual(result['new_rollouts'], 33*8)
        self.assertEqual(result['primary_window'], 3)

    def test_window1_exact_and_new_view_deduplicated(self):
        scores = np.full((97, 8, 5), 10.)
        scores[1, :, 0] = 0.
        scores[2, :, :] = 1.
        result = selected_conditions(self.views, self.extra, scores)
        self.assertEqual(result['chosen_indices'], [1, 2, 2, 2, 2])
        self.assertEqual(result['new_rollouts'], 35*8)
        self.assertEqual(len(result['evaluation_views']), len(set(result['evaluation_views'])))

    def test_incomplete_nonfinite_or_changed_scope_rejected(self):
        scores = np.ones((97, 8, 5))
        bad = scores.copy(); bad[0, 0, 2] = np.nan
        for value in [scores[:-1], scores[:, :4], bad]:
            with self.assertRaises(ValueError): selected_conditions(self.views, self.extra, value)
        with self.assertRaises(ValueError): selected_conditions(self.views, self.extra[:-1], scores)
        with self.assertRaises(ValueError): selected_conditions(self.views[::-1], self.extra, scores)

    def test_partial_population_freezes_without_endpoints_and_detects_tampering(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); probe = root/'probe'; output = root/'selections'
            source = root/'source/benchmarks/Robocasa365/short_view_probe.py'
            source.parent.mkdir(parents=True); source.write_text('# frozen fixture\n')
            initials = [f'initial{i}' for i in range(80)]
            reference = root/'reference.json'
            write(reference, dict(jobs=[dict(initial_id=i, task='Task', split='target', scene_seed=j)
                                       for j, i in enumerate(initials)],
                                  catalog=dict(views=[dict(view_id=v, role='extra_eval' if v in self.extra else 'train')
                                                      for v in self.views])))
            plan = root/'plan.json'; config = root/'config.json'
            write(plan, dict(initial_ids=initials, all_views=self.views, reference_plan=str(reference),
                             reference_plan_sha256=sha(reference)))
            write(config, dict(panel_plan=str(plan)))
            protocol = dict(model='official', initials=initials, views=self.views,
                            score_repeats=8, primary_window=3, windows=[1, 2, 3, 4, 5],
                            source=str(root/'source'), config_sha256=sha(config), panel_plan_sha256=sha(plan))
            write(probe/'protocol.json', protocol)
            worker = probe/'workers/first'
            write(worker/'identity.json', dict(model='official', initials=initials[:1], views=self.views,
                  config_sha256=sha(config), population_mode='fresh_confirmation', source_sha256=sha(source)))
            records = []
            for initial in initials:
                for view in self.views:
                    cell = worker/'cells'/initial/view
                    records.append(dict(initial=initial, view=view, cell=str(cell)))
                    if initial != initials[0]: continue
                    cell.mkdir(parents=True)
                    np.savez(cell/'probes.npz', scores=np.ones((8, 5)), prefix_means=np.ones(5))
                    np.savez(cell/'initial-anchor.npz', scores=np.ones(8))
                    write(cell/'initial-anchor.json', dict(initial=initial, view=view, model='official',
                          action_steps=0, score_repeats=8, arrays_sha256=sha(cell/'initial-anchor.npz')))
                    write(cell/'complete.json', dict(status='COMPLETE', window1_exact=True, seconds=1.,
                          arrays_sha256=sha(cell/'probes.npz'),
                          records=[dict(repeat=r, replan=k) for r in range(8) for k in range(5)]))
            write(probe/'ownership.json', dict(records=records))
            write(probe/'manifest.json', dict(runtime=dict(config=str(config)),
                  artifacts=[dict(path=str(p), sha256=sha(p)) for p in
                             [probe/'protocol.json', probe/'ownership.json']]))
            result = freeze_ready_initials(probe, output)
            self.assertEqual(result['completed_initials'], 1)
            self.assertEqual(len(result['pending_initials']), 79)
            saved = output/'initials/initial0.json'; digest = sha(saved)
            self.assertEqual(freeze_ready_initials(probe, output), result)
            self.assertEqual(sha(saved), digest)
            self.assertEqual(read(saved)['new_rollouts'], 264)
            anchor = worker/'cells/initial0/canonical/initial-anchor.json'
            write(anchor, dict(corrupted=True))
            with self.assertRaises(ValueError): freeze_ready_initials(probe, output)

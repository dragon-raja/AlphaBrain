import copy
import unittest
import json
from pathlib import Path
import tempfile

from AlphaBrain.research.dsol.data.matched_render import validate_episode_receipt, select_episode_root


class MatchedRenderTests(unittest.TestCase):
    def test_recovery_only_fills_missing_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = [Path(temp) / name for name in ('primary', 'recovery', 'duplicate')]
            for root in roots[1:]:
                dest = root / 'episodes' / 'j'
                dest.mkdir(parents=True)
                (dest / 'summary.json').write_text(json.dumps({'status': 'PASS'}))
            self.assertEqual(select_episode_root('j', roots[0], roots[1:2]), roots[1] / 'episodes/j')
            with self.assertRaises(ValueError):
                select_episode_root('j', roots[0], roots[1:])
            dest = roots[0] / 'episodes/j'
            dest.mkdir(parents=True)
            (dest / 'summary.json').write_text(json.dumps({'status': 'PASS'}))
            self.assertEqual(select_episode_root('j', roots[0]), dest)
            with self.assertRaises(ValueError):
                select_episode_root('j', roots[0], roots[1:2])

    def setUp(self):
        self.job = dict(job_id='j', episode_index=5, view_ids=['v1', 'v2'],
            rows=[dict(source_window_id='s', frame_index=7, training_pair_index=0)])
        def exposure(view):
            return dict(view_id=view, image_files=dict(left=view+'L', right=view+'R', wrist='W'))
        self.receipt = dict(status='PASS', job_id='j', episode_index=5, plan_sha256='sha',
            anchor_frame=0, wrist_shared_exact=True, records=[dict(source_window_id='s',
            frame_index=7, training_pair_index=0, canonical_exposures=[exposure('canonical')]*2,
            broad_exposures=[exposure('v1'), exposure('v2')])])

    def test_matched(self):
        self.assertEqual(len(validate_episode_receipt(self.job, self.receipt, 'sha')), 1)

    def test_changed_wrist_or_frame_block_release(self):
        changed = copy.deepcopy(self.receipt)
        changed['records'][0]['broad_exposures'][0]['image_files']['wrist'] = 'other'
        with self.assertRaises(ValueError):
            validate_episode_receipt(self.job, changed, 'sha')
        changed = copy.deepcopy(self.receipt)
        changed['records'][0]['frame_index'] += 1
        with self.assertRaises(ValueError):
            validate_episode_receipt(self.job, changed, 'sha')

    def test_wrong_plan_and_missing_source_block_release(self):
        with self.assertRaises(ValueError):
            validate_episode_receipt(self.job, self.receipt, 'other')
        self.receipt['records'] = []
        with self.assertRaises(ValueError):
            validate_episode_receipt(self.job, self.receipt, 'sha')

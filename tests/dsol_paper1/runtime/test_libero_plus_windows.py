import unittest
from unittest.mock import patch
import numpy as np

from benchmarks.LIBERO.plus_windows import model_without_external_pose, initial_scores
from benchmarks.LIBERO.plus_window_queue import partition_blocks
from benchmarks.LIBERO.plus_window_analysis import choices


class PlusWindowsTest(unittest.TestCase):
    def test_window1_and_cumulative_windows_with_explicit_permissions(self):
        scores=np.zeros((3,8,5)); scores[0]=1; scores[1]=2; scores[2]=3
        scores[2,:,1:]=0
        np.testing.assert_array_equal(choices(scores,include_canonical=True),[0,0,0,2,2])
        np.testing.assert_array_equal(choices(scores,include_canonical=False),[1,2,2,2,2])

    def test_partition_exact_and_capacity_weighted(self):
        counts = [50]*32
        pilots, groups = partition_blocks(counts, {'fresh': 4, 'gnu': 8})
        all_blocks = list(pilots.values()) + groups['fresh'] + groups['gnu']
        self.assertEqual(len(all_blocks), 64)
        self.assertEqual(len(set(all_blocks)), 64)
        self.assertEqual(set(all_blocks), {(m,i) for m in ('canonical','broad') for i in range(32)})
        self.assertLessEqual(abs(len(groups['fresh'])*2-len(groups['gnu'])), 2)

    def test_pose_only_model_equivalence(self):
        a = '<mujoco><worldbody><camera name="agentview" pos="1 2 3" quat="1 0 0 0" fovy="45"/></worldbody></mujoco>'
        b = a.replace('1 2 3','3 2 1').replace('1 0 0 0','0 1 0 0')
        self.assertEqual(model_without_external_pose(a), model_without_external_pose(b))
        self.assertNotEqual(model_without_external_pose(a), model_without_external_pose(b.replace('45','46')))
        with self.assertRaises(ValueError):
            model_without_external_pose('<mujoco/>')

    def test_initial_batch16_padding_and_eight_noise_members(self):
        class Connection:
            def __init__(self): self.sent=[]
            def send(self, request): self.sent.append(request)
            def recv(self): return {'velocity':np.ones((16,10,10,7))}
        cached = dict(views=np.arange(17))
        for key in ('external_images','wrist_images','robot_states','camera_intrinsics','camera_to_world_opencv'):
            cached[key] = np.arange(17)[:,None]
        noises=np.arange(8*10*7,dtype=np.float32).reshape(8,10,7); conn=Connection()
        with patch('benchmarks.LIBERO.plus_windows.compute_accel_scores',
                   side_effect=lambda trace,prefixes: {3:dict(score=np.ones(len(trace)),degenerate=np.zeros(len(trace),bool))}):
            scores=initial_scores(conn,cached,noises,'task')
        self.assertEqual(scores.shape,(17,8)); self.assertEqual(len(conn.sent),16)
        for j, request in enumerate(conn.sent):
            self.assertEqual(len(request['examples']),16)
            np.testing.assert_array_equal(request['noise'],np.repeat(noises[j%8][None],16,0))


if __name__ == '__main__':
    unittest.main()

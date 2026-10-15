import unittest

from benchmarks.LIBERO.plus_render import requests_for_initial


class PlusRenderRequestsTest(unittest.TestCase):
    def setUp(self):
        self.record = {'state': {'task_id': 'a', 'bddl_file': '/task.bddl',
                                'initialization': {'array_sha256': 'sha'}}}
        self.request = {'task_id': 'a', 'resolved_base_bddl': '/task.bddl',
                        'physical_initial': {'array_sha256': 'sha'}, 'request_id': 'one'}

    def test_selects_exact_task_and_initial(self):
        other = dict(self.request, task_id='b')
        self.assertEqual(requests_for_initial([other, self.request], self.record), [self.request])

    def test_missing_and_duplicate_fail(self):
        for rows in [[], [self.request, self.request]]:
            with self.assertRaises(ValueError):
                requests_for_initial(rows, self.record)

    def test_base_task_cannot_change(self):
        with self.assertRaises(ValueError):
            requests_for_initial([dict(self.request, resolved_base_bddl='/other.bddl')], self.record)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
from benchmarks.Robocasa365.attribution_dispatch import read_status


class DispatchStatusTests(unittest.TestCase):
    @patch('benchmarks.Robocasa365.attribution_dispatch.read',side_effect=FileNotFoundError)
    def test_missing_shared_status_is_not_success(self,_read):
        self.assertEqual(read_status('unused'),'STATUS_UNAVAILABLE')

    @patch('benchmarks.Robocasa365.attribution_dispatch.read',return_value={'status':'ERROR'})
    def test_real_worker_failure_is_preserved(self,_read):
        self.assertEqual(read_status('unused'),'ERROR')

    @patch('benchmarks.Robocasa365.attribution_dispatch.read',side_effect=ValueError('invalid JSON'))
    def test_malformed_status_is_not_ignored(self,_read):
        with self.assertRaises(ValueError):read_status('unused')


if __name__=='__main__':
    unittest.main()

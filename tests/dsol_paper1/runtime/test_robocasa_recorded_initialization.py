import unittest
from unittest.mock import Mock

from benchmarks.Robocasa365.recorded_initialization import PLACEMENT_EXHAUSTED, reset_for_recorded_replay


class RecordedInitializationTests(unittest.TestCase):
    def test_retry_only_known_failure(self):
        base = Mock()
        base.reset.side_effect = [RuntimeError(PLACEMENT_EXHAUSTED), None]
        result = reset_for_recorded_replay(base, 2)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(len(result['prior_errors']), 1)
        self.assertEqual(base.reset.call_count, 2)

    def test_unrelated_error_is_not_swallowed(self):
        base = Mock()
        base.reset.side_effect = RuntimeError('state dimension mismatch')
        with self.assertRaisesRegex(RuntimeError, 'dimension'):
            reset_for_recorded_replay(base, 5)
        self.assertEqual(base.reset.call_count, 1)

    def test_attempt_limit(self):
        base = Mock()
        base.reset.side_effect = RuntimeError(PLACEMENT_EXHAUSTED)
        with self.assertRaises(RuntimeError):
            reset_for_recorded_replay(base, 2)
        self.assertEqual(base.reset.call_count, 2)
        with self.assertRaises(ValueError):
            reset_for_recorded_replay(base, 6)

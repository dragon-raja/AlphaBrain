import unittest
from unittest import mock

from pi05_policy_server import (
    configure_torch_threads,
    runtime_identity,
    validate_policy_example,
)


class PolicyServerInputTest(unittest.TestCase):
    def test_rejects_oracle_or_branch_metadata(self):
        valid = {"image": [], "lang": "task", "language": "task", "state": []}
        validate_policy_example(valid)
        with self.assertRaises(ValueError):
            validate_policy_example({**valid, "branch_outcome": "slipped"})
        with self.assertRaises(ValueError):
            validate_policy_example({**valid, "action_divergence_time": 4})

    def test_runtime_identity_uses_builtin_strings(self):
        class VersionLike(str):
            pass

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_name(_index):
                return VersionLike("RTX 5090")

        class FakeTorch:
            __version__ = VersionLike("2.11.0")
            version = type("Version", (), {"cuda": VersionLike("12.8")})
            cuda = FakeCuda()

        identity = runtime_identity(FakeTorch())
        self.assertEqual(
            identity,
            {"torch_version": "2.11.0", "cuda_version": "12.8", "device_name": "RTX 5090"},
        )
        self.assertTrue(all(value is None or type(value) is str for value in identity.values()))

    def test_configures_explicit_torch_thread_limits(self):
        class FakeTorch:
            intraop = None
            interop = None

            @classmethod
            def set_num_threads(cls, value):
                cls.intraop = value

            @classmethod
            def set_num_interop_threads(cls, value):
                cls.interop = value

        with mock.patch.dict(
            "os.environ",
            {
                "FRESH_TORCH_NUM_THREADS": "2",
                "FRESH_TORCH_INTEROP_THREADS": "1",
            },
            clear=True,
        ):
            configure_torch_threads(FakeTorch)

        self.assertEqual(FakeTorch.intraop, 2)
        self.assertEqual(FakeTorch.interop, 1)

    def test_rejects_non_positive_torch_thread_limit(self):
        with mock.patch.dict(
            "os.environ",
            {"FRESH_TORCH_NUM_THREADS": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                configure_torch_threads(object())


if __name__ == "__main__":
    unittest.main()

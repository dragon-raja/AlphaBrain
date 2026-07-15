import unittest

from pi05_policy_server import runtime_identity, validate_policy_example


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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "dsol_paper1"
sys.path.insert(0, str(SCRIPT_ROOT))

from view_acquisition_inputs import (  # noqa: E402
    InputContractError,
    build_input_bundle,
    supported_input_contract,
)


def observation() -> dict:
    return {
        "agentview_image": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        "robot0_eye_in_hand_image": np.full((2, 5, 3), 93, dtype=np.uint8),
    }


def geometry() -> dict:
    return {
        "camera_intrinsics": np.array([[100, 0, 2], [0, 100, 1.5], [0, 0, 1]], dtype=np.float64),
        "camera_to_world_opencv": np.eye(4, dtype=np.float64),
    }


def build(obs=None, **changes):
    kwargs = {
        "language": "pick up the red mug",
        "input_contract": supported_input_contract(),
        "robot_state": np.arange(8, dtype=np.float32),
        "camera_geometry": geometry(),
    }
    kwargs.update(changes)
    return build_input_bundle(observation() if obs is None else obs, **kwargs)


class InputWhitelistTests(unittest.TestCase):
    def test_permission_difference_and_exact_keys(self):
        bundle = build()
        self.assertEqual(set(bundle.policy), {"external_rgb", "wrist_rgb", "language"})
        self.assertEqual(set(bundle.selector), set(bundle.policy) | {"robot_state", "camera_geometry"})
        self.assertEqual(bundle.selector["robot_state"].shape, (8,))
        self.assertEqual(bundle.selector["robot_state"].dtype, np.float32)
        self.assertNotIn("observation/state", bundle.policy)
        self.assertEqual(bundle.policy["language"], "pick up the red mug")

    def test_supported_contract_matches_current_draft(self):
        path = SCRIPT_ROOT.parents[1] / "configs" / "dsol_paper1" / "view_acquisition_a1_draft_v1.json"
        contract = json.loads(path.read_text())["input_contract"]
        self.assertEqual(contract, supported_input_contract())
        build(input_contract=contract)

    def test_polluted_observation_fields_never_forwarded_or_inspected(self):
        images = observation()

        class WhitelistOnlyObservation(Mapping):
            def __getitem__(self, key):
                if key not in images:
                    raise AssertionError("attempted privileged read: " + key)
                return images[key]

            def __iter__(self):
                raise AssertionError("raw observation enumeration is forbidden")

            def __len__(self):
                raise AssertionError("raw observation enumeration is forbidden")

        bundle = build(WhitelistOnlyObservation())
        self.assertEqual(len(bundle.policy), 3)
        contaminated = dict(images)
        for name in (
            "object-state", "sim_truth", "future_candidate_images", "history", "robot_state",
            "camera_geometry", "robot0_eef_pos", "_eval_noise", "prompt",
        ):
            contaminated[name] = object()
        polluted = build(contaminated)
        self.assertEqual(bundle.audit_record(), polluted.audit_record())

    def test_missing_either_camera_rejects_whole_bundle(self):
        for key in observation():
            obs = observation()
            del obs[key]
            with self.subTest(key=key), self.assertRaisesRegex(InputContractError, "missing"):
                build(obs)

    def test_bad_image_shape_or_dtype_rejected_without_conversion(self):
        bad_images = [
            np.zeros((3, 4), dtype=np.uint8),
            np.zeros((3, 4, 4), dtype=np.uint8),
            np.zeros((0, 4, 3), dtype=np.uint8),
            np.zeros((1, 3, 4, 3), dtype=np.uint8),
            np.zeros((3, 4, 3), dtype=np.float32),
            np.zeros((3, 4, 3), dtype=np.int16),
            [[[0, 0, 0]]],
        ]
        for value in bad_images:
            for key in observation():
                obs = observation()
                obs[key] = value
                with self.subTest(key=key, shape=np.shape(value)), self.assertRaises(InputContractError):
                    build(obs)

    def test_raw_images_are_not_flipped_resized_or_encoded(self):
        obs = observation()
        obs["agentview_image"] = obs["agentview_image"][::-1, ::-1]
        bundle = build(obs)
        np.testing.assert_array_equal(bundle.policy["external_rgb"], obs["agentview_image"])
        np.testing.assert_array_equal(bundle.policy["wrist_rgb"], obs["robot0_eye_in_hand_image"])
        self.assertEqual(bundle.policy["external_rgb"].shape, (3, 4, 3))
        self.assertTrue(bundle.policy["external_rgb"].flags.c_contiguous)
        self.assertFalse(bundle.audit_record()["policy_encoding_applied"])
        self.assertFalse(bundle.audit_record()["existing_websocket_request_ready"])

    def test_source_and_consumer_arrays_are_isolated_and_immutable(self):
        obs = observation()
        # Even two raw image keys sharing storage must be separated.
        obs["robot0_eye_in_hand_image"] = obs["agentview_image"]
        state = np.arange(8, dtype=np.float32)
        calibration = geometry()
        bundle = build(obs, robot_state=state, camera_geometry=calibration)
        before = bundle.audit_record()
        obs["agentview_image"][:] = 255
        state[:] = 100
        calibration["camera_intrinsics"][:] = 0
        arrays = [bundle.policy["external_rgb"], bundle.policy["wrist_rgb"],
                  bundle.selector["external_rgb"], bundle.selector["wrist_rgb"]]
        for index, array in enumerate(arrays):
            self.assertEqual(int(array[0, 0, 0]), 0)
            for other in arrays[index + 1:]:
                self.assertFalse(np.shares_memory(array, other))
            with self.assertRaises(ValueError):
                array[0, 0, 0] = 1
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertEqual(float(bundle.selector["robot_state"][0]), 0)
        self.assertEqual(float(bundle.selector["camera_geometry"]["camera_intrinsics"][0, 0]), 100)
        self.assertEqual(bundle.audit_record(), before)
        with self.assertRaises(TypeError):
            bundle.policy["robot_state"] = state
        with self.assertRaises(TypeError):
            bundle.selector["camera_geometry"]["object-state"] = state

    def test_explicit_state_and_geometry_are_required_no_raw_fallback(self):
        obs = observation()
        obs["robot_state"] = np.zeros(8, dtype=np.float32)
        obs["camera_geometry"] = geometry()
        for key in ("robot_state", "camera_geometry"):
            with self.subTest(key=key), self.assertRaisesRegex(InputContractError, "explicitly"):
                build(obs, **{key: None})

    def test_bad_states_rejected(self):
        states = [
            np.zeros(7), np.zeros((1, 8)), [0] * 7 + [np.nan], [0] * 7 + [np.inf],
            [0] * 7 + [1e100], ["0"] * 8, np.zeros(8, dtype=bool), np.zeros(8, dtype=complex),
        ]
        for state in states:
            with self.subTest(state=state), self.assertRaises(InputContractError):
                build(robot_state=state)

    def test_geometry_unknown_fields_and_invalid_matrices_rejected(self):
        cases = []
        for key in geometry():
            value = geometry()
            del value[key]
            cases.append(value)
        value = geometry()
        value["object-state"] = np.zeros(3)
        cases.append(value)
        for key, replacement in (
            ("camera_intrinsics", np.eye(4)),
            ("camera_intrinsics", np.full((3, 3), np.nan)),
            ("camera_intrinsics", -np.eye(3)),
            ("camera_to_world_opencv", np.eye(3)),
            ("camera_to_world_opencv", np.full((4, 4), np.inf)),
            ("camera_to_world_opencv", np.diag([-1, 1, 1, 1])),
            ("camera_to_world_opencv", np.diag([2, 1, 1, 1])),
            ("camera_to_world_opencv", np.diag([1, 1, 1, 2])),
        ):
            value = geometry()
            value[key] = replacement
            cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(InputContractError):
                build(camera_geometry=value)

    def test_unknown_or_expanded_contract_rejected(self):
        changes = {
            "id": "unreviewed_contract",
            "policy_inputs": ["external_rgb", "wrist_rgb", "language", "robot_state"],
            "selector_inputs": ["external_rgb", "wrist_rgb", "language", "robot_state", "candidate_images"],
            "base_policy_history": "all_frames",
            "extra_selector_state_declared": False,
            "along_path_images": 0,
            "bundle_image_keys": ["external_rgb"],
        }
        for key, value in changes.items():
            contract = supported_input_contract()
            contract[key] = value
            with self.subTest(key=key), self.assertRaises(InputContractError):
                build(input_contract=contract)
        contract = supported_input_contract()
        contract["unknown"] = True
        with self.assertRaises(InputContractError):
            build(input_contract=contract)
        del contract["unknown"]
        del contract["candidate_images"]
        with self.assertRaises(InputContractError):
            build(input_contract=contract)
        contract = supported_input_contract()
        contract["policy_inputs"][0] = np.array(["external_rgb", "object-state"])
        with self.assertRaises(InputContractError):
            build(input_contract=contract)

    def test_invalid_language_is_not_coerced(self):
        for language in (None, 12, "", "  ", "\ud800"):
            with self.subTest(language=repr(language)), self.assertRaises(InputContractError):
                build(language=language)

    def test_receipt_has_shapes_hashes_and_one_bundle_but_no_input_values(self):
        obs = observation()
        bundle = build(obs)
        receipt = bundle.audit_record()
        encoded = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("pick up the red mug", encoded)
        self.assertNotIn("pixels", encoded)
        self.assertEqual(receipt["bundle_image_count"], 2)
        self.assertEqual(receipt["query_units_per_observation_read"], 1)
        self.assertFalse(receipt["builder_performs_or_charges_queries"])
        image = receipt["policy"]["fields"]["external_rgb"]
        self.assertEqual(image["shape"], [3, 4, 3])
        self.assertEqual(image["dtype"], "uint8")
        self.assertEqual(image["sha256"], hashlib.sha256(obs["agentview_image"].tobytes()).hexdigest())
        self.assertEqual(image, receipt["selector"]["fields"]["external_rgb"])
        receipt["policy"]["fields"]["external_rgb"]["shape"][0] = 999
        self.assertEqual(bundle.audit_record()["policy"]["fields"]["external_rgb"]["shape"], [3, 4, 3])
        self.assertNotIn("array", repr(bundle))

    def test_hash_changes_only_with_explicitly_consumed_content(self):
        first = build().audit_record()
        self.assertEqual(first, build().audit_record())
        state_changed = build(robot_state=np.zeros(8)).audit_record()
        self.assertEqual(first["policy"]["sha256"], state_changed["policy"]["sha256"])
        self.assertNotEqual(first["selector"]["sha256"], state_changed["selector"]["sha256"])
        obs = observation()
        obs["agentview_image"][0, 0, 0] += 1
        image_changed = build(obs).audit_record()
        self.assertNotEqual(first["policy"]["sha256"], image_changed["policy"]["sha256"])
        self.assertNotEqual(first["receipt_sha256"], image_changed["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()

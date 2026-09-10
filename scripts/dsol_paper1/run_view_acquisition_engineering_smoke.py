#!/usr/bin/env python3
"""Real LIBERO camera-executor smoke, with NO VLA and NO scientific evaluation.

Runs exactly four short conditions on one archived development state: continue,
two matched holds, and one 5 cm kinematic camera move. Outputs cannot release a
formal protocol or be pooled with task-success experiments. Uses a fresh output
directory and private LIBERO config; never writes to an archived run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


def configure_imports() -> None:
    scripts = Path(__file__).resolve().parents[1]
    for path in (scripts / "dsol_paper1", scripts / "vla_shared"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _hash_array(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    header = json.dumps({"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True)
    return hashlib.sha256(header.encode() + array.tobytes()).hexdigest()


def run_condition(spec: dict[str, Any], args: argparse.Namespace, condition: str) -> dict[str, Any]:
    import h5py
    import numpy as np

    from audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths
    from view_acquisition_executor import (
        LiberoKinematicCameraBackend, PersistentOSCGoalHold, execute_acquisition,
    )
    from view_acquisition_motion import (
        BudgetLedger, MotionLimits, Pose, WorldAABB,
        no_acquisition, plan_camera_motion, plan_matched_hold,
    )

    hdf5 = Path(spec["hdf5"])
    _configure_runtime(args.runtime, hdf5.parent.parent, args.output_dir / "libero-config")
    from libero.libero.envs import OffScreenRenderEnv
    from libero_constructed_view import inject_static_visual_occluder

    with h5py.File(hdf5, "r") as handle:
        data = handle["data"]
        demo = data[spec["demo_name"]]
        state = np.asarray(demo["states"][int(spec["source_state_index"])])
        xml, _ = _rewrite_model_paths(_decode(demo.attrs["model_file"]), args.runtime)
        bddl_name = Path(_decode(data.attrs["bddl_file_name"])).name
    if spec.get("scene_construction") is not None:
        xml = inject_static_visual_occluder(xml, spec["scene_construction"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.runtime / "libero/libero/bddl_files" / spec["suite"] / bddl_name),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=args.render_gpu,
    )
    try:
        env.seed(20260907)
        env.reset()
        env.reset_from_xml_string(xml)
        initial = env.set_init_state(state)
        if env.check_success():
            raise ValueError("engineering source must not already satisfy the task")
        initial_state = np.asarray(env.get_sim_state()).copy()
        initial_gripper = env.env.robots[0].gripper.current_action.copy()
        gripper_ids = env.env.robots[0]._ref_joint_gripper_actuator_indexes
        initial_gripper_ctrl = np.asarray(env.env.sim.data.ctrl[gripper_ids]).copy()
        finger_ids = env.env.robots[0]._ref_gripper_joint_pos_indexes
        initial_finger_qpos = np.asarray(env.env.sim.data.qpos[finger_ids]).copy()
        dt = float(env.env.control_timestep)
        duration = args.steps * dt
        initial_time = float(env.env.sim.data.time)
        wall_started = time.monotonic()
        # HDF5 snapshots do not certify controller runtime restoration. Pin the
        # restored current EEF pose in this engineering-only test, explicitly.
        with PersistentOSCGoalHold(env, reference="current_pose") as hold:
            backend = LiberoKinematicCameraBackend(env, hold)
            start = backend.camera_pose()
            initial_calibration = backend.calibration()
            initial_eef = np.asarray(hold.controller.ee_pos).copy()
            bounds = WorldAABB(
                tuple(v - 0.25 for v in start.position),
                tuple(v + 0.25 for v in start.position),
            )
            limits = MotionLimits(0.5, 2.0, 1.0, 4.0, bounds)
            if condition == "continue":
                trajectory = no_acquisition(start, dt=dt, limits=limits)
            elif condition in ("hold-a", "hold-b"):
                trajectory = plan_matched_hold(start, duration=duration, dt=dt, limits=limits)
            elif condition == "move-x":
                goal = Pose(
                    (start.position[0] + 0.05, start.position[1], start.position[2]),
                    start.quaternion_wxyz,
                )
                trajectory = plan_camera_motion(start, goal, duration=duration, dt=dt, limits=limits)
            else:
                raise ValueError("unknown smoke condition")
            ledger = BudgetLedger(limit_env_steps=args.steps, limit_queries=1, limit_policy_calls=0)
            result = execute_acquisition(
                trajectory, backend, ledger, initial_observation=initial, clock_tolerance=1e-6,
            )
            hold.controller.update(force=True)
            final_eef = np.asarray(hold.controller.ee_pos).copy()
            final_calibration = backend.calibration()
            final_pose = backend.camera_pose()
        final_state = np.asarray(env.get_sim_state()).copy()
        observation = result.observation
        return {
            "condition": condition,
            "source_pair_key": spec["pair_key"],
            "source_hdf5": str(hdf5),
            "demo_name": spec["demo_name"],
            "source_state_index": int(spec["source_state_index"]),
            "hold_reference": "restored_current_pose_not_runtime_snapshot",
            "gripper_reference": "existing_sim_ctrl_not_certified_hdf5_runtime_target",
            "initial_gripper_ctrl": initial_gripper_ctrl.tolist(),
            "final_gripper_ctrl": np.asarray(env.env.sim.data.ctrl[gripper_ids]).tolist(),
            "initial_gripper_command": initial_gripper.tolist(),
            "final_gripper_command": env.env.robots[0].gripper.current_action.tolist(),
            "initial_finger_qpos": initial_finger_qpos.tolist(),
            "final_finger_qpos": np.asarray(env.env.sim.data.qpos[finger_ids]).tolist(),
            "max_finger_position_change_m": float(np.max(np.abs(
                initial_finger_qpos - env.env.sim.data.qpos[finger_ids]
            ))),
            "gripper_actuator_controls_unchanged": bool(np.array_equal(
                initial_gripper_ctrl, env.env.sim.data.ctrl[gripper_ids]
            )),
            "initial_physics_sha256": _hash_array(initial_state),
            "final_physics_sha256": _hash_array(final_state),
            "dt": dt,
            "planned_steps": trajectory.env_steps,
            "sim_time_delta": float(env.env.sim.data.time) - initial_time,
            "wall_seconds": time.monotonic() - wall_started,
            "initial_position": list(start.position),
            "final_position": list(final_pose.position),
            "camera_displacement_m": math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(start.position, final_pose.position)
            )),
            "eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
            "gripper_command_unchanged": bool(np.array_equal(
                initial_gripper, env.env.robots[0].gripper.current_action
            )),
            "controller_override_restored": "set_goal" not in vars(env.env.robots[0].controller)
                and "grip_action" not in vars(env.env.robots[0]),
            "initial_calibration": initial_calibration,
            "final_calibration": final_calibration,
            "initial_agent_image_sha256": _hash_array(initial["agentview_image"]),
            "final_agent_image_sha256": None if observation is None else _hash_array(observation["agentview_image"]),
            "budget": ledger.as_dict(),
            "execution": result.audit_record(),
        }
    finally:
        env.close()


def assess(records: list[dict[str, Any]]) -> dict[str, Any]:
    if {r["condition"] for r in records} != {"continue", "hold-a", "hold-b", "move-x"} or len(records) != 4:
        raise ValueError("engineering assessment requires the exact four conditions")
    rows = {r["condition"]: r for r in records}
    a, b, moved, continued = (rows[k] for k in ("hold-a", "hold-b", "move-x", "continue"))
    checks = {
        "identical_initial_physics": len({r["initial_physics_sha256"] for r in records}) == 1,
        "zero_policy_calls": all(r["budget"]["used"]["policy_calls"] == 0 for r in records),
        "continue_has_zero_cost": continued["budget"]["used"] == {"env_steps": 0, "queries": 0, "policy_calls": 0},
        "continue_preserves_physics": continued["initial_physics_sha256"] == continued["final_physics_sha256"],
        "continue_preserves_observation": continued["initial_agent_image_sha256"] == continued["final_agent_image_sha256"],
        "all_executions_complete": all(r["execution"]["status"] == "acquired" for r in (a, b, moved))
            and continued["execution"]["status"] == "continued_without_acquisition",
        "hold_repeats_identical_physics": a["final_physics_sha256"] == b["final_physics_sha256"],
        "move_hold_identical_physics": a["final_physics_sha256"] == moved["final_physics_sha256"],
        "actual_move_5cm": abs(moved["camera_displacement_m"] - 0.05) <= 1e-7,
        "holds_do_not_move_camera": all(r["camera_displacement_m"] <= 1e-7 for r in (a, b)),
        "move_changes_image": a["final_agent_image_sha256"] != moved["final_agent_image_sha256"],
        "move_changes_calibration": a["final_calibration"] != moved["final_calibration"],
        "same_physical_time_budget": all(abs(r["sim_time_delta"] - r["planned_steps"] * r["dt"]) <= 1e-6 for r in records),
        "gripper_command_state_unchanged": all(r["gripper_command_unchanged"] for r in records),
        "gripper_actuator_controls_unchanged": all(r["gripper_actuator_controls_unchanged"] for r in records),
        "controller_override_restored": all(r["controller_override_restored"] for r in records),
    }
    return {
        "schema": "dsol_view_acquisition_engineering_smoke_v1",
        "status": "PASS_ENGINEERING_ONLY" if all(checks.values()) else "FAIL_ENGINEERING_ONLY",
        "formal_release": False,
        "scientific_success_rate": None,
        "checks": checks,
        "scope": "one archived development state; no VLA; no selector; no information-value conclusion",
        "limitations": [
            "kinematic camera centre, not head/wrist mechanics or collision certification",
            "sampled reference path, not continuous camera actuator dynamics",
            "current-pose hold does not certify full controller/history snapshot restoration",
            "existing gripper ctrl is preserved, but HDF5 does not certify it as the original demonstration target",
            "matched physical trajectories do not prove negligible EEF/object drift",
        ],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--render-gpu", type=int, default=0)
    args = parser.parse_args()
    if not 20 <= args.steps <= 40:
        parser.error("engineering smoke is capped to 20..40 steps per condition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; never overwrite prior evidence")
    configure_imports()
    from evaluate_dsol_libero_hdf5_views import protocol_spec_at

    protocol = json.loads(args.source_protocol.read_text())
    spec = protocol_spec_at(protocol, 0)
    records = []
    for condition in ("continue", "hold-a", "hold-b", "move-x"):
        print("engineering_smoke_start " + condition, flush=True)
        row = run_condition(spec, args, condition)
        records.append(row)
        with (args.output_dir / (condition + ".json")).open("x", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print("engineering_smoke_done " + condition + " " + row["execution"]["status"], flush=True)
    report = assess(records)
    report["source_protocol"] = str(args.source_protocol.resolve())
    report["source_protocol_sha256"] = hashlib.sha256(args.source_protocol.read_bytes()).hexdigest()
    report["python"] = sys.version
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "report": str(args.output_dir / "report.json")}, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

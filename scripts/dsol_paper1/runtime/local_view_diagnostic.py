"""Opt-in runtime bridge for local image-gradient diagnostics, not rollouts.

Separate simulator/model environments communicate through explicit NPZ files.
No work happens at import; all artifacts are new and refuse overwrite.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np


def render(config):
    """Render +/- optical-body axes with static physics and AA0 RGB identity gate."""
    import sys
    import tempfile
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import initialize, load_initial
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import masked_policy_observation
    from scripts.dsol_paper1.runtime.scan_libero_hdf5_views import _restore_reference

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts/vla_shared"))
    release = json.loads(Path(config["release"]).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dsol-local-geometry-") as temporary:
        _configure_runtime(Path(config["runtime"]), Path(release["states"][0]["hdf5"]).parent.parent, Path(temporary))
        from libero.libero.envs import OffScreenRenderEnv
        from libero_camera_pose import capture_camera_reference, install_camera_pose, rotation_matrix_to_wxyz
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration, physics_state_sha256
        from robosuite.utils.camera_utils import get_real_depth_map
        for state_index in config["state_indices"]:
            state = release["states"][state_index]
            stored = np.load(Path(state["initial_asset_record"]).parent / "policy_inputs.npz")
            candidates = release["protocol"]["state_blocks"][state_index]["candidates"]
            env = None
            try:
                with render_protocol(0) as contexts:
                    env = OffScreenRenderEnv(bddl_file_name=state["bddl_file"],
                                             camera_names=("agentview", "robot0_eye_in_hand"),
                                             camera_heights=256, camera_widths=256,
                                             camera_depths=(True, False), render_gpu_device_id=config["render_gpu"])
                    _, receipt = initialize(env, state, load_initial(state))
                    physics = physics_state_sha256(env)
                    reference = capture_camera_reference(env, camera_name="agentview",
                        table_plane_z=json.loads(Path(release["catalog"]["path"]).read_text())["table_plane_z"])
                    sim = env.env.sim
                    cid = reference["camera_id"]
                    if int(sim.model.cam_bodyid[cid]) != 0:
                        raise ValueError("Diagnostic requires a world-attached external camera")
                    for candidate_index in config["candidate_indices"]:
                        dest = out / f"render-{state_index}-{candidate_index}.npz"
                        if dest.exists():
                            raise FileExistsError(dest)
                        _restore_reference(env, reference)
                        pose = candidates[candidate_index]["pose"]
                        if pose is not None:
                            install_camera_pose(env, reference, pose)
                        calibration = agentview_camera_calibration(env)
                        matrix = calibration["camera_to_world_opencv"]
                        axes = np.diag([1., -1., -1.])
                        steps = [config["translation_step_m"]]*3 + [config["rotation_step_rad"]]*3
                        twists = [np.zeros(6)]
                        for axis in range(6):
                            for sign in (-1, 1):
                                twist = np.zeros(6)
                                twist[axis] = sign*steps[axis]
                                twists.append(twist)
                        images, depths, render_seconds = [], [], []
                        for twist in twists:
                            # For axis-isolated probes this is the exact body exponential.
                            sim.model.cam_pos[cid] = matrix[:3, 3] + matrix[:3, :3] @ twist[:3]
                            rotation = matrix[:3, :3] @ Rotation.from_rotvec(twist[3:]).as_matrix() @ axes
                            sim.model.cam_quat[cid] = rotation_matrix_to_wxyz(rotation)
                            sim.forward()
                            start = time.perf_counter()
                            env.env._update_observables(force=True)
                            observation = env.env._get_observations()
                            example, rgb, wrist = masked_policy_observation(observation, prompt=state["prompt"],
                                resize_size=224, eval_seed=state["environment_seed"],
                                camera_calibration=agentview_camera_calibration(env), sensor_control="both")
                            render_seconds.append(time.perf_counter()-start)
                            images.append(rgb)
                            # Raw GL observation -> upright optical image; policy RGB is horizontally mirrored.
                            depth = get_real_depth_map(sim, observation["agentview_depth"][:, :, 0])[::-1]
                            depths.append(np.asarray(Image.fromarray(depth.astype(np.float32)).resize((224, 224), Image.NEAREST)))
                            if not np.array_equal(wrist, stored["wrist_images"][candidate_index]):
                                raise ValueError("Wrist input changed")
                            if not np.array_equal(np.asarray(example["observation/state"], np.float32), stored["robot_states"][candidate_index]):
                                raise ValueError("Proprioception changed")
                            if physics_state_sha256(env) != physics:
                                raise ValueError("Physical state changed")
                        if not np.array_equal(images[0], stored["external_images"][candidate_index]):
                            raise ValueError("Base RGB differs from frozen asset")
                        if set(contexts) != {0}:
                            raise ValueError("AA0 context mismatch")
                        intrinsic = stored["camera_intrinsics"][candidate_index].copy()
                        # Calibration uses pixel centres at u+.5; geometry uses integer array indices.
                        intrinsic[:2, 2] -= .5
                        np.savez_compressed(dest, probe_images=np.stack(images), depth_optical=np.stack(depths),
                                            twists=np.stack(twists), geometry_intrinsics=intrinsic)
                        audit = dict(state_index=state_index, candidate_index=candidate_index,
                                     base_rgb_exact=True, wrist_exact=True, physical_state_unchanged=True,
                                     physics_sha256=physics[0], initialization=receipt,
                                     render_seconds=render_seconds, policy_transform="horizontal mirror of optical upright",
                                     depth_resize="nearest, 256 to 224; approximate at silhouette boundaries")
                        write_new(dest.with_suffix(".json"), audit)
                        print(json.dumps(audit), flush=True)
            finally:
                if env is not None:
                    env.close()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def score(config, model_name, *, rendered=False):
    import torch
    from AlphaBrain.model.framework.base_framework import BaseFramework
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores, shared_flow_noise
    from AlphaBrain.research.dsol.metrics.accel_torch import compute_accel_score_tensor
    from AlphaBrain.research.dsol.metrics.pi05_differentiable import velocity_prefix

    release = json.loads(Path(config["release"]).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / (model_name + ("-rendered" if rendered else "-archived") + ".json")
    if result_path.exists():
        raise FileExistsError(result_path)
    device = torch.device("cuda:0")
    model = BaseFramework.from_pretrained(release["models"][model_name]["path"], strict_checkpoint=True)
    model = model.to(torch.bfloat16).to(device).eval()
    model.gripper_remap = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows = []
    for state_index in config["state_indices"]:
        state = release["states"][state_index]
        bank_path = Path(state["initial_asset_record"]).parent / "policy_inputs.npz"
        with np.load(bank_path) as stored:
            for candidate_index in config["candidate_indices"]:
                data = dict(stored)
                if rendered:
                    render_root = Path(config.get("render_input", config["output"]))
                    data.update(dict(np.load(render_root / f"render-{state_index}-{candidate_index}.npz")))
                    indices = range(len(data["probe_images"]))
                else:
                    indices = [0]
                base = {"lang": state["prompt"], "state": data["robot_states"][candidate_index],
                        "camera_intrinsics": data["camera_intrinsics"][candidate_index],
                        "camera_to_world_opencv": data["camera_to_world_opencv"][candidate_index]}
                for seed in config["score_seeds"]:
                    noise = shared_flow_noise(seed=seed, candidate_count=1,
                                              action_horizon=model.action_horizon, action_dim=model.action_dim)
                    for probe_index in indices:
                        rgb = (data["probe_images"][probe_index] if rendered
                               else data["external_images"][candidate_index])
                        wrist = data["wrist_images"][candidate_index]
                        example = dict(base, image=[rgb, wrist])
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        reference = model.predict_action(examples=[example], noise=noise, return_flow_trace=True)
                        torch.cuda.synchronize()
                        ref_seconds = time.perf_counter() - start
                        ref_trace = np.asarray(reference["flow_velocity_trace"], dtype=np.float32)
                        ref_score = float(compute_accel_scores(ref_trace, prefixes=(3,))[3]["score"][0])
                        row = dict(state_index=state_index, candidate_index=candidate_index,
                                   probe_index=probe_index, seed=seed, score=ref_score,
                                   full_forward_seconds=ref_seconds)
                        if probe_index == 0:
                            leaf = torch.tensor(rgb.transpose(2, 0, 1).copy(),
                                                dtype=torch.float32, requires_grad=True)
                            example = dict(base, image=[leaf, wrist])
                            torch.cuda.reset_peak_memory_stats()
                            start = time.perf_counter()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                trace, times = velocity_prefix(model, [example], noise)
                                metric = compute_accel_score_tensor(trace)
                            torch.cuda.synchronize()
                            forward_seconds = time.perf_counter() - start
                            delta = float(np.max(np.abs(trace.detach().float().cpu().numpy()-ref_trace[:, :3])))
                            if delta != 0 or not np.array_equal(times.detach().cpu().numpy(), np.array(reference["flow_times"], np.float32)[:3]):
                                raise ValueError(f"Golden trace/time mismatch: {delta}")
                            if bool(metric["degenerate"].any()):
                                raise ValueError("Degenerate Accel")
                            start = time.perf_counter()
                            grad, = torch.autograd.grad(metric["score"].mean(), leaf)
                            torch.cuda.synchronize()
                            backward_seconds = time.perf_counter() - start
                            gradient = grad.detach().float().cpu().numpy().transpose(1, 2, 0)
                            if not np.isfinite(gradient).all() or np.linalg.norm(gradient) == 0:
                                raise ValueError("Image gradient is nonfinite or zero")
                            filename = f"gradient-{model_name}-{state_index}-{candidate_index}-{seed}.npz"
                            gradient_path = out / filename
                            if gradient_path.exists():
                                if not np.array_equal(np.load(gradient_path)["gradient"], gradient):
                                    raise ValueError("Existing diagnostic gradient differs; do not overwrite")
                            else:
                                with gradient_path.open("xb") as stream:
                                    np.savez_compressed(stream, gradient=gradient)
                            row.update(trace_max_abs_error=delta, differentiable_score=float(metric["score"].item()),
                                       prefix_forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                       gradient_norm=float(np.linalg.norm(gradient)), gradient_file=str(gradient_path))
                            del trace, metric, grad, leaf
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    write_new(result_path, dict(model=model_name, diagnostic_only=True, config=config, rows=rows))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("mode", choices=("score", "render"))
    parser.add_argument("--model", choices=("canonical", "broad"), default="canonical")
    parser.add_argument("--rendered", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.mode == "score":
        score(config, args.model, rendered=args.rendered)
    else:
        render(config)


if __name__ == "__main__":
    main()

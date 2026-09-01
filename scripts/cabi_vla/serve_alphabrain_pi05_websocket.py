from __future__ import annotations

import argparse
import hashlib
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED_OBSERVATION_KEYS = {
    "observation/image",
    "observation/wrist_image",
    "observation/state",
    "prompt",
    "_eval_seed",
}
CAMERA_CALIBRATION_KEYS = (
    "camera_intrinsics",
    "camera_to_world_opencv",
)


def to_alphabrain_example(observation: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    missing = REQUIRED_OBSERVATION_KEYS - set(observation)
    if missing:
        raise KeyError(f"missing policy observation keys: {sorted(missing)}")
    seed = int(observation["_eval_seed"])
    agent = np.asarray(observation["observation/image"], dtype=np.uint8)
    wrist = np.asarray(observation["observation/wrist_image"], dtype=np.uint8)
    state = np.asarray(observation["observation/state"], dtype=np.float32)
    if agent.ndim != 3 or agent.shape[-1] != 3 or wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError("agent and wrist images must be HxWx3")
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"expected finite 8D LIBERO state, got {state.shape}")
    prompt = str(observation["prompt"])
    example = {
        "image": [np.ascontiguousarray(agent), np.ascontiguousarray(wrist)],
        "lang": prompt,
        "language": prompt,
        "state": state,
    }
    has_camera_calibration = tuple(key in observation for key in CAMERA_CALIBRATION_KEYS)
    if any(has_camera_calibration) and not all(has_camera_calibration):
        raise ValueError(
            "camera metadata requires both camera_intrinsics and "
            "camera_to_world_opencv"
        )
    if all(has_camera_calibration):
        intrinsics = np.asarray(observation["camera_intrinsics"])
        camera_to_world = np.asarray(observation["camera_to_world_opencv"])
        if intrinsics.shape != (3, 3) or camera_to_world.shape != (4, 4):
            raise ValueError("invalid camera calibration matrix shapes")
        if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(camera_to_world)):
            raise ValueError("camera calibration matrices must be finite")
        example.update(
            {
                "camera_intrinsics": np.ascontiguousarray(intrinsics),
                "camera_to_world_opencv": np.ascontiguousarray(camera_to_world),
            }
        )
    return example, seed


class AlphaBrainPi05Policy:
    def __init__(self, checkpoint: Path, device: str) -> None:
        from AlphaBrain.model.framework.base_framework import BaseFramework

        self._checkpoint = checkpoint.resolve()
        self._model = BaseFramework.from_pretrained(
            str(self._checkpoint),
            strict_checkpoint=True,
        )
        self._model = self._model.to(torch.bfloat16).to(device).eval()
        self._model.gripper_remap = False
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._horizon = int(self._model.action_horizon)

    @property
    def metadata(self) -> dict[str, Any]:
        device = next(self._model.parameters()).device
        return {
            "framework": "AlphaBrain",
            "policy": "Pi0.5",
            "checkpoint": str(self._checkpoint),
            "action_horizon": self._horizon,
            "action_dim": 7,
            "deterministic_eval_noise": True,
            "explicit_flow_noise": True,
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "device": str(device),
        }

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        example, seed = to_alphabrain_example(observation)
        explicit_noise = observation.get("_eval_noise")
        noise_sha256 = None
        if explicit_noise is not None:
            explicit_noise = np.ascontiguousarray(explicit_noise, dtype=np.float32)
            expected_noise_shape = (self._horizon, 7)
            if explicit_noise.shape != expected_noise_shape or not np.all(np.isfinite(explicit_noise)):
                raise ValueError(f"invalid explicit flow noise: {explicit_noise.shape}")
            noise_sha256 = hashlib.sha256(explicit_noise.tobytes(order="C")).hexdigest()
            claimed_sha256 = observation.get("_eval_noise_sha256")
            if claimed_sha256 is not None and str(claimed_sha256) != noise_sha256:
                raise ValueError("explicit flow-noise SHA-256 mismatch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.predict_action(
                examples=[example],
                noise=None if explicit_noise is None else explicit_noise[None],
            )
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["normalized_actions"][0], dtype=np.float32)
        expected = (self._horizon, 7)
        if actions.shape != expected or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid AlphaBrain Pi0.5 output: {actions.shape}")
        clipped_actions = np.ascontiguousarray(np.clip(actions, -1.0, 1.0), dtype=np.float32)
        return {
            "actions": clipped_actions,
            "predict_action_wall_seconds": elapsed,
            "explicit_flow_noise": explicit_noise is not None,
            "noise_sha256": noise_sha256,
            "action_chunk_sha256": hashlib.sha256(clipped_actions.tobytes(order="C")).hexdigest(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an AlphaBrain Pi0.5 checkpoint over OpenPI WebSocket")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from openpi.serving import websocket_policy_server

    args = parse_args()
    policy = AlphaBrainPi05Policy(args.checkpoint, args.device)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("serving AlphaBrain Pi0.5 from %s on port %d", args.checkpoint, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

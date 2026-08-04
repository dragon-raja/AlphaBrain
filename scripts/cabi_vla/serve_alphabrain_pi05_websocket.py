from __future__ import annotations

import argparse
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
    return (
        {
            "image": [np.ascontiguousarray(agent), np.ascontiguousarray(wrist)],
            "lang": prompt,
            "language": prompt,
            "state": state,
        },
        seed,
    )


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
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "device": str(device),
        }

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        example, seed = to_alphabrain_example(observation)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.predict_action(examples=[example])
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["normalized_actions"][0], dtype=np.float32)
        expected = (self._horizon, 7)
        if actions.shape != expected or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid AlphaBrain Pi0.5 output: {actions.shape}")
        return {
            "actions": np.clip(actions, -1.0, 1.0),
            "predict_action_wall_seconds": elapsed,
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

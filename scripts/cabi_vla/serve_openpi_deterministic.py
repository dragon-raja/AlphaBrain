from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np


class DeterministicNoisePolicy:
    """Attach request-keyed flow noise to an OpenPI policy."""

    def __init__(self, policy: Any, *, action_horizon: int, action_dim: int):
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action horizon and dimension must be positive")
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            **dict(self._policy.metadata),
            "deterministic_eval_noise": True,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
        }

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        values = dict(observation)
        if "_eval_seed" not in values:
            raise KeyError("deterministic policy requests require _eval_seed")
        seed = int(values.pop("_eval_seed"))
        generator = np.random.default_rng(seed)
        noise = generator.standard_normal(
            (self._action_horizon, self._action_dim),
            dtype=np.float32,
        )
        return self._policy.infer(values, noise=noise)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenPI with paired deterministic flow noise")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="none",
    )
    return parser.parse_args()


def main() -> None:
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    args = parse_args()
    config = training_config.get_config(args.config)
    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    if config.model.pytorch_compile_mode != compile_mode:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, pytorch_compile_mode=compile_mode),
        )
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint,
        pytorch_device=args.device,
    )
    model = policy._model
    wrapped = DeterministicNoisePolicy(
        policy,
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=wrapped.metadata,
    )
    logging.info(
        "serving deterministic %s on port %d from %s",
        args.config,
        args.port,
        args.checkpoint,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

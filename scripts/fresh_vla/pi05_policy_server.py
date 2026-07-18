from __future__ import annotations

import argparse
import os
import signal
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from AlphaBrain.model.framework.base_framework import BaseFramework


def runtime_identity(torch_module, device_index: int = 0) -> dict[str, str | None]:
    return {
        "torch_version": str(torch_module.__version__),
        "cuda_version": None if torch_module.version.cuda is None else str(torch_module.version.cuda),
        "device_name": (
            str(torch_module.cuda.get_device_name(device_index)) if torch_module.cuda.is_available() else "cpu"
        ),
    }


def validate_policy_example(example: dict) -> None:
    if set(example) != {"image", "lang", "language", "state"}:
        raise ValueError(f"unexpected policy example keys: {sorted(example)}")


def coupled_flow_noise(
    torch_module,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
    device,
):
    """Create one flow-noise draw and repeat it exactly across observations."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    one = torch_module.randn(1, horizon, action_dim, dtype=torch_module.float32, device=device)
    return one.expand(batch_size, -1, -1).clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Unix-socket Pi0.5 inference server")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")
    args.socket.unlink(missing_ok=True)
    model = BaseFramework.from_pretrained(str(args.checkpoint))
    model = model.to(torch.bfloat16).to(args.device).eval()
    model.gripper_remap = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    horizon = int(model.action_horizon)

    def predict(example, seed: int) -> tuple[np.ndarray, float]:
        validate_policy_example(example)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.predict_action(examples=[example])
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["normalized_actions"][0], dtype=np.float32)
        if actions.shape != (horizon, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid action output: shape={actions.shape}")
        return np.clip(actions, -1.0, 1.0), elapsed

    def predict_sample_batch(example, count: int, seed: int) -> tuple[np.ndarray, float]:
        validate_policy_example(example)
        if not 1 <= count <= 16:
            raise ValueError("predict_sample_batch requires count in [1, 16]")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.predict_action(examples=[example] * count)
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        if actions.shape != (count, horizon, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid batched action output: shape={actions.shape}")
        return np.clip(actions, -1.0, 1.0), elapsed

    def predict_observation_batch(examples, seed: int) -> tuple[np.ndarray, float]:
        if not 1 <= len(examples) <= 16:
            raise ValueError("predict_observation_batch requires 1 to 16 examples")
        for example in examples:
            validate_policy_example(example)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.predict_action(examples=examples)
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        expected = (len(examples), horizon, 7)
        if actions.shape != expected or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid observation-batch output: shape={actions.shape}")
        return np.clip(actions, -1.0, 1.0), elapsed

    def predict_observation_batch_coupled(examples, seed: int) -> tuple[np.ndarray, float]:
        if not 1 <= len(examples) <= 16:
            raise ValueError("predict_observation_batch_coupled requires 1 to 16 examples")
        for example in examples:
            validate_policy_example(example)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = next(model.parameters()).device
        noise = coupled_flow_noise(
            torch,
            batch_size=len(examples),
            horizon=horizon,
            action_dim=int(model.action_dim),
            device=device,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.predict_action(examples=examples, noise=noise)
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        expected = (len(examples), horizon, 7)
        if actions.shape != expected or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid coupled observation-batch output: shape={actions.shape}")
        return np.clip(actions, -1.0, 1.0), elapsed

    def extract_feature(example) -> tuple[np.ndarray, float]:
        from AlphaBrain.model.pi05_features import extract_pi05_image_feature

        validate_policy_example(example)
        started = time.perf_counter()
        with torch.inference_mode():
            feature = extract_pi05_image_feature(model, example)[0]
        elapsed = time.perf_counter() - started
        array = feature.detach().to(torch.float16).cpu().numpy()
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid frozen feature: shape={array.shape}")
        return array, elapsed

    def stop(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    listener = Listener(str(args.socket), family="AF_UNIX", authkey=b"fresh-vla-local")
    identity = runtime_identity(torch)
    print(f"policy_server_ready socket={args.socket} horizon={horizon}", flush=True)
    try:
        while True:
            connection = listener.accept()
            try:
                connection.send(
                    {
                        "horizon": horizon,
                        "checkpoint_realpath": str(args.checkpoint.resolve()),
                        "model_size_bytes": (args.checkpoint / "model.safetensors").stat().st_size,
                        **identity,
                    }
                )
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    if request.get("op") == "close":
                        break
                    if request.get("op") not in {
                        "predict",
                        "predict_many",
                        "predict_sample_batch",
                        "predict_observation_batch",
                        "predict_observation_batch_coupled",
                        "extract_feature",
                    }:
                        connection.send({"error": f"unknown operation: {request.get('op')!r}"})
                        continue
                    try:
                        if request["op"] == "predict":
                            actions, elapsed = predict(request["example"], int(request["seed"]))
                            connection.send({"actions": actions.tolist(), "predict_action_wall_seconds": elapsed})
                        elif request["op"] == "predict_many":
                            seeds = [int(seed) for seed in request["seeds"]]
                            if not 1 <= len(seeds) <= 16:
                                raise ValueError("predict_many requires 1 to 16 seeds")
                            outputs = [predict(request["example"], seed) for seed in seeds]
                            connection.send(
                                {
                                    "actions": [actions.tolist() for actions, _ in outputs],
                                    "predict_action_wall_seconds": sum(elapsed for _, elapsed in outputs),
                                    "per_call_wall_seconds": [elapsed for _, elapsed in outputs],
                                }
                            )
                        elif request["op"] == "predict_sample_batch":
                            actions, elapsed = predict_sample_batch(
                                request["example"], int(request["count"]), int(request["seed"])
                            )
                            connection.send(
                                {
                                    "actions": actions.tolist(),
                                    "predict_action_wall_seconds": elapsed,
                                }
                            )
                        elif request["op"] == "predict_observation_batch":
                            actions, elapsed = predict_observation_batch(
                                request["examples"], int(request["seed"])
                            )
                            connection.send(
                                {
                                    "actions": actions.tolist(),
                                    "predict_action_wall_seconds": elapsed,
                                }
                            )
                        elif request["op"] == "predict_observation_batch_coupled":
                            actions, elapsed = predict_observation_batch_coupled(
                                request["examples"], int(request["seed"])
                            )
                            connection.send(
                                {
                                    "actions": actions.tolist(),
                                    "predict_action_wall_seconds": elapsed,
                                    "coupled_flow_noise": True,
                                }
                            )
                        else:
                            feature, elapsed = extract_feature(request["example"])
                            connection.send(
                                {
                                    "feature": feature,
                                    "feature_wall_seconds": elapsed,
                                }
                            )
                    except Exception as error:
                        connection.send({"error": f"{type(error).__name__}: {error}"})
            finally:
                connection.close()
    finally:
        listener.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

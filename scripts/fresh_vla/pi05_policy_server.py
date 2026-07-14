from __future__ import annotations

import argparse
import os
import signal
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from AlphaBrain.model.framework.base_framework import BaseFramework


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

    def stop(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    listener = Listener(str(args.socket), family="AF_UNIX", authkey=b"fresh-vla-local")
    print(f"policy_server_ready socket={args.socket} horizon={horizon}", flush=True)
    try:
        while True:
            connection = listener.accept()
            try:
                connection.send({"horizon": horizon})
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    if request.get("op") == "close":
                        break
                    if request.get("op") != "predict":
                        connection.send({"error": f"unknown operation: {request.get('op')!r}"})
                        continue
                    seed = int(request["seed"])
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    example = request["example"]
                    with torch.inference_mode():
                        output = model.predict_action(examples=[example])
                    actions = np.asarray(output["normalized_actions"][0], dtype=np.float32)
                    if actions.shape != (horizon, 7) or not np.all(np.isfinite(actions)):
                        connection.send({"error": f"invalid action output: shape={actions.shape}"})
                        continue
                    connection.send({"actions": np.clip(actions, -1.0, 1.0).tolist()})
            finally:
                connection.close()
    finally:
        listener.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

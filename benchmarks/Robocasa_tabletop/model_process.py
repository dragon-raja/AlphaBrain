"""Isolated local model-call transport with a live parent grant per call.

The executable supports synthetic CPU calls and real CPU checkpoint loading /
preprocessing. Real GR1 inference remains behind its unregistered production gate. This is a worker
child connection, not a scheduler, network service or recovery daemon.
"""

from contextlib import redirect_stdout
from multiprocessing.connection import Listener, Client
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _ModelProcess:
    """One local child with fixed identity, live call grants and no reconnect."""

    def __init__(
        self,
        python,
        source_root,
        *,
        guard,
        log_path,
        timeout=30.0,
        model_config=None,
        import_paths=(),
        load_guard=lambda: None,
    ):
        if timeout <= 0 or timeout > 300:
            raise ValueError("Bounded local IPC timeout required")
        load_guard()
        self.guard, self.timeout = guard, float(timeout)
        self.load_guard = load_guard
        self.process = None
        self.connection = None
        self.closed = False
        self.sequence = 0
        self.model_config = model_config
        from .model_runtime import ScoreContract

        self.contract = (
            SimpleNamespace(execute_actions=16)
            if model_config is None
            else ScoreContract(**model_config["score_contract"])
        )
        self.temporary = tempfile.TemporaryDirectory(prefix="gr1-cpu-model-")
        address = str(Path(self.temporary.name) / "worker.sock")
        try:
            args = ["--cpu-contract"]
            if model_config is not None:
                path = Path(self.temporary.name) / "model.json"
                path.write_text(json.dumps(model_config, sort_keys=True))
                args = ["--model-config", str(path)]
            with Listener(address, family="AF_UNIX") as listener:
                with Path(log_path).open("xb") as log:
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES="",
                        JAX_PLATFORMS="cpu",
                        WANDB_MODE="disabled",
                        PYTHONPATH=os.pathsep.join(
                            [str(Path(source_root).resolve()), *map(str, import_paths)]
                        ),
                    )
                    self.process = subprocess.Popen(
                        [str(python), "-m", __name__, *args, "--socket", address],
                        cwd=source_root,
                        env=env,
                        stdout=log,
                        stderr=log,
                    )
                deadline = time.monotonic() + self.timeout
                while self.connection is None:
                    if self.process.poll() is not None:
                        raise RuntimeError("CPU child exited before handshake")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CPU child handshake timeout")
                    listener._listener._socket.settimeout(min(1.0, deadline - time.monotonic()))
                    try:
                        self.connection = listener.accept()
                    except TimeoutError:
                        pass
            hello = self._receive()
            expected = {
                "kind": "CPU_CONTRACT_ONLY" if model_config is None else "MODEL_LOADED_CPU_ONLY",
                "pid": self.process.pid,
                "source": source_identity(),
            }
            if model_config is not None:
                expected["config_sha256"] = hashlib.sha256(
                    json.dumps(model_config, sort_keys=True).encode()
                ).hexdigest()
            if hello != expected:
                raise RuntimeError("Child identity differs")
            load_guard()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError("Unknown local worker response")
        result = self.connection.recv()
        if not isinstance(result, dict) or result.get("kind") == "ERROR":
            raise RuntimeError("Local worker rejected request")
        return result

    def infer(self, example, noise, *, request):
        if self.closed:
            raise RuntimeError("Failed worker cannot reconnect or replay")
        try:
            if self.model_config is not None:
                from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production

                reject_production(
                    "benchmarks/Robocasa_tabletop/model_runtime.py:GR1ModelRuntime.infer"
                )
            self.sequence += 1
            self.connection.send(
                dict(
                    kind="INFER",
                    sequence=self.sequence,
                    example=example,
                    noise=noise,
                    request=request,
                )
            )
            intent = self._receive()
            if intent != {"kind": "AUTHORIZE", "sequence": self.sequence, "request": request}:
                raise RuntimeError("Worker call identity differs")
            self.guard(request)  # The caller's existing authority/ledger owns this grant.
            self.connection.send(dict(kind="GO", sequence=self.sequence))
            result = self._receive()
            if result.get("kind") != "RESULT" or result.get("sequence") != self.sequence:
                raise RuntimeError("Unknown worker result")
            return result["value"]
        except BaseException:
            self.close()
            raise

    def observation_shapes(self, example):
        """Actual processor only: no vision backbone, scoring or action model."""
        if self.closed or self.model_config is None:
            raise RuntimeError("Loaded real CPU model required")
        try:
            self.load_guard()
            self.sequence += 1
            self.connection.send(
                dict(kind="OBSERVATION_SHAPES", sequence=self.sequence, example=example)
            )
            result = self._receive()
            if (
                result.get("kind") != "OBSERVATION_SHAPES"
                or result.get("sequence") != self.sequence
            ):
                raise RuntimeError("Unknown preprocessing result")
            self.load_guard()
            return result["shapes"]
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        if self.connection is not None:
            self.connection.close()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CPUModelProcess(_ModelProcess):
    """Explicit synthetic worker for cross-interpreter integration/fault tests."""

    def __init__(self, python, source_root, *, guard, log_path, timeout=30.0):
        super().__init__(python, source_root, guard=guard, log_path=log_path, timeout=timeout)


class GR1ModelProcess(_ModelProcess):
    """Real checkpoint/processor in its own CPU interpreter; infer stays gated.

    The gate is enforced both here and in GR1ModelRuntime in the child. Loading
    weights and preprocessing alone do not grant a model execution permit.
    """

    def __init__(
        self,
        python,
        source_root,
        *,
        config,
        import_paths,
        guard,
        load_guard,
        log_path,
        timeout=180.0,
    ):
        from .model_runtime import ScoreContract

        if (
            set(config)
            != {
                "package",
                "score_contract",
                "expected_config_sha256",
                "expected_norm_sha256",
                "expected_tokenizer_sha256",
            }
            or not Path(config["package"]).is_absolute()
        ):
            raise ValueError("Explicit model paths and immutable processor identities required")
        ScoreContract(**config["score_contract"]).validate()
        super().__init__(
            python,
            source_root,
            guard=guard,
            log_path=log_path,
            timeout=timeout,
            model_config=config,
            import_paths=import_paths,
            load_guard=load_guard,
        )


def _cpu_worker(address):
    import numpy as np

    seen = set()
    sequence = 0
    with Client(address, family="AF_UNIX") as channel:
        channel.send(dict(kind="CPU_CONTRACT_ONLY", pid=os.getpid(), source=source_identity()))
        while True:
            try:
                row = channel.recv()
            except EOFError:
                return
            try:
                sequence += 1
                if row.get("kind") != "INFER" or row.get("sequence") != sequence:
                    raise ValueError("sequence")
                request = row["request"]
                key = (request["query_id"], request["epoch"], request["member"])
                if key in seen:
                    raise ValueError("duplicate")
                seen.add(key)
                channel.send(dict(kind="AUTHORIZE", sequence=sequence, request=request))
                if channel.recv() != dict(kind="GO", sequence=sequence):
                    raise ValueError("grant")
                state = np.asarray(row["example"]["observation/state"])
                if state.shape != (29,) or np.asarray(row["noise"]).shape != (16, 32):
                    raise ValueError("shape")
                value = dict(
                    feature=np.ones(4608, np.float32),
                    score=1.0 + float(np.square(state[:2] - 3).sum()) / 100,
                    actions=np.zeros((16, 29), np.float32),
                    synthetic=True,
                )
                channel.send(dict(kind="RESULT", sequence=sequence, value=value))
            except (EOFError, BrokenPipeError):
                return
            except Exception:
                channel.send(dict(kind="ERROR", sequence=sequence))
                return


def _model_worker(address, config):
    from .model_runtime import GR1ModelRuntime, ScoreContract
    import jax

    with Client(address, family="AF_UNIX") as channel:
        sequence = 0

        def grant(request):
            channel.send(dict(kind="AUTHORIZE", sequence=sequence, request=request))
            if channel.recv() != dict(kind="GO", sequence=sequence):
                raise PermissionError("Live parent grant missing")

        options = dict(config)
        options["score_contract"] = ScoreContract(**options["score_contract"])
        runtime = GR1ModelRuntime(**options, call_guard=grant, load_guard=lambda: None)
        channel.send(
            dict(
                kind="MODEL_LOADED_CPU_ONLY",
                pid=os.getpid(),
                source=source_identity(),
                config_sha256=hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest(),
            )
        )
        seen = set()
        while True:
            try:
                row = channel.recv()
            except EOFError:
                return
            try:
                sequence += 1
                if row.get("sequence") != sequence:
                    raise ValueError("sequence")
                if row.get("kind") == "OBSERVATION_SHAPES":
                    obs = runtime.observation(row["example"])
                    shapes = jax.tree.map(
                        lambda x: dict(shape=list(x.shape), dtype=str(x.dtype)), obs.to_dict()
                    )
                    channel.send(dict(kind="OBSERVATION_SHAPES", sequence=sequence, shapes=shapes))
                elif row.get("kind") == "INFER":
                    request = row["request"]
                    key = (request["query_id"], request["epoch"], request["member"])
                    if key in seen:
                        raise ValueError("duplicate")
                    seen.add(key)
                    value = runtime.infer(row["example"], row["noise"], request=request)
                    channel.send(dict(kind="RESULT", sequence=sequence, value=value))
                else:
                    raise ValueError("Unknown worker operation")
            except (EOFError, BrokenPipeError):
                return
            except Exception:
                channel.send(dict(kind="ERROR", sequence=sequence))
                return


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-contract", action="store_true")
    mode.add_argument("--model-config", type=Path)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    if args.cpu_contract:
        _cpu_worker(args.socket)
    else:
        _model_worker(args.socket, json.loads(args.model_config.read_text()))


if __name__ == "__main__":
    main()

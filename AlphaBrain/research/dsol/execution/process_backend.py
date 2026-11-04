"""Shared host-local process ownership and lifecycle for queue adapters.

No benchmark imports or scientific validation. Subclasses supply admission,
worker commands and result validation. Direct production launch is closed;
current live model workers use the separately authorized model_agent path.
"""

import os
from pathlib import Path
import signal
import subprocess
import time

from ..data.artifacts import read
from .queue import _durable_write


def process_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(
        pid=int(pid),
        start=fields[19],
        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        state=fields[0],
        group=int(fields[2]),
    )


def group_members(group):
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            row = process_identity(int(path.name))
            if row is not None and row["group"] == group and row["state"] != "Z":
                members.append(row)
    return members


class LocalProcessBackend:
    def admit(self, slot):
        raise NotImplementedError("Adapter must validate its scientific admission")

    def worker_command(self, task):
        raise NotImplementedError("Adapter must construct an explicit worker command")

    def validate(self, task, **kwargs):
        raise NotImplementedError("Adapter must validate its scientific results")

    def __init__(self, plan, root):
        self.plan, self.root = plan, Path(root)
        self.processes = {}
        self.logs = {}
        self.runtime = plan["runtime"]
        self.handles = self.root / "handles"
        self.handles.mkdir(exist_ok=True)

    def gpu_available(self, slot):
        """Fresh supervisor + /proc + GPU telemetry, including real memory use."""
        from AlphaBrain.research.dsol.execution.gpu_guard import preflight

        config = self.runtime.get("gpu_guard")
        if not isinstance(config, dict):
            raise RuntimeError("Explicit supervisor-backed gpu_guard configuration required")
        if config["profile"] != slot["host"] or slot["host"] != self.runtime["host"]:
            raise RuntimeError("GPU admission profile differs from queue host")
        config = dict(config)
        if self.runtime.get("workers_per_gpu", 1) > 1:
            if self.runtime["workers_per_gpu"] != 2:
                raise ValueError("Only admitted dual-worker capacity supported")
            config["minimum_free_mib"] = max(13500, config["minimum_free_mib"])
        owned = []
        state = self.root / "state.json"
        if state.exists():
            for row in read(state)["tasks"].values():
                if row["status"] == "RUNNING":
                    handle = row["handle"]
                    device = str(row["slot"].get("gpu_id", row["slot"]["id"]))
                    devices = {str(g["index"]): g["uuid"] for g in config["devices"]}
                    owned.append(
                        dict(
                            pid=handle["pid"],
                            start=handle["start"],
                            boot=handle["boot"],
                            group=handle["group"],
                            gpu_uuid=devices[device],
                        )
                    )
        result = preflight(config, owned=owned)
        self.last_gpu_admission = result
        if result["status"] == "MONITOR_ERROR":
            raise RuntimeError("GPU monitoring failed closed: " + result["error"])
        device = str(slot.get("gpu_id", slot["id"]))
        rows = [r for r in result["rows"] if str(r["index"]) == device]
        if len(rows) != 1:
            raise RuntimeError("Requested GPU missing from monitored inventory")
        return rows[0]["allowed"]

    def start(self, task, slot, attempt):
        from AlphaBrain.research.dsol.execution import dispatch_policy

        dispatch_policy.reject_production(
            "AlphaBrain/research/dsol/execution/process_backend.py:LocalProcessBackend.start"
        )
        receipt = self.handles / f'{task["key"]}-{attempt}.json'
        if receipt.exists() or Path(task["output"]).exists():
            raise ValueError("Existing work requires explicit recovery")
        args = self.worker_command(task)
        try:
            if not self.gpu_available(slot):
                raise RuntimeError("GPU ownership or memory changed before launch")
        except Exception as error:
            # Durable proof that Popen was never reached. Queue cleanup must not
            # mistake an intentionally refused launch for an unresolved gap.
            _durable_write(
                receipt,
                dict(key=task["key"], output=task["output"], not_started=True, reason=str(error)),
            )
            raise
        env = {
            **os.environ,
            **self.runtime["environment"],
            "CUDA_VISIBLE_DEVICES": slot.get("gpu_id", slot["id"]),
        }
        log = (self.root / f'{task["key"]}-{attempt}.log').open("xb")
        process = subprocess.Popen(
            args,
            env=env,
            cwd=self.runtime["source"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        identity = process_identity(process.pid)
        if identity is None:
            raise RuntimeError("Worker exited before durable identity; inspect launch log")
        handle = dict(
            **identity, output=task["output"], command=args, receipt=str(receipt), key=task["key"]
        )
        _durable_write(receipt, handle)
        self.processes[process.pid] = process
        self.logs[process.pid] = log
        return handle

    def locate(self, task, slot, attempt):
        receipt = self.handles / f'{task["key"]}-{attempt}.json'
        if not receipt.exists():
            raise RuntimeError(
                "Unresolved launch gap: manual process audit required, never auto-reissue"
            )
        handle = read(receipt)
        if handle["key"] != task["key"] or handle["output"] != task["output"]:
            raise ValueError("Launch receipt mismatch")
        if handle.get("not_started") is True:
            return None
        return handle

    def alive(self, handle):
        current = process_identity(handle["pid"])
        if current is not None and any(current[k] != handle[k] for k in ("start", "boot", "group")):
            raise RuntimeError("PID identity changed; no signal or recovery allowed")
        return bool(group_members(handle["group"]))

    def poll(self, handle):
        process = self.processes.get(handle["pid"])
        if process is not None:
            code = process.poll()
            if code is not None:
                self.logs[handle["pid"]].close()
            return code
        if self.alive(handle):
            return None
        path = Path(handle["output"]) / "status.json"
        return 0 if path.exists() and read(path)["status"] == "COMPLETE" else 1

    def cancel(self, handle):
        def reap():
            process = self.processes.get(handle["pid"])
            if process is not None:
                process.wait(timeout=10)
            log = self.logs.get(handle["pid"])
            if log is not None:
                log.close()

        if not self.alive(handle):
            reap()
            return
        current = process_identity(handle["pid"])
        if current is None:
            raise RuntimeError("Orphan process group: manual audit required")
        os.killpg(handle["group"], signal.SIGTERM)
        for _ in range(30):
            if not self.alive(handle):
                reap()
                return
            time.sleep(0.1)
        os.killpg(handle["group"], signal.SIGKILL)
        reap()

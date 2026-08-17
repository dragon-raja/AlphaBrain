from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 15
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
GIT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
GPU_TEXT_RE = re.compile(r"[A-Za-z0-9 ._()+/-]{1,128}\Z")
GPU_ID_RE = re.compile(r"[A-Za-z0-9-]{1,96}\Z")
INTEGER_RE = re.compile(r"[0-9]+\Z")
DEVICE_LIST_RE = re.compile(r"(?:all|none|void|[A-Za-z0-9_.:/-]+(?:,[A-Za-z0-9_.:/-]+)*)\Z")
LOCALE_RE = re.compile(r"[A-Za-z0-9_.@-]{1,64}\Z")
TIMEZONE_RE = re.compile(r"[A-Za-z0-9_+./-]{1,64}\Z")

SENSITIVE_TOKENS = frozenset(
    {
        "auth",
        "authorization",
        "credential",
        "credentials",
        "key",
        "password",
        "passwd",
        "secret",
        "token",
    }
)
SENSITIVE_COMPONENTS = frozenset(
    {
        ".codex",
        ".ssh",
        "authorized_keys",
        "clash",
        "codex",
        "env.sh",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "known_hosts",
    }
)
SENSITIVE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
SENSITIVE_COMPOUNDS = (
    "accesstoken",
    "apikey",
    "authtoken",
    "clientsecret",
    "privatekey",
    "refreshtoken",
)


def _one_of(*values: str) -> Callable[[str], bool]:
    allowed = frozenset(values)
    return lambda value: value in allowed


ENVIRONMENT_WHITELIST: dict[str, Callable[[str], bool]] = {
    "CUDA_DEVICE_ORDER": _one_of("PCI_BUS_ID", "FASTEST_FIRST"),
    "CUDA_VISIBLE_DEVICES": lambda value: bool(DEVICE_LIST_RE.fullmatch(value)),
    "LANG": lambda value: bool(LOCALE_RE.fullmatch(value)),
    "LC_ALL": lambda value: bool(LOCALE_RE.fullmatch(value)),
    "LOCAL_RANK": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "MKL_NUM_THREADS": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "NVIDIA_VISIBLE_DEVICES": lambda value: bool(DEVICE_LIST_RE.fullmatch(value)),
    "OMP_NUM_THREADS": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "PYTHONHASHSEED": lambda value: value == "random" or bool(INTEGER_RE.fullmatch(value)),
    "RANK": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "SLURM_ARRAY_JOB_ID": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "SLURM_ARRAY_TASK_ID": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "SLURM_JOB_ID": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "SLURM_LOCALID": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "SLURM_PROCID": lambda value: bool(INTEGER_RE.fullmatch(value)),
    "TZ": lambda value: bool(TIMEZONE_RE.fullmatch(value)),
    "WORLD_SIZE": lambda value: bool(INTEGER_RE.fullmatch(value)),
}


class UnsafeInputError(ValueError):
    """Raised without echoing an unsafe user-provided value."""


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    path: Path


def _contains_sensitive_name(value: str) -> bool:
    lowered = value.casefold()
    tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
    compact = "".join(tokens)
    return (
        bool(tokens & SENSITIVE_TOKENS)
        or lowered in SENSITIVE_COMPONENTS
        or lowered.startswith(".env")
        or lowered.endswith(SENSITIVE_SUFFIXES)
        or any(compound in compact for compound in SENSITIVE_COMPOUNDS)
    )


def validate_safe_label(label: str) -> None:
    if not LABEL_RE.fullmatch(label) or _contains_sensitive_name(label):
        raise UnsafeInputError("unsafe label rejected")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def validate_safe_path(path: Path) -> Path:
    absolute = _absolute_path(path)
    if any(_contains_sensitive_name(component) for component in absolute.parts):
        raise UnsafeInputError("unsafe path rejected")

    try:
        resolved = absolute.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise UnsafeInputError("unresolvable path rejected") from exc
    if any(_contains_sensitive_name(component) for component in resolved.parts):
        raise UnsafeInputError("unsafe resolved path rejected")
    return resolved


def parse_artifact_spec(value: str) -> ArtifactSpec:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise UnsafeInputError("artifact must use label=path syntax")
    validate_safe_label(label)
    return ArtifactSpec(label=label, path=validate_safe_path(Path(raw_path)))


def _subprocess_environment(*, git: bool = False) -> dict[str, str]:
    environment = {
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.getenv("PATH", os.defpath),
    }
    if git:
        environment.update(
            {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
    return environment


def _run_command(command: Sequence[str], *, git: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_subprocess_environment(git=git),
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(command, 127, "", None)


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return _run_command(("git", "-C", os.fspath(repo), *arguments), git=True)


def _safe_git_name(value: str | None) -> str | None:
    if value is None or not GIT_NAME_RE.fullmatch(value) or _contains_sensitive_name(value):
        return None
    return value


def _collect_repository(repo: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    record: dict[str, object] = {
        "branch": None,
        "clean": None,
        "head": None,
        "path": os.fspath(repo),
        "remote_names": [],
        "tree": None,
    }
    issues: list[dict[str, str]] = []

    inside = _git(repo, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        issues.append({"code": "repository_unavailable", "path": os.fspath(repo)})
        return record, issues

    head = _git(repo, "rev-parse", "--verify", "HEAD")
    tree = _git(repo, "rev-parse", "--verify", "HEAD^{tree}")
    branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    status_result = _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    remotes = _git(repo, "remote")

    if head.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40,64}", head.stdout.strip()):
        record["head"] = head.stdout.strip().lower()
    else:
        issues.append({"code": "repository_head_unavailable", "path": os.fspath(repo)})

    if tree.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40,64}", tree.stdout.strip()):
        record["tree"] = tree.stdout.strip().lower()
    else:
        issues.append({"code": "repository_tree_unavailable", "path": os.fspath(repo)})

    if branch.returncode == 0:
        safe_branch = _safe_git_name(branch.stdout.strip())
        if safe_branch is None:
            issues.append({"code": "repository_branch_rejected", "path": os.fspath(repo)})
        else:
            record["branch"] = safe_branch

    if status_result.returncode == 0:
        record["clean"] = not bool(status_result.stdout)
        if not record["clean"]:
            issues.append({"code": "repository_dirty", "path": os.fspath(repo)})
    else:
        issues.append({"code": "repository_cleanliness_unavailable", "path": os.fspath(repo)})

    if remotes.returncode == 0:
        names = [name for line in remotes.stdout.splitlines() if (name := _safe_git_name(line.strip()))]
        if len(names) != len([line for line in remotes.stdout.splitlines() if line.strip()]):
            issues.append({"code": "repository_remote_name_rejected", "path": os.fspath(repo)})
        record["remote_names"] = sorted(set(names))
    else:
        issues.append({"code": "repository_remote_names_unavailable", "path": os.fspath(repo)})

    return record, issues


def _collect_artifact(spec: ArtifactSpec) -> tuple[dict[str, object], list[dict[str, str]]]:
    record: dict[str, object] = {
        "exists": False,
        "label": spec.label,
        "path": os.fspath(spec.path),
        "sha256": None,
        "size_bytes": None,
    }
    issue = {"code": "artifact_missing_or_unreadable", "label": spec.label}

    try:
        with spec.path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return record, [issue]
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except (OSError, ValueError):
        return record, [issue]

    stable_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    final_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stable_identity != final_identity:
        return record, [{"code": "artifact_changed_during_capture", "label": spec.label}]

    record.update(
        {
            "exists": True,
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
        }
    )
    return record, []


def _collect_gpu_info() -> dict[str, object]:
    fields = "index,name,uuid,driver_version,memory.total"
    result = _run_command(
        ("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits")
    )
    if result.returncode != 0:
        return {"gpus": [], "query_status": "unavailable"}

    gpus: list[dict[str, object]] = []
    try:
        rows = csv.reader(result.stdout.splitlines(), skipinitialspace=True)
        for row in rows:
            if len(row) != 5:
                raise ValueError
            index, name, uuid, driver_version, memory_total = (value.strip() for value in row)
            if not (
                INTEGER_RE.fullmatch(index)
                and GPU_TEXT_RE.fullmatch(name)
                and GPU_ID_RE.fullmatch(uuid)
                and GPU_TEXT_RE.fullmatch(driver_version)
                and INTEGER_RE.fullmatch(memory_total)
            ):
                raise ValueError
            gpus.append(
                {
                    "driver_version": driver_version,
                    "index": int(index),
                    "memory_total_mib": int(memory_total),
                    "name": name,
                    "uuid": uuid,
                }
            )
    except (csv.Error, ValueError):
        return {"gpus": [], "query_status": "invalid_response"}

    gpus.sort(key=lambda gpu: (int(gpu["index"]), str(gpu["uuid"])))
    return {"gpus": gpus, "query_status": "ok"}


def _collect_environment() -> dict[str, str]:
    selected: dict[str, str] = {}
    for name, validator in sorted(ENVIRONMENT_WHITELIST.items()):
        value = os.getenv(name)
        if value is not None and validator(value):
            selected[name] = value
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def capture_runtime_identity(
    repositories: Iterable[Path],
    artifacts: Iterable[ArtifactSpec],
    *,
    allow_dirty_labels: Iterable[str] = (),
    created_at: str | None = None,
) -> dict[str, object]:
    repo_paths = sorted((validate_safe_path(path) for path in repositories), key=os.fspath)
    artifact_specs: list[ArtifactSpec] = []
    for spec in artifacts:
        validate_safe_label(spec.label)
        artifact_specs.append(ArtifactSpec(label=spec.label, path=validate_safe_path(spec.path)))
    artifact_specs.sort(key=lambda spec: (spec.label, os.fspath(spec.path)))
    dirty_labels = sorted(set(allow_dirty_labels))
    for label in dirty_labels:
        validate_safe_label(label)

    if len(repo_paths) != len(set(repo_paths)):
        raise UnsafeInputError("duplicate repository rejected")
    artifact_labels = [spec.label for spec in artifact_specs]
    if len(artifact_labels) != len(set(artifact_labels)):
        raise UnsafeInputError("duplicate artifact label rejected")

    repository_records: list[dict[str, object]] = []
    artifact_records: list[dict[str, object]] = []
    issues: list[dict[str, str]] = []
    exceptions: list[dict[str, object]] = []

    for repo in repo_paths:
        record, repo_issues = _collect_repository(repo)
        repository_records.append(record)
        issues.extend(repo_issues)
        if record["clean"] is False and dirty_labels:
            exceptions.append(
                {
                    "labels": dirty_labels,
                    "path": os.fspath(repo),
                    "type": "dirty_repository_hold_only",
                }
            )

    for spec in artifact_specs:
        record, artifact_issues = _collect_artifact(spec)
        artifact_records.append(record)
        issues.extend(artifact_issues)

    captured_at = created_at or _utc_now()
    receipt: dict[str, object] = {
        "artifacts": artifact_records,
        "created_at": captured_at,
        "exceptions": exceptions,
        "issues": issues,
        "repositories": repository_records,
        "runtime": {
            "environment": _collect_environment(),
            "gpu": _collect_gpu_info(),
            "hostname": socket.gethostname(),
            "platform": {
                "machine": platform.machine(),
                "release": platform.release(),
                "system": platform.system(),
                "version": platform.version(),
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "timezone": "UTC",
        },
        "schema_version": SCHEMA_VERSION,
        "status": "HOLD" if issues else "PASS",
    }
    return receipt


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a non-sensitive Paper 1 runtime/provenance receipt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--allow-dirty-label", action="append", default=[], metavar="LABEL")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = validate_safe_path(args.output)
        artifacts = [parse_artifact_spec(value) for value in args.artifact]
        if any(spec.path == output for spec in artifacts):
            raise UnsafeInputError("output and artifact paths must differ")
        receipt = capture_runtime_identity(
            args.repo,
            artifacts,
            allow_dirty_labels=args.allow_dirty_label,
        )
    except UnsafeInputError:
        print("error: unsafe label or path rejected", file=sys.stderr)
        return 2

    atomic_write_json(output, receipt)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

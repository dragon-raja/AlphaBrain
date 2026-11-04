"""CPU-only source collection and isolated import proof before queue binding."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil
import hashlib


def source_manifest(root, prefixes):
    """Hash the complete declared source areas without importing any runtime.

    Paths are relative to the release, so the same bytes on different hosts
    have the same identity. A source-area declaration is not an environment or
    asset identity: those remain explicit artifacts in the run binding.
    """
    root = Path(root).resolve()
    return {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in python_sources(root, prefixes)
    }


def source_tree_identity(root, prefixes):
    """Canonical identity of an explicitly collected immutable source tree."""
    from .manifest import digest

    return digest(source_manifest(root, prefixes))


def python_sources(root, prefixes):
    """Include new untracked modules in explicitly owned code directories.

    A previous release's file list is not a current dependency closure. This
    collector deliberately avoids home directories, runtime state and assets.
    """
    root = Path(root).resolve()
    result = set()
    for prefix in prefixes:
        folder = (root / prefix).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError("Source prefix must be an existing owned directory")
        for path in folder.rglob("*.py"):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Source symlink escapes the worktree")
            result.add(path.relative_to(root))
    if not result:
        raise ValueError("Empty source release")
    return sorted(result)


def isolated_identity(release, python, module, *, function="source_identity", timeout=60):
    """Import from the release with no working-tree/PYTHONPATH fallback.

    The selected module must expose a pure CPU identity function. Unknown import
    results fail before a controller/worker is created, not during GPU startup.
    """
    release = Path(release).resolve()
    script = (
        "import sys,json,importlib;from pathlib import Path;"
        "root=Path(sys.argv[1]);sys.path.insert(0,str(root));"
        "m=importlib.import_module(sys.argv[2]);"
        "assert Path(m.__file__).resolve().is_relative_to(root);"
        "print(json.dumps(getattr(m,sys.argv[3])()))"
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        WANDB_MODE="disabled",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
    )
    with tempfile.TemporaryDirectory(prefix="source-import-proof-") as cwd:
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", script, str(release), module, function],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if result.returncode:
        raise ValueError("Isolated release import failed: " + result.stderr[-4000:])
    return json.loads(result.stdout)


def materialize_verified_release(snapshot, destination):
    """Atomically install an exact code mirror on a local filesystem.

    Explicit preparation only, never a launch-time cache fallback. Every input
    and installed file is hashed; links, traversal, duplicates and existing
    destinations are rejected. The returned manifest changes paths, not bytes
    or identity. Workers must continue their ordinary per-request hash checks.
    """
    from .authority_wire import require_local

    source = Path(snapshot["release"])
    destination = Path(destination)
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("Absolute source and destination required")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Existing release cannot be replaced")
    if not destination.parent.is_dir():
        raise ValueError("Explicit existing owned destination parent required")
    require_local(destination.parent)
    entries = {}
    for item in snapshot["artifacts"]:
        path = Path(item["path"])
        try:
            relative = path.relative_to(source)
        except ValueError:
            raise ValueError("Artifact outside release") from None
        if (
            not relative.parts
            or ".." in relative.parts
            or relative in entries
            or path.is_symlink()
            or path.resolve() != source.resolve() / relative
            or not path.is_file()
        ):
            raise ValueError("Unsafe, missing or duplicate artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source hash mismatch")
        entries[relative] = item
    if not entries:
        raise ValueError("Empty source release")
    temporary = Path(tempfile.mkdtemp(prefix=".release-", dir=destination.parent))
    try:
        for relative, item in entries.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = Path(item["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("Source changed during copy")
            with target.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            target.chmod(0o444)
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Installed hash mismatch")
        # Destination reservation is exclusive across concurrent preparers.
        # Rename is never allowed to replace an existing published directory.
        lock = destination.parent / (destination.name + ".publish-lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if destination.exists() or destination.is_symlink():
                raise ValueError("Release appeared during preparation")
            temporary.rename(destination)
            parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            os.close(descriptor)
            lock.unlink()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return dict(
        snapshot,
        release=str(destination),
        artifacts=[
            dict(item, path=str(destination / relative)) for relative, item in entries.items()
        ],
    )


def libero_runtime_sources(root):
    """Include reviewed legacy renderer helpers as well as core and adapters.

    The Python3.8 simulator imports scripts.dsol_paper1.runtime lazily. A pure
    controller import cannot prove this dependency closure on its own.
    """
    return python_sources(root, ("AlphaBrain", "benchmarks", "scripts"))

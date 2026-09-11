"""Locate shared utilities without rewriting a historical frozen snapshot."""

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))

from pathlib import Path
import hashlib
import json

SHARED_SOURCE_NAMES = ('__init__.py', 'build_libero_plus_view_protocol.py',
                       'evaluate_pi05_libero_plus_views.py', 'libero_camera_pose.py',
                       'serve_alphabrain_pi05_websocket.py')


def shared_code_paths(repo):
    """Include implementation bytes, not just compatibility shims, in new receipts."""
    repo = Path(repo)
    shared = repo / 'scripts/vla_shared'
    if not shared.is_dir():
        return []
    paths = [shared / name for name in SHARED_SOURCE_NAMES]
    paths.append(repo / 'scripts/dsol_paper1/runtime/shared_runtime_paths.py')
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Incomplete shared source set')
    return paths


def shared_scripts(repo):
    repo = Path(repo)
    shared = repo / 'scripts/vla_shared'
    if shared.is_dir():
        return shared
    # Only frozen releases may use their recorded, hash-verified legacy copy.
    release = repo.parent / 'release.json'
    legacy = repo / 'scripts/cabi_vla'
    server = legacy / 'serve_alphabrain_pi05_websocket.py'
    if repo.name != 'repo' or not release.is_file() or not server.is_file():
        raise FileNotFoundError('Shared VLA utilities missing: ' + str(shared))
    expected = json.loads(release.read_text())['source_hashes'].get(str(server))
    if hashlib.sha256(server.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen legacy service identity mismatch: ' + str(server))
    return legacy

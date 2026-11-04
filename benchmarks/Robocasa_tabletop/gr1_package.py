"""CPU-only preparation of a pinned public GR1 inference package.

No downloaded Python is imported, no model is constructed, no credential is
accepted, and no shared environment is modified. Fetching is an explicit call.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote
from urllib.request import urlopen


def _relative(name):
    path = PurePosixPath(name)
    if (
        not isinstance(name, str)
        or not name
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or "\\" in name
    ):
        raise ValueError("Unsafe package-relative path")
    return name


def package_plan(catalog, manifest_text):
    """Cross-check publisher manifest against separately recorded file hashes."""
    rows = {}
    for row in catalog["Data"]["Files"]:
        if row["Type"] == "tree":
            continue
        name = _relative(row["Path"])
        if name in rows:
            raise ValueError("Duplicate catalog entry")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", row["Sha256"])
            or not re.fullmatch(r"[0-9a-f]{40}", row["Revision"])
            or type(row["Size"]) is not int
            or row["Size"] < 0
        ):
            raise ValueError("Explicit file size, SHA256 and revision required")
        rows[name] = row
    wanted = {}
    for line in manifest_text.splitlines():
        if not line.strip():
            continue
        sha, name = line.split(None, 1)
        name = _relative(name.removeprefix("*"))
        if name in wanted or name not in rows or rows[name]["Sha256"] != sha:
            raise ValueError("Publisher manifest and pinned catalog disagree")
        wanted[name] = rows[name]
    if "PACKAGE_MANIFEST.sha256" not in rows:
        raise ValueError("Missing pinned manifest identity")
    expected = rows["PACKAGE_MANIFEST.sha256"]["Sha256"]
    if hashlib.sha256(manifest_text.encode()).hexdigest() != expected:
        raise ValueError("Manifest bytes differ from pinned catalog")
    wanted["PACKAGE_MANIFEST.sha256"] = rows["PACKAGE_MANIFEST.sha256"]
    return sorted(wanted.values(), key=lambda row: (row["Size"], row["Path"]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _destination(root, name):
    root = Path(root).resolve()
    target = root / _relative(name)
    if not target.resolve().is_relative_to(root) or target.is_symlink():
        raise ValueError("Destination escapes package root")
    return target


def fetch_file(root, row, *, bytes_per_second, opener=urlopen, heartbeat=None):
    """One bounded streaming attempt; preserve partial evidence, no auto retry."""
    if not isinstance(bytes_per_second, int) or bytes_per_second <= 0:
        raise ValueError("Positive transfer limit required")
    target = _destination(root, row["Path"])
    if target.exists():
        if target.stat().st_size != row["Size"] or digest(target) != row["Sha256"]:
            raise ValueError("Existing asset differs; never overwrite")
        return {"path": row["Path"], "status": "REUSED_VERIFIED", "bytes": row["Size"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    # An interrupted transfer is retained for explicit audit, never auto resumed.
    if partial.exists() or partial.is_symlink():
        raise ValueError("Partial transfer needs explicit audit before retry")
    url = (
        "https://modelscope.cn/models/NewGain/pi05_robocasa_gr1_tabletop24/resolve/"
        + row["Revision"]
        + "/"
        + quote(row["Path"], safe="/")
    )
    start = time.monotonic()
    h, count = hashlib.sha256(), 0
    with opener(url, timeout=45) as response, partial.open("xb") as f:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > row["Size"]:
                raise ValueError("Response exceeds pinned file size")
            h.update(block)
            f.write(block)
            if heartbeat:
                heartbeat(row["Path"], count)
            delay = count / bytes_per_second - (time.monotonic() - start)
            if delay > 0:
                time.sleep(min(delay, 1.0))
        f.flush()
        os.fsync(f.fileno())
    if count != row["Size"] or h.hexdigest() != row["Sha256"]:
        raise ValueError("Downloaded bytes fail pinned size/SHA256")
    if target.exists():
        raise ValueError("Concurrent writer detected")
    partial.rename(target)
    return {"path": row["Path"], "status": "FETCHED_VERIFIED", "bytes": count}


def audit_package(root, plan):
    missing, invalid = [], []
    for row in plan:
        path = _destination(root, row["Path"])
        if not path.is_file():
            missing.append(row["Path"])
        elif path.stat().st_size != row["Size"] or digest(path) != row["Sha256"]:
            invalid.append(row["Path"])
    return {
        "status": "PASS_FILES_ONLY" if not missing and not invalid else "INCOMPLETE",
        "missing": missing,
        "invalid": invalid,
        "files": len(plan),
        "model_calls": 0,
        "runtime_admitted": False,
    }

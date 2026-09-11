"""Read-only repository boundary, source layout and upload checks.

No project modules are imported; no GPU, service, dataset, or credential is read.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTER = {"scripts", "deployment", "reports", "archive", "benchmarks"}


def files(root=ROOT):
    result = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=root)
    return sorted({root / name.decode() for name in result.split(b"\0") if name and (root / name.decode()).is_file()})


def check(root=ROOT):
    errors = []
    counts = Counter()
    source_count = 0
    for p in files(root):
        rel = p.relative_to(root)
        if rel.parts[0] == "archive":
            continue
        if p.suffix != ".py":
            continue
        source_count += 1
        counts[rel.parts[0]] += 1
        try:
            tree = ast.parse(p.read_text(), filename=str(rel))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Invalid Python: {rel}: {exc}")
            continue
        if rel.parts[0] == "docs":
            errors.append(f"Executable implementation inside documentation: {rel}")
        if (p.name.startswith("test_") or p.name.endswith("_test.py")) and rel.parts[0] not in {"tests", "tools"}:
            errors.append(f"Unit test outside tests/: {rel}")
        if rel.parts[0] != "AlphaBrain":
            continue
        for n in ast.walk(tree):
            modules = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) and not n.level else []
            for module in modules:
                if module.split(".")[0] in OUTER:
                    errors.append(f"Core package imports outer layer: {rel}:{n.lineno}: {module}")
                if rel.parts[1] == "common" and module.startswith("AlphaBrain.") and not module.startswith("AlphaBrain.common"):
                    errors.append(f"Common primitives depend on higher layer: {rel}:{n.lineno}")
    for name in ("AGENTS.md", "CONTRIBUTING.md", "docs/architecture/README.md", "tests/README.md", "scripts/README.md"):
        if not (root / name).is_file():
            errors.append("Missing repository entry: " + name)
    return dict(status="FAIL" if errors else "PASS", errors=errors, python_source_count=source_count,
                by_layer=dict(counts), scope="Static package boundaries, syntax and source ownership; not GPU correctness certification")


def staged_check(root=ROOT):
    """Check staged content only; findings never print matched secret values."""
    changed = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=root)
    errors = []
    patterns = [rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b", rb"\bhf_[A-Za-z0-9]{30,}\b",
                rb"\bAKIA[0-9A-Z]{16}\b"]
    for raw in changed.split(b"\0"):
        if not raw:
            continue
        name = raw.decode()
        p = Path(name)
        if (p.name.startswith(".env") and p.name != ".env.example") or p.name in {"id_rsa", "id_ed25519", ".hf_token", ".wandb_api_key"}:
            errors.append("Sensitive path staged: " + name)
        if p.suffix in {".safetensors", ".ckpt", ".pt", ".pth", ".npz"}:
            errors.append("Model/raw array staged; keep in artifact storage: " + name)
        if "phaser" in name.lower() or name.startswith("configs/continual_learning/stage_info/"):
            errors.append("Explicitly private research path staged: " + name)
        content = subprocess.check_output(["git", "show", ":" + name], cwd=root)
        if any(re.search(pattern, content) for pattern in patterns):
            errors.append("Possible credential in staged file (value redacted): " + name)
        if len(content) > 20 * 1024 * 1024:
            errors.append("Oversized staged file (>20 MiB): " + name)
    return dict(status="FAIL" if errors else "PASS", errors=errors,
                scope="Heuristic staged-content guard; not a guarantee that all sensitive information is detected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    result = staged_check() if args.staged else check()
    print(json.dumps(result, indent=2))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())

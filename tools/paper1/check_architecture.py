"""Enforce the new research-layer boundary without importing research modules."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools/paper1/architecture_registry.json"


def check(root=ROOT):
    errors = []
    registry = json.loads((root / "tools/paper1/architecture_registry.json").read_text())
    base = root / "AlphaBrain/research/dsol"
    for name in ["data", "metrics", "selectors", "analysis", "plotting", "acquisition"]:
        for p in (base / name).rglob("*.py"):
            tree = ast.parse(p.read_text(), filename=str(p))
            for node in ast.walk(tree):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [a.name for a in node.names]
                if isinstance(node, ast.ImportFrom) and not node.level:
                    imports = [node.module or ""]
                if any(m.split(".")[0] in {"scripts", "reports", "archive"} for m in imports):
                    errors.append(f"Core depends on orchestration/archive: {p.relative_to(root)}:{node.lineno}")
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith(("/share/", "/workspace/", "/root/", "/alphabrain/"))
                ):
                    errors.append(f"Machine path in core: {p.relative_to(root)}:{node.lineno}")
    actual = {p.name for p in (root / "scripts/dsol_paper1").iterdir() if p.suffix in {".py", ".sh"}}
    for name in sorted(actual - set(registry["legacy_entrypoints"])):
        errors.append("Unregistered flat script; implement in package first: " + name)
    migration = root / "archive/paper1/maintenance/script-layout-20260911/manifest.json"
    if migration.exists():
        record = json.loads(migration.read_text())
        for row in record["changes"]:
            if not (root / row["new"]).is_file():
                errors.append("Missing migrated source: " + row["new"])
            if row["old"] != row["new"] and (root / row["old"]).exists():
                errors.append("Moved implementation reappeared in flat directory: " + row["old"])
        allowed = {Path(row["path"]).name for row in record["protected"]
                   if Path(row["path"]).parent == Path("scripts/dsol_paper1")}
        errors.extend("New flat entry is not allowed: " + name for name in sorted(actual - allowed))
    for name in registry["thin_entrypoints"]:
        p = root / name
        if len(p.read_text().splitlines()) > 40:
            errors.append("Compatibility entry grew implementation: " + name)
    manifest = json.loads((root / "archive/research/code/manifest.json").read_text())
    exceptions = {"scripts/fresh_vla/" + n for n in ["__init__.py", "video_io.py", "pi05_policy_server.py", "README.md"]}
    for row in manifest["records"]:
        new = root / row["new"]
        old = root / row["old"]
        if not new.is_file() or hashlib.sha256(new.read_bytes()).hexdigest() != row["sha256"]:
            errors.append("Archived source missing/changed: " + row["new"])
        if old.exists() and row["old"] not in exceptions:
            errors.append("Retired side-project source reappeared: " + row["old"])
    for p in (root / "scripts/dsol_paper1").rglob("*.py"):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("archive"):
                errors.append("Mainline imports archive: " + str(p.relative_to(root)))
    return errors


def main():
    errors = check()
    print(
        json.dumps(
            {
                "status": "FAIL" if errors else "PASS",
                "errors": errors,
                "scope": "new module boundaries, registered legacy entries and byte-preserving side-project archives",
            },
            indent=2,
        )
    )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

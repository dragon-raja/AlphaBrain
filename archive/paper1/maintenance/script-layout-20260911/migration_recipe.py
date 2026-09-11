"""One-shot, guarded mechanical relocation; default is a read-only plan.

Does not execute research code. Preserves the current/active runtime dependency
closure byte-for-byte and snapshots every edited file before applying changes.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT))
from tools.paper1.check_layout import dependencies, RUNTIME_ROOTS

BASE = ROOT / "scripts/dsol_paper1"
RECEIPT = ROOT / "archive/paper1/maintenance/script-layout-20260911"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def category(name):
    if name.startswith("view_acquisition_"):
        return "acquisition"
    if "report" in name or "brief" in name or "pdf" in name:
        return "reporting"
    if name.startswith(("analyze_", "summarize_", "compare_", "plot_", "aggregate_", "join_")):
        return "analysis"
    if any(s in name for s in ("train", "pair_collection", "view_pairs", "libero_pair_records", "download_")):
        return "training"
    if name.startswith(("audit_", "diagnose_", "probe_", "trace_", "measure_", "capture_")):
        return "diagnostics"
    if name.startswith(("build_", "plan_", "select_", "expand_", "freeze_", "clone_", "package_")):
        return "protocols"
    return "operations"


def plan():
    graph = dependencies()
    todo = [BASE / n for n in (*RUNTIME_ROOTS, "run_full64_view_landscape_v2.py", "build_consolidated_view_analysis_v1.py")]
    protected = set()
    while todo:
        p = todo.pop()
        if p in protected:
            continue
        protected.add(p)
        todo.extend(graph.get(p, ()))
    # The current analysis CLI stays discoverable alongside the report entry.
    protected.add(BASE / "analyze_standard_initial_results_v1.py")
    mapping = {}
    for p in sorted(BASE.iterdir()):
        if p.suffix not in {".py", ".sh"} or p in protected:
            continue
        group = category(p.name)
        if group == "acquisition":
            dest = ROOT / "AlphaBrain/research/dsol/acquisition" / p.name.removeprefix("view_acquisition_")
        elif group == "reporting" and p.suffix == ".py":
            dest = ROOT / "reports/paper1/historical" / p.name
        elif group == "operations":
            dest = BASE / group / ("launchers" if p.suffix == ".sh" else "controllers") / p.name
        else:
            dest = BASE / group / p.name
        mapping[p] = dest
    return protected, mapping


def rewrite(text, old, new, mapping):
    modules = {p.stem: str(q.relative_to(ROOT).with_suffix("")).replace("/", ".")
               for p, q in mapping.items() if p.suffix == ".py"}
    known = {p.stem for p in BASE.glob("*.py")} | {p.stem for p in mapping if p.suffix == ".py"}
    # Explicit imports and mock-patch module names. Do not touch frozen source snapshots.
    for name, module in modules.items():
        text = re.sub(r"\bscripts\.dsol_paper1\." + re.escape(name) + r"\b", module, text)
    for p, q in mapping.items():
        a, b = str(p.relative_to(ROOT)), str(q.relative_to(ROOT))
        text = re.sub(r"(?<![\w/])" + re.escape(a) + r"\b", lambda m: b, text)
        for var in ("$REPO_ROOT/", "${REPO_ROOT}/", "$REPO/", "${REPO}/"):
            text = text.replace(var + a, var + b)
    if old.suffix != ".py":
        if old != new:
            depth = len(new.relative_to(ROOT).parts) - 1
            text = text.replace('$(dirname "$0")/../..', '$(dirname "$0")/' + '/'.join(['..'] * depth))
        return text
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        replacement = None
        if isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if name in known and (node.level == 0 or old.parent == BASE):
                module = modules.get(name, "scripts.dsol_paper1." + name)
                aliases = ", ".join(a.name + (" as " + a.asname if a.asname else "") for a in node.names)
                replacement = "from " + module + " import " + aliases
            elif name == "scripts.dsol_paper1":
                parts = []
                for a in node.names:
                    target = modules.get(a.name)
                    if target:
                        parts.append("import " + target + " as " + (a.asname or a.name))
                    else:
                        parts.append("from scripts.dsol_paper1 import " + a.name + (" as " + a.asname if a.asname else ""))
                replacement = ("; ").join(parts)
        elif isinstance(node, ast.Import):
            if any(a.name in known for a in node.names):
                replacement = "import " + ", ".join(
                    (modules.get(a.name, "scripts.dsol_paper1." + a.name) + " as " + (a.asname or a.name))
                    if a.name in known else (a.name + (" as " + a.asname if a.asname else ""))
                    for a in node.names)
        if replacement is not None:
            edits.append((offsets[node.lineno - 1] + node.col_offset,
                          offsets[node.end_lineno - 1] + node.end_col_offset, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    if old != new:
        depth = len(new.relative_to(ROOT).parts) - 1
        text = re.sub(r"Path\(__file__\)\.resolve\(\)\.parents\[([12])\]",
                      lambda m: f"Path(__file__).resolve().parents[{int(m[1]) + depth - 2}]", text)
        # Location-derived sibling references must keep targeting the actual file.
        for p in set(BASE.glob("*.py")) | {p for p in mapping if p.suffix == ".py"}:
            target = mapping.get(p, p)
            pattern = r"Path\(__file__\)(?:\.resolve\(\))?\.with_name\(([\"'])" + re.escape(p.name) + r"\1\)"
            text = re.sub(pattern, f"(Path(__file__).resolve().parents[{depth}] / {str(target.relative_to(ROOT))!r})", text)
        # CLI bootstrap only: the implementation imports are explicit, not a search
        # through all folders. Root contains the unchanged runtime helpers.
        if not new.is_relative_to(ROOT / "AlphaBrain"):
            tree = ast.parse(text)
            insert_line = 0
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and insert_line == 0:
                    insert_line = node.end_lineno
                elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    insert_line = node.end_lineno
                else:
                    break
            if insert_line == 0 and text.startswith("#!"):
                insert_line = 1
            lines = text.splitlines(keepends=True)
            lines.insert(insert_line, f"\n# Repository-local CLI bootstrap (no experiment is launched on import).\nimport sys as _layout_sys\nfrom pathlib import Path as _LayoutPath\n_layout_root = _LayoutPath(__file__).resolve().parents[{depth}]\nfor _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):\n    if str(_layout_path) not in _layout_sys.path:\n        _layout_sys.path.insert(0, str(_layout_path))\n\n")
            text = "".join(lines)
        ast.parse(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if RECEIPT.exists():
        raise SystemExit("Receipt already exists; do not repeat migration")
    protected, mapping = plan()
    print(json.dumps({"moves": len(mapping), "protected": len([p for p in protected if p.parent == BASE]),
                      "groups": dict(Counter(category(p.name) for p in mapping))}, indent=2))
    if not args.apply:
        return
    candidates = set(mapping)
    for base in (ROOT / "tests", ROOT / "scripts", ROOT / "reports"):
        candidates.update(p for p in base.rglob("*") if p.suffix in {".py", ".sh"}
                          and "__pycache__" not in p.parts)
    candidates -= protected
    changes = []
    for old in sorted(candidates):
        new = mapping.get(old, old)
        original = old.read_bytes()
        result = rewrite(original.decode(), old, new, mapping).encode()
        if new != old or result != original:
            changes.append((old, new, original, result))
    # Compute everything before touching files; destination collisions fail closed.
    for old, new, _, _ in changes:
        if old != new and new.exists():
            raise RuntimeError(f"Destination exists: {new}")
    RECEIPT.mkdir(parents=True)
    rows = []
    for old, new, original, result in changes:
        backup = RECEIPT / "before" / old.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
        mode = old.stat().st_mode
        new.parent.mkdir(parents=True, exist_ok=True)
        if new != old:
            shutil.move(str(old), str(new))
        new.write_bytes(result)
        new.chmod(mode)
        rows.append({"old": str(old.relative_to(ROOT)), "new": str(new.relative_to(ROOT)),
                     "before_sha256": digest(original), "after_sha256": digest(result),
                     "backup": str(backup.relative_to(ROOT))})
    keep = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p.read_bytes())}
            for p in sorted(protected) if p.is_file()]
    (RECEIPT / "manifest.json").write_text(json.dumps({"changes": rows, "protected": keep}, indent=2) + "\n")
    print(f"Relocated {len(mapping)} files; updated {len(changes)-len(mapping)} callers; saved original bytes")


if __name__ == "__main__":
    main()

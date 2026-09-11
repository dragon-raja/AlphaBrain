"""Read-only Paper 1 layout checks and conservative source navigation.

No research module is imported. Static reachability over-approximates imports
and literal script references; it is NOT a deletion-safety proof.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/dsol_paper1"
MANIFEST = ROOT / "archive/paper1/layout_manifest.json"
RUNTIME_ROOTS = (
    "dynamic_initial_scheduler_v1.py",
    "standard_initial_aa0_v1.py",
    "standard_initialization_v1.py",
    "dualhost_aa0_v1.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies():
    """Conservative Python imports / literal filename graph within scripts/."""
    files = sorted(p for p in (ROOT / "scripts").rglob("*") if p.suffix in {".py", ".sh"})
    by_stem = defaultdict(set)
    by_name = defaultdict(set)
    for p in files:
        by_stem[p.stem].add(p)
        by_name[p.name].add(p)
    def resolve(paths, caller):
        local = {p for p in paths if p.parent == caller.parent}
        shared = {p for p in paths if p.parent == ROOT / 'scripts/vla_shared'}
        return local or shared or set(paths)
    graph = {}
    for p in files:
        text = p.read_text()
        targets = set()
        if p.suffix == ".py":
            tree = ast.parse(text, filename=str(p))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or "", *[a.name for a in node.names]]
                for module in modules:
                    targets.update(resolve(by_stem.get(module.rsplit(".", 1)[-1], ()), p))
        for name, paths in by_name.items():
            if name in text:
                explicit = {q for q in paths if str(q.relative_to(ROOT)) in text}
                targets.update(explicit or resolve(paths, p))
        graph[p] = targets - {p}
    return graph


def runtime_closure(graph):
    todo = [SCRIPTS / n for n in RUNTIME_ROOTS]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        todo.extend(graph.get(path, ()))
    return seen


def navigation():
    closure = runtime_closure(dependencies())
    groups = defaultdict(list)
    for p in sorted(SCRIPTS.rglob("*")):
        if p.suffix not in {".py", ".sh"}:
            continue
        if p.name == "__init__.py":
            continue
        n = p.name
        if p.parent != SCRIPTS:
            group = "organized_" + p.relative_to(SCRIPTS).parts[0]
        elif p in closure:
            group = "runtime_dependencies_conservative"
        elif "view_acquisition" in n:
            group = "proposed_acquisition_not_current_experiment"
        elif any(t in n for t in ("brief", "report", "pdf", "unified_research_progress")):
            group = "reporting_including_historical_versions"
        elif n.startswith(("analyze_", "summarize_", "compare_", "plot_", "rank_")):
            group = "analysis_check_input_protocol_before_reuse"
        elif any(t in n for t in ("train", "collection", "view_pairs", "download_")):
            group = "training_and_data_provenance"
        else:
            group = "support_and_historical_protocols_retained"
        groups[group].append(str(p.relative_to(ROOT)))
    groups["cross_directory_runtime_dependencies"] = [
        str(p.relative_to(ROOT)) for p in sorted(closure) if p.parent != SCRIPTS
    ]
    groups["historical_report_builders"] = [str(p.relative_to(ROOT)) for p in
        sorted((ROOT / "reports/paper1/historical").glob("*.py")) if p.name != "__init__.py"]
    groups["acquisition_engineering_not_current_evaluator"] = [str(p.relative_to(ROOT)) for p in
        sorted((ROOT / "AlphaBrain/research/dsol/acquisition").glob("*.py")) if p.name != "__init__.py"]
    return dict(groups)


def check(verify_preserved_source=False):
    manifest = json.loads(MANIFEST.read_text())
    errors = []
    for move in manifest["moves"]:
        old, new = ROOT / move["old"], ROOT / move["new"]
        if old.exists():
            errors.append(f"Retired path reappeared: {move['old']}")
        if not new.is_file():
            errors.append(f"Missing relocated file: {move['new']}")
        elif move["new"].startswith("archive/") or verify_preserved_source:
            if sha(new) != move["sha256"]:
                errors.append(f"Preserved bytes changed: {move['new']}")
    for p in SCRIPTS.glob("*.py"):
        if p.name.startswith("test_") or p.name.endswith("_test.py"):
            errors.append(f"Test outside tests/: {p.name}")
    for p in (ROOT / "archive/paper1/operations").iterdir():
        if p.suffix != ".txt":
            errors.append(f"Executable-looking archive member: {p.name}")
    retired = [Path(m["old"]).name for m in manifest["moves"] if m["new"].startswith("archive/")]
    for base in (SCRIPTS, ROOT / "tests/dsol_paper1"):
        for p in base.iterdir():
            if p.suffix not in {".py", ".sh"}:
                continue
            text = p.read_text()
            for name in retired:
                if name in text or (name.endswith(".py") and Path(name).stem in text):
                    errors.append(f"Live source references retired operation: {p.name} -> {name}")
    if verify_preserved_source:
        # Cleanup acceptance only, not a permanent prohibition on future research edits.
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", manifest["baseline_commit"], "--",
             "AlphaBrain", "scripts", "configs", "schemas"], cwd=ROOT, text=True,
        ).splitlines()
        allowed = {m["old"] for m in manifest["moves"]} | {"scripts/dsol_paper1/README.md"}
        allowed.update(manifest.get('shared_extraction', {}).get('reviewed_followup_source_paths', []))
        errors.extend(f"Scientific source/config changed: {p}" for p in changed if p not in allowed)
        for name, expected in manifest.get('shared_extraction', {}).get('reviewed_source_sha256', {}).items():
            path = ROOT / name
            actual = sha(path) if path.is_file() else None
            if actual != expected:
                errors.append(f"Reviewed extraction source changed: {name}")
        original = subprocess.check_output(
            ["git", "show", f"{manifest['baseline_commit']}:docs/dsol_paper1/README.md"], cwd=ROOT,
        )
        if (ROOT / "docs/dsol_paper1/HISTORY.md").read_bytes() != original:
            errors.append("Historical index changed during relocation")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List scripts by conservative purpose")
    parser.add_argument("--verify-preserved-source", action="store_true", help="Verify this cleanup against its baseline")
    args = parser.parse_args()
    errors = check(args.verify_preserved_source)
    groups = navigation()
    output = {"status": "FAIL" if errors else "PASS", "errors": errors,
              "counts": {k: len(v) for k, v in groups.items()},
              "scope": "Static navigation, not complete dynamic dependency or deletion proof"}
    if args.list:
        output["groups"] = groups
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

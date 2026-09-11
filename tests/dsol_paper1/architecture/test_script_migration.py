"""Maintenance gates: no GPU or simulator is used by this test module."""
from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = REPOSITORY_ROOT
RECEIPT = ROOT / "archive/paper1/maintenance/script-layout-20260911/manifest.json"


def manifest():
    return json.loads(RECEIPT.read_text())


def test_retired_runtime_originals_and_new_source_identity_chain():
    from tools.repository.evidence import current_expected_hash
    runtime = json.loads((ROOT / 'archive/repository/20260911/runtime-modules/manifest.json').read_text())
    originals = {r['old']: r for r in runtime['records']}
    for row in manifest()["protected"]:
        if row['path'] in originals:
            assert hashlib.sha256((ROOT / originals[row['path']]['before']).read_bytes()).hexdigest() == row['sha256']
        path, expected = current_expected_hash(row['path'], row['sha256'])
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_every_original_source_is_recoverable_and_moved_path_is_absent():
    m = manifest()
    for row in m["changes"] + m["deferred"]:
        assert hashlib.sha256((ROOT / row["backup"]).read_bytes()).hexdigest() == row["before_sha256"]
    moved = [row for row in m["changes"] if row["old"] != row["new"]]
    assert len(moved) == 149
    for row in moved:
        assert (ROOT / row["new"]).is_file()
        assert not (ROOT / row["old"]).exists()
    assert not list((ROOT / "scripts/dsol_paper1").glob("*.py"))
    assert not list((ROOT / "scripts/dsol_paper1").glob("*.sh"))


def test_relocated_local_imports_resolve_without_flat_aliases():
    m = manifest()
    moved_names = {Path(r["old"]).stem for r in m["changes"] if r["old"] != r["new"] and r["old"].endswith(".py")}
    for row in m["changes"]:
        from tools.repository.evidence import current_path
        p = current_path(ROOT / row["new"])
        if not p.name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module or ""]
            for module in modules:
                assert module not in moved_names, (p, module, "stale bare import")
                if module.startswith(("scripts.dsol_paper1.", "AlphaBrain.research.dsol.acquisition.", "reports.paper1.historical.")):
                    target = ROOT / module.replace(".", "/")
                    assert target.is_dir() or target.with_suffix(".py").is_file(), (p, module)


@pytest.mark.parametrize("relative", [
    "scripts/dsol_paper1/analysis/analyze_accel_noise_stability.py",
    "scripts/dsol_paper1/protocols/build_libero_view_catalog.py",
    "scripts/dsol_paper1/training/audit_dsol_training_match.py",
    "scripts/dsol_paper1/diagnostics/capture_runtime_identity.py",
    "scripts/dsol_paper1/operations/controllers/run_libero_visibility_scan_plan.py",
])
def test_reviewed_cli_help_from_unrelated_directory(relative, tmp_path):
    # Only these reviewed parser-first entrypoints are allowed. Never invoke a
    # scheduler or historical recipe merely to discover whether it has help.
    env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(ROOT / relative), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())


def test_all_moved_shell_recipes_parse_and_resolve_repo_root():
    for row in manifest()["changes"]:
        p = ROOT / row["new"]
        if row["old"] == row["new"] or p.suffix != ".sh":
            continue
        result = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        # Inspect the root expression rather than source/execute the recipe.
        import re
        match = re.search(r'\$\(dirname "\$0"\)/((?:\.\./)*\.\.)', p.read_text())
        if match:
            assert (p.parent / match[1]).resolve() == ROOT


class NormalizeImportsAndRelocatedStrings(ast.NodeTransformer):
    def __init__(self, rows):
        self.replacements = [(r["new"], r["old"]) for r in rows if r["old"] != r["new"]]
        self.replacements += [(new.removeprefix("scripts/dsol_paper1/"), Path(old).name)
                              for new, old in list(self.replacements)
                              if new.startswith("scripts/dsol_paper1/")]

    def visit_Import(self, node):
        return ast.Pass()

    visit_ImportFrom = visit_Import

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for new, old in self.replacements:
                node.value = node.value.replace(new, old)
                if new.endswith(".py"):
                    node.value = node.value.replace(new[:-3].replace("/", "."), old[:-3].replace("/", "."))
        return node


def test_path_independent_function_bodies_preserved():
    """Exact AST comparison of non-path functions, not just successful imports.

    CLI path plumbing is covered separately; this compares numerical logic,
    constants, control flow, and argument defaults against original source.
    """
    m = manifest()
    checked = 0
    for row in m["changes"]:
        if row["old"] == row["new"] or not row["new"].endswith(".py"):
            continue
        before = ast.parse((ROOT / row["backup"]).read_text())
        from tools.repository.evidence import current_path, rows as later_rows
        after = ast.parse(current_path(ROOT / row["new"]).read_text())
        old_defs = {n.name: n for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new_defs = {n.name: n for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, old in old_defs.items():
            # Moving __file__-derived locations is deliberately not byte-equivalent.
            if any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(old)):
                continue
            new = new_defs[name]
            normalize = NormalizeImportsAndRelocatedStrings(m["changes"] + list(later_rows()))
            left = ast.dump(normalize.visit(copy.deepcopy(old)), include_attributes=False)
            right = ast.dump(normalize.visit(copy.deepcopy(new)), include_attributes=False)
            assert left == right, (row["new"], name)
            checked += 1
    assert checked > 700


def test_previously_import_executed_recipes_now_have_main_guards():
    for relative in [
        "scripts/dsol_paper1/operations/controllers/prepare_dynamic_scheduler_v1.py",
        "scripts/dsol_paper1/diagnostics/diagnose_standard_initial_input_v1.py",
        "scripts/dsol_paper1/diagnostics/measure_view_speed_resources_v1.py",
        "reports/paper1/historical/audit_consolidated_view_pdf_v1.py",
    ]:
        tree = ast.parse((ROOT / relative).read_text())
        assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
        assert any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) for n in tree.body)
        # Scientific operations are in main, not top-level with/open/loop blocks.
        assert not any(isinstance(n, (ast.With, ast.While)) for n in tree.body)

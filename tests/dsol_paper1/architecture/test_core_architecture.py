"""Portable, CPU-only checks suitable for public CI (no private experiment data)."""

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import ast
from pathlib import Path
import numpy as np
from tools.paper1.check_architecture import check
from AlphaBrain.research.dsol.analysis.statistics import hierarchy
from AlphaBrain.research.dsol.metrics.view_rules import metric_choices
from AlphaBrain.research.dsol.selectors.ridge import fit_ranker


def test_repository_boundaries_and_archive_integrity():
    assert check() == []


def test_oracle_noise_order():
    y = np.array([[[[1, 0], [0, 1]]]], dtype=float)
    assert hierarchy(y.mean(axis=-1))["initial_oracle"] == 0.5


def test_rule_tie_and_top10():
    a = np.zeros((2, 12))
    v = np.zeros_like(a)
    v[:, -1] = 1
    result = metric_choices(a, v)
    np.testing.assert_array_equal(result["accel10_visibility"], [0, 0])


def test_numerical_bodies_preserved_from_pre_refactor_baseline():
    root = REPOSITORY_ROOT
    source = root / "archive/paper1/maintenance/baseline-20260911/analyze_standard_initial_results_v1.py"
    old = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    for module in ["analysis/statistics.py", "metrics/view_rules.py", "selectors/ridge.py", "analysis/initial_study.py"]:
        for node in ast.parse((root / "AlphaBrain/research/dsol" / module).read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert ast.dump(node, include_attributes=False) == old[node.name]


def test_test_labels_cannot_change_selection_or_alpha():
    rng = np.random.default_rng(12)
    geometry = rng.normal(size=(32, 7, 3))
    context = rng.normal(size=(32, 10))
    images = rng.normal(size=(32, 7, 10))
    tasks = np.repeat(np.arange(8), 4)
    initials = np.tile(np.arange(4), 8)
    q = rng.random((32, 7))
    for family in ["geometry_context_ridge", "candidate_image_ridge"]:
        a = fit_ranker(geometry, context, images, tasks, initials, q, family)
        changed = q.copy()
        changed[initials >= 2] = 10000
        b = fit_ranker(geometry, context, images, tasks, initials, changed, family)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[2] == b[2]

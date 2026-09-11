"""Optional full-data golden check; no simulator, VLA training or publication."""

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import json
from pathlib import Path
import tempfile
import importlib.util
import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.initial_study import summarize


def test_full_initial32_statistics_and_selections_match_published_analysis():
    root = Path("/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908")
    archive = root / "analysis/research-v1/history/20260911T023803721133Z"
    if not (archive / "summary.json").exists():
        pytest.skip("Private golden matrix unavailable")
    r = json.loads((root / "release.json").read_text())
    with np.load(archive / "matrix.npz") as a:
        y, acc, vis, camera = [a[k] for k in ["success", "accel_members", "visibility", "camera"]]
    with np.load(archive / "embeddings.npz") as a:
        images, context = a["external"], a["context"]
    with tempfile.TemporaryDirectory(prefix="paper1-golden-") as tmp:
        out = Path(tmp) / "new"
        out.mkdir()
        before = Path(tmp) / "before"
        before.mkdir()
        source = (
            REPOSITORY_ROOT
            / "archive/paper1/maintenance/baseline-20260911/analyze_standard_initial_results_v1.py"
        )
        spec = importlib.util.spec_from_file_location("initial_analysis_before_refactor", source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        expected_same_environment = old.summarize(before, r, y, acc, vis, camera, images, context)
        actual = summarize(out, r, y, acc, vis, camera, images, context)
        assert actual == expected_same_environment
        assert actual == json.loads((archive / "summary.json").read_text())
        for name in ["selections.json", "per-initial.json"]:
            assert json.loads((out / name).read_text()) == json.loads((archive / name).read_text())
        for model in ["canonical", "broad"]:
            for family in ["geometry_context_ridge", "candidate_image_ridge"]:
                with np.load(out / f"{model}-{family}.npz") as a, np.load(archive / f"{model}-{family}.npz") as b:
                    np.testing.assert_array_equal(a["choices"], b["choices"])
                    with np.load(before / f"{model}-{family}.npz") as same:
                        np.testing.assert_array_equal(a["predictions"], same["predictions"])
                    # Published features/PCA are float32. The CPU test runner uses
                    # one BLAS thread; publication used two. Allow float32-scale
                    # roundoff only against that run; same-environment old/new
                    # predictions, all choices and all reported statistics are exact.
                    np.testing.assert_allclose(a["predictions"], b["predictions"], rtol=1e-6, atol=1e-6)

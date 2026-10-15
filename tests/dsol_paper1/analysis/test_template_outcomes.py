import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.template_outcomes import choose_endpoints, summarize_endpoints


def inputs():
    a = np.random.default_rng(5).uniform(size=(2, 32, 97, 8))
    return a, np.repeat(np.arange(8), 4), np.tile(np.arange(4), 8)


def test_dev_templates_cannot_read_any_evaluation_scores():
    a, t, i = inputs()
    c, refs = choose_endpoints(a, t, i)
    altered = a.copy()
    altered[:, i >= 2] = np.random.default_rng(7).uniform(size=altered[:, i >= 2].shape)*100
    c2, _ = choose_endpoints(altered, t, i)
    for k in ["dev_global", "dev_same_task", "dev_other_tasks"]:
        np.testing.assert_array_equal(c[k], c2[k])
        assert all(i[j] < 2 for r in refs if r["method"] == k for j in r["reference_initials"])
        assert (c[k] > 0).all()


def test_global_is_fixed_and_previous_template_excludes_target():
    a, t, i = inputs()
    c, refs = choose_endpoints(a, t, i)
    assert all(len(np.unique(row)) == 1 for row in c["dev_global"])
    for r in refs:
        if r["method"].startswith("previous_"):
            assert r["initial"] not in r["reference_initials"]
        if r["method"] == "previous_other_tasks":
            assert all(t[j] != t[r["initial"]] for j in r["reference_initials"])


def test_outcomes_join_fixed_endpoints_and_keep_populations_separate():
    a, t, i = inputs()
    c, _ = choose_endpoints(a, t, i)
    y = np.zeros((2, 32, 97, 32), dtype=np.int8)
    y[:, :, 0, :16] = 1
    y[:, :, 65:, :8] = 1
    ids = np.array(["canonical"]+[f"broad_train_{k:03d}" for k in range(64)]+[f"broad_heldout_{k:03d}" for k in range(32)])
    r = summarize_endpoints(y, c, ids, t, i)
    for m in r["models"]:
        assert "dev_global" not in m["populations"]["all32"]
        p = m["populations"]["evaluation16"]
        assert p["known_canonical"]["success"]["mean"] == .5
        assert p["stay_heldout32"]["success"]["mean"] == .25
        assert p["known_canonical"]["gain_vs_stay"]["mean"] == .25
        assert len(p["known_canonical"]["per_initial_success"]) == 16


def test_only_actual_binary_full_task_labels_accepted():
    a, t, i = inputs()
    c, _ = choose_endpoints(a, t, i)
    with pytest.raises(ValueError):
        summarize_endpoints(np.full((2, 32, 97, 32), .5), c, [], t, i)

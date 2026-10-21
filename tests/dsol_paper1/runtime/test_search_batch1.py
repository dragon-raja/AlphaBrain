import numpy as np
import pytest

from benchmarks.LIBERO.search_batch1 import ADMISSION_INPUTS, admit, require_same
from benchmarks.LIBERO.search_features import run


def result(initial, candidate, member):
    value = np.asarray([initial + candidate + member], np.float32)
    return dict(trace=value, actions=value, score=value, external=value,
                wrist=value, seconds=0.1)


def test_reference_coverage_and_exact_gate(tmp_path):
    calls = []

    def query(i, c, m, capture, stage):
        calls.append((i, c, m, capture, stage))
        return result(i, c, m)

    reference = admit(query, result(0, 0, 0), tmp_path)
    assert len(calls) == 66
    assert set(reference) == {(i, c, m) for i, c in ADMISSION_INPUTS for m in range(4)}
    assert all(x[-1] == "admission" for x in calls)
    with np.load(tmp_path / "reference.npz") as z:
        assert z["identities"].shape == (32, 3)


def test_reference_drift_stops_before_next_input(tmp_path):
    calls = []

    def query(i, c, m, capture, stage):
        calls.append((i, c, m))
        value = result(i, c, m)
        if capture:
            value["trace"] = value["trace"] + 1
        return value

    with pytest.raises(ValueError, match="trace"):
        admit(query, result(0, 0, 0), tmp_path)
    assert len(calls) == 4
    assert not (tmp_path / "admission.json").exists()


def test_previous_process_reference_is_required(tmp_path):
    prior = result(0, 0, 1)
    with pytest.raises(ValueError, match="trace"):
        admit(lambda i,c,m,*a: result(i,c,m), prior, tmp_path)


def test_new_mode_cannot_be_mixed_or_implicitly_executed(tmp_path):
    with pytest.raises(ValueError, match="Explicit"):
        run("missing", tmp_path / "output", batch1_reference=True)
    with pytest.raises(ValueError, match="own bounded"):
        run("missing", tmp_path / "output", execute=True, batch1_reference=True, export=True)


def test_visual_reference_is_exact():
    a, b = result(0,0,0), result(0,0,0)
    b["external"] = np.asarray([np.nan], np.float32)
    with pytest.raises(ValueError, match="external"):
        require_same(a, b, keys=("external",))

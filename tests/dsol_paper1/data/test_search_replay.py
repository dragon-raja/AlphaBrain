"""Portable synthetic provenance/partition tests; no private assets."""

import json
import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import sha
from AlphaBrain.research.dsol.data.search_replay import load_initial32


@pytest.fixture
def bank(tmp_path):
    ids = ["canonical"] + [f"broad_train_{i:03d}" for i in range(64)] + [
        f"broad_heldout_{i:03d}" for i in range(32)]
    states = [dict(pair_key=f"task{t}-initial{i}", task_id=f"task{t}", task_ordinal=t,
                   initialization=dict(init_state_index=i, kind="official_libero_initial_state_v1"))
              for t in range(8) for i in range(4)]
    (tmp_path / "states.json").write_text(json.dumps(states))
    np.savez(tmp_path / "matrix.npz", success=np.zeros((2, 32, 97, 32), dtype=np.int8),
             accel_members=np.ones((2, 32, 97, 8)), camera=np.tile(np.eye(4), (32, 97, 1, 1)),
             candidate_ids=ids)
    context = np.concatenate([np.full((32, 512), 999), np.full((32, 512), 3)], axis=1)
    np.savez(tmp_path / "embeddings.npz", external=np.ones((32, 97, 512)), context=context)
    refresh_hashes(tmp_path)
    return tmp_path


def refresh_hashes(path):
    (path / "provenance.json").write_text(json.dumps({"artifacts": {
        name: sha(path / name) for name in ("matrix.npz", "states.json", "embeddings.npz")}}))


def test_partition_labels_and_no_canonical_context(bank):
    data, outcomes = load_initial32(bank)
    assert len(data.development) == len(data.evaluation) == 16
    assert not set(data.development) & set(data.evaluation)
    assert np.all(data.wrist == 3)
    assert data.score_scales == (1., 1.)
    assert outcomes.paired(0, 0, 0, 1)["paired_repeats"] == 32
    assert not hasattr(data.queries(0, 0), "success")


def test_mutated_source_rejected(bank):
    with (bank / "states.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Audited source changed"):
        load_initial32(bank)


def test_duplicate_initial_rejected(bank):
    states = json.loads((bank / "states.json").read_text())
    states[1] = states[0]
    (bank / "states.json").write_text(json.dumps(states))
    refresh_hashes(bank)
    with pytest.raises(ValueError, match="Duplicate"):
        load_initial32(bank)


def test_evaluation_scores_do_not_set_reward_scale(bank):
    with np.load(bank / "matrix.npz") as z:
        values = {name: z[name].copy() for name in z.files}
    values["accel_members"][:, np.tile([False, False, True, True], 8)] = 1000
    np.savez(bank / "matrix.npz", **values)
    refresh_hashes(bank)
    data, _ = load_initial32(bank)
    assert data.score_scales == (1., 1.)

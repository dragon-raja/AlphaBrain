"""Native-policy search table; historical outcomes remain evaluation-only."""

from pathlib import Path

import numpy as np

from .artifacts import read, sha
from .search_replay import ReplayInputs, OutcomeTable, frozen_array


def load_native_search(source, native, inputs_file):
    source, native = Path(source), Path(native)
    provenance = read(source / "provenance.json")
    receipt = read(native / "features.json")
    if (receipt["status"] != "COMPLETE" or receipt["batch_size"] != 1
            or receipt["noise_count"] != 2 or receipt["attempted"] != {"admission":66, "export":6208}):
        raise ValueError("Complete admitted batch1/two-noise table required")
    hashes = {str(source / "provenance.json"):sha(source / "provenance.json"),
              str(native / "features.json"):sha(native / "features.json")}
    expected = {source / name:provenance["artifacts"][name] for name in ("matrix.npz", "states.json")}
    expected.update({native / "siglip-scores.npz":receipt["asset_sha256"],
                     native / "reference.npz":receipt["reference_sha256"],
                     Path(inputs_file):receipt["inputs_sha256"]})
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError("Native search source changed: " + str(path))
        hashes[str(path)] = digest
    states = read(source / "states.json")
    inputs = read(inputs_file)["records"]
    with np.load(source / "matrix.npz", allow_pickle=False) as z:
        poses, ids, labels = z["camera"].copy(), tuple(z["candidate_ids"].tolist()), z["success"].copy()
    with np.load(native / "siglip-scores.npz", allow_pickle=False) as z:
        scores, visual, wrist = z["scores"].copy(), z["external"].copy(), z["wrist"].copy()
    if (scores.shape != (32,97,2) or visual.shape != (32,97,4608) or wrist.shape != visual.shape
            or poses.shape != (32,97,4,4) or labels.shape != (2,32,97,32)):
        raise ValueError("Frozen INITIAL32 native axes required")
    if (any(not np.isfinite(x).all() for x in (scores, visual, wrist, poses))
            or (scores < 0).any() or not np.isin(labels,[0,1]).all()):
        raise ValueError("Invalid native inputs or incomplete evaluation labels")
    expected_ids = (("canonical",) + tuple(f"broad_train_{i:03d}" for i in range(64))
                    + tuple(f"broad_heldout_{i:03d}" for i in range(32)))
    if ids != expected_ids or len(states) != 32 or len(inputs) != 32:
        raise ValueError("Initial/candidate identity mismatch")
    for i, (state, record) in enumerate(zip(states, inputs)):
        if (state["pair_key"] != record["state"]["pair_key"] or tuple(record["candidate_ids"]) != ids
                or state["task_ordinal"] != i // 4
                or state["initialization"]["init_state_index"] != i % 4):
            raise ValueError("Native table is not aligned with historical outcomes")
    if not np.array_equal(wrist, np.repeat(wrist[:, :1],97,axis=1)):
        raise ValueError("Static wrist context varies across external camera candidates")
    dev, evaluation = tuple(i for i in range(32) if i % 4 < 2), tuple(i for i in range(32) if i % 4 >= 2)
    scale = float(np.median(scores[list(dev)].mean(-1)))
    if scale <= 0:
        raise ValueError("Degenerate development-only scale")
    data = ReplayInputs(ids, tuple(states), frozen_array(poses), frozen_array(visual, np.float32),
                        frozen_array(wrist[:,0], np.float32), frozen_array(scores[None]),
                        dev, evaluation, (scale,), hashes)
    return data, OutcomeTable(labels)

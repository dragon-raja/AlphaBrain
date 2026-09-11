"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .artifacts import read, sha, write

MODELS = ["canonical", "broad"]


def audit_and_load(root, out):
    r = read(root / "release.json")
    rh = sha(root / "release.json")
    states = r["states"]
    cids = [c["selected_candidate_id"] for c in r["protocol"]["state_blocks"][0]["candidates"]]
    assert len(states) == 32 and len(cids) == 97 and cids[0] == "canonical"
    inputs = {str(root / "release.json"): rh}
    for h in r["hosts"]:
        for path in [
            root / f"hosts/{h}/dense-complete.json",
            root / f"hosts/{h}/gate-pass.json",
            root / f"scheduling/dynamic-v1/{h}/cross-host-gate-pass.json",
        ]:
            inputs[str(path)] = sha(path)
        assert read(root / f"hosts/{h}/dense-complete.json")["episodes"] == 99328
    vis = np.empty((32, 97))
    acc = np.empty((2, 32, 97, 8))
    camera = []
    artifact_shas = []
    for i in range(32):
        path = root / f"assets/aa0-a/{i:02d}/render.json"
        a = read(path)
        inputs[str(path)] = sha(path)
        assert a["release_sha256"] == rh and a["candidate_count"] == 97
        artifact = Path(a["artifact"])
        assert sha(artifact) == a["artifact_sha256"]
        # The rollout receipt hashes render.json (which in turn hashes the NPZ).
        inputs[str(artifact)] = a["artifact_sha256"]
        artifact_shas.append(sha(path))
        with np.load(artifact) as z:
            assert z["candidate_ids"].tolist() == cids
            camera.append(z["camera_to_world_opencv"])
        vd = {v["candidate_id"]: v["per_camera"]["agentview"]["score"] for v in a["visibility"]}
        vis[i] = [vd[c] for c in cids]
        for mi, model in enumerate(MODELS):
            path = root / f"scores/{model}/aa0-a/{i:02d}/ranking.json"
            sc = read(path)
            complete = read(path.parent / "completion.json")
            assert complete["ranking_sha256"] == sha(path) and complete["release_sha256"] == rh
            assert sc["pair_key"] == states[i]["pair_key"] and sc["physics_state_sha256"] == a["physics_state_sha256"]
            inputs[str(path)] = sha(path)
            sd = {v["candidate_id"]: v["member_accel_3"] for v in sc["ranking"]}
            acc[mi, i] = [sd[c] for c in cids]

    def cell(key):
        mi, si = key
        model = MODELS[mi]
        s = states[si]
        ii = s["initialization"]["init_state_index"]
        h = next(h for h, v in r["hosts"].items() if si in v["state_indices"])
        folder = root / f"hosts/{h}/dense-init-{ii}/{model}/{si:02d}"
        files = sorted(folder.glob("*.json"))
        assert len(files) == 3104
        y = np.full((97, 32), -1, dtype=np.int8)
        hashes = []
        lengths = np.empty((97, 32), dtype=np.int16)
        for f in files:
            raw = f.read_bytes()
            a = json.loads(raw)
            idx = a["aa0"]["index"]
            ss, rem = divmod(idx, 3104)
            c, n = divmod(rem, 32)
            assert ss == si and int(f.stem) == idx and y[c, n] == -1
            assert a["aa0"]["release_sha256"] == rh and a["aa0"]["model"] == model and a["aa0"]["host"] == h
            assert a["aa0"]["checkpoint_sha256"] == r["models"][model]["weights_sha256"]
            assert (
                a["pair_key"] == s["pair_key"] and a["selected_candidate_id"] == cids[c] and a["policy_repeat_id"] == n
            )
            assert a["noise_bank_manifest_sha256"] == r["noise_bank"]["manifest_sha256"] and a["explicit_flow_noise"]
            assert (
                a["initialization"] == s["initialization"]
                and not a["initialization_receipt"]["demonstration_state_used"]
            )
            assert a["initialization_receipt"]["matched_metric_asset_sha256"] == artifact_shas[si]
            assert set(a["aa0"]["render_samples"]) == {0} and a["status"] == "complete"
            assert a["scene_construction"] is None and isinstance(a["success"], bool)
            y[c, n] = a["success"]
            lengths[c, n] = a["completion_steps"]
            hashes.append((str(f.relative_to(root)), hashlib.sha256(raw).hexdigest()))
        assert (y >= 0).all()
        return mi, si, y, lengths, hashes

    Y = np.empty((2, 32, 97, 32), dtype=np.int8)
    steps = np.empty_like(Y, dtype=np.int16)
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for k, (m, s, y, length, hashes) in enumerate(pool.map(cell, [(m, s) for m in range(2) for s in range(32)])):
            Y[m, s] = y
            steps[m, s] = length
            records.extend(hashes)
            if (k + 1) % 8 == 0:
                print("Audit cells", k + 1, "/64", flush=True)
    write(out / "record-hashes.json", records)
    write(out / "inputs.json", inputs)
    np.savez_compressed(
        out / "matrix.npz",
        success=Y,
        completion_steps=steps,
        accel_members=acc,
        visibility=vis,
        camera=np.stack(camera),
        candidate_ids=np.array(cids),
    )
    write(out / "states.json", states)
    return r, Y, acc, vis, np.stack(camera)

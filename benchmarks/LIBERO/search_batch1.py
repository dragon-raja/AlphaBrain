"""Admit native batch-one queries, then materialize their score/feature table.

Only called by the explicitly executed search_features backend. The old batch16
table is provenance, not a numerical target for this user-approved reference.
"""

from contextlib import nullcontext
import json
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha, write
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from AlphaBrain.research.dsol.metrics.accel_inference import _validated_trace_output


ADMISSION_INPUTS = tuple((i, c) for i in (0, 4, 8, 12) for c in (0, 65))


def require_same(left, right, keys=("trace", "actions", "score")):
    for key in keys:
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Batch1 exact reference mismatch: {key}")


def admit(query, prior, output):
    """66 trajectories; paired hook check for every member; no lucky retries."""
    for _ in range(2):
        query(0, 0, 0, False, "admission")
    reference, timing = {}, []
    for initial, candidate in ADMISSION_INPUTS:
        off_times, on_times = [], []
        for member in range(4):
            off = query(initial, candidate, member, False, "admission")
            on = query(initial, candidate, member, True, "admission")
            require_same(off, on)
            if (initial, candidate, member) == (0, 0, 0):
                require_same(on, prior, keys=("trace", "actions"))
            reference[initial, candidate, member] = on
            off_times.append(off["seconds"])
            on_times.append(on["seconds"])
        timing.append(dict(initial=initial, candidate=candidate,
                           seconds_off={str(n):sum(off_times[:n]) for n in (1, 2, 4)},
                           seconds_on={str(n):sum(on_times[:n]) for n in (1, 2, 4)}))
    identities = np.asarray(list(reference), dtype=np.int64)
    np.savez_compressed(output / "reference.npz", identities=identities,
        **{k: np.stack([v[k] for v in reference.values()])
           for k in ("trace", "actions", "score", "external", "wrist")})
    write(output / "admission.json", dict(status="PASS", trajectories=66,
        inputs=8, members_per_input=4, hook_exact=True, prior_process_exact=True,
        timing=timing, timing_scope="Sequential prefix sums from four member calls; no repeated latency benchmark",
        reference_sha256=sha(output / "reference.npz")))
    return reference


def run_batch1(policy, tower, inputs, noises, old_reference, config, output):
    import torch
    from AlphaBrain.model.pi05_features import SiglipFeatureCapture
    from .search_features import load_observations, policy_example

    if len(inputs) != 32 or noises.shape != (32, 8, 10, 7):
        raise ValueError("Frozen 32-initial noise axes required")
    prior_path = config["batch1_reference"]["path"]
    if sha(prior_path) != config["batch1_reference"]["sha256"]:
        raise ValueError("Prior batch1 artifact changed")
    with np.load(prior_path, allow_pickle=False) as z:
        prior = {k:z[k].copy() for k in ("trace", "actions")}
    counters = {"admission":0, "export":0}
    limits = {"admission":66, "export":6208}
    cache = {}
    vision_calls = [0]

    def counter(module, args, result):
        vision_calls[0] += 1

    handle = tower.register_forward_hook(counter)

    def query(initial, candidate, member, capture, stage):
        if counters[stage] >= limits[stage]:
            raise ValueError("Frozen trajectory ceiling reached")
        if cache.get("initial") != initial:
            cache.clear()
            cache.update(initial=initial, arrays=load_observations(inputs[initial]))
        example = policy_example(inputs[initial], cache["arrays"], candidate)
        noise = noises[initial, member:member + 1]
        counters[stage] += 1
        with (output / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(stage=stage, index=counters[stage], initial=initial,
                                         candidate=candidate, member=member, capture=capture)) + "\n")
        tap = SiglipFeatureCapture(tower, patch_grid=(16, 16),
                                   view_names=("external", "wrist", "masked_dummy"))
        before = vision_calls[0]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), (tap if capture else nullcontext()):
            prediction = policy.predict_action(examples=[example], noise=noise, return_flow_trace=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        trace, _, _, actions = _validated_trace_output(
            prediction, noise, candidate_count=1, action_horizon=10, action_dim=7)
        metric = compute_accel_scores(trace, prefixes=[3])[3]
        if metric["degenerate"].any() or vision_calls[0] - before != 3:
            raise ValueError("Invalid Accel or extra/missing vision tower invocation")
        result = dict(trace=trace, actions=actions, score=metric["score"], seconds=elapsed)
        if capture:
            result.update({k:v.cpu().numpy().reshape(-1) for k,v in tap.features().items()
                           if k in ("external", "wrist")})
            if any(result[k].shape != (4608,) or not np.isfinite(result[k]).all()
                   for k in ("external", "wrist")):
                raise ValueError("Invalid native feature")
        return result

    try:
        reference = admit(query, prior, output)
        if counters["admission"] != 66:
            raise ValueError("Admission coverage changed")
        scores = np.empty((32, 97, 2), np.float64)
        external = np.empty((32, 97, 4608), np.float32)
        wrist = np.empty_like(external)
        durations = np.empty((32, 97, 2), np.float64)
        (output / "initials").mkdir()
        for initial in range(32):
            traces, actions = [], []
            for candidate in range(97):
                pair = [query(initial, candidate, member, True, "export") for member in (0, 1)]
                require_same(pair[0], pair[1], keys=("external", "wrist"))
                for member, item in enumerate(pair):
                    key = (initial, candidate, member)
                    if key in reference:
                        require_same(item, reference[key], keys=("trace", "actions", "score", "external", "wrist"))
                    scores[initial, candidate, member] = item["score"][0]
                    durations[initial, candidate, member] = item["seconds"]
                external[initial, candidate] = pair[0]["external"]
                wrist[initial, candidate] = pair[0]["wrist"]
                traces.append(np.stack([x["trace"][0] for x in pair]))
                actions.append(np.stack([x["actions"][0] for x in pair]))
            np.savez_compressed(output / "initials" / f"{initial:02d}.npz",
                trace=np.stack(traces), actions=np.stack(actions), scores=scores[initial],
                external=external[initial], wrist=wrist[initial], seconds=durations[initial])
            write(output / "progress.json", dict(completed_initials=initial + 1, attempted=counters))
            print(f"batch1 export initial={initial} complete, trajectories={counters['export']}", flush=True)
        if counters["export"] != 6208:
            raise ValueError("Export coverage changed")
        np.savez(output / "siglip-scores.npz", scores=scores, external=external, wrist=wrist,
                 seconds=durations, historical_scores=old_reference[:, :, :2])
        write(output / "features.json", dict(status="COMPLETE", batch_size=1, noise_count=2,
            attempted=counters, total_cumulative_trajectories=36 + sum(counters.values()),
            asset_sha256=sha(output / "siglip-scores.npz"), inputs_sha256=sha(config["inputs"]),
            weights_sha256=config["models"]["canonical"]["weights_sha256"],
            reference_sha256=sha(output / "reference.npz"), feature_source="native_siglip_2x2_pre_projector",
            observations=3104, vision_slot_evaluations=counters["export"] * 3,
            historical_scores_are_diagnostic_only=True, downstream_started=False))
        return dict(status="COMPLETE", attempted=counters)
    except BaseException as error:
        write(output / "failure.json", dict(status="FAILED_STOPPED", attempted=counters,
            total_cumulative_trajectories=36 + sum(counters.values()), error=repr(error), downstream_started=False))
        raise
    finally:
        handle.remove()

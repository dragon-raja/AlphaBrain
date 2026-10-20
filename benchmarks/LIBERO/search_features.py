"""Explicit, bounded policy-native feature/timing admission and export.

No simulator or training runs here. One process owns one GPU. Historical images
are hashed inputs to the actual frozen policy, never substitutes for its hidden
features. Failure stops this invocation and preserves all attempted calls.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from AlphaBrain.research.dsol.metrics.accel_inference import _validated_trace_output


def policy_example(record, cached, candidate):
    return dict(
        image=[cached["external_images"][candidate], cached["wrist_images"][candidate]],
        lang=record["state"]["prompt"], language=record["state"]["prompt"],
        state=cached["robot_states"][candidate],
        camera_intrinsics=cached["camera_intrinsics"][candidate],
        camera_to_world_opencv=cached["camera_to_world_opencv"][candidate])


def load_observations(record):
    asset = record["observation_cache"]
    if sha(asset["path"]) != asset["sha256"]:
        raise ValueError("Historical policy input hash mismatch")
    with np.load(asset["path"], allow_pickle=False) as arrays:
        cached = {k: arrays[k].copy() for k in arrays.files}
    if cached["candidate_ids"].tolist() != record["candidate_ids"]:
        raise ValueError("Candidate order changed")
    return cached


def batch_check_inputs(record, cached, noise, duplicate=False):
    """Historical candidate batch versus same-shape duplicate-view batch."""
    candidates = [0] * 16 if duplicate else list(range(16))
    return [policy_example(record, cached, j) for j in candidates], np.repeat(noise[None], 16, axis=0)


def run(config_path, output, *, export=False, execute=False, batch_check=False):
    if not execute:
        raise ValueError("Explicit --execute required")
    if batch_check and export:
        raise ValueError("Diagnostic cannot trigger feature export or downstream work")
    import torch
    from AlphaBrain.model.framework.base_framework import BaseFramework
    from AlphaBrain.model.pi05_features import SiglipFeatureCapture

    config = read(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    attempted = 0
    rows = []
    try:
        for path, digest in config["input_hashes"].items():
            if sha(path) != digest:
                raise ValueError("Bound input changed: " + path)
        checkpoint = config["models"]["canonical"]
        if sha(Path(checkpoint["path"]) / "model.safetensors") != checkpoint["weights_sha256"]:
            raise ValueError("Frozen policy weights changed")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        policy = BaseFramework.from_pretrained(
            checkpoint["path"], strict_checkpoint=True).to(torch.bfloat16).to("cuda").eval()
        policy.gripper_remap = False
        policy.requires_grad_(False)
        cfg = policy.config.framework.paligemma
        if cfg.num_images != 3 or list(cfg.image_mask) != [True, True, False]:
            raise ValueError("Unexpected policy image slots/masks")
        tower = policy.vlm_interface.model.vision_tower
        vision_config = tower.config
        grid = vision_config.image_size // vision_config.patch_size
        if grid != 16 or vision_config.hidden_size != 1152:
            raise ValueError("Unadmitted SigLIP token geometry")
        inputs = read(config["inputs"])["records"]
        with np.load(config["anchors"], allow_pickle=False) as arrays:
            noises = arrays["first_score_noise"].copy()
            reference = arrays["member_scores"][:, 0].copy()
        write(output / "runtime.json", dict(
            torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            weights_sha256=checkpoint["weights_sha256"], grid=grid, dim=1152,
            slots=list(cfg.image_mask), batch_size=16 if batch_check else 1))
        if batch_check:
            cached = load_observations(inputs[0])
            diagnostics = []
            for duplicate in (False, True):
                examples, noise_batch = batch_check_inputs(
                    inputs[0], cached, noises[0, 0], duplicate=duplicate)
                attempted += 16
                write(output / "attempts.json", dict(attempted=attempted, duplicate=duplicate))
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    result = policy.predict_action(
                        examples=examples, noise=noise_batch, return_flow_trace=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                trace, _, _, actions = _validated_trace_output(
                    result, noise_batch, candidate_count=16, action_horizon=10, action_dim=7)
                scores = compute_accel_scores(trace, prefixes=[3])[3]
                expected = np.repeat(reference[0, 0, 0], 16) if duplicate else reference[0, :16, 0]
                exact = bool(np.array_equal(scores["score"], expected))
                np.savez_compressed(output / ("duplicate.npz" if duplicate else "historical.npz"),
                                    trace=trace, actions=actions, scores=scores["score"], expected=expected)
                diagnostics.append(dict(duplicate=duplicate, exact=exact,
                    max_score_difference=float(np.max(abs(scores["score"] - expected))), seconds=elapsed))
                write(output / "batch-check.json", dict(status="IN_PROGRESS", attempted=attempted,
                                                        rows=diagnostics, downstream_started=False))
                if not exact or scores["degenerate"].any():
                    raise ValueError("Batch16 diagnostic disagrees with historical score")
            write(output / "batch-check.json", dict(status="PASS", attempted=attempted,
                rows=diagnostics, downstream_started=False,
                scope="Batch-shape diagnosis only; single-view rewards still unadmitted"))
            return dict(status="DIAGNOSIS_ONLY", attempted=attempted)
        vision_calls = [0]

        def counter(module, args, result):
            vision_calls[0] += 1

        handle = tower.register_forward_hook(counter)

        def infer(example, noise, capture, identity):
            nonlocal attempted
            if attempted >= 114:
                raise ValueError("114-noise admission ceiling")
            attempted += 1
            with (output / "calls.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(attempt=attempted, **identity)) + "\n")
            before = vision_calls[0]
            tap = SiglipFeatureCapture(tower, patch_grid=(16, 16),
                                       view_names=("external", "wrist", "masked_dummy"))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode(), (tap if capture else nullcontext()):
                result = policy.predict_action(
                    examples=[example], noise=noise[None], return_flow_trace=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            trace, times, actual_noise, actions = _validated_trace_output(
                result, noise[None], candidate_count=1, action_horizon=10, action_dim=7)
            metric = compute_accel_scores(trace, prefixes=[3])[3]
            if metric["degenerate"].any() or vision_calls[0] - before != 3:
                raise ValueError("Degenerate score or unexpected vision calls")
            features = {k: v.cpu().numpy() for k, v in tap.features().items()} if capture else None
            return dict(trace=trace, actions=actions, score=metric["score"],
                        seconds=elapsed, features=features)

        first = load_observations(inputs[0])
        example = policy_example(inputs[0], first, 0)
        for repeat in range(2):
            infer(example, noises[0, 0], False, dict(phase="warmup", repeat=repeat))
        for initial in (0, 4, 8, 12):
            cached = load_observations(inputs[initial])
            for candidate in (0, 65):
                example = policy_example(inputs[initial], cached, candidate)
                for count in (1, 2, 4):
                    paired = []
                    for capture in (False, True):
                        paired.append([
                            infer(example, noises[initial, j], capture,
                                  dict(phase="admission", initial=initial, candidate=candidate,
                                       count=count, member=j, capture=capture))
                            for j in range(count)])
                    for j, (old, new) in enumerate(zip(*paired)):
                        exact = all(np.array_equal(old[key], new[key]) for key in ("trace", "actions", "score"))
                        reference_exact = bool(np.array_equal(
                            new["score"], reference[initial, candidate, j:j + 1]))
                        row = dict(initial=initial, candidate=candidate, count=count, member=j,
                                   hook_exact=exact, historical_score_exact=reference_exact,
                                   historical_score=float(reference[initial, candidate, j]),
                                   actual_score=float(new["score"][0]),
                                   seconds_off=old["seconds"], seconds_on=new["seconds"])
                        rows.append(row)
                        write(output / "progress.json", dict(attempted=attempted, rows=rows))
                        if not exact:
                            raise ValueError("Feature capture changed policy outputs")
                        if not reference_exact:
                            np.savez_compressed(output / "mismatch.npz", actions=new["actions"],
                                                trace=new["trace"], noise=noises[initial, j])
                            raise ValueError("Online batch1 score differs from historical batch16; do not train on unvalidated rewards")
                print(f"admission initial={initial} candidate={candidate} passed", flush=True)
        handle.remove()
        if attempted != 114:
            raise ValueError("Admission coverage mismatch")
        write(output / "admission.json", dict(status="PASS", attempted=attempted, rows=rows,
                                             weights_sha256=checkpoint["weights_sha256"]))
        if export:
            external = np.empty((32, 97, 4608), np.float32)
            wrist = np.empty_like(external)
            for initial, record in enumerate(inputs):
                cached = load_observations(record)
                for candidate in range(97):
                    example = policy_example(record, cached, candidate)
                    with torch.inference_mode(), SiglipFeatureCapture(
                            tower, patch_grid=(16, 16),
                            view_names=("external", "wrist", "masked_dummy")) as tap:
                        policy._prepare_prefix_paligemma([example])
                    features = tap.features()
                    external[initial, candidate] = features["external"].cpu().numpy().reshape(-1)
                    wrist[initial, candidate] = features["wrist"].cpu().numpy().reshape(-1)
                write(output / "export-progress.json", dict(completed_initials=initial + 1))
                print(f"features initial={initial} complete", flush=True)
            np.savez(output / "siglip.npz", external=external, wrist=wrist)
            write(output / "features.json", dict(status="COMPLETE", observations=3104,
                vision_slot_evaluations=9312, new_action_denoising=0,
                weights_sha256=checkpoint["weights_sha256"], inputs_sha256=sha(config["inputs"]),
                feature_sha256=sha(output / "siglip.npz"), pooling=[2, 2], hidden_dim=1152))
        return dict(status="COMPLETE", attempted=attempted)
    except BaseException as error:
        write(output / "failure.json", dict(status="FAILED_STOPPED", attempted=attempted,
            error=repr(error), completed_comparisons=len(rows), downstream_started=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--batch-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, export=args.export, execute=args.execute, batch_check=args.batch_check)

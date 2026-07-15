from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORT_TASK = "recovery_support_view"
ARMS = ("clean_recovery_replay", "policy_state_recovery")


def first_stable_true(
    values: Sequence[bool],
    *,
    start: int,
    dwell_steps: int,
) -> int | None:
    """Return the inclusive index that completes the first stable true run."""
    if dwell_steps < 1:
        raise ValueError("dwell_steps must be positive")
    if not 0 <= start <= len(values):
        raise ValueError("start is outside the sequence")
    run = 0
    for index in range(start, len(values)):
        run = run + 1 if bool(values[index]) else 0
        if run >= dwell_steps:
            return index
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _absolute_observation(row: Mapping[str, Any], source_root: Path) -> dict[str, str]:
    result = {}
    for key in ("agentview_path", "wrist_path"):
        value = Path(str(row["observation"][key]))
        result[key] = str(value if value.is_absolute() else (source_root / value).resolve())
    return result


def _validate_pool_row(row: Mapping[str, Any], *, horizon: int) -> None:
    action = np.asarray(row["action_chunk"], dtype=np.float32)
    state = np.asarray(row["robot_state"], dtype=np.float32)
    if action.shape != (horizon, 7) or not np.all(np.isfinite(action)):
        raise ValueError(f"invalid action chunk for {row.get('sample_id')}: {action.shape}")
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid robot state for {row.get('sample_id')}: {state.shape}")


def clean_feedback_to_regrasp_pool(
    episode_root: Path,
    window_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    dwell_steps: int,
) -> dict[str, list[tuple[dict[str, Any], Path]]]:
    manifest = json.loads((episode_root / "manifest.json").read_text())
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row["split"] == "train" and row["branch_id"] == "slipped":
            by_pair[str(row["pair_id"])].append(row)

    result: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for group in manifest["groups"]:
        if group["split"] != "train":
            continue
        pair_id = str(group["pair_id"])
        feedback_index = int(group["feedback_reveal_time"])
        episode_path = episode_root / str(group["episode_files"]["slipped"])
        with np.load(episode_path, allow_pickle=False) as episode:
            grasped = np.asarray(episode["grasped"], dtype=bool)
        endpoint = first_stable_true(
            grasped,
            start=feedback_index,
            dwell_steps=dwell_steps,
        )
        if endpoint is None:
            raise RuntimeError(f"clean teacher never reaches stable regrasp: {pair_id}")
        pool = []
        for source in sorted(by_pair[pair_id], key=lambda row: int(row["frame_index"])):
            frame_index = int(source["frame_index"])
            if feedback_index <= frame_index < endpoint:
                copied = copy.deepcopy(source)
                _validate_pool_row(copied, horizon=horizon)
                pool.append((copied, window_root))
        if not pool:
            raise RuntimeError(f"no clean feedback-to-regrasp windows for {pair_id}")
        result[pair_id] = pool
    return result


def correction_pool(
    correction_root: Path,
    *,
    horizon: int,
) -> dict[str, list[tuple[dict[str, Any], Path]]]:
    quality_path = correction_root / "quality_report.json"
    if not quality_path.is_file() or not json.loads(quality_path.read_text()).get("passed"):
        raise RuntimeError(f"policy-state correction quality gate did not pass: {quality_path}")
    result: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for source in _load_jsonl(correction_root / "records.jsonl"):
        if source["split"] != "train":
            raise ValueError("policy-state correction data must contain train rows only")
        copied = copy.deepcopy(source)
        _validate_pool_row(copied, horizon=horizon)
        result[str(copied["pair_id"])].append((copied, correction_root))
    if not result:
        raise ValueError("policy-state correction pool is empty")
    return dict(result)


def _balanced_group_sequence(
    group_ids: Sequence[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    if not group_ids:
        raise ValueError("group_ids must not be empty")
    result = []
    ordered = np.asarray(sorted(group_ids), dtype=object)
    while len(result) < count:
        result.extend(str(value) for value in ordered[rng.permutation(len(ordered))])
    return result[:count]


def _view_row(
    source: Mapping[str, Any],
    source_root: Path,
    *,
    arm: str,
    seed: int,
    slot: int,
    slot_type: str,
    horizon: int,
) -> dict[str, Any]:
    row = copy.deepcopy(source)
    source_sample_id = str(row["sample_id"])
    row.update(
        {
            "sample_id": f"support::{arm}::seed{seed}::slot{slot:05d}",
            "window_group_id": f"support::seed{seed}::slot{slot:05d}",
            "task": SUPPORT_TASK,
            "split": "train",
            "schedule_slot": slot,
            "slot_type": slot_type,
            "support_arm": arm,
            "source_sample_id": source_sample_id,
            "source_pair_id": str(row["pair_id"]),
            "oracle_feedback_horizon": horizon,
        }
    )
    row["observation"] = _absolute_observation(source, source_root)
    return row


def build_matched_rows(
    anchor_pool: Sequence[tuple[dict[str, Any], Path]],
    clean_by_group: Mapping[str, Sequence[tuple[dict[str, Any], Path]]],
    policy_by_group: Mapping[str, Sequence[tuple[dict[str, Any], Path]]],
    *,
    seed: int,
    steps: int,
    horizon: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if steps < 2 or steps % 2:
        raise ValueError("steps must be a positive even number")
    if not anchor_pool:
        raise ValueError("anchor pool is empty")
    common_groups = sorted(set(clean_by_group) & set(policy_by_group))
    if not common_groups:
        raise ValueError("clean and policy-state pools have no common groups")
    if any(not clean_by_group[group] or not policy_by_group[group] for group in common_groups):
        raise ValueError("a common group has an empty target pool")

    target_count = steps // 2
    slot_rng = np.random.default_rng(seed + 10_001)
    slot_types = np.asarray(["anchor"] * target_count + ["target"] * target_count, dtype=object)
    slot_types = slot_types[slot_rng.permutation(steps)]
    group_sequence = iter(
        _balanced_group_sequence(
            common_groups,
            target_count,
            np.random.default_rng(seed + 20_003),
        )
    )
    anchor_rng = np.random.default_rng(seed + 30_011)
    clean_rng = np.random.default_rng(seed + 40_009)
    policy_rng = np.random.default_rng(seed + 50_021)

    rows = {arm: [] for arm in ARMS}
    schedule = []
    for slot, raw_slot_type in enumerate(slot_types):
        slot_type = str(raw_slot_type)
        if slot_type == "anchor":
            source, source_root = anchor_pool[int(anchor_rng.integers(len(anchor_pool)))]
            pair_id = str(source["pair_id"])
            clean_ref = policy_ref = (source, source_root)
        else:
            pair_id = next(group_sequence)
            clean_pool = clean_by_group[pair_id]
            policy_pool = policy_by_group[pair_id]
            clean_ref = clean_pool[int(clean_rng.integers(len(clean_pool)))]
            policy_ref = policy_pool[int(policy_rng.integers(len(policy_pool)))]
        for arm, reference in zip(ARMS, (clean_ref, policy_ref), strict=True):
            source, source_root = reference
            rows[arm].append(
                _view_row(
                    source,
                    source_root,
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    slot_type=slot_type,
                    horizon=horizon,
                )
            )
        schedule.append(
            {
                "slot": slot,
                "slot_type": slot_type,
                "source_pair_id": pair_id,
                "clean_source_sample_id": rows[ARMS[0]][-1]["source_sample_id"],
                "policy_source_sample_id": rows[ARMS[1]][-1]["source_sample_id"],
            }
        )

    target_counts = Counter(
        row["source_pair_id"] for row in rows[ARMS[0]] if row["slot_type"] == "target"
    )
    checks = {
        "equal_view_lengths": len(rows[ARMS[0]]) == len(rows[ARMS[1]]) == steps,
        "half_anchor_half_target": all(
            Counter(row["slot_type"] for row in rows[arm])
            == {"anchor": target_count, "target": target_count}
            for arm in ARMS
        ),
        "slot_types_aligned": all(
            left["slot_type"] == right["slot_type"]
            for left, right in zip(rows[ARMS[0]], rows[ARMS[1]], strict=True)
        ),
        "anchor_samples_identical": all(
            left["source_sample_id"] == right["source_sample_id"]
            for left, right in zip(rows[ARMS[0]], rows[ARMS[1]], strict=True)
            if left["slot_type"] == "anchor"
        ),
        "target_groups_identical": all(
            left["source_pair_id"] == right["source_pair_id"]
            for left, right in zip(rows[ARMS[0]], rows[ARMS[1]], strict=True)
            if left["slot_type"] == "target"
        ),
        "target_groups_balanced": max(target_counts.values()) - min(target_counts.values()) <= 1,
        "train_only": all(row["split"] == "train" for arm in ARMS for row in rows[arm]),
        "unique_sample_ids": all(
            len({row["sample_id"] for row in rows[arm]}) == steps for arm in ARMS
        ),
    }
    metadata = {
        "seed": seed,
        "steps": steps,
        "horizon": horizon,
        "anchor_count": target_count,
        "target_count": target_count,
        "common_group_count": len(common_groups),
        "target_group_counts": dict(sorted(target_counts.items())),
        "schedule": schedule,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return rows, metadata


def _write_view(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    metadata: Mapping[str, Any],
    source_roots: Mapping[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = {
        str(row["sample_id"]): {
            "full_h": int(metadata["horizon"]),
            "oracle_feedback_horizon": int(metadata["horizon"]),
        }
        for row in rows
    }
    missing_images = []
    for row in rows:
        for path in row["observation"].values():
            if not Path(path).is_file():
                missing_images.append(path)
    quality = {
        "passed": bool(metadata["passed"]) and not missing_images,
        "checks": {
            **metadata["checks"],
            "all_referenced_images_exist": not missing_images,
            "all_labels_full_horizon": all(
                value["oracle_feedback_horizon"] == metadata["horizon"]
                for value in labels.values()
            ),
        },
        "metrics": {
            "record_count": len(rows),
            "anchor_count": sum(row["slot_type"] == "anchor" for row in rows),
            "target_count": sum(row["slot_type"] == "target" for row in rows),
            "source_group_count": len({row["source_pair_id"] for row in rows}),
            "missing_image_count": len(missing_images),
        },
    }
    manifest = {
        "schema_version": 1,
        "generator": "build_recovery_support_views.py",
        "arm": arm,
        "task": SUPPORT_TASK,
        "seed": metadata["seed"],
        "steps": metadata["steps"],
        "horizon": metadata["horizon"],
        "shuffle": False,
        "sampling": "pre-randomized matched slot schedule",
        "source_roots": dict(source_roots),
        "policy_input_fields": ["agentview", "wrist", "robot_state", "language_instruction"],
        "loss_only_fields": [],
    }
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (output_dir / "training_labels.json").write_text(
        json.dumps(
            {"schema_version": 1, "horizon": metadata["horizon"], "records": labels},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    if not quality["passed"]:
        raise RuntimeError(f"recovery support view failed quality gate: {output_dir}")
    return quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched clean/on-policy recovery replay views")
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dwell-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")
    records = _load_jsonl(args.window_root / "records.jsonl")
    anchors = [
        (copy.deepcopy(row), args.window_root)
        for row in records
        if row["split"] == "train"
    ]
    for row, _ in anchors:
        _validate_pool_row(row, horizon=args.horizon)
    clean = clean_feedback_to_regrasp_pool(
        args.episode_root,
        args.window_root,
        records,
        horizon=args.horizon,
        dwell_steps=args.dwell_steps,
    )
    policy = correction_pool(args.correction_root, horizon=args.horizon)
    rows, metadata = build_matched_rows(
        anchors,
        clean,
        policy,
        seed=args.seed,
        steps=args.steps,
        horizon=args.horizon,
    )
    staging = args.output_root.parent / f".{args.output_root.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    source_roots = {
        "episode_root": str(args.episode_root),
        "window_root": str(args.window_root),
        "correction_root": str(args.correction_root),
    }
    try:
        qualities = {
            arm: _write_view(
                staging / arm,
                rows[arm],
                arm=arm,
                metadata=metadata,
                source_roots=source_roots,
            )
            for arm in ARMS
        }
        (staging / "matched_schedule.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        (staging / "quality_report.json").write_text(
            json.dumps(
                {
                    "passed": metadata["passed"] and all(value["passed"] for value in qualities.values()),
                    "matched_schedule": metadata,
                    "views": qualities,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        staging.rename(args.output_root)
    except Exception:
        raise
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "record_count_per_arm": args.steps,
                "common_group_count": metadata["common_group_count"],
                "passed": metadata["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

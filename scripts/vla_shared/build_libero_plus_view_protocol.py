from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


CAMERA_TASK_PATTERN = re.compile(
    r"^(?P<base>.+)_view_"
    r"(?P<orbit_yaw>-?\d+)_"
    r"(?P<orbit_pitch>-?\d+)_"
    r"(?P<radius_percent>\d+)_"
    r"(?P<look_yaw>-?\d+)_"
    r"(?P<look_pitch>-?\d+)_"
    r"initstate_(?P<robot_variant>\d+)"
    r"(?:_noise_(?P<noise>\d+))?$"
)

DEFAULT_CANDIDATE_VIEWS = (
    {
        "name": "canonical",
        "orbit_yaw_deg": 0,
        "orbit_pitch_deg": 0,
        "radius_percent": 100,
        "look_yaw_deg": 0,
        "look_pitch_deg": 0,
    },
    {
        "name": "yaw_m30",
        "orbit_yaw_deg": -30,
        "orbit_pitch_deg": 0,
        "radius_percent": 100,
        "look_yaw_deg": 0,
        "look_pitch_deg": 0,
    },
    {
        "name": "yaw_p30",
        "orbit_yaw_deg": 30,
        "orbit_pitch_deg": 0,
        "radius_percent": 100,
        "look_yaw_deg": 0,
        "look_pitch_deg": 0,
    },
    {
        "name": "pitch_p15",
        "orbit_yaw_deg": 0,
        "orbit_pitch_deg": 15,
        "radius_percent": 100,
        "look_yaw_deg": 0,
        "look_pitch_deg": 0,
    },
    {
        "name": "near_90",
        "orbit_yaw_deg": 0,
        "orbit_pitch_deg": 0,
        "radius_percent": 90,
        "look_yaw_deg": 0,
        "look_pitch_deg": 0,
    },
    {
        "name": "far_110",
        "orbit_yaw_deg": 0,
        "orbit_pitch_deg": 0,
        "radius_percent": 110,
        "look_yaw_deg": 0,
        "look_pitch_deg": 0,
    },
)


def normalize_degrees(value: int) -> int:
    normalized = (int(value) + 180) % 360 - 180
    return 180 if normalized == -180 and int(value) > 0 else normalized


def parse_camera_task_name(name: str) -> dict[str, Any]:
    match = CAMERA_TASK_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"not a LIBERO-Plus camera task name: {name}")
    raw = {key: int(value) for key, value in match.groupdict(default="0").items() if key != "base"}
    parsed = {
        "base_task": match.group("base"),
        "orbit_yaw_deg": normalize_degrees(raw["orbit_yaw"]),
        "orbit_pitch_deg": normalize_degrees(raw["orbit_pitch"]),
        "radius_percent": raw["radius_percent"],
        "look_yaw_deg": normalize_degrees(raw["look_yaw"]),
        "look_pitch_deg": normalize_degrees(raw["look_pitch"]),
        "robot_variant": raw["robot_variant"],
        "noise": raw["noise"],
    }
    changed = [
        axis
        for axis, value, canonical in (
            ("orbit_yaw", parsed["orbit_yaw_deg"], 0),
            ("orbit_pitch", parsed["orbit_pitch_deg"], 0),
            ("radius", parsed["radius_percent"], 100),
            ("look_yaw", parsed["look_yaw_deg"], 0),
            ("look_pitch", parsed["look_pitch_deg"], 0),
        )
        if value != canonical
    ]
    parsed["perturbation_axes"] = changed or ["canonical"]
    parsed["perturbation_family"] = changed[0] if len(changed) == 1 else "combined"
    return parsed


def synthetic_camera_task_name(base_task: str, view: Mapping[str, Any]) -> str:
    def encoded_angle(value: Any) -> int:
        angle = int(value)
        return angle % 360 if angle < 0 else angle

    return (
        f"{base_task}_view_"
        f"{encoded_angle(view['orbit_yaw_deg'])}_"
        f"{encoded_angle(view['orbit_pitch_deg'])}_"
        f"{int(view['radius_percent'])}_"
        f"{encoded_angle(view['look_yaw_deg'])}_"
        f"{encoded_angle(view['look_pitch_deg'])}_initstate_0"
    )


def _rank_key(row: Mapping[str, Any], *, seed: int) -> str:
    identity = f"{seed}::{row['suite']}::{row['difficulty_level']}::{row['name']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def camera_task_population(
    classification: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    population: list[dict[str, Any]] = []
    for suite, items in sorted(classification.items()):
        for item in items:
            if item.get("category") != "Camera Viewpoints":
                continue
            parsed = parse_camera_task_name(str(item["name"]))
            population.append(
                {
                    "suite": str(suite),
                    "task_id": int(item["id"]),
                    "task_index": int(item["id"]) - 1,
                    "name": str(item["name"]),
                    "category": "Camera Viewpoints",
                    "difficulty_level": int(item["difficulty_level"]),
                    **parsed,
                }
            )
    return population


def stratified_camera_sample(
    classification: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_suite_difficulty: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_suite_difficulty <= 0:
        raise ValueError("per-suite-difficulty must be positive")
    selected: list[dict[str, Any]] = []
    population = camera_task_population(classification)

    strata: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in population:
        strata.setdefault((row["suite"], row["difficulty_level"]), []).append(row)
    for key, rows in sorted(strata.items()):
        if len(rows) < per_suite_difficulty:
            raise ValueError(f"stratum {key} only contains {len(rows)} camera tasks")

    suites = sorted({suite for suite, _ in strata})
    for suite in suites:
        difficulties = sorted(difficulty for name, difficulty in strata if name == suite)
        best: tuple[int, list[dict[str, Any]]] | None = None
        for order in itertools.permutations(difficulties):
            used_base_tasks: set[str] = set()
            candidate: list[dict[str, Any]] = []
            for difficulty in order:
                ranked = sorted(
                    strata[(suite, difficulty)],
                    key=lambda row: _rank_key(row, seed=seed),
                )
                chosen: list[dict[str, Any]] = []
                for row in ranked:
                    if row["base_task"] in used_base_tasks:
                        continue
                    chosen.append(row)
                    used_base_tasks.add(str(row["base_task"]))
                    if len(chosen) == per_suite_difficulty:
                        break
                if len(chosen) < per_suite_difficulty:
                    for row in ranked:
                        if row in chosen:
                            continue
                        chosen.append(row)
                        if len(chosen) == per_suite_difficulty:
                            break
                candidate.extend(chosen)
            score = len({str(row["base_task"]) for row in candidate})
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            raise ValueError(f"suite {suite} contains no difficulty strata")
        selected.extend(best[1])

    population_counts = Counter(
        (row["suite"], row["difficulty_level"]) for row in population
    )
    summary = {
        "camera_population_count": len(population),
        "selected_count": len(selected),
        "selected_unique_base_task_count": len(
            {(row["suite"], row["base_task"]) for row in selected}
        ),
        "population_by_suite_difficulty": [
            {"suite": suite, "difficulty_level": difficulty, "count": count}
            for (suite, difficulty), count in sorted(population_counts.items())
        ],
    }
    return sorted(
        selected,
        key=lambda row: (row["suite"], row["difficulty_level"], row["task_id"]),
    ), summary


def build_protocol(
    classification: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_suite_difficulty: int,
    seed: int,
) -> dict[str, Any]:
    selected, summary = stratified_camera_sample(
        classification,
        per_suite_difficulty=per_suite_difficulty,
        seed=seed,
    )
    population = camera_task_population(classification)
    representatives: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(
        population,
        key=lambda item: (
            int(item["robot_variant"]) != 0,
            _rank_key(item, seed=seed),
        ),
    ):
        representatives.setdefault((str(row["suite"]), str(row["base_task"])), row)
    candidates = [
        {
            **view,
            "synthetic_suffix": synthetic_camera_task_name("TASK", view).removeprefix("TASK_"),
        }
        for view in DEFAULT_CANDIDATE_VIEWS
    ]
    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_view_generalization_and_active_sensing",
        "selection_seed": seed,
        "per_suite_difficulty": per_suite_difficulty,
        "summary": summary,
        "official_camera_tasks": selected,
        "candidate_views": candidates,
        "candidate_matrix_base_tasks": [
            {
                "suite": suite,
                "base_task": base_task,
                "task_index": int(representative["task_index"]),
                "representative_task_id": int(representative["task_id"]),
                "representative_camera_name": str(representative["name"]),
            }
            for (suite, base_task), representative in sorted(representatives.items())
        ],
        "metrics": {
            "view_generalization_gap": "canonical_success - official_camera_success",
            "static_view_gain": "selected_static_view_success - canonical_success",
            "active_view_gain": "active_selector_success - fixed_start_view_success",
            "oracle_view_headroom": "per_episode_best_view_success - canonical_success",
        },
    }


def build_full_camera_protocol(
    classification: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Build an official-compatible protocol containing every Camera task."""
    protocol = build_protocol(
        classification,
        per_suite_difficulty=1,
        seed=seed,
    )
    population = sorted(
        camera_task_population(classification),
        key=lambda row: (row["suite"], row["task_id"]),
    )
    counts = Counter((row["suite"], row["difficulty_level"]) for row in population)
    protocol.update(
        {
            "study": "pi05_libero_plus_camera_full",
            "protocol_scope": "libero_plus_camera_full",
            "official_camera_tasks": population,
            "summary": {
                "camera_population_count": len(population),
                "selected_count": len(population),
                "selected_unique_base_task_count": len(
                    {(row["suite"], row["base_task"]) for row in population}
                ),
                "population_by_suite_difficulty": [
                    {"suite": suite, "difficulty_level": difficulty, "count": count}
                    for (suite, difficulty), count in sorted(counts.items())
                ],
            },
        }
    )
    protocol.pop("per_suite_difficulty", None)
    return protocol


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic stratified LIBERO-Plus camera protocol"
    )
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-suite-difficulty", type=int, default=2)
    parser.add_argument(
        "--full-camera",
        action="store_true",
        help="include all official Camera Viewpoints tasks instead of a stratified subset",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {args.output}")
    classification = json.loads(args.classification.read_text())
    if args.full_camera:
        protocol = build_full_camera_protocol(classification, seed=args.seed)
    else:
        protocol = build_protocol(
            classification,
            per_suite_difficulty=args.per_suite_difficulty,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), **protocol["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()

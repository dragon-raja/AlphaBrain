from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from build_libero_plus_view_protocol import parse_camera_task_name


BACKGROUND_TASK_PATTERN = re.compile(
    r"^(?P<base>.+)_(?P<kind>table|tb)_(?P<texture_index>\d+)$"
)


def parse_background_task_name(name: str) -> dict[str, Any]:
    match = BACKGROUND_TASK_PATTERN.fullmatch(str(name))
    if match is None:
        raise ValueError(f"not a LIBERO-Plus background task name: {name}")
    return {
        "base_task": match.group("base"),
        "background_kind": match.group("kind"),
        "background_texture_index": int(match.group("texture_index")),
    }


def compose_camera_background_task_name(
    camera_task_name: str,
    background_task_name: str,
) -> str:
    camera = parse_camera_task_name(camera_task_name)
    background = parse_background_task_name(background_task_name)
    if camera["base_task"] != background["base_task"]:
        raise ValueError(
            "camera and background tasks have different bases: "
            f"{camera['base_task']} != {background['base_task']}"
        )
    suffix = str(camera_task_name)[len(str(camera["base_task"])) :]
    return f"{background_task_name}{suffix}"


def background_task_population(
    classification: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    background_kind: str = "tb",
) -> list[dict[str, Any]]:
    if background_kind not in {"table", "tb"}:
        raise ValueError("background_kind must be table or tb")
    population: list[dict[str, Any]] = []
    for suite, items in sorted(classification.items()):
        for item in items:
            if item.get("category") != "Background Textures":
                continue
            parsed = parse_background_task_name(str(item["name"]))
            if parsed["background_kind"] != background_kind:
                continue
            population.append(
                {
                    "suite": str(suite),
                    "task_id": int(item["id"]),
                    "task_index": int(item["id"]) - 1,
                    "name": str(item["name"]),
                    "category": "Background Textures",
                    "difficulty_level": int(item["difficulty_level"]),
                    **parsed,
                }
            )
    return population


def _rank_background(
    row: Mapping[str, Any],
    *,
    camera_task: Mapping[str, Any],
    seed: int,
) -> str:
    identity = (
        f"{seed}::{camera_task['suite']}::{camera_task['name']}::"
        f"{row['name']}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_composition_protocol(
    view_protocol: Mapping[str, Any],
    classification: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    background_kind: str = "tb",
) -> dict[str, Any]:
    population = background_task_population(
        classification,
        background_kind=background_kind,
    )
    by_base: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in population:
        by_base[(str(row["suite"]), str(row["base_task"]))].append(row)

    tasks = []
    excluded = []
    exact_matches = 0
    for camera in view_protocol["official_camera_tasks"]:
        key = (str(camera["suite"]), str(camera["base_task"]))
        candidates = by_base.get(key, [])
        if not candidates:
            excluded.append(
                {
                    "suite": str(camera["suite"]),
                    "base_task": str(camera["base_task"]),
                    "camera_task_name": str(camera["name"]),
                    "reason": f"no_official_{background_kind}_background_variant",
                }
            )
            continue
        target_difficulty = int(camera["difficulty_level"])
        minimum_distance = min(
            abs(int(row["difficulty_level"]) - target_difficulty)
            for row in candidates
        )
        pool = [
            row
            for row in candidates
            if abs(int(row["difficulty_level"]) - target_difficulty)
            == minimum_distance
        ]
        selected = min(
            pool,
            key=lambda row: _rank_background(
                row,
                camera_task=camera,
                seed=seed,
            ),
        )
        exact_match = minimum_distance == 0
        exact_matches += int(exact_match)
        tasks.append(
            {
                **dict(camera),
                "camera_task_name": str(camera["name"]),
                "camera_difficulty_level": target_difficulty,
                "background_task_id": int(selected["task_id"]),
                "background_task_name": str(selected["name"]),
                "background_difficulty_level": int(selected["difficulty_level"]),
                "background_difficulty_distance": int(minimum_distance),
                "background_kind": str(selected["background_kind"]),
                "background_texture_index": int(selected["background_texture_index"]),
                "background_exact_difficulty_match": exact_match,
                "camera_background_task_name": compose_camera_background_task_name(
                    str(camera["name"]),
                    str(selected["name"]),
                ),
            }
        )

    counts = Counter((row["suite"], row["difficulty_level"]) for row in tasks)
    distance_counts = Counter(row["background_difficulty_distance"] for row in tasks)
    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_camera_background_composition",
        "selection_seed": int(seed),
        "background_kind": background_kind,
        "source_view_protocol_study": view_protocol.get("study"),
        "summary": {
            "source_camera_task_count": len(view_protocol["official_camera_tasks"]),
            "selected_count": len(tasks),
            "selected_unique_base_task_count": len(
                {(row["suite"], row["base_task"]) for row in tasks}
            ),
            "background_population_count": len(population),
            "excluded_camera_task_count": len(excluded),
            "exact_difficulty_match_count": exact_matches,
            "background_difficulty_distance_counts": {
                str(distance): count
                for distance, count in sorted(distance_counts.items())
            },
            "selected_by_suite_difficulty": [
                {
                    "suite": suite,
                    "difficulty_level": difficulty,
                    "count": count,
                }
                for (suite, difficulty), count in sorted(counts.items())
            ],
        },
        "composition_tasks": tasks,
        "excluded_camera_tasks": excluded,
        "conditions": {
            "canonical": "canonical camera and canonical scene",
            "camera_only": "official camera perturbation and canonical scene",
            "background_only": "canonical camera and unseen table/background textures",
            "camera_background": (
                "official camera perturbation and the paired unseen table/background textures"
            ),
        },
        "estimands": {
            "camera_gap_canonical_background": "success(canonical) - success(camera_only)",
            "camera_gap_unseen_background": (
                "success(background_only) - success(camera_background)"
            ),
            "combined_gap": "success(canonical) - success(camera_background)",
            "negative_composition_interaction": (
                "camera_gap_unseen_background - camera_gap_canonical_background"
            ),
        },
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paired LIBERO-Plus camera by background protocol"
    )
    parser.add_argument("--view-protocol", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--background-kind", choices=("table", "tb"), default="tb")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    protocol = build_composition_protocol(
        json.loads(args.view_protocol.read_text()),
        json.loads(args.classification.read_text()),
        seed=args.seed,
        background_kind=args.background_kind,
    )
    protocol["source_view_protocol"] = str(args.view_protocol)
    protocol["source_view_protocol_sha256"] = hashlib.sha256(
        args.view_protocol.read_bytes()
    ).hexdigest()
    protocol["source_classification"] = str(args.classification)
    protocol["source_classification_sha256"] = hashlib.sha256(
        args.classification.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

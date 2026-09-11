from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    phrase: str
    object_name: str
    object_type: str


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    phrase: str
    object_name: str


SOURCES = (
    SourceSpec("red", "red mug", "red_coffee_mug_1", "red_coffee_mug"),
    SourceSpec("white", "white mug", "porcelain_mug_1", "porcelain_mug"),
    SourceSpec(
        "yellow_white",
        "yellow and white mug",
        "white_yellow_mug_1",
        "white_yellow_mug",
    ),
)

TARGETS = (
    TargetSpec("left", "left plate", "plate_1"),
    TargetSpec("right", "right plate", "plate_2"),
)

SUPERVISED_EDGES = frozenset(
    {
        ("red", "left"),
        ("red", "right"),
        ("white", "left"),
        ("yellow_white", "right"),
    }
)

CANONICAL_INIT = (
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate.pruned_init"
)

REGIONS = """    (:regions
      (plate_left_region
          (:target living_room_table)
          (:ranges ((-0.025 -0.325 0.025 -0.27499999999999997)))
          (:yaw_rotation ((0.0 0.0)))
      )
      (plate_right_region
          (:target living_room_table)
          (:ranges ((-0.025 0.27499999999999997 0.025 0.325)))
          (:yaw_rotation ((0.0 0.0)))
      )
      (porcelain_mug_init_region
          (:target living_room_table)
          (:ranges ((-0.125 -0.175 -0.07500000000000001 -0.125)))
          (:yaw_rotation ((0.0 0.0)))
      )
      (white_yellow_mug_init_region
          (:target living_room_table)
          (:ranges ((-0.07500000000000001 0.07500000000000001 -0.025 0.125)))
          (:yaw_rotation ((0.0 0.0)))
      )
      (red_coffee_mug_init_region
          (:target living_room_table)
          (:ranges ((-0.225 -0.025 -0.17500000000000002 0.025)))
          (:yaw_rotation ((0.0 0.0)))
      )
    )"""


def edge_id(source: SourceSpec, target: TargetSpec) -> str:
    return f"{source.source_id}-{target.target_id}"


def render_bddl(source: SourceSpec, target: TargetSpec) -> str:
    language = f"put the {source.phrase} on the {target.phrase}"
    return f"""(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language})
{REGIONS}

  (:fixtures
    living_room_table - living_room_table
  )

  (:objects
    porcelain_mug_1 - porcelain_mug
    red_coffee_mug_1 - red_coffee_mug
    white_yellow_mug_1 - white_yellow_mug
    plate_1 plate_2 - plate
  )

  (:obj_of_interest
    {source.object_name}
    {target.object_name}
  )

  (:init
    (On plate_1 living_room_table_plate_left_region)
    (On plate_2 living_room_table_plate_right_region)
    (On red_coffee_mug_1 living_room_table_red_coffee_mug_init_region)
    (On white_yellow_mug_1 living_room_table_white_yellow_mug_init_region)
    (On porcelain_mug_1 living_room_table_porcelain_mug_init_region)
  )

  (:goal
    (And (On {source.object_name} {target.object_name}))
  )
)
"""


def state_split(index: int) -> str:
    if 0 <= index <= 34:
        return "train"
    if 35 <= index <= 39:
        return "val"
    if 40 <= index <= 49:
        return "test"
    raise ValueError(f"state index must be in [0, 49], got {index}")


def build_manifest(
    *,
    bddl_dir: Path,
    canonical_init_path: Path,
    state_count: int = 50,
) -> dict:
    if state_count != 50:
        raise ValueError("LIBERO-Bind v0 is sealed to the 50-state canonical bank")
    edges = []
    for source in SOURCES:
        for target in TARGETS:
            supervised = (source.source_id, target.target_id) in SUPERVISED_EDGES
            name = edge_id(source, target)
            edges.append(
                {
                    "edge_id": name,
                    "source_id": source.source_id,
                    "target_id": target.target_id,
                    "language_instruction": f"put the {source.phrase} on the {target.phrase}",
                    "source_object": source.object_name,
                    "target_object": target.object_name,
                    "bddl": str(bddl_dir / f"{name}.bddl"),
                    "action_supervised": supervised,
                    "representation_anchor_allowed": True,
                    "sealed_action_evaluation_only": not supervised,
                }
            )
    states = [
        {"canonical_state_index": index, "split": state_split(index)}
        for index in range(state_count)
    ]
    return {
        "schema_version": 1,
        "benchmark": "LIBERO-Bind-v0",
        "scene": "LIVING_ROOM_SCENE5",
        "canonical_init_states": str(canonical_init_path),
        "sources": [asdict(value) for value in SOURCES],
        "targets": [asdict(value) for value in TARGETS],
        "edges": edges,
        "states": states,
        "policy_input_fields": ["agentview", "wrist", "robot_state", "language_instruction"],
        "training_only_fields": [
            "edge_id",
            "source_id",
            "target_id",
            "canonical_state_index",
            "action_supervised",
        ],
        "leakage_rule": (
            "withheld edges expose only the common initial observation and instruction "
            "during training; expert actions and post-initial rollout states remain sealed"
        ),
    }


def write_suite(output_dir: Path, canonical_init_path: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing suite: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    bddl_dir = staging / "bddl"
    bddl_dir.mkdir(parents=True, exist_ok=False)
    try:
        for source in SOURCES:
            for target in TARGETS:
                path = bddl_dir / f"{edge_id(source, target)}.bddl"
                path.write_text(render_bddl(source, target))
        manifest = build_manifest(
            bddl_dir=output_dir / "bddl",
            canonical_init_path=canonical_init_path,
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    return manifest


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the sealed LIBERO-Bind v0 task matrix")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share/longjunyu/cabi-vla/libero-bind-v0"),
    )
    parser.add_argument(
        "--canonical-init-states",
        type=Path,
        default=Path(
            "/share/longjunyu/capt-vla/vendor/LIBERO/libero/libero/init_files/libero_90"
        )
        / CANONICAL_INIT,
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if not args.canonical_init_states.is_file():
        raise FileNotFoundError(args.canonical_init_states)
    manifest = write_suite(args.output_dir, args.canonical_init_states)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "edge_count": len(manifest["edges"]),
                "supervised_edges": sum(edge["action_supervised"] for edge in manifest["edges"]),
                "withheld_edges": sum(not edge["action_supervised"] for edge in manifest["edges"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

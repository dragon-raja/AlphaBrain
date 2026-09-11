#!/usr/bin/env python3
"""Freeze a visibility-selected, outcome-blind strong-information pilot."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "dsol_visibility_selected_strong_gate_protocol_v1"
CONDITIONS = (
    ("canonical_both", "canonical", "both"),
    ("strong_info_both", "strong_info", "both"),
    ("matched_control_both", "matched_control", "both"),
    ("canonical_external_only", "canonical", "external_only"),
    ("strong_info_external_only", "strong_info", "external_only"),
    ("matched_control_external_only", "matched_control", "external_only"),
    ("canonical_wrist_only", "canonical", "wrist_only"),
    ("all_camera_blackout", "canonical", "all_blackout"),
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Iterable[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def canonical_record(scan: Mapping[str, Any]) -> Mapping[str, Any]:
    values = [row for row in scan["records"] if row.get("pose_id") == "canonical"]
    if len(values) != 1:
        raise ValueError("scan must contain exactly one canonical record")
    return values[0]


def passing_pair(
    scan: Mapping[str, Any],
    *,
    minimum_strong_delta: float,
    maximum_control_abs_delta: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    paired: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in scan["records"]:
        if record.get("group") != "constructed_task_orbit":
            continue
        pose = record["pose"]
        paired[str(pose["pair_id"])][str(pose["pair_member"])] = record
    options = []
    for pair_id, members in paired.items():
        if set(members) != {"negative", "positive"}:
            raise ValueError(f"incomplete mirrored pair {pair_id}")
        negative, positive = members["negative"], members["positive"]
        strong, control = (
            (negative, positive)
            if float(negative["delta_visibility"]) >= float(positive["delta_visibility"])
            else (positive, negative)
        )
        strong_delta = float(strong["delta_visibility"])
        control_delta = float(control["delta_visibility"])
        if (
            strong_delta >= minimum_strong_delta
            and abs(control_delta) <= maximum_control_abs_delta
        ):
            options.append((strong_delta - control_delta, strong_delta, pair_id, strong, control))
    if not options:
        return None
    _specificity, _strong_delta, _pair_id, strong, control = max(options)
    return strong, control


def select(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    minimum_strong_delta: float,
    maximum_control_abs_delta: float,
    maximum_canonical_external: float,
    maximum_canonical_wrist: float,
) -> list[
    tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
    ]
]:
    selected = []
    for row in rows:
        if row.get("split") != split or row.get("status") != "PASS":
            continue
        scan = json.loads((Path(row["output_dir"]) / "scan.json").read_text())
        if bool(scan.get("initial_task_success")):
            continue
        canonical = canonical_record(scan)
        if (
            float(canonical["per_camera_scores"]["agentview"])
            > maximum_canonical_external
            or float(canonical["per_camera_scores"]["robot0_eye_in_hand"])
            > maximum_canonical_wrist
        ):
            continue
        pair = passing_pair(
            scan,
            minimum_strong_delta=minimum_strong_delta,
            maximum_control_abs_delta=maximum_control_abs_delta,
        )
        if pair is not None:
            selected.append((row, scan, canonical, pair[0], pair[1]))
    return selected


def build(
    rows: Sequence[Mapping[str, Any]],
    *,
    catalog: Path,
    minimum_strong_delta: float,
    maximum_control_abs_delta: float,
    maximum_canonical_external: float,
    maximum_canonical_wrist: float,
    minimum_validation_episodes: int,
    minimum_test_episodes: int,
) -> dict[str, Any]:
    kwargs = {
        "minimum_strong_delta": minimum_strong_delta,
        "maximum_control_abs_delta": maximum_control_abs_delta,
        "maximum_canonical_external": maximum_canonical_external,
        "maximum_canonical_wrist": maximum_canonical_wrist,
    }
    validation = select(rows, split="val", **kwargs)
    test = select(rows, split="test", **kwargs)
    validation_episodes = {str(row[0]["episode_id"]) for row in validation}
    test_episodes = {str(row[0]["episode_id"]) for row in test}
    status = (
        "HOLD_MANUAL_AUDIT"
        if len(validation_episodes) >= minimum_validation_episodes
        and len(test_episodes) >= minimum_test_episodes
        else "FAIL"
    )
    specs = []
    selected_states = []
    for row, scan, canonical, strong, control in test:
        roles = {
            "canonical": canonical,
            "strong_info": strong,
            "matched_control": control,
        }
        pair_key = str(row["scan_id"])
        visibility_selection = {
            role: {
                "pose_id": str(record["pose_id"]),
                "visibility_score": float(record["visibility_score"]),
                "delta_visibility": float(record["delta_visibility"]),
                "per_camera_scores": record.get("per_camera_scores"),
            }
            for role, record in roles.items()
        }
        common = {
            "pair_key": pair_key,
            "scan_id": pair_key,
            "task_id": str(row["task_id"]),
            "diagnostic_role": str(row["diagnostic_role"]),
            "suite": str(row["suite"]),
            "hdf5": str(row["hdf5"]),
            "episode_id_source": str(row["episode_id"]),
            "demo_name": str(row["demo_name"]),
            "demo_index": int(row["demo_index"]),
            "split": "test",
            "source_state_index": int(row["frame"]),
            "stage_fraction": float(row["stage_fraction"]),
            "scene_construction": scan["scene_construction"],
            "visibility_selection": visibility_selection,
            "selection_used_policy_outcomes": False,
        }
        for condition, role, sensor_control in CONDITIONS:
            identity = f"{pair_key}::{condition}"
            specs.append(
                {
                    **common,
                    "condition": condition,
                    "pose": roles[role].get("pose"),
                    "sensor_control": sensor_control,
                    "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                }
            )
        selected_states.append(
            {
                "pair_key": pair_key,
                "task_id": str(row["task_id"]),
                "source_episode_id": str(row["episode_id"]),
                "source_frame": int(row["frame"]),
                "stage_fraction": float(row["stage_fraction"]),
                "scan_output_dir": str(row["output_dir"]),
                "strong_pose_id": str(strong["pose_id"]),
                "control_pose_id": str(control["pose_id"]),
                "strong_delta_visibility": float(strong["delta_visibility"]),
                "control_delta_visibility": float(control["delta_visibility"]),
                "canonical_external_visibility": float(
                    canonical["per_camera_scores"]["agentview"]
                ),
                "canonical_wrist_visibility": float(
                    canonical["per_camera_scores"]["robot0_eye_in_hand"]
                ),
                "strong_visibility_pass": True,
                "control_visibility_pass": True,
            }
        )
    return {
        "schema": SCHEMA,
        "status": status,
        "analysis_role": "expanded_A_strong_information_mechanism_pilot",
        "selection_policy": "per_state_mirrored_pair_visibility_threshold",
        "policy_outcomes_used_for_selection": False,
        "test_threshold_retuning": False,
        "statistical_unit": "source_episode",
        "thresholds_frozen_before_test_policy_evaluation": kwargs,
        "validation_passing_state_count": len(validation),
        "validation_passing_source_episode_count": len(validation_episodes),
        "test_passing_state_count": len(test),
        "test_passing_source_episode_count": len(test_episodes),
        "minimum_validation_source_episode_count": minimum_validation_episodes,
        "minimum_test_source_episode_count": minimum_test_episodes,
        "catalog": str(catalog.resolve()),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "condition_count": len(CONDITIONS),
        "episode_count": len(specs),
        "selected_states": selected_states,
        "specs": specs,
    }


def contact_sheet(protocol: Mapping[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw

    cards = []
    for state in protocol["selected_states"]:
        source = Path(state["scan_output_dir"]) / "visibility_extremes.png"
        image = Image.open(source).convert("RGB")
        width = 1400
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height + 54), "white")
        canvas.paste(image, (0, 54))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            (
                f"{state['pair_key']} | strong={state['strong_pose_id']} "
                f"{100 * state['strong_delta_visibility']:.1f}pp | "
                f"control={state['control_pose_id']} "
                f"{100 * state['control_delta_visibility']:.1f}pp"
            ),
            fill="black",
        )
        cards.append(canvas)
    sheet = Image.new("RGB", (1400, sum(card.height for card in cards)), "white")
    offset = 0
    for card in cards:
        sheet.paste(card, (0, offset))
        offset += card.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--minimum-strong-delta", type=float, default=0.05)
    parser.add_argument("--maximum-control-abs-delta", type=float, default=0.02)
    parser.add_argument("--maximum-canonical-external", type=float, default=0.005)
    parser.add_argument("--maximum-canonical-wrist", type=float, default=0.02)
    parser.add_argument("--minimum-validation-episodes", type=int, default=2)
    parser.add_argument("--minimum-test-episodes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        load_rows(args.inputs),
        catalog=args.catalog,
        minimum_strong_delta=args.minimum_strong_delta,
        maximum_control_abs_delta=args.maximum_control_abs_delta,
        maximum_canonical_external=args.maximum_canonical_external,
        maximum_canonical_wrist=args.maximum_canonical_wrist,
        minimum_validation_episodes=args.minimum_validation_episodes,
        minimum_test_episodes=args.minimum_test_episodes,
    )
    atomic_json(args.output, result)
    if result["selected_states"]:
        contact_sheet(result, args.contact_sheet)
    print(
        json.dumps(
            {
                "status": result["status"],
                "validation_states": result["validation_passing_state_count"],
                "test_states": result["test_passing_state_count"],
                "episodes": result["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from analyze_libero_bind_camera_viewpoints import paired_bootstrap_delta, summarize_rows


def select_rule_rows(
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    default_pose = str(rule["default_pose"])
    edge_pose = {str(key): str(value) for key, value in rule.get("edge_pose", {}).items()}
    lookup = {
        (
            str(row["camera_pose"]),
            str(row["edge_id"]),
            int(row["canonical_state_index"]),
            int(row["execution_horizon"]),
        ): row
        for row in rows
    }
    baseline_keys = sorted(
        (
            str(row["edge_id"]),
            int(row["canonical_state_index"]),
            int(row["execution_horizon"]),
        )
        for row in rows
        if str(row["camera_pose"]) == default_pose
    )
    if not baseline_keys:
        raise ValueError("evaluation does not contain the rule's default pose")
    baseline_rows = []
    selected_rows = []
    for edge_id, state_index, horizon in baseline_keys:
        baseline_key = (default_pose, edge_id, state_index, horizon)
        selected_pose = edge_pose.get(edge_id, default_pose)
        selected_key = (selected_pose, edge_id, state_index, horizon)
        if selected_key not in lookup:
            raise KeyError(f"missing rule episode: {selected_key}")
        baseline_rows.append(lookup[baseline_key])
        selected_rows.append(
            {
                **lookup[selected_key],
                "source_camera_pose": selected_pose,
                "camera_pose": "camera_rule",
            }
        )
    return baseline_rows, selected_rows


def summarize_edges(
    baseline_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = []
    for label, rows in (("baseline", baseline_rows), ("camera_rule", selected_rows)):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["edge_id"])].append(row)
        for edge_id, edge_rows in sorted(grouped.items()):
            summary = summarize_rows(
                [{**row, "camera_pose": label} for row in edge_rows]
            )[0]
            values.append({"policy": label, "edge_id": edge_id, **summary})
    return values


def plot_rule_comparison(summaries: Sequence[Mapping[str, Any]], output: Path) -> None:
    labels = ["baseline", "camera_rule"]
    by_label = {str(row["camera_pose"]): row for row in summaries}
    metrics = [
        ("success", "Task success"),
        ("progress", "Progress"),
        ("source_selection_success", "Correct grasp"),
        ("transport_success", "Transport"),
    ]
    width, height = 760, 460
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), "Held-out task-conditioned camera rule", fill="black")
    left, right, top, bottom = 70, 730, 70, 390
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - int(tick * (bottom - top))
        draw.line((left, y, right, y), fill="#DDDDDD")
        draw.text((28, y - 6), f"{tick:.2g}", fill="#555555")
    colors = {"baseline": "#0072B2", "camera_rule": "#E69F00"}
    group_width = (right - left) / len(metrics)
    for metric_index, (metric, metric_label) in enumerate(metrics):
        center = left + (metric_index + 0.5) * group_width
        for label_index, label in enumerate(labels):
            value = float(by_label[label][metric])
            bar_width = 48
            x0 = int(center - 54 + label_index * 58)
            y0 = bottom - int(value * (bottom - top))
            draw.rectangle(
                (x0, y0, x0 + bar_width, bottom),
                fill=colors[label],
            )
            draw.text((x0 + 8, y0 - 18), f"{value:.2f}", fill="black")
        draw.text((int(center - 48), bottom + 15), metric_label, fill="black")
    draw.rectangle((500, 20, 515, 35), fill=colors["baseline"])
    draw.text((522, 20), "Baseline", fill="black")
    draw.rectangle((610, 20, 625, 35), fill=colors["camera_rule"])
    draw.text((632, 20), "Camera rule", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def evaluate_rule(
    evaluation: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rows, selected_rows = select_rule_rows(evaluation["rows"], rule)
    labeled_rows = [
        *[{**row, "camera_pose": "baseline"} for row in baseline_rows],
        *selected_rows,
    ]
    summaries = summarize_rows(labeled_rows)
    rule_summary = next(
        summary for summary in summaries if summary["camera_pose"] == "camera_rule"
    )
    rule_summary.update(
        {
            "azimuth_deg": None,
            "elevation_deg": None,
            "radius_scale": None,
            "source_camera_poses": sorted(
                {str(row["source_camera_pose"]) for row in selected_rows}
            ),
        }
    )
    paired = [
        paired_bootstrap_delta(
            labeled_rows,
            candidate="camera_rule",
            baseline="baseline",
            metric=metric,
        )
        for metric in ("success", "progress", "completion_steps")
    ]
    return {
        "baseline_rows": baseline_rows,
        "selected_rows": selected_rows,
        "summaries": summaries,
        "edge_summaries": summarize_edges(baseline_rows, selected_rows),
        "paired_group_bootstrap": paired,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen task-conditioned camera rule"
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    evaluation = json.loads(args.evaluation.read_text())
    rule = json.loads(args.rule.read_text())
    if int(rule.get("schema_version", -1)) != 1:
        raise ValueError("camera rule must use schema_version=1")
    result = evaluate_rule(evaluation, rule)
    report = {
        "schema_version": 1,
        "evaluation": str(args.evaluation),
        "rule_path": str(args.rule),
        "rule": rule,
        "episode_count_per_policy": len(result["baseline_rows"]),
        "summaries": result["summaries"],
        "edge_summaries": result["edge_summaries"],
        "paired_group_bootstrap": result["paired_group_bootstrap"],
        "selected_episode_provenance": [
            {
                "edge_id": row["edge_id"],
                "canonical_state_index": row["canonical_state_index"],
                "source_camera_pose": row["source_camera_pose"],
            }
            for row in result["selected_rows"]
        ],
    }
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        with (staging / "edge_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(result["edge_summaries"][0]),
            )
            writer.writeheader()
            writer.writerows(result["edge_summaries"])
        plot_rule_comparison(
            result["summaries"],
            staging / "camera_rule_comparison.png",
        )
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summaries": result["summaries"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

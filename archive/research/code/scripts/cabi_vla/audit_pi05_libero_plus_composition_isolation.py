from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_CONDITIONS = frozenset({"camera_only", "camera_background"})
EXPECTED_CONDITIONS = frozenset(
    {"canonical", "camera_only", "background_only", "camera_background"}
)


def matrix_key(matrix: Any, *, decimals: int = 4) -> tuple[float, ...]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("camera pose must be a finite 4x4 matrix")
    return tuple(np.round(value.reshape(-1), decimals=decimals).tolist())


def eval_camera_to_world(row: Mapping[str, Any]) -> np.ndarray:
    metrics = row["initial_metrics"]
    rotation_mujoco = np.asarray(
        metrics["agent_camera_rotation"], dtype=np.float64
    ).reshape(3, 3)
    position = np.asarray(
        metrics["agent_camera_position"], dtype=np.float64
    ).reshape(3)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.column_stack(
        (
            rotation_mujoco[:, 0],
            -rotation_mujoco[:, 1],
            -rotation_mujoco[:, 2],
        )
    )
    camera_to_world[:3, 3] = position
    return camera_to_world


def load_episode_rows(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        paths = sorted(glob.glob(str(run_dir / "episodes-shard-*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"no episode shards found in {run_dir}")
        for path in paths:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def audit_isolation(
    training_manifest: Mapping[str, Any],
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    training_split: str,
    budget_fraction: float,
    pose_decimals: int = 4,
) -> dict[str, Any]:
    if not 0.0 < budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be in (0, 1]")
    episodes = list(training_manifest.get("episodes", []))
    selected_training = [
        row
        for row in episodes
        if str(row.get("split")) == training_split
        and float(row.get("budget_percentile", 0.0)) <= budget_fraction
    ]
    if not selected_training:
        raise ValueError("training selection is empty")
    if not eval_rows:
        raise ValueError("evaluation rows are empty")

    training_pose_keys = {
        matrix_key(row["camera_to_world_opencv"], decimals=pose_decimals)
        for row in selected_training
    }
    eval_pose_keys = {
        matrix_key(eval_camera_to_world(row), decimals=pose_decimals)
        for row in eval_rows
        if str(row.get("condition")) in CAMERA_CONDITIONS
    }

    condition_counts = Counter(str(row.get("condition")) for row in eval_rows)
    rows_by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        rows_by_pair[str(row["pair_key"])].append(row)
    invalid_condition_pairs = []
    physical_mismatch_pairs = []
    for pair_key, rows in rows_by_pair.items():
        conditions = {str(row.get("condition")) for row in rows}
        if conditions != EXPECTED_CONDITIONS:
            invalid_condition_pairs.append(pair_key)
        physics_hashes = {
            str(row["initial_metrics"]["physics_state_sha256"])
            for row in rows
        }
        if len(physics_hashes) != 1:
            physical_mismatch_pairs.append(pair_key)

    training_languages = {
        str(row.get("language_instruction", "")).strip().lower()
        for row in selected_training
        if str(row.get("language_instruction", "")).strip()
    }
    eval_languages = {
        str(row.get("language", row.get("prompt", ""))).strip().lower()
        for row in eval_rows
        if str(row.get("language", row.get("prompt", ""))).strip()
    }
    background_keys = {
        key
        for row in selected_training
        for key in row
        if "background" in str(key).lower() or "texture" in str(key).lower()
    }
    pose_overlap = training_pose_keys & eval_pose_keys
    camera_isolation_pass = not pose_overlap
    pairing_pass = not invalid_condition_pairs and not physical_mismatch_pairs
    background_factor_observed = bool(background_keys)
    strict_seen_factor_composition = (
        camera_isolation_pass and background_factor_observed and pairing_pass
    )

    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_composition_isolation_audit",
        "training": {
            "split": training_split,
            "budget_fraction": budget_fraction,
            "selected_episode_count": len(selected_training),
            "unique_camera_pose_count": len(training_pose_keys),
            "language_count": len(training_languages),
            "background_descriptor_keys": sorted(background_keys),
        },
        "evaluation": {
            "episode_count": len(eval_rows),
            "pair_count": len(rows_by_pair),
            "condition_counts": dict(sorted(condition_counts.items())),
            "unique_perturbed_camera_pose_count": len(eval_pose_keys),
            "language_count": len(eval_languages),
        },
        "isolation": {
            "camera_pose_overlap_count": len(pose_overlap),
            "camera_pose_isolation_pass": camera_isolation_pass,
            "task_language_overlap_count": len(training_languages & eval_languages),
            "task_identity_disjoint": not bool(training_languages & eval_languages),
            "background_factor_observed_in_training_manifest": (
                background_factor_observed
            ),
            "paired_physics_pass": pairing_pass,
            "invalid_condition_pair_count": len(invalid_condition_pairs),
            "physical_mismatch_pair_count": len(physical_mismatch_pairs),
        },
        "classification": {
            "strict_seen_factor_composition": strict_seen_factor_composition,
            "label": (
                "STRICT_SEEN_FACTOR_COMPOSITION"
                if strict_seen_factor_composition
                else "PAIRED_JOINT_OOD_STRESS_TEST"
            ),
            "reason": (
                "camera poses are held out, but the training manifest does not "
                "contain a background factor; task identities are also shared"
                if not strict_seen_factor_composition
                else "camera and background factors are represented in training "
                "while the evaluated pairing is held out"
            ),
        },
    }


def render_report(result: Mapping[str, Any]) -> str:
    training = result["training"]
    evaluation = result["evaluation"]
    isolation = result["isolation"]
    classification = result["classification"]
    return "\n".join(
        [
            "# Pi0.5 LIBERO-Plus 组合评测隔离审计",
            "",
            f"**分类：`{classification['label']}`**",
            "",
            "## 数据边界",
            "",
            f"- 训练选择：{training['selected_episode_count']} episodes，"
            f"{training['unique_camera_pose_count']} 个相机位姿。",
            f"- 闭环评测：{evaluation['episode_count']} episodes，"
            f"{evaluation['pair_count']} 个四条件配对组。",
            f"- 评测扰动相机：{evaluation['unique_perturbed_camera_pose_count']} 个。",
            "",
            "## 泄漏检查",
            "",
            f"- 训练/评测相机外参重合：{isolation['camera_pose_overlap_count']}。",
            f"- 四条件物理初态配对：{'通过' if isolation['paired_physics_pass'] else '失败'}。",
            f"- 训练/评测任务语言重合：{isolation['task_language_overlap_count']}。",
            f"- 训练清单是否包含背景因子："
            f"{'是' if isolation['background_factor_observed_in_training_manifest'] else '否'}。",
            "",
            "## 解释",
            "",
            classification["reason"] + "。",
            "",
        ]
    )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit train/eval isolation for the LIBERO-Plus composition gate"
    )
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, action="append", required=True)
    parser.add_argument("--training-split", default="train")
    parser.add_argument("--budget-fraction", type=float, default=0.25)
    parser.add_argument("--pose-decimals", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.training_manifest.read_text())
    result = audit_isolation(
        manifest,
        load_episode_rows(args.eval_dir),
        training_split=args.training_split,
        budget_fraction=args.budget_fraction,
        pose_decimals=args.pose_decimals,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(render_report(result))
    print(json.dumps(result["classification"], sort_keys=True))


if __name__ == "__main__":
    main()

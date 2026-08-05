from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from evaluate_pi05_libero_plus_views import (
    DUMMY_ACTION,
    build_episode_specs,
    physics_state_sha256,
    stable_seed,
)


CONDITIONS = (
    "canonical",
    "camera_only",
    "background_only",
    "camera_background",
)


def image_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
        )
    )


def render_spec(
    spec: Mapping[str, Any],
    *,
    task_suite: Any,
    bddl_root: Path,
    seed: int,
    wait_steps: int,
    render_gpu: int,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    from libero.libero.envs import OffScreenRenderEnv

    init_states = task_suite.get_task_init_states(int(spec["task_index"]))
    bddl = bddl_root / str(spec["suite"]) / f"{spec['camera_task_name']}.bddl"
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=render_gpu,
    )
    episode_seed = stable_seed(str(spec["pair_key"]), seed=seed)
    env.seed(episode_seed)
    env.reset()
    observation = env.set_init_state(init_states[int(spec["init_state_index"])])
    for _ in range(wait_steps):
        observation, _, _, _ = env.step(DUMMY_ACTION)
    env.env.sim.forward()
    state_hash, state_size = physics_state_sha256(env)
    agent = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1])
    wrist = np.ascontiguousarray(
        np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1]
    )
    return (
        {
            "condition": str(spec["condition"]),
            "bddl_file": str(bddl),
            "physics_state_sha256": state_hash,
            "physics_state_size": state_size,
            "agent": agent,
            "wrist": wrist,
        },
        env,
        np.concatenate([agent, wrist], axis=1),
    )


def make_contact_sheet(
    rows: list[tuple[str, dict[str, np.ndarray]]],
    output: Path,
) -> None:
    title_height = 30
    cell_width = 512
    cell_height = 256 + title_height
    canvas = Image.new(
        "RGB",
        (cell_width * len(CONDITIONS), cell_height * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (task, images) in enumerate(rows):
        for column_index, condition in enumerate(CONDITIONS):
            x = column_index * cell_width
            y = row_index * cell_height
            draw.text((x + 6, y + 6), f"{condition} | {task}", fill="black")
            canvas.paste(Image.fromarray(images[condition]), (x, y + title_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    from libero.libero import benchmark

    protocol = json.loads(args.protocol.read_text())
    specs = build_episode_specs(
        protocol,
        modes=["composition"],
        init_state_count=1,
    )
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        by_pair.setdefault(str(spec["pair_key"]), []).append(spec)
    selected = list(sorted(by_pair.items()))[: args.task_count]
    if len(selected) < args.task_count:
        raise ValueError("protocol has fewer pairs than requested")

    environments = []
    contact_rows = []
    summaries = []
    suites: dict[str, Any] = {}
    for pair_key, pair_specs in selected:
        pair_specs = sorted(pair_specs, key=lambda row: CONDITIONS.index(row["condition"]))
        suite_name = str(pair_specs[0]["suite"])
        task_suite = suites.setdefault(
            suite_name,
            benchmark.get_benchmark_dict()[suite_name](),
        )
        rendered = {}
        images = {}
        for spec in pair_specs:
            row, environment, combined = render_spec(
                spec,
                task_suite=task_suite,
                bddl_root=args.bddl_root,
                seed=args.seed,
                wait_steps=args.wait_steps,
                render_gpu=args.render_gpu,
            )
            environments.append(environment)
            rendered[str(spec["condition"])] = row
            images[str(spec["condition"])] = combined
        hashes = {rendered[condition]["physics_state_sha256"] for condition in CONDITIONS}
        if len(hashes) != 1:
            raise ValueError(f"physics state mismatch for {pair_key}")
        diagnostics = {
            "agent_canonical_vs_camera_mae": image_mae(
                rendered["canonical"]["agent"], rendered["camera_only"]["agent"]
            ),
            "wrist_canonical_vs_camera_mae": image_mae(
                rendered["canonical"]["wrist"], rendered["camera_only"]["wrist"]
            ),
            "agent_canonical_vs_background_mae": image_mae(
                rendered["canonical"]["agent"], rendered["background_only"]["agent"]
            ),
            "wrist_canonical_vs_background_mae": image_mae(
                rendered["canonical"]["wrist"], rendered["background_only"]["wrist"]
            ),
            "agent_background_vs_composition_mae": image_mae(
                rendered["background_only"]["agent"],
                rendered["camera_background"]["agent"],
            ),
            "wrist_background_vs_composition_mae": image_mae(
                rendered["background_only"]["wrist"],
                rendered["camera_background"]["wrist"],
            ),
        }
        if diagnostics["agent_canonical_vs_camera_mae"] <= 1.0:
            raise ValueError(f"camera perturbation is not visible for {pair_key}")
        if diagnostics["agent_canonical_vs_background_mae"] <= 1.0:
            raise ValueError(f"background perturbation is not visible for {pair_key}")
        if diagnostics["wrist_canonical_vs_camera_mae"] > 1e-6:
            raise ValueError(f"external camera perturbation changed wrist pixels for {pair_key}")
        if diagnostics["wrist_background_vs_composition_mae"] > 1e-6:
            raise ValueError(f"external camera perturbation changed wrist pixels for {pair_key}")
        summaries.append(
            {
                "pair_key": pair_key,
                "base_task": str(pair_specs[0]["base_task"]),
                "physics_state_sha256": next(iter(hashes)),
                "diagnostics": diagnostics,
            }
        )
        contact_rows.append((str(pair_specs[0]["base_task"]), images))

    make_contact_sheet(contact_rows, args.output_figure)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "study": "libero_plus_camera_background_protocol_smoke",
        "protocol": str(args.protocol),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "task_count": len(summaries),
        "physics_pairing_valid": True,
        "external_camera_isolation_valid": True,
        "pairs": summaries,
        "contact_sheet": str(args.output_figure),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and validate paired LIBERO-Plus composition cells"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=2)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--render-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate(args)
    print(json.dumps({key: payload[key] for key in ("status", "task_count")}, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

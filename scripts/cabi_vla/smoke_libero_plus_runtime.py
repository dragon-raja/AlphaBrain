from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def runtime_config(
    *,
    runtime: Path,
    dataset_root: Path,
) -> dict[str, str]:
    benchmark_root = runtime / "libero" / "libero"
    return {
        "assets": str(benchmark_root / "assets"),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "benchmark_root": str(benchmark_root),
        "datasets": str(dataset_root),
        "init_states": str(benchmark_root / "init_files"),
    }


def write_runtime_config(
    *,
    runtime: Path,
    dataset_root: Path,
    config_root: Path,
) -> Path:
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    config_path.write_text(
        json.dumps(
            runtime_config(runtime=runtime, dataset_root=dataset_root),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return config_path


def smoke(
    *,
    runtime: Path,
    overlay: Path,
    dataset_root: Path,
    config_root: Path,
    output: Path,
    render_gpu: int,
) -> dict[str, Any]:
    manifest_path = runtime / "libero_plus_runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"runtime manifest is incomplete: {manifest_path}")
    write_runtime_config(
        runtime=runtime,
        dataset_root=dataset_root,
        config_root=config_root,
    )

    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(overlay))

    import numpy as np
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = (
        runtime
        / "libero"
        / "libero"
        / "bddl_files"
        / "libero_goal"
        / "put_the_cream_cheese_in_the_bowl.bddl"
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=224,
        camera_widths=224,
        render_gpu_device_id=render_gpu,
    )
    try:
        observation = env.reset()
        observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        external = np.flipud(observation["agentview_image"])
        wrist = np.flipud(observation["robot0_eye_in_hand_image"])
        frame = np.concatenate((external, wrist), axis=1).astype(np.uint8)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(output)
        result = {
            "schema_version": 1,
            "status": "complete",
            "study": "libero_plus_runtime_smoke",
            "task": "put_the_cream_cheese_in_the_bowl",
            "external_shape": list(external.shape),
            "wrist_shape": list(wrist.shape),
            "output": str(output),
            "runtime_source_commit": manifest["source_commit"],
        }
        smoke_manifest = output.with_suffix(".json")
        result["manifest"] = str(smoke_manifest)
        smoke_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        env.close()
    return result


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the LIBERO-Plus runtime")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-gpu", type=int, default=0)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    result = smoke(
        runtime=args.runtime,
        overlay=args.overlay,
        dataset_root=args.dataset_root,
        config_root=args.config_root,
        output=args.output,
        render_gpu=args.render_gpu,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

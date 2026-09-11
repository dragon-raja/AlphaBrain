from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any


REQUIRED_DEMO_DATASETS = (
    "actions",
    "dones",
    "rewards",
    "robot_states",
    "states",
)
REQUIRED_OBSERVATIONS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _runtime_config(runtime: Path, dataset_root: Path) -> dict[str, str]:
    benchmark_root = runtime / "libero" / "libero"
    return {
        "assets": str(benchmark_root / "assets"),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "benchmark_root": str(benchmark_root),
        "datasets": str(dataset_root),
        "init_states": str(benchmark_root / "init_files"),
    }


def _configure_runtime(runtime: Path, dataset_root: Path, config_root: Path) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "config.yaml").write_text(
        json.dumps(
            _runtime_config(runtime=runtime, dataset_root=dataset_root),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(runtime))


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _rewrite_model_paths(model_xml: str, runtime: Path) -> tuple[str, dict[str, int]]:
    import robosuite

    root = ET.fromstring(model_xml)
    robosuite_root = Path(robosuite.__file__).resolve().parent
    libero_assets = runtime / "libero" / "libero" / "assets"
    counts = {"robosuite": 0, "libero": 0}

    for element in root.findall(".//*[@file]"):
        old_path = element.get("file")
        if not old_path:
            continue
        parts = Path(old_path).parts
        if "robosuite" in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index("robosuite")
            element.set("file", str(robosuite_root.joinpath(*parts[index + 1 :])))
            counts["robosuite"] += 1
            continue
        marker = "/chiliocosm/assets/"
        if marker in old_path:
            element.set("file", str(libero_assets / old_path.split(marker, 1)[1]))
            counts["libero"] += 1

    return ET.tostring(root, encoding="unicode"), counts


def _select_frames(length: int, count: int) -> list[int]:
    import numpy as np

    if length <= 0:
        return []
    return sorted({int(value) for value in np.linspace(0, length - 1, min(count, length))})


def _image_metrics(rendered: Any, reference: Any) -> dict[str, float]:
    import numpy as np

    rendered_array = np.asarray(rendered, dtype=np.float32)
    reference_array = np.asarray(reference, dtype=np.float32)
    error = np.abs(rendered_array - reference_array)
    return {
        "mae": float(error.mean()),
        "p99_abs_error": float(np.quantile(error, 0.99)),
        "max_abs_error": float(error.max()),
    }


def _audit_file_schema(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"missing data group: {path}")
        data = handle["data"]
        demos = sorted(data.keys())
        if not demos:
            raise ValueError(f"no demonstrations: {path}")
        sample = data[demos[0]]
        missing = [name for name in REQUIRED_DEMO_DATASETS if name not in sample]
        if "obs" not in sample:
            missing.append("obs")
        else:
            missing.extend(
                f"obs/{name}" for name in REQUIRED_OBSERVATIONS if name not in sample["obs"]
            )
        if missing:
            raise ValueError(f"missing required datasets in {path}: {missing}")
        state_dims = sorted({int(data[name]["states"].shape[1]) for name in demos})
        action_dims = sorted({int(data[name]["actions"].shape[1]) for name in demos})
        return {
            "path": str(path),
            "demo_count": len(demos),
            "sample_count": int(sum(data[name]["states"].shape[0] for name in demos)),
            "state_dims": state_dims,
            "action_dims": action_dims,
            "bddl_file_name": Path(_decode(data.attrs["bddl_file_name"])).name,
        }


def _audit_restore(
    *,
    path: Path,
    runtime: Path,
    demos_per_file: int,
    frames_per_demo: int,
    render_gpu: int,
) -> dict[str, Any]:
    import h5py
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv

    suite = path.parent.name
    with h5py.File(path, "r") as handle:
        bddl_name = Path(_decode(handle["data"].attrs["bddl_file_name"])).name
    bddl = runtime / "libero" / "libero" / "bddl_files" / suite / bddl_name
    if not bddl.is_file():
        raise FileNotFoundError(f"missing BDDL file: {bddl}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=render_gpu,
    )
    records: list[dict[str, Any]] = []
    rewrite_counts = {"robosuite": 0, "libero": 0}
    try:
        env.reset()
        with h5py.File(path, "r") as handle:
            demos = sorted(handle["data"].keys())[:demos_per_file]
            for demo_name in demos:
                demo = handle["data"][demo_name]
                model_xml, counts = _rewrite_model_paths(
                    _decode(demo.attrs["model_file"]), runtime
                )
                for name, count in counts.items():
                    rewrite_counts[name] += count
                env.reset_from_xml_string(model_xml)
                frames = _select_frames(int(demo["states"].shape[0]), frames_per_demo)
                for frame in frames:
                    state = np.asarray(demo["states"][frame])
                    observation = env.set_init_state(state)
                    restored_state = np.asarray(env.get_sim_state())
                    records.append(
                        {
                            "demo": demo_name,
                            "frame": frame,
                            "state_max_abs_error": float(
                                np.max(np.abs(restored_state - state))
                            ),
                            "agentview": _image_metrics(
                                observation["agentview_image"],
                                demo["obs/agentview_rgb"][frame],
                            ),
                            "wrist": _image_metrics(
                                observation["robot0_eye_in_hand_image"],
                                demo["obs/eye_in_hand_rgb"][frame],
                            ),
                        }
                    )
    finally:
        env.close()

    max_state_error = max(record["state_max_abs_error"] for record in records)
    max_agent_mae = max(record["agentview"]["mae"] for record in records)
    max_wrist_mae = max(record["wrist"]["mae"] for record in records)
    mean_agent_mae = sum(record["agentview"]["mae"] for record in records) / len(
        records
    )
    mean_wrist_mae = sum(record["wrist"]["mae"] for record in records) / len(
        records
    )
    state_restore_status = "PASS" if max_state_error <= 1e-8 else "FAIL"
    render_equivalence_status = (
        "PASS" if max_agent_mae <= 3.0 and max_wrist_mae <= 5.0 else "RENDER_DRIFT"
    )
    return {
        "path": str(path),
        "bddl": str(bddl),
        "asset_path_rewrites": rewrite_counts,
        "records": records,
        "max_state_abs_error": max_state_error,
        "max_agentview_mae": max_agent_mae,
        "max_wrist_mae": max_wrist_mae,
        "mean_agentview_mae": mean_agent_mae,
        "mean_wrist_mae": mean_wrist_mae,
        "state_restore_status": state_restore_status,
        "source_rgb_equivalence_status": render_equivalence_status,
        "pair_generation_status": state_restore_status,
        "status": (
            "PASS"
            if state_restore_status == "PASS" and render_equivalence_status == "PASS"
            else "PASS_WITH_RENDER_DRIFT"
            if state_restore_status == "PASS"
            else "FAIL"
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    runtime = args.runtime.resolve()
    files = sorted(
        path
        for path in dataset_root.glob("*/*.hdf5")
        if path.is_file() and path.stat().st_size > 0
    )
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"no HDF5 files found under {dataset_root}")

    restore_files = files
    if args.restore_files_per_suite is not None:
        if args.restore_files_per_suite <= 0:
            raise ValueError("restore-files-per-suite must be positive")
        suites = sorted({path.parent.name for path in files})
        selected: list[Path] = []
        for suite in suites:
            selected.extend(
                [path for path in restore_files if path.parent.name == suite][
                    : args.restore_files_per_suite
                ]
            )
        restore_files = selected

    _configure_runtime(runtime, dataset_root, args.config_root.resolve())
    schemas = [_audit_file_schema(path) for path in files]
    restores = []
    if not args.schema_only:
        restores = [
            _audit_restore(
                path=path,
                runtime=runtime,
                demos_per_file=args.demos_per_file,
                frames_per_demo=args.frames_per_demo,
                render_gpu=args.render_gpu,
            )
            for path in restore_files
        ]

    status = (
        "PASS"
        if all(item.get("status") == "PASS" for item in restores)
        else "PASS_WITH_RENDER_DRIFT"
        if restores
        and all(item.get("pair_generation_status") == "PASS" for item in restores)
        else "FAIL"
    )
    if args.schema_only:
        status = "PASS"
    result = {
        "schema": "dsol_libero_hdf5_restore_audit_v1",
        "status": status,
        "output": str(args.output.resolve()),
        "dataset_root": str(dataset_root),
        "runtime": str(runtime),
        "file_count": len(files),
        "restore_file_count": len(restore_files) if not args.schema_only else 0,
        "schema_audits": schemas,
        "restore_audits": restores,
    }
    _write_json_atomic(args.output.resolve(), result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit canonical LIBERO HDF5 schema and exact-state rendering."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--demos-per-file", type=int, default=1)
    parser.add_argument("--frames-per-demo", type=int, default=3)
    parser.add_argument("--restore-files-per-suite", type=int)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--schema-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = audit(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "file_count": result["file_count"],
                "output": result.get("output"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

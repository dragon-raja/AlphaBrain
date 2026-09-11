from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_candidate_support import load_episode
from evaluate_libero_closed_loop import _frame, _restore_recorded_state
from libero_full_episode_collector import FullEpisodeTeacher, object_grasped
from libero_snapshot_collector import _step
from video_io import write_h264_video


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    value = np.asarray(frame, dtype=np.uint8).copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 28), (255, 255, 255), -1)
    cv2.putText(value, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return value


def contact_sheet(path: Path, frames: list[np.ndarray]) -> None:
    indices = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    chosen = [frames[index] for index in indices]
    sheet = np.concatenate(chosen, axis=1)
    Image.fromarray(sheet).save(path)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    parser = argparse.ArgumentParser(description="Render Gate 1 label-audit candidate rollouts")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-max-steps", type=int, default=320)
    args = parser.parse_args()
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = {group["pair_id"]: group for group in manifest["groups"]}
    examples = json.loads(args.examples.read_text())["examples"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest["bddl"])),
        camera_heights=224,
        camera_widths=224,
    )
    rendered = []
    try:
        for example_index, example in enumerate(examples):
            group = groups[example["pair_id"]]
            outcome = example["outcome"]
            reference = load_episode(args.episode_root, group, outcome)
            observation = _restore_recorded_state(env, reference, int(example["frame_index"]))
            with np.load(example["candidate_file"], allow_pickle=False) as candidates:
                chunk = np.asarray(candidates["chunks"][int(example["candidate_index"])], dtype=np.float32)
            title = (
                f"{outcome} s{example['checkpoint_seed']} c{example['candidate_index']} "
                f"{example['visualization_reason']}"
            )
            frames = [caption(_frame(observation), title + " start")]
            for step, action in enumerate(chunk[:2], start=1):
                observation = _step(env, action)
                frames.append(caption(_frame(observation), title + f" candidate K={step}"))
            teacher = FullEpisodeTeacher(observation)
            teacher_steps = 0
            success = bool(env.check_success())
            while not success and not teacher.done and teacher_steps < args.teacher_max_steps:
                decision = teacher.decide(
                    observation,
                    grasped=object_grasped(env),
                    success=success,
                )
                observation = _step(env, decision.action)
                teacher_steps += 1
                success = bool(env.check_success())
                frames.append(caption(_frame(observation), title + f" teacher={decision.phase}"))
            stem = f"{example_index:02d}-{outcome}-s{example['checkpoint_seed']}-{example['pair_id']}-c{example['candidate_index']}"
            video = args.output_dir / f"{stem}.mp4"
            sheet = args.output_dir / f"{stem}.jpg"
            write_h264_video(video, frames, fps=10.0)
            contact_sheet(sheet, frames)
            rendered.append(
                {
                    **example,
                    "video": str(video),
                    "contact_sheet": str(sheet),
                    "teacher_steps": teacher_steps,
                    "success": success,
                }
            )
            print(json.dumps({"rendered": stem, "success": success}, sort_keys=True), flush=True)
    finally:
        env.close()
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"encoding": "H.264/avc1 yuv420p faststart", "examples": rendered}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

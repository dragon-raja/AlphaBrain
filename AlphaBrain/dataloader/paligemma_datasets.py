"""
PaliGemmaOFT Data Pipeline

Adapts VLAE's existing LeRobot data loading to PaliGemmaOFT format.
Reuses the existing lerobot_datasets.py infrastructure, adding Pi0-specific transforms.

Pi0 expects:
  - images: dict of {camera_name: [B, H, W, 3] uint8 tensors}
  - image_masks: dict of {camera_name: [B] bool tensors} 
  - state: [B, state_dim] float32
  - tokenized_prompt: [B, max_token_len] int32
  - tokenized_prompt_mask: [B, max_token_len] bool
  - actions: [B, action_horizon, action_dim] float32
"""

import json
import logging
from typing import Any, Mapping, Optional, Dict, List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Pi0DataConfig:
    """Configuration for Pi0-specific data processing."""
    image_resolution: tuple = (224, 224)  # H, W for SigLIP
    max_token_len: int = 200  # max language token length (pi05 default)
    action_horizon: int = 50
    action_dim: int = 7
    camera_names: tuple = ("image_0",)  # maps to observation camera keys
    include_state: bool = True
    state_dim: int = 7
    feedback_horizon_key: str = "feedback_horizon"


class Pi0DataTransform:
    """
    Transform VLAE LeRobot data samples into PaliGemmaOFT format.
    
    Input (from LeRobot dataloader):
        dict with keys: image (List[PIL.Image]), lang (str), action (np.ndarray), state (np.ndarray)
        
    Output (for PaliGemmaOFT.forward()):
        dict with same keys, but images resized and ready for Pi0 processing
    """
    
    def __init__(self, config: Pi0DataConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer  # PaliGemma/Gemma tokenizer
        
    def __call__(self, sample: dict) -> dict:
        """Transform a single sample."""
        result = {}
        
        # ── Images ──
        images = sample.get("image", [])
        if isinstance(images, (list, tuple)):
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = img.resize(
                        (self.config.image_resolution[1], self.config.image_resolution[0]),
                        Image.BILINEAR
                    )
                    img = np.array(img)
                elif isinstance(img, np.ndarray):
                    # Resize numpy image
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize(
                        (self.config.image_resolution[1], self.config.image_resolution[0]),
                        Image.BILINEAR
                    )
                    img = np.array(pil_img)
                elif isinstance(img, torch.Tensor):
                    img = img.numpy()
                processed_images.append(img)
            result["image"] = processed_images
        else:
            result["image"] = images
        
        # ── Language ──
        result["lang"] = sample.get("lang", "")
        
        # ── Actions ──
        action = sample.get("action", None)
        if action is not None:
            if isinstance(action, np.ndarray):
                action = action.astype(np.float32)
            elif isinstance(action, torch.Tensor):
                action = action.float().numpy()
            
            # Ensure shape is [action_horizon, action_dim]
            if action.ndim == 1:
                action = action.reshape(1, -1)
            
            # Pad/truncate to action_horizon
            if action.shape[0] < self.config.action_horizon:
                pad = np.zeros(
                    (self.config.action_horizon - action.shape[0], action.shape[1]),
                    dtype=np.float32
                )
                action = np.concatenate([action, pad], axis=0)
            elif action.shape[0] > self.config.action_horizon:
                action = action[:self.config.action_horizon]
                
            result["action"] = action
        
        # ── State ──
        if self.config.include_state and "state" in sample:
            state = sample["state"]
            if isinstance(state, np.ndarray):
                state = state.astype(np.float32)
            elif isinstance(state, torch.Tensor):
                state = state.float().numpy()
            result["state"] = state

        feedback_key = self.config.feedback_horizon_key
        if feedback_key in sample:
            result[feedback_key] = int(sample[feedback_key])

        for key in (
            "action_supervised",
            "cabi_tetrad_id",
            "cabi_corner",
            "cabi_transport_roles",
            "cabi_decision_point",
            "sample_id",
            "edge_id",
            "canonical_state_index",
            "camera_pose",
        ):
            if key in sample:
                result[key] = sample[key]
        for key in (
            "camera_intrinsics",
            "camera_to_world_opencv",
            "camera_intrinsics_by_view",
            "camera_to_world_opencv_by_view",
        ):
            if key in sample:
                result[key] = np.asarray(sample[key], dtype=np.float32)
        
        return result


class FreshSnapshotDataset:
    """Read pre-chunked counterfactual samples without inventing future frames."""

    def __init__(
        self,
        root: Path | str,
        *,
        split: str = "train",
        feedback_label: str = "oracle_feedback_horizon",
        feedback_output_key: str = "oracle_feedback_horizon",
        tasks: tuple[str, ...] = ("grasp_slip",),
    ):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        records_path = self.root / "records.jsonl"
        splits_path = self.root / "splits.json"
        labels_path = self.root / "training_labels.json"
        self.snapshots_path = self.root / "policy_observation_snapshots.npz"
        required = (manifest_path, records_path, splits_path, labels_path, self.snapshots_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete FRESH snapshot dataset; missing: {missing}")

        self.manifest = json.loads(manifest_path.read_text())
        split_map = json.loads(splits_path.read_text())["pair_splits"]
        labels = json.loads(labels_path.read_text())["records"]
        pair_tasks = {row["pair_id"]: row["task"] for row in self.manifest["pairs"]}
        allowed_tasks = set(tasks)
        rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]

        selected = []
        record_ids = set()
        for row in rows:
            pair_id = row["pair_id"]
            if split_map.get(pair_id) != split or pair_tasks.get(pair_id) not in allowed_tasks:
                continue
            record_id = f"{pair_id}::{row['branch_id']}"
            if record_id in record_ids:
                raise ValueError(f"duplicate snapshot record: {record_id}")
            record_ids.add(record_id)
            if record_id not in labels or feedback_label not in labels[record_id]:
                raise KeyError(f"missing label {feedback_label!r} for {record_id}")
            selected.append((row, int(labels[record_id][feedback_label]), pair_tasks[pair_id]))
        if not selected:
            raise ValueError(f"no FRESH snapshot records for split={split!r}, tasks={sorted(allowed_tasks)}")

        with np.load(self.snapshots_path, allow_pickle=False) as snapshots:
            snapshot_keys = set(snapshots.files)
        for row, _, _ in selected:
            snapshot_key = row["observation"]["snapshot_key"]
            expected = {f"{snapshot_key}_agentview", f"{snapshot_key}_wrist"}
            missing_keys = sorted(expected - snapshot_keys)
            if missing_keys:
                raise KeyError(f"missing image arrays for {row['pair_id']}: {missing_keys}")

        self.rows = selected
        self.feedback_output_key = feedback_output_key
        self._snapshots = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self._snapshots is None:
            self._snapshots = np.load(self.snapshots_path, allow_pickle=False)
        row, feedback_horizon, task = self.rows[idx]
        snapshot_key = row["observation"]["snapshot_key"]
        return {
            "image": [
                np.asarray(self._snapshots[f"{snapshot_key}_agentview"]),
                np.asarray(self._snapshots[f"{snapshot_key}_wrist"]),
            ],
            "fresh_sample_id": f"{row['pair_id']}::{row['branch_id']}",
            "pair_id": row["pair_id"],
            "branch_id": row["branch_id"],
            "branch_outcome": row.get("branch_outcome", row["branch_id"]),
            "task": task,
            "oracle_feedback_horizon": int(row["oracle_feedback_horizon"]),
            "lang": row["language_instruction"],
            "language": row["language_instruction"],
            "action": np.asarray(row["action_chunk"], dtype=np.float32),
            "state": np.asarray(row["robot_state"], dtype=np.float32),
            self.feedback_output_key: feedback_horizon,
        }


class FreshEpisodeWindowDataset:
    """Read full-episode sliding windows while keeping oracle labels loss-only."""

    def __init__(
        self,
        root: Path | str,
        *,
        split: str = "train",
        feedback_label: str = "oracle_feedback_horizon",
        feedback_output_key: str = "oracle_feedback_horizon",
        tasks: tuple[str, ...] = ("grasp_slip_full_episode",),
    ):
        self.root = Path(root)
        records_path = self.root / "records.jsonl"
        labels_path = self.root / "training_labels.json"
        missing = [str(path) for path in (records_path, labels_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete FRESH episode-window dataset; missing: {missing}")
        labels = json.loads(labels_path.read_text())["records"]
        allowed_tasks = set(tasks)
        rows = []
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != split or row["task"] not in allowed_tasks:
                continue
            sample_id = row["sample_id"]
            if sample_id not in labels or feedback_label not in labels[sample_id]:
                raise KeyError(f"missing label {feedback_label!r} for {sample_id}")
            rows.append((row, int(labels[sample_id][feedback_label])))
        if not rows:
            raise ValueError(f"no FRESH episode windows for split={split!r}, tasks={sorted(allowed_tasks)}")
        self.rows = rows
        self.feedback_output_key = feedback_output_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from PIL import Image

        row, feedback_horizon = self.rows[idx]
        observation = row["observation"]
        agent = np.asarray(Image.open(self.root / observation["agentview_path"]).convert("RGB"))
        wrist = np.asarray(Image.open(self.root / observation["wrist_path"]).convert("RGB"))
        return {
            "image": [agent, wrist],
            "fresh_sample_id": row["sample_id"],
            "pair_id": row["pair_id"],
            "branch_id": row["branch_id"],
            "branch_outcome": row["branch_outcome"],
            "frame_index": int(row["frame_index"]),
            "task": row["task"],
            "oracle_feedback_horizon": int(row["oracle_feedback_horizon"]),
            "lang": row["language_instruction"],
            "language": row["language_instruction"],
            "action": np.asarray(row["action_chunk"], dtype=np.float32),
            "state": np.asarray(row["robot_state"], dtype=np.float32),
            self.feedback_output_key: feedback_horizon,
        }


class LiberoBindTrainingDataset:
    """Mix full-trajectory BC windows with action-free fourth-corner tetrads."""

    def __init__(
        self,
        root: Path | str,
        *,
        split: str = "train",
        anchor_period: int = 1,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        records_path = self.root / "records.jsonl"
        anchors_path = self.root / "anchors.npz"
        missing = [str(path) for path in (manifest_path, records_path, anchors_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete LIBERO-Bind training view: {missing}")
        if anchor_period <= 0:
            raise ValueError("anchor_period must be positive")

        self.manifest = json.loads(manifest_path.read_text())
        self.source_collection = Path(self.manifest["source_collection"])
        self.action_horizon = int(self.manifest["action_horizon"])
        self.action_dim = int(self.manifest.get("action_dim", 7))
        self.anchor_period = int(anchor_period)
        self.edge_instructions = dict(self.manifest["edge_instructions"])
        self.records = [
            json.loads(line)
            for line in records_path.read_text().splitlines()
            if line.strip() and json.loads(line)["split"] == split
        ]
        self.tetrads = [row for row in self.manifest["tetrads"] if row["split"] == split]
        if not self.records:
            raise ValueError(f"no LIBERO-Bind action windows for split={split!r}")
        if not self.tetrads:
            raise ValueError(f"no LIBERO-Bind tetrads for split={split!r}")
        self.anchors_path = anchors_path
        self._anchors = None
        self._episode_path = None
        self._episode = None
        self._camera_view_path = None
        self._camera_view = None
        self.camera_training_view = self.manifest.get("camera_training_view")
        if self.camera_training_view is not None:
            self._validate_camera_training_records()

    def _validate_camera_training_records(self) -> None:
        """Fail before training if randomized images and calibration disagree."""

        rows_by_shard: dict[str, list[Mapping[str, Any]]] = {}
        for index, row in enumerate(self.records):
            required = {
                "camera_view_file",
                "camera_view_index",
                "camera_intrinsics",
                "camera_to_world_opencv",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"camera training record {index} is missing fields: {missing}"
                )
            intrinsics = np.asarray(row["camera_intrinsics"], dtype=np.float64)
            camera_to_world = np.asarray(
                row["camera_to_world_opencv"],
                dtype=np.float64,
            )
            if (
                intrinsics.shape != (3, 3)
                or camera_to_world.shape != (4, 4)
                or not np.all(np.isfinite(intrinsics))
                or not np.all(np.isfinite(camera_to_world))
            ):
                raise ValueError(
                    f"camera training record {index} has invalid calibration"
                )
            rotation = camera_to_world[:3, :3]
            if (
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
                or not np.allclose(
                    camera_to_world[3],
                    [0.0, 0.0, 0.0, 1.0],
                )
            ):
                raise ValueError(
                    f"camera training record {index} has an invalid rigid transform"
                )
            relative = Path(str(row["camera_view_file"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"camera training record {index} has an unsafe shard path"
                )
            rows_by_shard.setdefault(str(relative), []).append(row)

        for relative, rows in sorted(rows_by_shard.items()):
            path = self.root / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing camera training shard: {path}")
            with np.load(path, allow_pickle=False) as archive:
                required_arrays = {"agentview", "wrist", "robot_state"}
                if set(archive.files) != required_arrays:
                    raise ValueError(
                        f"camera training shard {path} has unexpected arrays: "
                        f"{sorted(archive.files)}"
                    )
                agentview = archive["agentview"]
                wrist = archive["wrist"]
                robot_state = archive["robot_state"]
                count = len(agentview)
                if (
                    agentview.shape != (count, 224, 224, 3)
                    or wrist.shape != (count, 224, 224, 3)
                    or robot_state.shape != (count, 8)
                    or agentview.dtype != np.uint8
                    or wrist.dtype != np.uint8
                    or not np.all(np.isfinite(robot_state))
                ):
                    raise ValueError(
                        f"camera training shard {path} has invalid array schema"
                    )
                for row in rows:
                    camera_index = int(row["camera_view_index"])
                    if not 0 <= camera_index < count:
                        raise IndexError(
                            f"camera index {camera_index} is outside "
                            f"{path} length {count}"
                        )
        logger.info(
            "Validated %d randomized-camera records across %d shards",
            len(self.records),
            len(rows_by_shard),
        )

    @staticmethod
    def _anchor_key(
        edge_id: str,
        state_index: int,
        field: str,
        decision_point: str | None = None,
    ) -> str:
        decision = "" if decision_point is None else f"{decision_point}__"
        return f"{edge_id}__state_{state_index:02d}__{decision}{field}"

    def _load_episode(self, relative_path: str) -> dict[str, np.ndarray]:
        path = self.source_collection / relative_path
        if path != self._episode_path:
            if self._episode is not None:
                self._episode.close()
            self._episode = np.load(path, allow_pickle=False)
            self._episode_path = path
        return self._episode

    def _load_camera_view(self, relative_path: str) -> dict[str, np.ndarray]:
        path = self.root / relative_path
        if path != self._camera_view_path:
            if self._camera_view is not None:
                self._camera_view.close()
            self._camera_view = np.load(path, allow_pickle=False)
            self._camera_view_path = path
        return self._camera_view

    @staticmethod
    def _camera_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "camera_pose",
            "camera_intrinsics",
            "camera_to_world_opencv",
            "camera_azimuth_deg",
            "camera_elevation_deg",
            "camera_radius_scale",
        )
        return {key: row[key] for key in keys if key in row}

    def _action_chunk(self, actions: np.ndarray, start: int) -> np.ndarray:
        chunk = np.asarray(actions[start : start + self.action_horizon], dtype=np.float32)
        if len(chunk) < self.action_horizon:
            chunk = np.concatenate(
                [
                    chunk,
                    np.zeros(
                        (self.action_horizon - len(chunk), actions.shape[1]),
                        dtype=np.float32,
                    ),
                ]
            )
        return chunk

    def _action_example(self, row: Mapping[str, Any]) -> dict:
        episode = self._load_episode(row["episode_file"])
        frame = int(row["frame_index"])
        agentview = np.asarray(episode["agentview"][frame])
        wrist = np.asarray(episode["wrist"][frame])
        state = np.asarray(episode["robot_state"][frame], dtype=np.float32)
        if "camera_view_file" in row:
            camera_view = self._load_camera_view(str(row["camera_view_file"]))
            camera_index = int(row["camera_view_index"])
            agentview = np.asarray(camera_view["agentview"][camera_index])
            wrist = np.asarray(camera_view["wrist"][camera_index])
            state = np.asarray(
                camera_view["robot_state"][camera_index],
                dtype=np.float32,
            )
        return {
            "image": [agentview, wrist],
            "lang": row["language_instruction"],
            "action": self._action_chunk(episode["actions"], frame),
            "state": state,
            "action_supervised": True,
            "sample_id": row["sample_id"],
            "edge_id": row["edge_id"],
            "canonical_state_index": int(row["canonical_state_index"]),
            **self._camera_metadata(row),
        }

    def _anchor_example(
        self,
        tetrad: Mapping[str, Any],
        corner_name: str,
        *,
        instance_id: str,
    ) -> dict:
        if self._anchors is None:
            self._anchors = np.load(self.anchors_path, allow_pickle=False)
        corner = tetrad["corners"][corner_name]
        physical_edge = corner["physical_edge"]
        instruction_edge = corner["instruction_edge"]
        state_index = int(tetrad["canonical_state_index"])
        decision_point = tetrad.get("decision_point")
        field = lambda name: self._anchors[
            self._anchor_key(
                physical_edge,
                state_index,
                name,
                None if decision_point is None else str(decision_point),
            )
        ]
        supervised = bool(corner["action_supervised"])
        action = (
            np.asarray(field("action"), dtype=np.float32)
            if supervised
            else np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        )
        example = {
            "image": [np.asarray(field("agentview")), np.asarray(field("wrist"))],
            "lang": self.edge_instructions[instruction_edge],
            "action": action,
            "state": np.asarray(field("state"), dtype=np.float32),
            "action_supervised": supervised,
            "cabi_tetrad_id": instance_id,
            "cabi_corner": corner_name,
            "sample_id": f"{instance_id}--{corner_name}",
            "edge_id": instruction_edge,
            "canonical_state_index": state_index,
            **(
                {"cabi_transport_roles": list(tetrad["transport_roles"])}
                if "transport_roles" in tetrad
                else {}
            ),
            **(
                {"cabi_decision_point": str(tetrad["decision_point"])}
                if "decision_point" in tetrad
                else {}
            ),
        }
        camera_training_view = getattr(self, "camera_training_view", None)
        if camera_training_view is not None:
            baseline = camera_training_view["baseline_camera"]
            example.update(
                {
                    "camera_pose": "baseline",
                    "camera_intrinsics": baseline["camera_intrinsics"],
                    "camera_to_world_opencv": baseline[
                        "camera_to_world_opencv"
                    ],
                    "camera_azimuth_deg": 0.0,
                    "camera_elevation_deg": 0.0,
                    "camera_radius_scale": 1.0,
                }
            )
        return example

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> list[dict]:
        bundle = [self._action_example(self.records[index])]
        if index % self.anchor_period == 0:
            tetrad = self.tetrads[(index // self.anchor_period) % len(self.tetrads)]
            instance_id = f"{tetrad['tetrad_id']}--item-{index:07d}"
            for corner in ("base", "source_anchor", "target_anchor", "fourth_anchor"):
                bundle.append(
                    self._anchor_example(tetrad, corner, instance_id=instance_id)
                )
        return bundle


def get_pi0_dataset(data_cfg, mode="train", **kwargs):
    """
    Get dataset for PaliGemmaOFT training.
    
    Reuses VLAE's existing LeRobot data loading, wrapping it with Pi0-specific transforms.
    
    Args:
        data_cfg: dataset config (same as used by other VLAE frameworks)
        mode: "train" or "eval"
        
    Returns:
        dataset wrapped with Pi0DataTransform
    """
    dataset_format = getattr(data_cfg, 'dataset_format', 'lerobot')
    action_horizon = getattr(data_cfg, 'action_horizon', 50)
    if dataset_format == 'libero_bind':
        base_dataset = LiberoBindTrainingDataset(
            getattr(data_cfg, 'data_root_dir'),
            split=getattr(data_cfg, 'split', mode),
            anchor_period=int(getattr(data_cfg, 'cabi_anchor_period', 1)),
        )
    elif dataset_format in {'fresh_snapshot', 'fresh_episode_window'}:
        configured_tasks = getattr(data_cfg, 'snapshot_tasks', ('grasp_slip',))
        dataset_class = FreshSnapshotDataset if dataset_format == 'fresh_snapshot' else FreshEpisodeWindowDataset
        base_dataset = dataset_class(
            getattr(data_cfg, 'data_root_dir'),
            split=getattr(data_cfg, 'split', mode),
            feedback_label=getattr(data_cfg, 'feedback_horizon_column', 'oracle_feedback_horizon'),
            feedback_output_key=getattr(data_cfg, 'feedback_horizon_key', 'oracle_feedback_horizon'),
            tasks=tuple(configured_tasks),
        )
    else:
        from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset
        from AlphaBrain.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP

        # Override action_indices in LIBERO data config if action_horizon > default
        libero_cfg = ROBOT_TYPE_CONFIG_MAP.get("libero_franka", None)
        if libero_cfg is not None:
            if action_horizon > 8:  # default LIBERO action_indices is range(8)
                libero_cfg.action_indices = list(range(action_horizon))
                logger.info(f"[pi0_data] Overriding action_indices to range({action_horizon})")

            # Skip data-level q99 normalization; Pi0 handles MEAN_STD in the model.
            skip_action_norm = getattr(data_cfg, 'skip_action_norm', True)
            if skip_action_norm:
                from AlphaBrain.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor
                from AlphaBrain.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform

                def raw_transform(self=libero_cfg):
                    transforms = [StateActionToTensor(apply_to=self.action_keys)]
                    return ComposedModalityTransform(transforms=transforms)

                libero_cfg.transform = raw_transform
                logger.info("[pi0_data] Disabled data-level action normalization (model handles MEAN_STD)")
        # Get the base LeRobot dataset
        base_dataset = get_vla_dataset(data_cfg, mode=mode, **kwargs)
    
    # Create Pi0 transform config from data_cfg
    pi0_config = Pi0DataConfig(
        action_horizon=getattr(data_cfg, 'action_horizon', 50),
        action_dim=getattr(data_cfg, 'action_dim', 7),
        include_state=getattr(data_cfg, 'include_state', True),
        state_dim=getattr(data_cfg, 'state_dim', 7),
        feedback_horizon_key=getattr(data_cfg, 'feedback_horizon_key', 'feedback_horizon'),
    )
    
    transform = Pi0DataTransform(config=pi0_config)
    
    # Wrap the dataset with Pi0 transforms
    return Pi0DatasetWrapper(base_dataset, transform)


class Pi0DatasetWrapper:
    """Wraps a LeRobot dataset with Pi0-specific transforms."""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, list):
            return [self.transform(value) for value in sample]
        return self.transform(sample)
    
    def __iter__(self):
        for sample in self.base_dataset:
            if isinstance(sample, list):
                yield [self.transform(value) for value in sample]
            else:
                yield self.transform(sample)


def pi0_collate_fn(batch):
    """Flatten optional CABI bundles while preserving normal Pi0 batches."""

    flattened = []
    for item in batch:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


# ── LIBERO-specific config ──

LIBERO_PI0_CONFIG = Pi0DataConfig(
    image_resolution=(224, 224),
    max_token_len=200,
    action_horizon=10,     # LIBERO uses shorter horizon
    action_dim=7,          # 6 DOF + gripper
    camera_names=("image_0",),
    include_state=True,
    state_dim=7,
)

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
from typing import Optional, Dict, List
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
    if dataset_format == 'fresh_snapshot':
        configured_tasks = getattr(data_cfg, 'snapshot_tasks', ('grasp_slip',))
        base_dataset = FreshSnapshotDataset(
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
        return self.transform(sample)
    
    def __iter__(self):
        for sample in self.base_dataset:
            yield self.transform(sample)


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

"""Optional PPO dependency boundary. Import only in the isolated RL environment.

No fitting or environment construction occurs at import. Actor and critic see
the same masked dictionary observations; endpoint labels are never accepted.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback


class SearchAuditCallback(BaseCallback):
    """Check every sampled action and reward; retain all completed search paths."""

    def __init__(self, output, initial_ids):
        super().__init__()
        self.output = Path(output)
        self.initial_ids = list(initial_ids)
        self.checked_actions = 0

    def _on_step(self):
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        masks = np.asarray(self.locals["action_masks"], dtype=bool)
        if not masks[np.arange(len(actions)), actions].all():
            raise ValueError("PPO sampled an invalid masked action")
        if not np.isfinite(self.locals["rewards"]).all():
            raise ValueError("Nonfinite reward")
        values = self.locals["values"].detach()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite critic prediction")
        self.checked_actions += len(actions)
        with (self.output / "episodes.jsonl").open("a") as stream:
            for slot, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
                if not done:
                    continue
                record = {key: info[key] for key in (
                    "queries", "scoring_members", "visited", "path", "selected",
                    "translation_m", "rotation_deg", "motion_units", "total_reward",
                    "stop_reason", "initial_accel", "best_accel")}
                record.update(step=int(self.model.num_timesteps), initial=int(self.initial_ids[slot]))
                self.model.audit_episodes.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return True


class AuditedPPO(MaskablePPO):
    """Standard MaskablePPO updates, with finite gradients and update-level logs."""

    def train(self):
        super().train()
        actor_norm = critic_norm = 0.
        for name, parameter in self.policy.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError("Nonfinite trained parameter: " + name)
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError("Nonfinite gradient: " + name)
                value = float(parameter.grad.square().sum())
                if "value_net" in name:
                    critic_norm += value
                else:
                    actor_norm += value
        record = {
            "steps": int(self.num_timesteps),
            "actor_gradient_norm": actor_norm ** .5,
            "critic_gradient_norm": critic_norm ** .5,
        }
        for key, value in self.logger.name_to_value.items():
            if key.startswith("train/") and isinstance(value, (int, float, np.number)):
                # explained_variance can be undefined for a constant return batch.
                record[key] = float(value) if np.isfinite(value) else None
        episodes = self.audit_episodes
        record["completed_episodes"] = len(episodes)
        for key in ("total_reward", "queries", "initial_accel", "best_accel", "motion_units"):
            record["episode_mean_" + key] = float(np.mean([e[key] for e in episodes])) if episodes else None
        self.audit_episodes = []
        with (Path(self.audit_output) / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"steps={self.num_timesteps} reward={record['episode_mean_total_reward']:.4f} "
              f"queries={record['episode_mean_queries']:.2f} "
              f"value_loss={record.get('train/value_loss')}", flush=True)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["audit_episodes", "audit_output"]


def build_model(env, seed, config, output):
    model = AuditedPPO(
        "MultiInputPolicy", env, seed=seed, device="cpu", verbose=0,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        **config,
    )
    model.audit_output = str(output)
    model.audit_episodes = []
    return model

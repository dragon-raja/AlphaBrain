"""Data-only GR1 trial contracts, checked before resource construction.

These plans bind scientific inputs, not permission. They neither register a
production worker nor infer defaults from a LIBERO campaign. Noise files are
hash-checked before loading and never use pickle.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import numpy as np

from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig
from AlphaBrain.research.dsol.execution.manifest import file_digest
from .model_runtime import ScoreContract
from .policy_binding import validate as validate_policy_binding


@dataclass(frozen=True)
class NoiseFile:
    path: str
    sha256: str

    def load(self, shape):
        if (
            not Path(self.path).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or file_digest(self.path) != self.sha256
        ):
            raise ValueError("Frozen absolute noise file identity required")
        # Read and hash the same bytes to avoid a check/load race on shared storage.
        import hashlib
        import io

        raw = Path(self.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("Noise file changed during preparation")
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("Frozen finite float32 noise dimensions differ")
        value.flags.writeable = False
        return value


@dataclass(frozen=True)
class TrialPlan:
    episode_id: str
    instruction: str
    search_config: dict
    score_noise: NoiseFile
    operation_noise: NoiseFile
    search_decisions: int
    search_seconds: float
    operation_actions: int
    operation_seconds: float
    score_scale: float
    seed: int
    settling_tolerance_deg: float | None

    def prepare(self, policy_binding, calls):
        binding = validate_policy_binding(policy_binding)
        score = ScoreContract(**binding["score"])
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+", self.episode_id)
            or self.episode_id in (".", "..")
            or not isinstance(self.instruction, str)
            or not self.instruction.strip()
            or type(self.seed) is not int
            or self.seed < 0
            or any(
                type(v) is not int or v <= 0
                for v in (self.search_decisions, self.operation_actions, calls)
            )
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in (self.search_seconds, self.operation_seconds, self.score_scale)
            )
        ):
            raise ValueError("Explicit bounded trial identity, clocks and scale required")
        # Every scientific parameter is supplied, even when it equals a library default.
        if set(self.search_config) != set(LocalViewConfig.__dataclass_fields__):
            raise ValueError("Complete search configuration required")
        config = LocalViewConfig(**self.search_config)
        tol = self.settling_tolerance_deg
        if tol is not None and (not math.isfinite(tol) or not 0 < tol < config.step_limit_deg / 4):
            raise ValueError("Explicit bounded measured-arrival tolerance required")
        chunks = math.ceil(self.operation_actions / score.execute_actions)
        upper = config.query_budget * config.score_samples + chunks
        if calls < upper:
            raise ValueError(
                "Reservation must cover search, return recheck and all operation replans"
            )
        if binding["task"] is not None:
            from AlphaBrain.research.dsol.data.task_condition import task_vector

            task_vector(binding["task"], self.instruction)
        return dict(
            config=config,
            score_noise=self.score_noise.load((config.score_samples, 16, 32)),
            operation_noise=self.operation_noise.load((chunks, 16, 32)),
            max_model_calls=upper,
            policy_binding=binding,
        )

    def request(self):
        return asdict(self)

    @classmethod
    def from_request(cls, request):
        values = dict(request)
        values["score_noise"] = NoiseFile(**values["score_noise"])
        values["operation_noise"] = NoiseFile(**values["operation_noise"])
        return cls(**values)

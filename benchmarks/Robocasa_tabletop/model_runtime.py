"""Local-only GR1 pi05 loader and native visual/velocity interface.

All optional model imports and checkpoint IO occur on explicit construction.
The caller owns real-time authorization and budget charging. CPU loading alone
does not admit infer(), numerical equivalence, rendering or operation rollout.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MethodType
import numpy as np

from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


@dataclass(frozen=True)
class ScoreContract:
    denoise_steps: int
    prefix: int
    execute_actions: int
    score_actions: int
    active_dim: int = 29
    generated_actions: int = 16
    model_dim: int = 32

    def validate(self):
        if (
            any(type(x) is not int for x in vars(self).values())
            or not 2 <= self.prefix <= self.denoise_steps
            or self.denoise_steps < 2
            or not 1 <= self.execute_actions <= self.generated_actions
            or self.score_actions != self.execute_actions
            or (self.active_dim, self.generated_actions, self.model_dim) != (29, 16, 32)
        ):
            raise ValueError(
                "Explicit GR1 denoise and matching score/execution action prefixes required"
            )

    def score(self, trace):
        self.validate()
        x = np.asarray(trace)
        if x.shape != (1, self.denoise_steps, 16, 32) or not np.isfinite(x).all():
            raise ValueError("Full native velocity trace required")
        result = compute_accel_scores(x[:, :, : self.score_actions, :29], prefixes=[self.prefix])[
            self.prefix
        ]
        if bool(result["degenerate"][0]):
            raise ValueError("Degenerate score is not a valid low-Accel candidate")
        return float(result["score"][0])


def _native_sample(model, rng, observation, *, noise, num_steps):
    """Tap the existing image pass; never run a second vision forward."""
    from flax import linen as nn
    import jax.numpy as jnp

    captured = []

    def intercept(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        if (
            context.method_name == "__call__"
            and isinstance(out, tuple)
            and len(out) == 2
            and isinstance(out[1], dict)
            and "pre_logits_2d" in out[1]
        ):
            captured.append(out[1]["pre_logits_2d"])
        return out

    with nn.intercept_methods(intercept):
        actions, trace = model.sample_actions_with_trace(
            rng, observation, noise=noise, num_steps=num_steps
        )
    names = list(observation.images)
    if len(captured) != len(names) or "base_0_rgb" not in names:
        raise ValueError("Native SigLIP tap count/camera order differs")
    grid = captured[names.index("base_0_rgb")]
    if grid.shape != (1, 16, 16, 1152):
        raise ValueError("Expected original pre-language 16x16x1152 ego grid")
    pooled = jnp.mean(grid.astype(jnp.float32).reshape(1, 2, 8, 2, 8, 1152), axis=(2, 4)).reshape(
        1, 4608
    )
    return actions, trace, pooled


class GR1ModelRuntime:
    def __init__(
        self,
        package,
        *,
        score_contract,
        call_guard,
        load_guard,
        expected_config_sha256,
        expected_norm_sha256,
        expected_tokenizer_sha256,
    ):
        score_contract.validate()
        load_guard()
        if not callable(call_guard):
            raise ValueError("Explicit per-model-call authority required")
        self.package = Path(package).resolve()
        self.contract = score_contract
        self.call_guard = call_guard
        config_path = self.package / "checkpoint/assets/_train_config_model.json"
        norm_path = (
            self.package / "checkpoint/assets/robocasa_gr1_tabletop24_global/norm_stats.json"
        )
        tokenizer_path = self.package / "openpi_cache/big_vision/paligemma_tokenizer.model"
        for path, stamp in [
            (config_path, expected_config_sha256),
            (norm_path, expected_norm_sha256),
            (tokenizer_path, expected_tokenizer_sha256),
        ]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != stamp:
                raise ValueError("Runtime input identity differs: " + str(path))
        from openpi.models import model as upstream, pi0_config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.gr1_tabletop_policy import Gr1TabletopInputs, Gr1TabletopOutputs
        from openpi.shared import normalize, nnx_utils
        from openpi import transforms
        import sentencepiece
        import jax
        import jax.numpy as jnp

        config = json.loads(config_path.read_text())
        if (
            config["pi05"] is not True
            or config["use_qha"] is not False
            or (config["action_horizon"], config["action_dim"]) != (16, 32)
        ):
            raise ValueError("Expected frozen plain GR1 pi05 checkpoint")
        self.config = pi0_config.Pi0Config(**config)

        # The original tokenizer algorithm with an explicitly local constructor.
        class LocalTokenizer(PaligemmaTokenizer):
            def __init__(self):
                self._max_len = config["max_token_len"]
                self._tokenizer = sentencepiece.SentencePieceProcessor(
                    model_file=str(tokenizer_path)
                )

        stats = normalize.load(norm_path.parent)
        self.input_transform = transforms.compose(
            [
                Gr1TabletopInputs(),
                transforms.Normalize(stats, use_quantiles=True),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    LocalTokenizer(), discrete_state_input=config["discrete_state_input"]
                ),
                transforms.PadStatesAndActions(32),
            ]
        )
        self.output_transform = transforms.compose(
            [transforms.Unnormalize(stats, use_quantiles=True), Gr1TabletopOutputs()]
        )
        params = upstream.restore_params(self.package / "checkpoint/params", dtype=jnp.bfloat16)
        # No silent dropping of extra checkpoint tensors.
        self.model = self.config.load(params, remove_extra_params=False)
        self.parameter_shapes = jax.tree.map(lambda x: list(x.shape), params)
        self.sample = nnx_utils.module_jit(
            MethodType(_native_sample, self.model), static_argnames=("num_steps",)
        )
        load_guard()

    def observation(self, example):
        """Preprocessing only. No model evaluation or mutable inference RNG."""
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation

        if not isinstance(example.get("prompt"), str) or not example["prompt"].strip():
            raise ValueError("Actual task instruction required")
        value = self.input_transform(jax.tree.map(lambda x: x, example))
        return Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None, ...], value))

    def infer(self, example, noise, *, request):
        """One authorized and charged scoring call, explicitly supplied noise."""
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production

        # GR1 is intentionally not yet on the production entry allowlist. CPU
        # loading/abstract shape tracing cannot accidentally admit inference.
        reject_production("benchmarks/Robocasa_tabletop/model_runtime.py:GR1ModelRuntime.infer")
        noise = np.asarray(noise)
        if noise.shape != (16, 32) or noise.dtype != np.float32 or not np.isfinite(noise).all():
            raise ValueError("Explicit finite float32 model noise required")
        if set(request) != {"query_id", "epoch", "member"}:
            raise ValueError("Explicit query, cache epoch and paired noise member required")
        self.call_guard(request)  # Charge/GO before VLA execution; failure remains unknown/spent.
        import jax
        import jax.numpy as jnp

        obs = self.observation(example)
        actions, trace, features = self.sample(
            jax.random.key(0),
            obs,
            noise=jnp.asarray(noise)[None],
            num_steps=self.contract.denoise_steps,
        )
        actions, trace, features = jax.device_get((actions, trace, features))
        raw = self.output_transform({"actions": actions[0]})["actions"]
        if raw.shape != (16, 29) or not np.isfinite(raw).all():
            raise ValueError("Invalid GR1 action output")
        score = self.contract.score(trace["v"])
        return dict(
            actions=raw[: self.contract.execute_actions].copy(),
            full_actions=raw.copy(),
            trace=np.asarray(trace["v"]),
            deltas=np.asarray(trace["ds"]),
            feature=np.asarray(features[0], dtype=np.float32),
            score=score,
            request=dict(request),
        )

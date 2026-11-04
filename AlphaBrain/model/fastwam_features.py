"""Opt-in eager FastWAM feature capture; no model loading or extra forward.

The caller must own a serialized inference request. Compiled/concurrent model
use is not supported. This captures prepare's pre-transformer joint image tokens,
not independent camera embeddings or generated/future frames. Actual upstream
GPU equivalence and the model/preprocessing identity require separate admission.
"""
import torch


class FastWAMFeatureCapture:
    """Temporarily observe video_expert.prepare for exactly one query."""

    _marker = "_dsol_native_prepare_capture"

    def __init__(self, video_expert, *, query_id, token_shape):
        if (not isinstance(query_id, str) or not query_id.strip()
                or not isinstance(token_shape, tuple) or len(token_shape) != 3
                or token_shape[0] != 1
                or any(type(n) is not int or n <= 0 for n in token_shape)):
            raise ValueError("Explicit query and batch-one token shape required")
        self.expert = video_expert
        self.query_id, self.token_shape = query_id, token_shape
        self._active = self._used = self._failed = False
        self._calls, self._tokens = 0, None

    def __enter__(self):
        if self._used or hasattr(self.expert, self._marker):
            raise RuntimeError("Single-use serialized capture required")
        original = self.expert.prepare
        if not callable(original):
            raise ValueError("Expected callable video_expert.prepare")
        self._had_override = "prepare" in vars(self.expert)
        self._override = vars(self.expert).get("prepare")

        def observe(*args, **kwargs):
            self._calls += 1
            try:
                x, timestep = kwargs.get("x"), kwargs.get("timestep")
                if (self._calls != 1 or args or not isinstance(x, torch.Tensor)
                        or x.ndim != 5 or x.shape[0] != 1 or x.shape[2] != 1
                        or not isinstance(timestep, torch.Tensor)
                        or tuple(timestep.shape) != (1,)
                        or not torch.isfinite(timestep).all() or torch.any(timestep != 0)
                        or "action" not in kwargs or kwargs["action"] is not None):
                    raise ValueError("One real first-frame, zero-time prepare call required")
                output = original(*args, **kwargs)
                if not isinstance(output, tuple) or len(output) != 10:
                    raise ValueError("Unexpected FastWAM prepare contract")
                tokens = output[0]
                if (not isinstance(tokens, torch.Tensor)
                        or tuple(tokens.shape) != self.token_shape
                        or tokens.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                        or not torch.isfinite(tokens).all()):
                    raise ValueError("Invalid native token shape, dtype or values")
                self._tokens = tokens.detach().clone()
                return output  # Preserve original output identity, dtype and graph.
            except BaseException:
                self._failed, self._tokens = True, None
                raise

        setattr(self.expert, self._marker, True)
        self.expert.prepare = observe
        self._active = self._used = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._had_override:
            self.expert.prepare = self._override
        else:
            delattr(self.expert, "prepare")
        delattr(self.expert, self._marker)
        self._active = False
        if exc_type is not None:
            self._failed, self._tokens = True, None
        return False

    def features(self, *, query_id):
        if self._active or not self._used:
            raise RuntimeError("Read only after the inference request completes")
        if self._failed or self._calls != 1 or self._tokens is None or query_id != self.query_id:
            raise ValueError("Missing, failed or different-query native features")
        return self._tokens.clone()

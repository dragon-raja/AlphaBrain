"""GR1 bindings for the shared configurable Actor/Critic and collector.

This module does not select a winning structure, load an operation model or
authorize dispatch. All core variants and optional text use the same shared
implementation. A checkpoint envelope binds semantic provenance, not just size.
"""

from copy import deepcopy
from dataclasses import asdict
import re
import json

from AlphaBrain.research.dsol.data.search_core import validate_core_spec
from AlphaBrain.research.dsol.data.task_condition import validate_binding, task_vector
from AlphaBrain.research.dsol.data.native_features import layout_spec, read_layout
from AlphaBrain.research.dsol.execution.manifest import digest
from .model_runtime import ScoreContract
from .search_runtime import STREAMS, collector_session


PROVENANCE = {
    "model_sha256",
    "producer_sha256",
    "history_sha256",
    "processor_sha256",
    "normalization_sha256",
    "source_sha256",
}


def make_binding(core, provenance, score, *, task=None):
    result = dict(
        schema="GR1_SHARED_POLICY_BINDING_V1",
        core=deepcopy(core),
        provenance=deepcopy(provenance),
        score=asdict(score),
        layout=layout_spec(STREAMS),
        projection_groups=["ego"],
        task=deepcopy(task),
    )
    return validate(result)


def validate(binding):
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {"schema", "core", "provenance", "score", "layout", "projection_groups", "task"}
        or binding["schema"] != "GR1_SHARED_POLICY_BINDING_V1"
        or read_layout(binding["layout"]) != STREAMS
        or binding["projection_groups"] != ["ego"]
    ):
        raise ValueError("Explicit single-ego shared policy binding required")
    validate_core_spec(binding["core"])
    p = binding["provenance"]
    if (
        not isinstance(p, dict)
        or set(p) != PROVENANCE
        or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in p.values())
    ):
        raise ValueError("Complete model/processor/source provenance required")
    ScoreContract(**binding["score"]).validate()
    if binding["task"] is not None:
        validate_binding(binding["task"])
    return json.loads(json.dumps(binding))


def build_policy(binding):
    from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic

    b = validate(binding)
    return UnifiedActorCritic(
        b["core"],
        streams=STREAMS,
        groups=b["projection_groups"],
        producer_sha256=b["provenance"]["producer_sha256"],
        task_spec=None if b["task"] is None else b["task"]["spec"],
    )


def validate_policy(policy, binding):
    b = validate(binding)
    expected = dict(
        core=b["core"],
        visual_dim=None,
        streams=[asdict(s) for s in STREAMS],
        projection_groups=b["projection_groups"],
        producer_sha256=b["provenance"]["producer_sha256"],
    )
    if b["task"] is not None:
        expected["task_spec"] = b["task"]["spec"]
    if policy.identity != expected:
        raise ValueError("Policy computation differs from binding")
    return b


def checkpoint_envelope(policy, binding):
    """In-memory envelope; the existing store owns atomic durable publication."""
    b = validate_policy(policy, binding)
    return dict(binding=b, binding_sha256=digest(b), state_dict=deepcopy(policy.state_dict()))


def load_policy(envelope, expected_binding):
    b = validate(expected_binding)
    if (
        set(envelope) != {"binding", "binding_sha256", "state_dict"}
        or envelope["binding"] != b
        or envelope["binding_sha256"] != digest(b)
    ):
        raise ValueError("Checkpoint semantic provenance differs before loading weights")
    policy = build_policy(b)
    policy.load_state_dict(envelope["state_dict"], strict=True)
    return policy


def bind_collector(source, binding, *, instruction, **session_options):
    """Validate before the first charged query; optionally add real cached text."""
    b = validate(binding)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Actual model input instruction required")
    if source.queries or source.status != "READY":
        raise ValueError("Binding must precede episode acquisition")
    for key in ("model_sha256", "producer_sha256", "history_sha256"):
        if getattr(source, key) != b["provenance"][key]:
            raise ValueError("Collector provenance differs from policy binding")
    if not hasattr(source.runtime, "contract") or asdict(source.runtime.contract) != b["score"]:
        raise ValueError("Collector scoring/action contract differs")
    # Resolve the exact cache entry before any motion or model call. No zero/text-ID fallback.
    vector = None if b["task"] is None else task_vector(b["task"], instruction)
    source.expected_instruction = instruction
    env, options = collector_session(source, **session_options)
    if vector is not None:
        from AlphaBrain.research.dsol.acquisition.task_condition import TaskConditionedEnv

        env = TaskConditionedEnv(env, vector)
    return env, options

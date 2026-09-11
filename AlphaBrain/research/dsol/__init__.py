"""DSOL modules with lazy, backward-compatible protocol exports.

The frozen Python 3.8 simulator imports leaf data/metric modules only. It must
not eagerly load modern protocol dataclasses or unrelated analysis machinery.
The protocol API itself retains its existing interpreter requirements.
"""

from importlib import import_module

__all__ = [
    "ED_EVIDENCE_GATE_NAMES",
    "FROZEN_FORBIDDEN_DEPLOYMENT_KEYS",
    "HOLD",
    "PASS",
    "TEMPORAL_STRATA",
    "TEMPORAL_TESTS",
    "ObservationContract",
    "Preregistration",
    "ProtocolValidationError",
    "RelationRecord",
    "RelationType",
    "TaskTemplate",
    "TemporalGate",
    "evaluate_release",
    "validate_deployment_inputs",
    "validate_observation_contract",
    "validate_preregistration",
    "validate_relation_record",
    "validate_task_template",
    "validate_temporal_gate",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module('.protocol', __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

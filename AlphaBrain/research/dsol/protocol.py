"""Fail-closed protocol records for the DSOL Paper 1 release process.

The records in this module are declarations, not trusted evidence.  Call
``validate`` before using one, or pass a complete :class:`Preregistration` to
:func:`evaluate_release`.  Validation never fills in missing evidence.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any


PASS = "PASS"
HOLD = "HOLD"
_PROTOCOL_GATE_NAMES = (
    "task_templates",
    "observation_contracts",
    "relations",
    "temporal_gates",
    "preregistration",
)


class ProtocolValidationError(ValueError):
    """Raised when a Paper 1 protocol declaration is incomplete or unsafe."""


class RelationType(str, Enum):
    """Relation classes admitted by the Paper 1 protocol."""

    N = "N"
    E_D = "E_D"
    MATCHED_CONTROL = "MATCHED_CONTROL"


TEMPORAL_TESTS = frozenset({"T1", "T2", "T3"})
TEMPORAL_STRATA = frozenset({"S0", "S1", "S2"})
ED_EVIDENCE_GATE_NAMES = frozenset(
    {
        "current_full_observation_ambiguous",
        "latent_decision_sets_incompatible",
        "no_shared_safe_recovery_action",
        "candidate_snapshot_disambiguates",
        "matched_uninformative_control_exists",
    }
)

# This list is part of the preregistered observation boundary.  Removing an
# entry is a protocol change, not a runtime configuration choice.
FROZEN_FORBIDDEN_DEPLOYMENT_KEYS = frozenset(
    {
        "branch_outcome",
        "branch_outcomes",
        "camera_label",
        "camera_labels",
        "file_name",
        "file_names",
        "filename",
        "filenames",
        "ground_truth_relation",
        "object_id",
        "object_ids",
        "oracle_relation",
        "oracle_relations",
        "oracle_route",
        "privileged_state",
        "candidate_rank",
        "future_state",
        "r_hat_target",
        "relation_oracle",
        "scan_order",
        "timestamp",
        "timestamps",
    }
)


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolValidationError(f"{field_name} must be an explicit boolean")
    return value


def _normalise_key(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_").lower()


def _forbidden_key(key: str) -> str | None:
    normalised = _normalise_key(key)
    for forbidden in FROZEN_FORBIDDEN_DEPLOYMENT_KEYS:
        if normalised == forbidden or normalised.startswith(forbidden + "_"):
            return forbidden
        if normalised.endswith("_" + forbidden) or f"_{forbidden}_" in normalised:
            return forbidden
    return None


def _scan_deployment_mapping(value: Mapping[object, Any], path: str) -> None:
    for raw_key, child in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ProtocolValidationError(f"{path} contains a non-string or empty key")
        forbidden = _forbidden_key(raw_key)
        child_path = f"{path}.{raw_key}"
        if forbidden is not None:
            raise ProtocolValidationError(
                f"deployment input {child_path!r} uses frozen forbidden key {forbidden!r}"
            )
        if isinstance(child, Mapping):
            _scan_deployment_mapping(child, child_path)
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _scan_deployment_mapping(item, f"{child_path}[{index}]")


def validate_deployment_inputs(deployment_inputs: object) -> None:
    """Reject forbidden deployment features, including nested schema keys."""

    if isinstance(deployment_inputs, Mapping):
        if not deployment_inputs:
            raise ProtocolValidationError("deployment_inputs must not be empty")
        _scan_deployment_mapping(deployment_inputs, "deployment_inputs")
        return

    if isinstance(deployment_inputs, Collection) and not isinstance(
        deployment_inputs, (str, bytes, bytearray)
    ):
        if not deployment_inputs:
            raise ProtocolValidationError("deployment_inputs must not be empty")
        seen: set[str] = set()
        for raw_key in deployment_inputs:
            key = _require_nonempty_string(raw_key, "deployment input key")
            normalised = _normalise_key(key)
            if normalised in seen:
                raise ProtocolValidationError(f"duplicate deployment input key {key!r}")
            seen.add(normalised)
            forbidden = _forbidden_key(key)
            if forbidden is not None:
                raise ProtocolValidationError(
                    f"deployment input {key!r} uses frozen forbidden key {forbidden!r}"
                )
        return

    raise ProtocolValidationError("deployment_inputs must be a mapping or collection of keys")


def _string_collection(value: object, field_name: str, *, allow_empty: bool = False) -> frozenset[str]:
    if not isinstance(value, Collection) or isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ProtocolValidationError(f"{field_name} must be a collection of strings")
    strings: list[str] = []
    for item in value:
        strings.append(_require_nonempty_string(item, field_name))
    if not strings and not allow_empty:
        raise ProtocolValidationError(f"{field_name} must not be empty")
    if len(strings) != len(set(strings)):
        raise ProtocolValidationError(f"{field_name} must not contain duplicates")
    return frozenset(strings)


@dataclass(frozen=True, slots=True)
class ObservationContract:
    """The complete feature-name boundary visible to a deployed policy."""

    contract_id: str
    deployment_inputs: Mapping[str, Any] | Collection[str]

    def validate(self) -> ObservationContract:
        _require_nonempty_string(self.contract_id, "contract_id")
        validate_deployment_inputs(self.deployment_inputs)
        return self


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    """A preregistered task family and its admissible action vocabulary."""

    template_id: str
    instruction: str
    action_set: Collection[str]
    observation_contract_id: str

    def validate(self) -> TaskTemplate:
        _require_nonempty_string(self.template_id, "template_id")
        _require_nonempty_string(self.instruction, "instruction")
        _string_collection(self.action_set, "action_set")
        _require_nonempty_string(self.observation_contract_id, "observation_contract_id")
        return self


@dataclass(frozen=True, slots=True)
class RelationRecord:
    """Evidence declaration for an N, E_D, or matched-control relation."""

    relation_id: str
    relation_type: RelationType | str
    source_task_id: str
    target_task_id: str
    action_set_equivalent: bool | None = None
    action_set_equivariant: bool | None = None
    r_hat: float | None = None
    r_hat_threshold: float = 0.1
    evidence_gates: Mapping[str, bool] | None = None
    matched_control_id: str | None = None
    expert_decision_sets: Sequence[Collection[str]] | None = None
    shared_safe_recovery_actions: Collection[str] | None = None

    @property
    def kind(self) -> RelationType:
        try:
            return RelationType(self.relation_type)
        except (TypeError, ValueError) as error:
            raise ProtocolValidationError(
                "relation_type must be N, E_D, or MATCHED_CONTROL"
            ) from error

    def validate(self) -> RelationRecord:
        _require_nonempty_string(self.relation_id, "relation_id")
        _require_nonempty_string(self.source_task_id, "source_task_id")
        _require_nonempty_string(self.target_task_id, "target_task_id")

        if self.kind is RelationType.N:
            self._validate_n_relation()
        elif self.kind is RelationType.E_D:
            self._validate_ed_relation()
        else:
            self._validate_matched_control()
        return self

    def _validate_n_relation(self) -> None:
        equivalent = _require_exact_bool(self.action_set_equivalent, "action_set_equivalent")
        equivariant = _require_exact_bool(self.action_set_equivariant, "action_set_equivariant")
        if not (equivalent or equivariant):
            raise ProtocolValidationError(
                "N requires action-set equivalence or equivariance"
            )
        if isinstance(self.r_hat, bool) or not isinstance(self.r_hat, (int, float)):
            raise ProtocolValidationError("N requires a numeric r_hat")
        if isinstance(self.r_hat_threshold, bool) or not isinstance(
            self.r_hat_threshold, (int, float)
        ):
            raise ProtocolValidationError("r_hat_threshold must be numeric")
        r_hat = float(self.r_hat)
        threshold = float(self.r_hat_threshold)
        if not math.isfinite(r_hat) or r_hat < 0:
            raise ProtocolValidationError("r_hat must be finite and non-negative")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ProtocolValidationError("r_hat_threshold must be finite and positive")
        if r_hat > threshold:
            raise ProtocolValidationError(
                f"N requires low r_hat (got {r_hat}, threshold {threshold})"
            )

    def _validate_ed_relation(self) -> None:
        if not isinstance(self.evidence_gates, Mapping):
            raise ProtocolValidationError("E_D requires exactly five named evidence gates")
        actual_gate_names = frozenset(self.evidence_gates)
        if actual_gate_names != ED_EVIDENCE_GATE_NAMES:
            missing = sorted(ED_EVIDENCE_GATE_NAMES - actual_gate_names)
            extra = sorted(actual_gate_names - ED_EVIDENCE_GATE_NAMES)
            raise ProtocolValidationError(
                "E_D evidence gates must match the frozen five-gate contract; "
                f"missing={missing}, extra={extra}"
            )
        for gate_name, passed in self.evidence_gates.items():
            _require_nonempty_string(gate_name, "E_D evidence gate name")
            if type(passed) is not bool:
                raise ProtocolValidationError(
                    f"E_D evidence gate {gate_name!r} must be an explicit boolean"
                )
            if not passed:
                raise ProtocolValidationError(f"E_D evidence gate {gate_name!r} did not pass")

        _require_nonempty_string(self.matched_control_id, "matched_control_id")
        if not isinstance(self.expert_decision_sets, Sequence) or isinstance(
            self.expert_decision_sets, (str, bytes, bytearray)
        ):
            raise ProtocolValidationError("E_D requires expert_decision_sets")
        if len(self.expert_decision_sets) < 2:
            raise ProtocolValidationError("E_D requires at least two expert decision sets")

        decisions = [
            _string_collection(decision_set, f"expert_decision_sets[{index}]")
            for index, decision_set in enumerate(self.expert_decision_sets)
        ]
        for left_index, left in enumerate(decisions):
            for right_index in range(left_index + 1, len(decisions)):
                overlap = left.intersection(decisions[right_index])
                if overlap:
                    raise ProtocolValidationError(
                        "E_D expert decision sets must be incompatible; shared decisions: "
                        + ", ".join(sorted(overlap))
                    )

        if self.shared_safe_recovery_actions is None:
            raise ProtocolValidationError(
                "E_D requires an explicit shared_safe_recovery_actions audit"
            )
        recovery = _string_collection(
            self.shared_safe_recovery_actions,
            "shared_safe_recovery_actions",
            allow_empty=True,
        )
        if recovery:
            raise ProtocolValidationError("E_D permits no shared safe recovery action")

    def _validate_matched_control(self) -> None:
        if self.matched_control_id is not None:
            raise ProtocolValidationError(
                "MATCHED_CONTROL must not itself reference a matched_control_id"
            )


@dataclass(frozen=True, slots=True)
class TemporalGate:
    """A preregistered temporal test and its analysis stratum."""

    test_id: str
    stratum: str
    ordered_clip_advantage: bool
    include_in_main_analysis: bool

    @property
    def eligible_for_main_analysis(self) -> bool:
        return (
            self.test_id in TEMPORAL_TESTS
            and self.stratum == "S0"
            and self.ordered_clip_advantage is False
        )

    def validate(self) -> TemporalGate:
        if self.test_id not in TEMPORAL_TESTS:
            raise ProtocolValidationError("test_id must be T1, T2, or T3")
        if self.stratum not in TEMPORAL_STRATA:
            raise ProtocolValidationError("stratum must be S0, S1, or S2")
        advantage = _require_exact_bool(
            self.ordered_clip_advantage, "ordered_clip_advantage"
        )
        included = _require_exact_bool(
            self.include_in_main_analysis, "include_in_main_analysis"
        )
        if advantage and self.stratum != "S1":
            raise ProtocolValidationError("ordered-clip advantage must be classified S1")
        if included and not self.eligible_for_main_analysis:
            raise ProtocolValidationError("only S0 may enter the main analysis")
        return self


@dataclass(frozen=True, slots=True)
class Preregistration:
    """The complete Paper 1 declaration evaluated before release."""

    preregistration_id: str
    task_templates: Sequence[TaskTemplate]
    observation_contracts: Sequence[ObservationContract]
    relations: Sequence[RelationRecord]
    temporal_gates: Sequence[TemporalGate]
    release_gates: Mapping[str, Collection[str]] = field(default_factory=dict)

    def validate(self) -> Preregistration:
        _require_nonempty_string(self.preregistration_id, "preregistration_id")
        templates = self._validate_records(
            self.task_templates, TaskTemplate, "task_templates", "template_id"
        )
        contracts = self._validate_records(
            self.observation_contracts,
            ObservationContract,
            "observation_contracts",
            "contract_id",
        )
        relations = self._validate_records(
            self.relations, RelationRecord, "relations", "relation_id"
        )
        temporal = self._validate_records(
            self.temporal_gates, TemporalGate, "temporal_gates", "test_id"
        )

        contract_ids = {contract.contract_id for contract in contracts}
        for template in templates:
            if template.observation_contract_id not in contract_ids:
                raise ProtocolValidationError(
                    f"task {template.template_id!r} references unknown observation contract"
                )

        task_ids = {template.template_id for template in templates}
        for relation in relations:
            if relation.source_task_id not in task_ids or relation.target_task_id not in task_ids:
                raise ProtocolValidationError(
                    f"relation {relation.relation_id!r} references an unknown task"
                )

        relation_by_id = {relation.relation_id: relation for relation in relations}
        relation_kinds = {relation.kind for relation in relations}
        if relation_kinds != set(RelationType):
            missing = sorted(kind.value for kind in set(RelationType) - relation_kinds)
            raise ProtocolValidationError(
                "preregistration must contain N, E_D, and MATCHED_CONTROL relations; "
                f"missing {missing}"
            )
        for relation in relations:
            if relation.kind is RelationType.E_D:
                control = relation_by_id.get(relation.matched_control_id or "")
                if control is None or control.kind is not RelationType.MATCHED_CONTROL:
                    raise ProtocolValidationError(
                        f"E_D relation {relation.relation_id!r} must reference a registered "
                        "MATCHED_CONTROL"
                    )

        if {gate.test_id for gate in temporal} != TEMPORAL_TESTS:
            raise ProtocolValidationError("temporal_gates must contain T1, T2, and T3 exactly once")

        if not isinstance(self.release_gates, Mapping) or not self.release_gates:
            raise ProtocolValidationError("release_gates must be a non-empty mapping")
        for gate_name, required_artifacts in self.release_gates.items():
            gate_name = _require_nonempty_string(gate_name, "release gate name")
            if gate_name in _PROTOCOL_GATE_NAMES:
                raise ProtocolValidationError(
                    f"release gate name {gate_name!r} is reserved by the protocol"
                )
            _string_collection(required_artifacts, f"release_gates[{gate_name!r}]")
        return self

    @staticmethod
    def _validate_records(
        records: object,
        expected_type: type,
        field_name: str,
        identifier_name: str,
    ) -> tuple[Any, ...]:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
            raise ProtocolValidationError(f"{field_name} must be a sequence")
        if not records:
            raise ProtocolValidationError(f"{field_name} must not be empty")
        identifiers: set[str] = set()
        validated: list[Any] = []
        for index, record in enumerate(records):
            if not isinstance(record, expected_type):
                raise ProtocolValidationError(
                    f"{field_name}[{index}] must be {expected_type.__name__}"
                )
            record.validate()
            identifier = getattr(record, identifier_name)
            if identifier in identifiers:
                raise ProtocolValidationError(
                    f"{field_name} contains duplicate identifier {identifier!r}"
                )
            identifiers.add(identifier)
            validated.append(record)
        return tuple(validated)

    def main_analysis_temporal_gates(self) -> tuple[TemporalGate, ...]:
        """Return validated, explicitly included S0 temporal gates only."""

        self.validate()
        return tuple(
            gate
            for gate in self.temporal_gates
            if gate.include_in_main_analysis and gate.eligible_for_main_analysis
        )


def _section_passes(records: object, expected_type: type) -> bool:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return False
    if not records:
        return False
    try:
        for record in records:
            if not isinstance(record, expected_type):
                return False
            record.validate()
    except Exception:
        return False
    return True


def _artifact_passes(value: object) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().upper() == PASS
    if isinstance(value, Mapping):
        if "status" in value:
            status = value["status"]
            return isinstance(status, str) and status.strip().upper() == PASS
        if "valid" in value:
            return type(value["valid"]) is bool and value["valid"]
    return False


def evaluate_release(prereg: object, artifacts: object) -> dict[str, str]:
    """Evaluate protocol and artifact gates without ever defaulting to PASS.

    Artifact evidence is explicit: a value must be ``True``, ``"PASS"``, a
    mapping with ``{"status": "PASS"}``, or a mapping with ``{"valid": True}``.
    Missing keys and unrecognised evidence formats produce ``HOLD``.
    """

    decisions = {gate_name: HOLD for gate_name in _PROTOCOL_GATE_NAMES}
    if not isinstance(prereg, Preregistration):
        return decisions

    decisions["task_templates"] = (
        PASS if _section_passes(prereg.task_templates, TaskTemplate) else HOLD
    )
    decisions["observation_contracts"] = (
        PASS
        if _section_passes(prereg.observation_contracts, ObservationContract)
        else HOLD
    )
    decisions["relations"] = (
        PASS if _section_passes(prereg.relations, RelationRecord) else HOLD
    )
    decisions["temporal_gates"] = (
        PASS if _section_passes(prereg.temporal_gates, TemporalGate) else HOLD
    )

    prereg_valid = False
    try:
        prereg.validate()
    except Exception:
        for gate_name in _PROTOCOL_GATE_NAMES:
            decisions[gate_name] = HOLD
    else:
        prereg_valid = True
        for gate_name in _PROTOCOL_GATE_NAMES:
            decisions[gate_name] = PASS

    release_gates = prereg.release_gates
    if not isinstance(release_gates, Mapping):
        return decisions
    artifact_mapping = artifacts if isinstance(artifacts, Mapping) else {}
    for raw_gate_name, required_artifacts in release_gates.items():
        gate_name = str(raw_gate_name)
        decisions[gate_name] = HOLD
        if not prereg_valid:
            continue
        try:
            requirements = _string_collection(
                required_artifacts, f"release_gates[{gate_name!r}]"
            )
        except (ProtocolValidationError, TypeError, ValueError):
            continue
        if all(
            artifact_name in artifact_mapping
            and _artifact_passes(artifact_mapping[artifact_name])
            for artifact_name in requirements
        ):
            decisions[gate_name] = PASS
    return decisions


def validate_task_template(value: TaskTemplate) -> TaskTemplate:
    if not isinstance(value, TaskTemplate):
        raise ProtocolValidationError("value must be a TaskTemplate")
    return value.validate()


def validate_observation_contract(value: ObservationContract) -> ObservationContract:
    if not isinstance(value, ObservationContract):
        raise ProtocolValidationError("value must be an ObservationContract")
    return value.validate()


def validate_relation_record(value: RelationRecord) -> RelationRecord:
    if not isinstance(value, RelationRecord):
        raise ProtocolValidationError("value must be a RelationRecord")
    return value.validate()


def validate_temporal_gate(value: TemporalGate) -> TemporalGate:
    if not isinstance(value, TemporalGate):
        raise ProtocolValidationError("value must be a TemporalGate")
    return value.validate()


def validate_preregistration(value: Preregistration) -> Preregistration:
    if not isinstance(value, Preregistration):
        raise ProtocolValidationError("value must be a Preregistration")
    return value.validate()


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

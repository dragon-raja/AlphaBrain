"""CPU metadata contracts for native search observations (not runtime admission).

No model loading, tensor conversion or legacy-policy changes. Image/tensor bytes,
geometry, and capture-on/off equivalence remain the runtime adapter's obligation.
Times use one explicitly named, adapter-normalized clock, not raw mixed clocks.
"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class NativeStream:
    """One truthful feature stream; joint streams keep their camera membership."""
    name: str
    channels: int
    fusion_scope: str
    camera_ids: tuple[str,...]
    camera_roles: tuple[str,...]

    def validate(self):
        if (not _name(self.name) or type(self.channels) is not int or self.channels<1
                or self.fusion_scope not in ('per_view','joint')
                or not isinstance(self.camera_ids,tuple) or not self.camera_ids
                or any(not _name(k) for k in self.camera_ids)
                or len(set(self.camera_ids))!=len(self.camera_ids)
                or not isinstance(self.camera_roles,tuple)
                or len(self.camera_roles)!=len(self.camera_ids)
                or any(r not in ('movable','context') for r in self.camera_roles)
                or (self.fusion_scope=='per_view' and len(self.camera_ids)!=1)):
            raise ValueError('Explicit native channels and real camera mapping required')


def validate_streams(streams):
    if not isinstance(streams,tuple) or not streams:raise ValueError('Explicit stream tuple required')
    for stream in streams:stream.validate()
    cameras=[k for s in streams for k in s.camera_ids]
    if len({s.name for s in streams})!=len(streams):raise ValueError('Duplicate stream')
    if len(set(cameras))!=len(cameras):raise ValueError('A camera cannot be counted twice across feature streams')
    if not any('movable' in s.camera_roles for s in streams):raise ValueError('Movable camera required')


def _name(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _time(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    role: str
    valid: bool
    captured_at: float | None
    image_sha256: str | None


@dataclass(frozen=True)
class ObservationBundle:
    query_id: str
    clock_id: str
    settled_at: float
    observed_at: float
    max_age: float
    max_skew: float
    preprocessing_sha256: str
    history_sha256: str
    cameras: tuple[CameraFrame, ...]

    def validate(self):
        if not all(_name(x) for x in (self.query_id, self.clock_id)):
            raise ValueError("Explicit query and common clock identities required")
        if not all(_digest(x) for x in (self.preprocessing_sha256, self.history_sha256)):
            raise ValueError("Preprocessing and causal history hashes required")
        if (not all(_time(x) for x in (self.settled_at, self.observed_at,
                                       self.max_age, self.max_skew))
                or min(self.max_age, self.max_skew) < 0
                or self.observed_at < self.settled_at):
            raise ValueError("Finite ordered times and nonnegative tolerances required")
        if not isinstance(self.cameras, tuple) or not self.cameras:
            raise ValueError("Explicit ordered camera tuple required")
        ids, times, movable = set(), [], False
        for frame in self.cameras:
            if (not isinstance(frame, CameraFrame) or not _name(frame.camera_id)
                    or frame.camera_id in ids or frame.role not in ("movable", "context")
                    or type(frame.valid) is not bool):
                raise ValueError("Unique real camera identities, roles and boolean masks required")
            ids.add(frame.camera_id)
            if not frame.valid:
                if frame.captured_at is not None or frame.image_sha256 is not None:
                    raise ValueError("Missing cameras cannot carry a fabricated frame")
                continue
            if not _digest(frame.image_sha256) or not _time(frame.captured_at):
                raise ValueError("Valid frames require content hashes and capture times")
            if not self.settled_at <= frame.captured_at <= self.observed_at:
                raise ValueError("Frame predates settling or comes from the future")
            if self.observed_at - frame.captured_at > self.max_age:
                raise ValueError("Stale frame")
            times.append(frame.captured_at)
            movable |= frame.role == "movable"
        if not movable:
            raise ValueError("At least one observed movable camera required")
        if max(times) - min(times) > self.max_skew:
            raise ValueError("Unsynchronized cameras")


@dataclass(frozen=True)
class FeatureSpec:
    source_kind: str
    model_sha256: str
    tap: str
    fusion_scope: str
    camera_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    preprocessing_sha256: str


def validate_feature_metadata(spec, observation, *, query_id, history_sha256,
                              observed_shape, observed_dtype):
    """Reject mismatched feature/cache metadata; does not validate tensor contents.

    A per-view record describes exactly one real view. A joint record explicitly
    covers all currently valid views in bundle order; never infer a split by shape.
    This first contract describes a settled current observation only. Generated
    or temporal features require a separate explicit time/noise contract.
    """
    observation.validate()
    if query_id != observation.query_id or history_sha256 != observation.history_sha256:
        raise ValueError("Features belong to another query or history branch")
    if (spec.source_kind not in ("target_native", "shared_source_control")
            or not _digest(spec.model_sha256) or not _name(spec.tap)
            or spec.preprocessing_sha256 != observation.preprocessing_sha256):
        raise ValueError("Feature source, tap or preprocessing identity mismatch")
    valid_ids = tuple(c.camera_id for c in observation.cameras if c.valid)
    if (not isinstance(spec.camera_ids, tuple) or not spec.camera_ids
            or any(not _name(x) for x in spec.camera_ids)
            or len(set(spec.camera_ids)) != len(spec.camera_ids)):
        raise ValueError("Explicit unique feature camera mapping required")
    if spec.fusion_scope == "per_view":
        if len(spec.camera_ids) != 1 or spec.camera_ids[0] not in valid_ids:
            raise ValueError("Per-view features require one available camera")
    elif spec.fusion_scope == "joint":
        if spec.camera_ids != valid_ids:
            raise ValueError("Joint features must preserve declared valid camera order")
    else:
        raise ValueError("Unknown fusion scope")
    if (not isinstance(spec.shape, tuple) or not spec.shape
            or any(type(n) is not int or n < 1 for n in spec.shape)
            or tuple(observed_shape) != spec.shape
            or spec.dtype not in ("float32", "float16", "bfloat16")
            or observed_dtype != spec.dtype):
        raise ValueError("Feature shape or dtype mismatch")

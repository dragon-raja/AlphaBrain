"""Next-protocol execution adapter, injected backend only; old executor untouched.

The caller owns append-only recording and fail-closed task scheduling. This
module never creates a simulator, loads a model or launches a task on import.
"""
import hashlib
import json
import numpy as np

from AlphaBrain.research.dsol.data.local_search_plan import validate_plan


def input_identity(example):
    """Hash the exact policy input, including dtype/shape and both language keys."""
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object arrays are not valid policy inputs')
            return dict(dtype=value.dtype.str, shape=list(value.shape),
                        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        if isinstance(value, dict):
            return {k: encode(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    return hashlib.sha256(json.dumps(encode(example), sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def rollout_local_episode(request, infer, example, bank, state_key, repeat, identity,
                          *, plan, suite, record):
    """Task-specific horizon; record full clipped prediction before executing five.

    `record` must durably append each event in the real backend. Exceptions
    propagate; there is no automatic retry, reset or replacement trajectory.
    """
    validate_plan(plan)
    if repeat not in plan['evaluation']['execution_repeat_ids'] or not callable(record):
        raise ValueError('Declared paired repeat and explicit recorder required')
    horizon = plan['evaluation']['horizons'][suite]
    op = plan['operation_policy']
    if (op['predicted_actions'], op['executed_actions_per_replan'], op['clip']) != (10, 5, [-1, 1]):
        raise ValueError('Frozen ten-action/five-executed contract required')
    calibration = {k: np.array(example[k], copy=True) for k in ('camera_intrinsics', 'camera_to_world_opencv')}
    steps = replans = 0
    success = False
    chunks, executed, input_hashes, noise_hashes, offsets = [], [], [], [], []
    native_traces,native_noises=[],[]
    while steps < horizon and not success:
        noise = bank.get(state_key, repeat, replans)
        if np.asarray(noise['noise']).shape != (10, 7) or not np.isfinite(noise['noise']).all():
            raise ValueError('Finite explicit execution noise required')
        digest = input_identity(example)
        address = dict(identity, repeat=repeat, replan=replans, state_key=state_key)
        record(dict(kind='inference_attempt', identity=address, input_sha256=digest,
                    noise_sha256=noise['noise_sha256'], action_offset=steps))
        result = infer(example, noise['noise'], address, capture=False)
        prediction = np.asarray(result['actions'])
        if 'full_trace' in result:
            native_traces.append(np.array(result['full_trace'],copy=True))
            native_noises.append(np.array(result['full_noise'],copy=True))
        if prediction.shape != (10, 7) or not np.isfinite(prediction).all():
            raise ValueError('Full finite 10x7 action block required')
        full = np.clip(prediction, -1, 1)
        chunk = full[:min(5, horizon - steps)].copy()
        record(dict(kind='prediction', identity=address, full_clipped_actions=full.copy(),
                    robot_state=np.array(example['state'], copy=True), input_sha256=digest))
        response = request(dict(kind='step', actions=chunk))
        count = response['executed']
        done = response['success']
        if (type(count) is not int or not 1 <= count <= len(chunk) or type(done) is not bool
                or (count != len(chunk) and not done)):
            raise ValueError('Partial execution without successful termination is invalid')
        record(dict(kind='executed', identity=address, actions=chunk[:count].copy(), success=done))
        example = response['example']
        if any(not np.array_equal(example[k], value) for k, value in calibration.items()):
            raise ValueError('External camera changed during manipulation')
        offsets.append(steps)
        chunks.append(full.copy()); executed.extend(chunk[:count])
        input_hashes.append(digest); noise_hashes.append(noise['noise_sha256'])
        steps += count; replans += 1; success = done
    arrays=dict(full_clipped_action_chunks=np.asarray(chunks), actions=np.asarray(executed),
                replan_action_offsets=np.asarray(offsets))
    if native_traces:
        if len(native_traces)!=replans:raise ValueError('Missing native full execution trace')
        arrays.update(native_full_velocity_traces=np.asarray(native_traces),native_full_noises=np.asarray(native_noises))
    return dict(success=success, steps=steps, replans=replans, horizon=horizon,
                repeat=repeat, state_key=state_key, input_sha256=input_hashes,
                execution_noise_sha256=noise_hashes), arrays

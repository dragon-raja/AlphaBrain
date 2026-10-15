"""Portable work identities and immutable scientific/execution manifests."""
import hashlib
import json
from pathlib import Path


FORMAT = 'dsol_work_queue_v1'
ADDRESS_FIELDS = {'model', 'task', 'initial', 'view', 'stream', 'repeat', 'noise'}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def task_key(experiment, protocol_sha256, addresses):
    """Block identity includes every original trial address, not the host/GPU."""
    return digest([experiment, protocol_sha256, addresses])


def validate(plan):
    if plan.get('format') != FORMAT or not isinstance(plan.get('experiment_id'), str) or not plan['experiment_id']:
        raise ValueError('Explicit queue format and experiment identity required')
    fingerprint = plan.get('protocol_sha256', '')
    if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('A frozen scientific protocol SHA256 is required')
    if not plan.get('slots') or not plan.get('tasks') or not plan.get('artifacts'):
        raise ValueError('Nonempty slots, tasks and source artifact fingerprints required')
    slots = set()
    for slot in plan['slots']:
        if not all(isinstance(slot.get(k), str) and slot[k] for k in ('host', 'id')):
            raise ValueError('Slots require explicit host and id')
        key = (slot['host'], slot['id'])
        if key in slots:
            raise ValueError('Duplicate host/slot')
        slots.add(key)
    keys, addresses, outputs = set(), set(), []
    for task in plan['tasks']:
        if not task.get('addresses') or not isinstance(task.get('payload'), dict):
            raise ValueError('Each block requires trial addresses and an adapter payload')
        for address in task['addresses']:
            if set(address) != ADDRESS_FIELDS or address['stream'] not in ('score', 'execution'):
                raise ValueError('Incomplete trial address or unknown noise stream')
            if type(address['repeat']) is not int or address['repeat'] < 0:
                raise ValueError('Repeat must be a nonnegative integer')
            if not all(isinstance(address[k], str) and address[k]
                       for k in ADDRESS_FIELDS - {'repeat'}):
                raise ValueError('Trial identities/noise addresses must be explicit strings')
            identity = digest(address)
            if identity in addresses:
                raise ValueError('Duplicate trial ownership')
            addresses.add(identity)
        expected = task_key(plan['experiment_id'], fingerprint, task['addresses'])
        if task.get('key') != expected or expected in keys:
            raise ValueError('Changed or duplicate block identity')
        keys.add(expected)
        output = Path(task['output'])
        if not output.is_absolute() or '..' in output.parts or output == Path(output.anchor):
            raise ValueError('Task output requires an explicit, non-root absolute path')
        if any(output == old or output in old.parents or old in output.parents for old in outputs):
            raise ValueError('Overlapping task output ownership')
        outputs.append(output)
    paths = set()
    for artifact in plan['artifacts']:
        path = Path(artifact['path'])
        value = artifact['sha256']
        if not path.is_absolute() or str(path) in paths or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Artifacts require unique absolute paths and SHA256 fingerprints')
        paths.add(str(path))
    if fingerprint not in {a['sha256'] for a in plan['artifacts']}:
        raise ValueError('Scientific protocol must be one of the fingerprinted artifacts')
    backend = plan.get('backend', {})
    if not isinstance(backend.get('module'), str) or not backend['module'] or not isinstance(backend.get('factory'), str) or not backend['factory']:
        raise ValueError('An explicit, trusted execution adapter factory is required')
    return plan


def verify_artifacts(plan):
    for artifact in plan['artifacts']:
        if file_digest(artifact['path']) != artifact['sha256']:
            raise ValueError('Frozen artifact changed: ' + artifact['path'])

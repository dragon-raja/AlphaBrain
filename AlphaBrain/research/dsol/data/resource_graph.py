"""Portable project resource graph plus separate machine-local path bindings.

Checks declared relationships and path presence, not numerical equivalence or
third-party license permission. It never initializes a model or simulator.
"""
from pathlib import Path


def audit_graph(graph, locations=None):
    errors, rows = [], []
    resources = graph['resources']
    ids = [item['id'] for item in resources]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate resource ID')
    by_id = {item['id']: item for item in resources}
    mapping = (locations or {}).get('resource_paths', {})
    retired = (locations or {}).get('retired_roots', [])
    for item in resources:
        for parent in item.get('parents', []):
            if parent not in by_id:
                errors.append(f"Unknown parent: {item['id']} -> {parent}")
        row = {'id': item['id'], 'kind': item['kind'], 'availability': item['availability']}
        if locations is not None:
            value = mapping.get(item['id'])
            row['path'] = value
            row['exists'] = bool(value and Path(value).exists())
            if value and (not Path(value).is_absolute() or any(
                    value == root or value.startswith(root + '/') for root in retired)):
                errors.append(f"Invalid local path: {item['id']}")
            if not row['exists'] and item['availability'] == 'present':
                errors.append(f"Missing present resource: {item['id']}")
        rows.append(row)
    # Reject cycles in the provenance graph, including self-reference.
    active, done = set(), set()
    def visit(key):
        if key in active:
            raise ValueError('Resource provenance cycle')
        if key in done or key not in by_id:
            return
        active.add(key)
        for parent in by_id[key].get('parents', []):
            visit(parent)
        active.remove(key)
        done.add(key)
    try:
        for key in by_id:
            visit(key)
    except ValueError as exc:
        errors.append(str(exc))
    for binding in graph.get('execution_bindings', []):
        refs = binding['resources']
        for key, revision in binding.get('required_revisions', {}).items():
            if key not in refs or by_id.get(key, {}).get('revision') != revision:
                errors.append(f"Execution revision mismatch: {binding['id']} -> {key}")
        for key in refs:
            if key not in by_id:
                errors.append(f"Unknown execution resource: {binding['id']} -> {key}")
        for key in binding.get('checkpoints', []):
            checkpoint = by_id.get(key, {})
            if key not in refs or checkpoint.get('kind') != 'checkpoint':
                errors.append(f"Invalid checkpoint binding: {binding['id']}")
            if checkpoint.get('backend') != binding['backend']:
                errors.append(f"Checkpoint backend mismatch: {binding['id']} -> {key}")
    return {'status': 'PASS' if not errors else 'INCOMPLETE', 'errors': errors,
            'resources': rows, 'execution_bindings': graph.get('execution_bindings', []),
            'scope': 'declared provenance/backend bindings and optional local presence; not payload or policy-equivalence certification'}

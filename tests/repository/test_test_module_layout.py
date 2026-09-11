"""Prevent flat test accumulation and lost assertions during classification."""
import ast
import hashlib
import json
import subprocess
import copy

from tests.dsol_paper1.paths import ROOT


def baseline_ast_dump(tree):
    """Render the original Python 3.12 receipt on Python 3.10 as well.

    PEP 695 added the empty type_params field to definitions in 3.12.
    Preserve the recorded digest instead of regenerating historical receipts.
    These sources use no generic type-parameter syntax; no executable field is
    removed or ignored here.
    """
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and 'type_params' not in node._fields:
            node._fields = (*node._fields, 'type_params')
            node.type_params = []
    return ast.dump(tree, include_attributes=False)


def test_baseline_ast_representation_handles_pre_312_definition_fields():
    node = ast.parse('def probe():\n    assert 1 == 1\n').body[0]
    expected = baseline_ast_dump(node)
    old_style = copy.deepcopy(node)
    old_style._fields = tuple(f for f in old_style._fields if f != 'type_params')
    assert baseline_ast_dump(old_style) == expected
    changed = ast.parse('def probe():\n    assert 1 == 2\n').body[0]
    assert baseline_ast_dump(changed) != expected


def test_all_85_modules_have_an_owner_and_no_flat_forwarders():
    manifest = json.loads((ROOT / 'archive/repository/20260911/test-modules/manifest.json').read_text())
    assert len(manifest['records']) == 85
    for row in manifest['records']:
        assert not (ROOT / row['old']).exists()
        assert (ROOT / row['new']).is_file()
    ignored = subprocess.run(['git', 'check-ignore', '--stdin'],
                             input='\n'.join(r['new'] for r in manifest['records']) + '\n',
                             cwd=ROOT, text=True, capture_output=True)
    assert not ignored.stdout, ignored.stdout
    assert {p.name for p in (ROOT / 'tests/dsol_paper1').glob('*.py')} == {'conftest.py', 'paths.py'}


def test_scientific_test_bodies_and_assertions_were_not_removed_or_weakened():
    manifest = json.loads((ROOT / 'archive/repository/20260911/test-modules/manifest.json').read_text())
    runtime = json.loads((ROOT / 'archive/repository/20260911/runtime-modules/manifest.json').read_text())
    previous = {r['new']: ROOT / r['before'] for r in runtime['records']}
    moves = [(r['new'], r['old']) for r in runtime['records'] if r['new'] != r['old']]

    class NormalizePaths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for new, old in moves:
                    node.value = node.value.replace(new, old)
                    if new.endswith('.py'):
                        node.value = node.value.replace(new[:-3].replace('/', '.'), old[:-3].replace('/', '.'))
                    if node.value == new.rsplit('/', 1)[-1]:
                        node.value = old.rsplit('/', 1)[-1]
            return node

        def visit_Import(self, node):
            return ast.Pass()

        visit_ImportFrom = visit_Import

    def cases(tree):
        return {str(index) + ':' + node.name: node for index, node in enumerate(
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_'))}

    checked = 0
    for row in manifest['records']:
        if '/architecture/' in row['new']:
            continue  # Layout tests explicitly follow the new relocation ledger.
        path = ROOT / row['new']
        tree = ast.parse(previous.get(row['new'], path).read_text())
        actual = {
            str(index) + ':' + node.name: hashlib.sha256(baseline_ast_dump(node).encode()).hexdigest()
            for index, node in enumerate(n for n in ast.walk(tree)
                                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_'))
        }
        assert actual == row['test_bodies'], row['new']
        before_cases, after_cases = cases(tree), cases(ast.parse(path.read_text()))
        assert before_cases.keys() == after_cases.keys()
        for key in before_cases:
            left = ast.dump(NormalizePaths().visit(copy.deepcopy(before_cases[key])), include_attributes=False)
            right = ast.dump(NormalizePaths().visit(copy.deepcopy(after_cases[key])), include_attributes=False)
            assert left == right, (row['new'], key)
        checked += len(actual)
    assert checked > 300

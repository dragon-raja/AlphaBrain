"""Compare extracted numerical bodies to the preserved pre-refactor sources."""
import ast
import hashlib
import json
from pathlib import Path
import inspect
from copy import deepcopy
from benchmarks.LIBERO import simulation_inputs


class ImportsOnly(ast.NodeTransformer):
    def visit_Import(self, node):
        return None
    def visit_ImportFrom(self, node):
        return None


def normalized_body(tree):
    """Match the preserved Python 3.12 AST format on Python 3.10 as well.

    Python 3.12 added type_params to function/class nodes, even for nongeneric
    definitions. Add only that missing empty field; do not erase real type
    parameters, imports aside from the established extraction rule, or bodies.
    """
    tree = ImportsOnly().visit(deepcopy(tree))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if "type_params" not in node._fields:
                node._fields = (*node._fields, "type_params")
                node.type_params = []
    return ast.dump(tree, include_attributes=False)


def test_extracted_math_and_state_paths_match_original_bodies():
    root=Path(__file__).resolve().parents[1]
    gold=json.loads((root/'helpers/runtime_extraction_goldens.json').read_text())
    for name,expected in gold.items():
        function=inspect.unwrap(getattr(simulation_inputs,name))
        tree=ast.parse(inspect.getsource(function)).body[0]
        normalized=normalized_body(tree)
        assert hashlib.sha256(normalized.encode()).hexdigest()==expected,name


def test_ast_version_normalization_retains_body_changes():
    tree = ast.parse("def example(x):\n    return x + 1\n").body[0]
    old_tree = deepcopy(tree)
    for node in ast.walk(old_tree):
        if "type_params" in node._fields:
            node._fields = tuple(k for k in node._fields if k != "type_params")
            del node.type_params
    assert normalized_body(old_tree) == normalized_body(tree)
    changed = ast.parse("def example(x):\n    return x + 2\n").body[0]
    assert normalized_body(changed) != normalized_body(tree)


def test_ast_version_normalization_does_not_erase_type_parameters():
    tree = ast.parse("def example(x):\n    return x\n").body[0]
    original = normalized_body(tree)
    if "type_params" not in tree._fields:
        tree._fields = (*tree._fields, "type_params")
    tree.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert normalized_body(tree) != original


def test_legacy_exports_use_the_same_runtime_implementation():
    from scripts.dsol_paper1.runtime import standard_initialization_v1 as old
    assert old.initialize is simulation_inputs.initialize
    assert old.load_initial is simulation_inputs.load_initial

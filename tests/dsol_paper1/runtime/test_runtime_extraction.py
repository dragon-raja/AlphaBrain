"""Compare extracted numerical bodies to the preserved pre-refactor sources."""
import ast
import hashlib
import json
from pathlib import Path
import inspect
from benchmarks.LIBERO import simulation_inputs


class ImportsOnly(ast.NodeTransformer):
    def visit_Import(self, node):
        return None
    def visit_ImportFrom(self, node):
        return None


def test_extracted_math_and_state_paths_match_original_bodies():
    root=Path(__file__).resolve().parents[1]
    gold=json.loads((root/'helpers/runtime_extraction_goldens.json').read_text())
    for name,expected in gold.items():
        function=inspect.unwrap(getattr(simulation_inputs,name))
        tree=ast.parse(inspect.getsource(function)).body[0]
        normalized=ast.dump(ImportsOnly().visit(tree),include_attributes=False)
        assert hashlib.sha256(normalized.encode()).hexdigest()==expected,name


def test_legacy_exports_use_the_same_runtime_implementation():
    from scripts.dsol_paper1.runtime import standard_initialization_v1 as old
    assert old.initialize is simulation_inputs.initialize
    assert old.load_initial is simulation_inputs.load_initial

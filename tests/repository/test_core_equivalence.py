import ast
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest

from AlphaBrain.common import images, pair_records

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "archive/repository/20260911/core/before"


def original(relative, name):
    spec = importlib.util.spec_from_file_location(name, BEFORE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape,target", [((10, 20, 3), (15, 15)), ((2, 7, 19, 3), (8, 16)), ((16, 16, 3), (16, 16))])
def test_image_preprocessing_exactly_matches_original(shape, target):
    before = original("deployment/model_server/tools/image_tools.py", "images_before_extraction")
    rng = np.random.default_rng(41)
    a = rng.integers(0, 256, size=shape, dtype=np.uint8)
    np.testing.assert_array_equal(images.resize_with_pad(a, *target), before.resize_with_pad(a, *target))
    f = rng.random(shape).astype(np.float32)
    np.testing.assert_array_equal(images.convert_to_uint8(f), before.convert_to_uint8(f))
    if len(shape) == 3:
        left, right = images.to_pil_preserve([f, (a,)]), before.to_pil_preserve([f, (a,)])
        np.testing.assert_array_equal(np.asarray(left[0]), np.asarray(right[0]))
        np.testing.assert_array_equal(np.asarray(left[1][0]), np.asarray(right[1][0]))


def test_pair_codec_function_bodies_are_unchanged():
    old = ast.parse((BEFORE / "scripts/dsol_paper1/training/libero_pair_records.py").read_text())
    new = ast.parse(Path(pair_records.__file__).read_text())
    definitions = lambda t: {n.name: ast.dump(n, include_attributes=False) for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert definitions(old) == definitions(new)


def test_pair_record_serialization_exactly_matches_original():
    before = original("scripts/dsol_paper1/training/libero_pair_records.py", "pair_codec_before_extraction")
    rng = np.random.default_rng(41)
    views = {name: rng.integers(0, 256, size=(12, 10, 3), dtype=np.uint8) for name in pair_records.IMAGE_ORDER}
    buffers = []
    returns = []
    for module in (before, pair_records):
        output = io.BytesIO()
        module.initialize_shard(output)
        returns.append(module.write_record(output, header={"task": "fixture", "version": 1}, images=views))
        buffers.append(output.getvalue())
    assert buffers[0] == buffers[1]
    assert returns[0] == returns[1]

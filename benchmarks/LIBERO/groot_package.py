"""Prepare an isolated, auditable GR00T import surface without changing kernels.

Upstream imports every robot driver, dataset and model at package import time.
This candidate registers only the GR00T N1.5 model dependencies. Model functions
remain byte-identical; graph observation is a separate optional overlay. Neither
preparation nor import grants a GPU execution permit.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

from .groot_flow_capture import UPSTREAM_SHA256


IMPORT_SURFACE = {
    'fluxvla/engines/__init__.py': 'from .utils import *\n',
    'fluxvla/models/__init__.py': 'from .backbones import *\nfrom .heads import *\nfrom .vlas import *\n',
    'fluxvla/models/backbones/__init__.py': 'from .vlms.eagle import EagleBackbone, EagleInferenceBackbone\n',
    'fluxvla/models/backbones/vlms/__init__.py': '',
    'fluxvla/models/heads/__init__.py': 'from .flow_matching_head import FlowMatchingHead\nfrom .flow_matching_inference_head import FlowMatchingInferenceHead\n',
    'fluxvla/models/vlas/__init__.py': 'from .llava_vla import LlavaVLA\n',
    'fluxvla/transforms/__init__.py': '',
    'fluxvla/datasets/__init__.py': '',
    'fluxvla/datasets/utils/__init__.py': '',
    'fluxvla/tokenizers/__init__.py': 'from .pretrained_tokenizer import PretrainedTokenizer\n',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    if digest(source/'fluxvla/models/heads/flow_matching_inference_head.py') != UPSTREAM_SHA256:
        raise ValueError('Unreviewed upstream inference head')
    if destination.exists():
        raise ValueError('Candidate destination must be new')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'build', '*.so'))
    replacements = dict(IMPORT_SURFACE)
    root = (source/'fluxvla/__init__.py').read_text()
    replacements['fluxvla/__init__.py'] = root.split('from .collators import *')[0] + 'from . import engines, models\n'
    changed = {}
    for name, code in replacements.items():
        original = (source/name).read_text(encoding='utf-8-sig')
        # Keep upstream copyright/license comment preamble for replaced initializers.
        header = []
        for line in original.splitlines(keepends=True):
            if line.strip() and not line.lstrip().startswith('#'): break
            header.append(line)
        replacement = code if name == 'fluxvla/__init__.py' else ''.join(header) + code
        ast.parse(replacement)
        (destination/name).write_text(replacement)
        changed[name] = dict(original=digest(source/name), candidate=digest(destination/name))
    preserved = {}
    for path in source.rglob('*.py'):
        name = str(path.relative_to(source))
        if name in replacements: continue
        if digest(path) != digest(destination/name):
            raise ValueError('Non-initializer source changed: '+name)
        preserved[name] = digest(path)
    manifest = dict(changed_initializers=changed, preserved_python=preserved,
                    model_calls=0, gpu_admitted=False, source=str(source))
    (destination/'dsol-import-surface.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

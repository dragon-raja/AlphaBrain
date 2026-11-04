"""Explicit, versioned render intervention shared by bridge-only workers."""

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np

REPO=_REPOSITORY_ROOT
ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/render-protocol-bridge-v1')
RUNTIME=Path('/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus')

def read(path): return json.loads(Path(path).read_text())

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): value.update(chunk)
    return value.hexdigest()

def write_new(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2); stream.write('\n')

def array_identity(value):
    a=np.ascontiguousarray(value)
    return {'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}

def cell_key(spec): return (spec['pair_key'],spec['selected_candidate_id'],int(spec['policy_repeat_id']))

@contextmanager
def render_protocol(offsamples):
    if offsamples not in (0,4): raise ValueError('Only preregistered AA0/AA4 protocols are permitted')
    from robosuite.utils.binding_utils import MjRenderContext
    original=MjRenderContext._set_mujoco_context_and_buffers
    contexts=[]
    def initialize(context):
        if offsamples==4 and int(context.model.vis.quality.offsamples)!=4:
            raise ValueError('Historical XML does not have the expected default MSAA4 setting')
        context.model.vis.quality.offsamples=offsamples
        result=original(context)
        if int(context.con.offSamples)!=offsamples: raise ValueError('Actual framebuffer sample count differs')
        contexts.append(int(context.con.offSamples))
        return result
    MjRenderContext._set_mujoco_context_and_buffers=initialize
    try: yield contexts
    finally: MjRenderContext._set_mujoco_context_and_buffers=original

def verify_release():
    release=read(ROOT/'release.json')
    for item in release['sources']:
        if sha(item['path'])!=item['sha256']: raise ValueError('Frozen bridge source changed: '+item['path'])
    return release

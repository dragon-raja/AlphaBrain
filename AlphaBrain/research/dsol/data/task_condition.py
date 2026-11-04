"""Model-independent text-only task representation and immutable cache contract."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'TEXT_ONLY_CONTEXTUAL_TASK_V1'


def instruction_key(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Actual nonempty task instruction required')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_spec(spec):
    required={'schema','family','model_sha256','tokenizer_sha256','encoder_source_sha256','tap','pooling','dimension','inputs','attention'}
    if (not isinstance(spec,dict) or set(spec)!=required or spec['schema']!=SCHEMA
            or spec['inputs']!=['instruction'] or spec['pooling']!='valid_instruction_tokens_mean_float32'
            or spec['attention']!='bidirectional_text_only' or spec['tap']!='language_backbone_last_hidden'
            or type(spec['dimension']) is not int or spec['dimension']<=0
            or not isinstance(spec['family'],str) or not spec['family']):
        raise ValueError('Explicit text-only contextual feature contract required')
    for field in ('model_sha256','tokenizer_sha256','encoder_source_sha256'):
        value=spec[field]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid task feature provenance')
    return spec


def validate_binding(binding):
    if not isinstance(binding,dict) or set(binding)!={'spec','cache','cache_sha256'}:raise ValueError('Complete task cache binding required')
    validate_spec(binding['spec'])
    if not isinstance(binding['cache'],str) or not binding['cache']:raise ValueError('Explicit cache path required')
    value=binding['cache_sha256']
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid task cache digest')
    return binding


@lru_cache(maxsize=16)
def _load(path,stamp):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=stamp:raise ValueError('Changed task feature cache')
    cache=json.loads(raw);validate_spec(cache['spec'])
    if set(cache)!={'spec','instructions'} or not cache['instructions']:raise ValueError('Invalid text feature cache')
    for key,row in cache['instructions'].items():
        if set(row)!={'text','vector'} or instruction_key(row['text'])!=key:raise ValueError('Task text/cache identity mismatch')
        x=np.asarray(row['vector'],dtype=np.float32)
        if x.shape!=(cache['spec']['dimension'],) or not np.isfinite(x).all():raise ValueError('Malformed task feature')
    return cache


def task_vector(binding,text):
    validate_binding(binding);cache=_load(binding['cache'],binding['cache_sha256'])
    if cache['spec']!=binding['spec']:raise ValueError('Task encoder provenance differs')
    row=cache['instructions'].get(instruction_key(text))
    if row is None or row['text']!=text:raise ValueError('Unregistered instruction; no fabricated or fallback task feature')
    return np.asarray(row['vector'],dtype=np.float32).copy()

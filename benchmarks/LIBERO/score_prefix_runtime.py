"""Opt-in scoring decoration; native forward, features, actions and RNG unchanged."""
from copy import deepcopy
from AlphaBrain.research.dsol.data.score_prefix import validate
from AlphaBrain.research.dsol.metrics.score_prefix import score


class PrefixRuntime:
    def __init__(self, runtime, spec):
        object.__setattr__(self, 'runtime', runtime)
        object.__setattr__(self, 'score_contract', validate(spec))

    def __getattr__(self, name): return getattr(self.runtime, name)

    def __setattr__(self, name, value):
        if name in ('runtime', 'score_contract'): raise AttributeError('Immutable prefix/runtime binding')
        setattr(self.runtime, name, value)

    def infer(self, *args, **kwargs):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/score_prefix_runtime.py:PrefixRuntime.infer')
        value = self.runtime.infer(*args, **kwargs)
        capture = kwargs.get('capture', args[3] if len(args) > 3 else True)
        if capture:
            value = dict(value)
            value['score'] = score(value['trace'], self.score_contract)
            value['score_contract'] = deepcopy(self.score_contract)
        return value

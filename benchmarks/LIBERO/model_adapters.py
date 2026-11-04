"""Reviewed LIBERO model capabilities; the PPO core treats family IDs as opaque.

New models register their runtime contract and operation adapter here. Imports
are lazy; configuration cannot name an arbitrary module or callable.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelAdapter:
    family: str
    implementation: str
    runtime_module: str
    runtime_class: str
    constructor_style: str
    rollout_name: str
    warmup_calls: int
    suite_specific: bool

    def validate_runtime(self,path,sha):
        return import_module('.'+self.runtime_module,__package__).validate_contract(path,sha)

    def rollout(self,*args,**kwargs):
        module='.rldx_search_runtime' if self.family=='rldx' else '.local_search_rollout'
        return getattr(import_module(module,__package__),self.rollout_name)(*args,**kwargs)

    def create_runtime(self,config,output,limit):
        cls=getattr(import_module('.'+self.runtime_module,__package__),self.runtime_class)
        if self.constructor_style=='config':return cls(config,output,limit)
        if self.constructor_style=='contract':
            model=config['models'][config['operation_model']]
            return cls(model['runtime_contract'],model['runtime_contract_sha256'],output,limit,
                       instrument=True,native_features=config['native_features'])
        raise ValueError('Unreviewed native runtime constructor')


_ADAPTERS={
    'rldx':ModelAdapter('rldx','rldx_native','rldx_search_runtime','RLDXSearchRuntime','config','rollout_rldx_episode',0,False),
    'fastwam':ModelAdapter('fastwam','fastwam_native','legacy.fastwam_runtime','FastWAMRuntime','config','rollout_fastwam_episode',0,False),
    'groot_n15':ModelAdapter('groot_n15','groot_native','groot_runtime','GrootRuntime','contract','rollout_groot_episode',1,True),
}


def model_adapter(family):
    try:return _ADAPTERS[family]
    except (KeyError,TypeError):raise ValueError('Unregistered target model adapter') from None

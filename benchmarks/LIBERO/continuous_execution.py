"""Explicit one-way search-to-manipulation gate for continuous evaluation.

Training/admission configs cannot unlock manipulation. Each episode uses a
fresh simulator; reset, catalog queries and camera motion after unlock remain
forbidden. The normal env.step observation timing is shared with old rollouts.
"""
class ContinuousExecutionGate:
    def __init__(self, config):
        self.authorized = config.get('continuous_execution') == 'PAIRED_EVALUATION_ONLY'
        self.executing = False

    def allow(self, kind):
        if kind == 'close':
            return
        if self.executing:
            if kind != 'step':
                raise ValueError('Camera/search commands forbidden after execution begins')
        elif kind == 'begin_execution':
            if not self.authorized:
                raise ValueError('Continuous manipulation not authorized')
        elif kind not in ('continuous_info', 'move_pose', 'query_pose'):
            raise ValueError('Continuous search forbids catalog/reset/manipulation commands')

    def begin(self):
        self.allow('begin_execution')
        self.executing = True

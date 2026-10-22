"""One persistent continuous-training worker on the standard fail-closed queue.

This is not a discrete catalog evaluator or a multi-worker PPO collector.
GPU ownership, keepalive identity and the pre-start recheck reuse ProbeBackend.
"""
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read,sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


class ContinuousTrainingBackend(ProbeBackend):
    def _scope(self):
        if (self.runtime['mode']!='continuous_ppo_training' or self.runtime.get('workers_per_gpu')!=1
                or len(self.plan['slots'])!=1 or len(self.plan['tasks'])!=1):
            raise ValueError('Single persistent training worker required')
        config=read(self.runtime['config'])
        if config.get('authorization')!='EXPLICIT_TRAINING_ONLY':
            raise ValueError('Training is not authorized by this prepared manifest')
        return config

    def admit(self,slot):
        self._scope()
        if slot['host']!=self.runtime['host']:raise ValueError('Wrong coordinator host')
        proof=self.runtime['continuous_admission']
        if sha(proof['path'])!=proof['sha256'] or read(proof['path'])['status']!='PASS_ENGINEERING_ONLY':
            raise ValueError('Continuous admission identity missing')
        return self.gpu_available(slot)

    def worker_command(self,task):
        self._scope()
        return [self.runtime['python'],'-u','-m','benchmarks.LIBERO.continuous_training',
                '--config',self.runtime['config'],'--output',task['output'],'--execute']

    def validate(self,task):
        path=Path(task['output'])/'status.json'
        if not path.exists():raise MissingResult(str(path))
        row=read(path)
        if (row['status']!='COMPLETE' or row['config_sha256']!=sha(self.runtime['config'])
                or row['source_kind']!='LIVE_RENDER_BATCH1_TWO_NOISES' or row['episodes']!=512
                or row['updates']!=16 or not 512<=row['queries']<=4096
                or row['calls']!=row['queries']*2 or row['manipulation_actions']!=0):
            raise ValueError('Incomplete/incorrect training receipt')
        latest=read(path.parent/'latest.json')
        if latest['update']!=16 or sha(path.parent/latest['path'])!=latest['sha256']:
            raise ValueError('Final checkpoint mismatch')
        return dict(status='PASS_TRAINING_ONLY',receipt_sha256=sha(path),checkpoint_sha256=latest['sha256'])


def factory(plan,root):
    return ContinuousTrainingBackend(plan,root)

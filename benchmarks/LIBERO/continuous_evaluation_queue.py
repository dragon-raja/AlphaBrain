"""Standard fail-closed queue adapter for one frozen paired evaluation worker."""
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


class ContinuousEvaluationBackend(ProbeBackend):
    def _scope(self):
        cfg = read(self.runtime['config'])
        if (self.runtime['mode']!='continuous_paired_evaluation' or len(self.plan['slots'])!=1
                or len(self.plan['tasks'])!=1 or self.runtime['workers_per_gpu']!=1
                or cfg['authorization']!='EXPLICIT_PAIRED_EVALUATION_ONLY'):
            raise ValueError('Single persistent paired evaluation worker required')
        proof=self.runtime['continuous_admission']
        if sha(proof['path'])!=proof['sha256'] or read(proof['path'])['status']!='PASS_ENGINEERING_ONLY':
            raise ValueError('Prior continuous engineering admission missing')

    def admit(self,slot):
        self._scope()
        return self.gpu_available(slot)

    def worker_command(self,task):
        self._scope()
        return [self.runtime['python'],'-u','-m','benchmarks.LIBERO.continuous_evaluation',
            '--config',self.runtime['config'],'--output',task['output'],'--execute']

    def validate(self,task):
        path=Path(task['output'])/'status.json'
        if not path.exists():raise MissingResult(str(path))
        value=read(path)
        if (value['status']!='COMPLETE' or value['tasks']!=192 or value['searches']!=48
                or value['training_updates']!=0 or value['config_sha256']!=sha(self.runtime['config'])
                or value['calls']['score']>544 or value['calls']['execution']>10560
                or value['summary']['complete_pairs']!=64):
            raise ValueError('Incomplete or inconsistent paired evaluation')
        return dict(status='PASS_PAIRED_EVALUATION',receipt_sha256=sha(path))


def factory(plan,root):return ContinuousEvaluationBackend(plan,root)

"""Rendering backend: reuse durable process ownership, no alternate scheduler."""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read,sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from .probe_queue import ProbeBackend
from .confirmation_render import validate_output


class RenderBackend(ProbeBackend):
    def admit(self,slot):
        if slot['host']!=self.runtime['host']:raise ValueError('Wrong host coordinator')
        receipt=read(self.runtime['admission'])
        if (receipt['status']!='COMPLETE' or receipt['candidate_count']!=97
                or receipt['plan_sha256']!=sha(self.runtime['plan'])
                or receipt['source_sha256']!=sha(Path(self.runtime['source'])/'benchmarks/Robocasa365/confirmation_render.py')):
            raise ValueError('Actual new-population renderer admission missing')
        return self.gpu_available(slot)

    def worker_command(self,task):
        return [self.runtime['python'],'-u','-m','benchmarks.Robocasa365.confirmation_render',
                '--plan',self.runtime['plan'],'--initial',task['payload']['initial'],
                '--output',task['output']]

    def validate(self,task):
        if not (Path(task['output'])/'status.json').exists():raise MissingResult(task['output'])
        state=validate_output(self.runtime['plan'],task['payload']['initial'],task['output'])
        return dict(status='PASS',status_sha256=sha(Path(task['output'])/'status.json'),
                    candidate_summary_sha256=state['candidate_summary_sha256'])


def factory(plan,root):return RenderBackend(plan,root)

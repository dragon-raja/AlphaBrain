"""Search workers on the existing durable host-local queue/process adapter."""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


class SearchBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Wrong host coordinator')
        if self.runtime['mode'] != 'capacity':
            raise ValueError('Only capacity is admitted; production requires its own full-task adapter')
        return self.gpu_available(slot)

    def worker_command(self, task):
        module = {'capacity':'benchmarks.LIBERO.search_capacity'}[self.runtime['mode']]
        command=[self.runtime['python'],'-u','-m',module,'--config',self.runtime['config'],
                 '--output',task['output'],'--execute']
        return command

    def validate(self, task):
        output=Path(task['output']); path=output/'status.json'
        if not path.is_file():
            raise MissingResult(str(output))
        receipt=read(path)
        if receipt['status']!='COMPLETE' or receipt['config_sha256']!=sha(self.runtime['config']):
            raise ValueError('Worker identity/status mismatch')
        if self.runtime['mode']=='capacity':
            if (receipt['calls']!=2 or receipt['exact'] is not True
                    or sha(output/'timing.json')!=receipt['timing_sha256']):
                raise ValueError('Capacity result invalid')
        else:
            raise ValueError('Unsupported search worker mode')
        return dict(status='PASS',receipt_sha256=sha(path))


def factory(plan, root):
    return SearchBackend(plan,root)

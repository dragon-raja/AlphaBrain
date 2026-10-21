"""Search workers on the existing durable host-local queue/process adapter."""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


class SearchBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Wrong host coordinator')
        if self.runtime['mode'] not in ('capacity','domain_admission'):
            raise ValueError('Only capacity/domain admission allowed; production requires its own full-task adapter')
        if self.runtime['mode']=='domain_admission':
            proof=self.runtime['capacity_admission']
            if sha(proof['path'])!=proof['sha256'] or read(proof['path'])['status']!='PASS':
                raise ValueError('Measured capacity admission required')
        return self.gpu_available(slot)

    def worker_command(self, task):
        module = {'capacity':'benchmarks.LIBERO.search_capacity',
                  'domain_admission':'benchmarks.LIBERO.search_domain'}[self.runtime['mode']]
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
        elif self.runtime['mode']=='domain_admission':
            if receipt['calls']!=64 or [c['initial'] for c in receipt['cases']]!=list(range(16)):
                raise ValueError('Domain admission coverage mismatch')
            for case in receipt['cases']:
                if (case['reset_pixels_exact'] is not True or case['feature_hook_exact'] is not True
                        or sha(case['reference'])!=case['sha256']):
                    raise ValueError('Domain input/model admission failed')
        else:
            raise ValueError('Unsupported search worker mode')
        return dict(status='PASS',receipt_sha256=sha(path))


def factory(plan, root):
    return SearchBackend(plan,root)

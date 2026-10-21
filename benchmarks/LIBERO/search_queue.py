"""Search workers on the existing durable host-local queue/process adapter."""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


class SearchBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Wrong host coordinator')
        if self.runtime['mode'] not in ('capacity','domain_admission','full_tasks','continuous_admission'):
            raise ValueError('Unsupported search queue mode; production requires the full_tasks adapter')
        proofs=['capacity_admission'] if self.runtime['mode']=='domain_admission' else []
        if self.runtime['mode'] in ('full_tasks','continuous_admission'):
            proofs=['capacity_admission','domain_admission']
        for key in proofs:
            proof=self.runtime[key]
            if sha(proof['path'])!=proof['sha256'] or read(proof['path'])['status']!='PASS':
                raise ValueError('Frozen prior admission required: '+key)
        return self.gpu_available(slot)

    def worker_command(self, task):
        module = {'capacity':'benchmarks.LIBERO.search_capacity',
                  'domain_admission':'benchmarks.LIBERO.search_domain',
                  'continuous_admission':'benchmarks.LIBERO.local_view_runtime',
                  'full_tasks':'benchmarks.LIBERO.search_rollout'}[self.runtime['mode']]
        config=task['payload']['config'] if self.runtime['mode']=='full_tasks' else self.runtime['config']
        command=[self.runtime['python'],'-u','-m',module,'--config',config,
                 '--output',task['output'],'--execute']
        if self.runtime['mode']=='full_tasks':command+=['--block',task['payload']['block']]
        return command

    def validate(self, task):
        output=Path(task['output']); path=output/'status.json'
        if not path.is_file():
            raise MissingResult(str(output))
        receipt=read(path)
        config=task['payload']['config'] if self.runtime['mode']=='full_tasks' else self.runtime['config']
        if receipt['status']!='COMPLETE' or receipt['config_sha256']!=sha(config):
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
        elif self.runtime['mode']=='continuous_admission':
            from .local_view_runtime import TARGETS
            expected={(c,i,tuple(a)) for c,i in [('A',2),('A',10),('B',0),('B',4)] for a in TARGETS}
            if receipt['calls']!=64 or len(receipt['cases'])!=16 or {
                (c['cohort'],c['initial'],tuple(c['angles'])) for c in receipt['cases']}!=expected:
                raise ValueError('Continuous admission coverage mismatch')
            for case in receipt['cases']:
                if (case['pose_input_return_exact'] is not True or case['model_return_exact'] is not True
                        or sha(case['reference'])!=case['sha256']):
                    raise ValueError('Continuous admission input/model verification failed')
        elif self.runtime['mode']=='full_tasks':
            from .search_rollout import validate_block
            block=validate_block(read(task['payload']['block']))
            if receipt['block_sha256']!=sha(task['payload']['block']):
                raise ValueError('Evaluation block receipt mismatch')
            expected={(m,r) for m in block['methods'] for r in block['repeats']}
            actual={(t['method'],t['repeat']) for t in receipt['tasks']}
            if actual!=expected or len(receipt['tasks'])!=len(expected):
                raise ValueError('Paired execution coverage mismatch')
            for episode in receipt['tasks']:
                if (episode['cohort']!=block['cohort'] or episode['initial']!=block['initial']
                        or episode['start']!=block['start'] or type(episode['success']) is not bool
                        or len(episode['execution_noise_sha256'])!=episode['replans']
                        or sha(episode['trajectory'])!=episode['trajectory_sha256']):
                    raise ValueError('Full task identity or trajectory mismatch')
        else:
            raise ValueError('Unsupported search worker mode')
        return dict(status='PASS',receipt_sha256=sha(path))


def factory(plan, root):
    return SearchBackend(plan,root)

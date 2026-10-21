"""Search workers on the existing durable host-local queue/process adapter."""
import hashlib
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend, process_identity


def verified_coexisting_process(binding, current, command, executable_sha256, gpu_uuid):
    """Pin an existing non-experiment process; a familiar filename is not enough."""
    return (current is not None and current['state']!='Z'
        and all(binding.get(k)==current[k] for k in ('pid','start','boot','group'))
        and binding.get('gpu_uuid')==gpu_uuid
        and binding.get('command_sha256')==hashlib.sha256(command).hexdigest()
        and binding.get('executable_sha256')==executable_sha256)


class SearchBackend(ProbeBackend):
    def allowed_external_process(self,pid,args,gpu_uuid):
        for binding in self.runtime.get('coexisting_processes',[]):
            if binding['pid']!=pid:
                continue
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes()
                executable=sha(f'/proc/{pid}/exe')
            except FileNotFoundError:
                return False
            return verified_coexisting_process(binding,process_identity(pid),command,executable,gpu_uuid)
        return False

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

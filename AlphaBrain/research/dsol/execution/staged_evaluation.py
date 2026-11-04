"""One-shot staged execution with durable attempts and an explicit first gate.

Callbacks compose admitted backends. This layer never loads models, grants
budgets, retries unknown work, or changes a backend's scientific contract.
"""
from pathlib import Path

from .manifest import file_digest
from .queue import _durable_write, exclusive


def run(allocation, root, *, contract_phase, stage, completion, gate, report,
        status, verify_sources):
    phases = allocation['phases']
    if not phases or len({p['phase'] for p in phases}) != len(phases):
        raise ValueError('Nonempty unique phases required')
    if contract_phase in {p['phase'] for p in phases}:
        raise ValueError('CPU contract needs a separate phase identity')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    with exclusive(root):
        calls = episodes = searches = 0
        limit = allocation['limits']
        call_cap = sum(limit[k] for k in ('score', 'paired_execution', 'nominal_execution'))

        def execute(name, phase, cpu):
            verify_sources()
            if file_digest(phase['campaign']) != phase['sha256']:
                raise ValueError('Frozen campaign changed')
            # Any exception after this point retains the intent. Reconciliation
            # must inspect the backend ledger; invoking this runner again fails.
            _durable_write(root / (name + '.attempt.json'),
                           dict(phase=name, cpu=cpu, campaign_sha256=phase['sha256']))
            stage(name, phase['campaign'], cpu=cpu)
            proof = completion(name, cpu)
            if proof['status'] != 'COMPLETE':
                raise ValueError('Incomplete stage receipt')
            return proof

        try:
            status('CPU_CONTRACT_RUNNING')
            proof = execute(contract_phase, phases[0], True)
            if proof['actual_model_calls'] != 0 or proof['actual_manipulation_episodes'] != 0:
                raise ValueError('CPU contract consumed scientific work')
            for index, phase in enumerate(phases):
                status('RECOVERY_RUNNING', phase=phase['phase'], completed_searches=searches,
                       completed_episodes=episodes, calls_settled=calls)
                proof = execute(phase['phase'], phase, False)
                count = proof['actual_model_calls']
                phase_cap = sum(phase['limits'][k] for k in ('score', 'paired_execution', 'nominal_execution'))
                if (type(count) is not int or not 0 <= count <= phase_cap
                        or proof['actual_manipulation_episodes'] != phase['limits']['episodes']):
                    raise ValueError('Stage receipt exceeds or differs from frozen budget')
                calls += count
                episodes += proof['actual_manipulation_episodes']
                searches += phase['searches']
                if calls > call_cap:
                    raise ValueError('Cumulative call budget exceeded')
                if index == 0:
                    gate(phase['phase'])
                _durable_write(root / (phase['phase'] + '.accepted.json'), proof)
            if episodes != limit['episodes'] or searches != sum(p['searches'] for p in phases):
                raise ValueError('Final coverage differs')
            status('AGGREGATING_ALL_OUTCOMES', completed_searches=searches,
                   completed_episodes=episodes, calls_settled=calls)
            report()
            status('COMPLETE', completed_searches=searches,
                   completed_episodes=episodes, calls_settled=calls)
        except BaseException as error:
            status('HALTED_NO_RETRY', error=repr(error), completed_searches=searches,
                   completed_episodes=episodes, calls_settled=calls)
            raise

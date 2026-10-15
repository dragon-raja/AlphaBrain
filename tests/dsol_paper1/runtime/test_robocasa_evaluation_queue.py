"""No simulator: verify that full cache construction isolates every initial."""

import json
import sys
import pytest
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import sha, write
from scripts.dsol_paper1.operations import prepare_robocasa_evaluation as cli


def test_queue_uses_isolated_children_and_does_not_release_holds(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    write(plan, dict(implementation_sha256={}, jobs=[{'initial_id': 'one'}, {'initial_id': 'two'}]))
    output = tmp_path / 'cache'
    commands = []

    def fake_run(command, check):
        commands.append(command)
        assert check
        initial = command[command.index('--initial-id') + 1]
        write(output / 'initials' / initial / 'summary.json', dict(initial_id=initial,
              plan_sha256=sha(plan), status='PASS' if initial == 'one' else 'HOLD_CANONICAL_PIXELS',
              seconds=1, files={}))

    monkeypatch.setattr(cli.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['prepare', 'render', '--plan', str(plan),
                                     '--output', str(output), '--execute'])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert len(commands) == 2
    assert all(c[2] == 'render-one' and c[-1] == '--execute' for c in commands)
    receipt = json.loads((output / 'status.json').read_text())
    assert receipt['complete'] and receipt['full_population']
    assert not receipt['eligible_for_formal_release']
    assert not (output / 'worker.lock').exists()
    # A resumed queue verifies matching completed records without more GPU work.
    with pytest.raises(SystemExit):
        cli.main()
    assert len(commands) == 2


def test_queue_keeps_error_in_population_and_continues(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    write(plan, dict(implementation_sha256={}, jobs=[{'initial_id': 'bad'}, {'initial_id': 'good'}]))
    output = tmp_path / 'cache'
    calls = []
    def fake_run(command, check):
        initial = command[command.index('--initial-id') + 1]
        calls.append(initial)
        if initial == 'bad':
            raise RuntimeError('physical/pixel gate failure')
        write(output / 'initials' / initial / 'summary.json', dict(initial_id=initial,
              plan_sha256=sha(plan), status='PASS', seconds=1, files={}))
    monkeypatch.setattr(cli.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['prepare', 'render', '--plan', str(plan),
        '--output', str(output), '--execute', '--continue-on-error'])
    with pytest.raises(SystemExit):
        cli.main()
    record = json.loads((output / 'status.json').read_text())
    assert calls == ['bad', 'good']
    assert record['counts'] == {'ERROR': 1, 'PASS': 1}
    assert record['full_population'] and not record['eligible_for_formal_release']

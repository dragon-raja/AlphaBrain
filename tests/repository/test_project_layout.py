import hashlib
import json
from pathlib import Path
import pytest

from tools.paper1.project import check, init_run


def test_current_ownership_and_private_boundary():
    assert check()['errors'] == []


def test_prepare_records_identity_but_never_starts_or_reopens(tmp_path):
    protocol = tmp_path/'protocol.json'; protocol.write_text('{"steps": 5}\n')
    target = init_run(tmp_path/'artifacts', 'test-run', protocol, 'CPU fixture')
    d = json.loads((target/'run.json').read_text())
    assert d['status'] == 'PREPARED_NOT_RUNNING' and d['automatic_execution'] is False
    assert d['protocol_sha256'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert (target/'protocol/design.json').read_bytes() == protocol.read_bytes()
    with pytest.raises(FileExistsError): init_run(tmp_path/'artifacts', 'test-run', protocol, 'CPU fixture')
    with pytest.raises(ValueError): init_run(tmp_path/'artifacts', '../escape', protocol, 'CPU fixture')


def test_document_move_ledger_preserves_original_bytes():
    root = Path(__file__).resolve().parents[2]
    ledger = json.loads((root/'archive/repository/20261022/project-layout/manifest.json').read_text())
    for row in ledger['records']:
        assert hashlib.sha256((root/row['before']).read_bytes()).hexdigest() == row['before_sha256']
        assert (root/row['new']).exists()
        if row['old'] != row['new']:
            assert not (root/row['old']).exists()
            assert hashlib.sha256((root/row['new']).read_bytes()).hexdigest() == row['after_sha256']

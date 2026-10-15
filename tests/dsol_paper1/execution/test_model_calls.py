import pytest
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.model_calls import ModelCallLedger


def test_failed_dispatch_spends_budget_and_cannot_resume(tmp_path):
    path=tmp_path/'ledger.json'
    write(path,dict(limit=172,inherited_reserved=171,reserved=0,completed=0,calls=[]))
    ledger=ModelCallLedger(path)
    assert ledger.reserve({'cell':'first'})==0
    assert read(path)['reserved']==1
    with pytest.raises(ValueError,match='cap'):ledger.reserve({'cell':'second'})
    with pytest.raises(ValueError,match='resume'):ModelCallLedger(path)
    ledger.complete(0,'hash')
    assert read(path)['completed']==1
    with pytest.raises(ValueError):ledger.complete(0,'again')

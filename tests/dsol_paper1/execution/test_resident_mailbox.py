import json
import pytest
from AlphaBrain.research.dsol.execution.resident_mailbox import ResidentMailbox


def setup(tmp_path):
    m=ResidentMailbox(tmp_path/'mail');m.root.mkdir()
    q=tmp_path/'request.json';q.write_text('{}')
    receipt=tmp_path/'receipt.json';receipt.write_text('{"status":"COMPLETE"}')
    return m,q,receipt


def test_two_versions_then_explicit_stop(tmp_path):
    m,q,r=setup(tmp_path)
    for v in range(2):
        m.publish(v,q);request=m.next(v,timeout=1)
        m.complete(v,request['sha256'],r)
        assert m.result(v)['version']==v
    m.close(2);assert m.next(2,timeout=1) is None
    with pytest.raises(ValueError,match='closed'):m.publish(2,q)


def test_no_skips_replays_or_tampering(tmp_path):
    m,q,r=setup(tmp_path)
    with pytest.raises(ValueError,match='Previous'):m.publish(1,q)
    m.publish(0,q)
    with pytest.raises(ValueError,match='already'):m.publish(0,q)
    request=m.next(0,timeout=1);m.complete(0,request['sha256'],r)
    with pytest.raises(ValueError,match='replayed'):m.next(0,timeout=1)
    with pytest.raises(ValueError,match='Duplicate'):m.complete(0,request['sha256'],r)
    r.write_text('changed')
    with pytest.raises(ValueError,match='changed'):m.result(0)


def test_lost_coordinator_and_stale_request_fail_closed(tmp_path):
    m,q,r=setup(tmp_path)
    with pytest.raises(RuntimeError,match='alive'):m.next(0,timeout=1,alive=lambda:False)
    m.publish(0,q);q.write_text('changed')
    with pytest.raises(ValueError,match='hash'):m.next(0,timeout=1)


def test_wait_is_bounded_and_cannot_stop_inflight(tmp_path):
    m,q,r=setup(tmp_path)
    with pytest.raises(TimeoutError):m.next(0,timeout=.01)
    m.publish(0,q)
    with pytest.raises(ValueError,match='completes'):m.close(1)

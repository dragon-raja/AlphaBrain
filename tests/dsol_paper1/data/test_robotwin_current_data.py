from benchmarks.RoboTwin.current_data import audit_clean_task


def test_empty_official_current_dataset_is_not_complete(tmp_path):
    result = audit_clean_task(tmp_path)
    assert result['status'] == 'INCOMPLETE'
    assert result['episodes'] == 0
    assert len(result['errors']) == 2

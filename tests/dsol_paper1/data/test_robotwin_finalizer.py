def test_finalizer_is_dry_without_execute(tmp_path, monkeypatch, capsys):
    from scripts.dsol_paper1.operations.finalize_robotwin_official import main
    monkeypatch.setattr('sys.argv', ['finalizer', '--data-run', 'absent', '--checkpoint-run', 'absent',
                                   '--source-report', 'absent', '--output', str(tmp_path/'out'),
                                   '--record', str(tmp_path/'record')])
    main()
    assert not (tmp_path/'out').exists()
    assert not (tmp_path/'record').exists()
    assert '"execute": false' in capsys.readouterr().out

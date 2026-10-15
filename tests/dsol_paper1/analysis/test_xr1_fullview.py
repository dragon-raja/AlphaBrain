"""Synthetic matrix tests; no simulator, GPU or private research inputs."""
from pathlib import Path

import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis import xr1_fullview as module


def test_assembly_all97_and_camera_specific_hashes(tmp_path, monkeypatch):
    initials = [f'i{x}' for x in range(40)]
    views = ['canonical', *[f'v{x}' for x in range(96)]]
    panel = dict(initials=initials, views=views, models=['official', 'canonical', 'broad'],
                 y=np.zeros((3, 40, 97, 4)), accel=np.ones((3, 40, 97)),
                 tasks=np.array(['t'+str(i//4) for i in range(40)]),
                 splits=np.array(['pretrain', 'target']*20), extra=np.arange(65, 97))
    monkeypatch.setattr(module, 'load_panel', lambda _: panel)
    partitions = {'a': views[:1], 'b': views[1:5], 'c': views[5:]}

    def fake_read(path):
        parts = Path(path).parts
        phase, filename = parts[0], parts[-1]
        if filename == 'plan.json':
            return dict(initial_ids=initials, checkpoint_receipt_sha256='weight', panel_sha256='panel',
                        noise_version='v1', native_replan_steps=16, native_flow_steps=5, native_observation_history=4)
        if filename == 'launch.json':
            return {'workers': [{'output': phase+'/worker'}]}
        if filename == 'status.json':
            return {'status': 'COMPLETE'}
        if filename == 'identity.json':
            return {'initials': initials, 'views': partitions[phase]}
        if filename == 'scores.json':
            return {'rows': [{'accel_p3': .1}] * 8}
        return dict(initial_physical_sha256=parts[-2], horizon=100, success=True,
                    plans=[{'accel_p3': .1}] * 5)

    monkeypatch.setattr(module, 'read', fake_read)
    monkeypatch.setattr(module, 'sha', lambda _: 'sha')
    monkeypatch.setattr(module, 'model_statistics', lambda *args: {})
    monkeypatch.setattr(module, 'interval', lambda *args: {})
    from AlphaBrain.research.dsol.plotting import view_landscape
    monkeypatch.setattr(view_landscape, 'plot', lambda *args: None)
    output = tmp_path / 'out'
    result = module.assemble('panel', ['a', 'b', 'c'], output)
    assert result['splits']['all']['xr1']['empirical_hierarchy']['canonical'] == 1
    with np.load(output / 'panel_arrays.npz') as data:
        assert data['success'].shape == (4, 40, 97, 4)
        assert data['success'][-1].sum() == 15520
        assert data['score_repeats'].shape == (40, 97, 8)
    partitions['c'] = views[4:]
    with pytest.raises(ValueError, match='Duplicate'):
        module.assemble('panel', ['a', 'b', 'c'], tmp_path / 'bad')


def test_seen_fixed_choices_do_not_use_heldout_labels():
    q=np.array([[.5,.9,.1],[.8,.2,.7],[.1,.2,.9],[.2,.9,.1]])
    tasks=np.array(['a','b','a','b'])
    splits=np.array(['pretrain','pretrain','target','target'])
    first=module.seen_fixed_choices(q,tasks,splits)
    changed=q.copy();changed[2:]=1-q[2:]
    second=module.seen_fixed_choices(changed,tasks,splits)
    assert first['global_view']==second['global_view']
    assert first['task_views']==second['task_views']=={'a':1,'b':0}
    np.testing.assert_array_equal(first['per_initial_task_choices'],second['per_initial_task_choices'])


def test_seen_fixed_choices_reject_unknown_target_task():
    with pytest.raises(ValueError,match='known tasks'):
        module.seen_fixed_choices(np.zeros((2,3)),np.array(['a','b']),np.array(['pretrain','target']))


def test_plot_completion_preserves_input_bytes(tmp_path,monkeypatch):
    models=['official','canonical','broad']
    initials=[f'i{i}' for i in range(40)];views=['canonical',*[f'v{i}' for i in range(96)]]
    panel=dict(models=models,initials=initials,views=views,y=np.zeros((3,40,97,4)),accel=np.ones((3,40,97)),
               tasks=np.array(['a']*40),splits=np.array(['pretrain']*20+['target']*20),
               coordinates=np.zeros((96,3)),extra=np.arange(65,97))
    monkeypatch.setattr(module,'load_panel',lambda _:panel.copy())
    root=tmp_path/'input';root.mkdir()
    from AlphaBrain.research.dsol.data.artifacts import write,sha
    write(root/'summary.json',dict(sources={}))
    np.savez(root/'panel_arrays.npz',models=np.array([*models,'xr1']),initials=np.array(initials),views=np.array(views),
             success=np.zeros((4,40,97,4)),accel=np.ones((4,40,97)))
    before={p.name:sha(p) for p in root.iterdir()}
    from AlphaBrain.research.dsol.plotting import view_landscape
    def fake_plot(_panel,_result,out):
        for name in ['landscape_all','landscape_pretrain','landscape_target','regional_and_task_diagnostic']:
            (out/(name+'.png')).write_bytes(b'fake-figure')
    monkeypatch.setattr(view_landscape,'plot',fake_plot)
    module.render_completed(root,'unused',tmp_path/'plots')
    assert before=={p.name:sha(p) for p in root.iterdir()}
    assert (tmp_path/'plots/completion.json').is_file()

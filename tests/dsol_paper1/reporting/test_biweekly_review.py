import pytest

from reports.paper1.biweekly_review import content, window_panel_legend, comparison_panels


def inputs():
    cfg=dict(author='测试',student_id='1',report_date='2026-10-15',evidence_cutoff='2026-10-13')
    gain=dict(mean=0,ci95=[-.1,.1])
    summary={'robocasa_original':{'models':{m:{'populations':{'all':{'methods':{'window3':{'gain_vs_stay':gain}}}}}
               for m in ('official','xr1')}}}
    cell={'all':dict(counts=dict(improved=1,degraded=1,equal=2),delta=0)}
    transition=dict(baseline_matching={m:dict(custom4=.2,plus4=.4) for m in ('canonical','broad')},
        panels={m:{'include':{p:cell for p in ('W1_W3','W3_W5')}}
                for m in ('libero97_canonical','robocasa_official','robocasa_xr1')})
    audit=dict(new_rollouts=0,models={'official':{'execution_trials':22416},'xr1':{'execution_trials':21960}})
    return cfg,summary,transition,audit


def test_report_displays_report_date_without_preparation_notes():
    pages=content(*inputs())
    text='\n'.join(b['text'] for page in pages for b in page)
    assert len(pages)==3
    assert '汇报日期：2026年10月15日' in text
    assert '提前' not in text and '截至' not in text and '10月13' not in text


def test_new_model_comparison_and_conditional_window_benefit():
    pages=content(*inputs())
    text='\n'.join(b['text'] for page in pages for b in page)
    assert '新增XR-1模型对照' in text
    assert '多步诊断有作用，但更长不一定更好' in text
    assert '不能算作另一份独立验证' in text


def test_counts_use_audited_execution_trials_not_cpu_audit_new_rollouts():
    pages=content(*inputs())
    tables=[b for p in pages for b in p if b['kind']=='table']
    assert '44,376' in tables[0]['rows'][0][1]


def test_benchmark_explanation_is_a_hypothesis_not_a_causal_conclusion():
    pages=content(*inputs())
    text='\n'.join(b['text'] for page in pages for b in page)
    assert '任务能力、视角敏感性及相机设置' in text
    assert '尚不能认定差异由benchmark本身造成' in text
    assert '不是视角恢复的上限' in text
    assert '待核实：任务与评测设置的影响' in text
    assert '待验证：这种误选是否' in text
    assert 'RoboCasa相机仍随底盘运动' in text


def test_future_evidence_rejected():
    args=list(inputs());args[0]['evidence_cutoff']='2026-10-16'
    with pytest.raises(ValueError):content(*args)


def test_window_panel_legend_explains_all_four_lines():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from AlphaBrain.research.dsol.plotting.window_comparison import curves
    fig, ax = plt.subplots()
    try:
        curves(ax, [.2, .9, .6, .7, .8, .7, .7], [.2, .9, .5, .6, .7, .6, .6])
        legend = window_panel_legend(ax)
        assert [t.get_text() for t in legend.get_texts()] == [
            '允许选择规范位', '排除规范位', '扰动不调整 20.00%', '规范视角 90.00%']
    finally:
        plt.close(fig)


def test_three_rows_use_distinct_sources_and_matched_bridge_repeats():
    summary = {'panels': {}, 'bridge': {}}
    for key in ('libero_canonical','libero_broad','plus_canonical','plus_broad'):
        summary['panels'][key] = {'methods': {m: {'scopes': {'all': {'mean': [key,m]}}}
                                             for m in ('include','exclude')}}
    for model in ('canonical','broad'):
        summary['bridge'][model] = {m: {'execution_repeats': 4, 'mean': ['bridge',model,m]}
                                     for m in ('include','exclude')}
    panels = comparison_panels(summary)
    assert len(panels) == 6
    assert panels[0]['include'] == ['libero_canonical','include']
    assert panels[2]['include'] == ['plus_canonical','include']
    assert panels[4]['include'] == ['bridge','canonical','include']
    assert panels[5]['exclude'] == ['bridge','broad','exclude']
    assert '复用4次执行' in panels[4]['title']
    summary['bridge']['canonical']['include']['execution_repeats'] = 32
    with pytest.raises(ValueError):
        comparison_panels(summary)

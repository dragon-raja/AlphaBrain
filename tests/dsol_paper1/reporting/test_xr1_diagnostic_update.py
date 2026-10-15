"""CPU publication/version and figure-page contracts; no model evaluation."""
from pathlib import Path

import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from reports.paper1.xr1_diagnostic_update import XR1Brief, publish, PdfPages, fitz, percentage


def test_percentage_rounding_matches_user_facing_values():
    assert percentage(.48125)=='48.13'
    assert percentage(.88125)=='88.13'
    assert percentage(.66875)=='66.88'


def test_publication_uses_explicit_page_count_and_added_evidence(tmp_path):
    public, output = tmp_path/'public', tmp_path/'build'
    public.mkdir();output.mkdir();previous=output/'previous-publication';previous.mkdir()
    old=dict(original_pdf=dict(path='historical',sha256='immutable'))
    write(public/'publication.json',old)
    write(previous/'publication.json',old)
    for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md'):
        (public/name).write_bytes(b'old')
        (output/name).write_bytes(b'new')
    before={p.name:sha(p) for p in public.iterdir()}
    artifacts={name:sha(output/name) for name in ('vla_view_research_progress_zh.pdf','vla_view_research_progress_zh.md')}
    write(output/'provenance.json',dict(before=before,artifacts=artifacts,pages=19,
          added_closed_loop_results=15520,scope='XR-1 matched diagnostic',layout_pass=True))
    publish(output,public)
    record=read(public/'publication.json')
    assert record['current']['pages']==19
    assert record['current']['added_closed_loop_results']==15520
    assert record['original_pdf']==old['original_pdf']


def test_new_pages_are_cjk_and_keep_calibration_exploratory(tmp_path):
    if not Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').exists():
        pytest.skip('Optional report font unavailable')
    row=dict(reference=.66875,stay=.53515625,selected=.48125,
             empirical_hierarchy=dict(initial_oracle=.88125),quartiles=[.524,.535,.501,.405],
             association=dict(spatial_mean_spearman=.835,pooled_within_initial_rank_correlation=.17))
    summary=dict(splits={'all':{'official':row,'xr1':row}},
                 seen_rules={'models':{'xr1':{'stay':.5421875,'global_fixed':.675}}})
    coordinates=np.array([[a,b,r] for a in range(4) for b in range(4) for r in (.8,.9,1,1.1,1.2,1.3)])
    panel=dict(models=['official','xr1'],coordinates=coordinates,
               accel=np.ones((2,40,97)),y=np.ones((2,40,97,4)))
    brief=XR1Brief(summary,panel)
    with PdfPages(tmp_path/'pages.pdf') as pdf:
        brief.stronger_model(pdf);brief.implications(pdf)
    assert len(brief.notes)==2
    assert all(not row['issues'] for row in brief.qa)
    assert '初态 Accel' in brief.notes[1]['title']
    with fitz.open(tmp_path/'pages.pdf') as document:
        assert len(document)==2
        assert any(f[2]=='Type3' for f in document[0].get_fonts())

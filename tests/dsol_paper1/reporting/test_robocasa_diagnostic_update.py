"""Publication guards and explicit population labels for diagnostic updates."""
import json

import pytest

from reports.paper1.robocasa_diagnostic_update import DiagnosticBrief, evidence_rows, publish, fitz, PdfPages


def test_population_and_diagnostic_are_not_interchanged():
    def groups(models, n):
        return {s: {m: {'initials': n, 'gain': {'mean': n/10000}}
                    for m in models} for s in ('pretrain', 'target')}
    summary = {'splits': groups(('official', 'canonical', 'broad'), 20),
               'population_sensitivity': {'splits': groups(('canonical', 'broad'), 200)}}
    small, full = evidence_rows(summary), evidence_rows(summary, population=True)
    assert len(small) == 6 and len(full) == 4
    assert all(r['initials'] == 20 for _, _, r in small)
    assert all(r['initials'] == 200 for _, _, r in full)
    assert not any(model == 'official' for _, model, _ in full)


def test_publication_refuses_concurrent_change(tmp_path):
    output, public = tmp_path/'build', tmp_path/'current'
    output.mkdir(); public.mkdir()
    name = 'vla_view_research_progress_zh.pdf'
    (public/name).write_bytes(b'newer publication')
    (output/'provenance.json').write_text(json.dumps({'before': {name: 'old-hash'}}))
    with pytest.raises(ValueError, match='Current publication changed'):
        publish(output, public)
    assert (public/name).read_bytes() == b'newer publication'


def test_new_pages_embed_correct_cjk_outline_format(tmp_path):
    from pathlib import Path
    if not Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').exists():
        pytest.skip('Optional report CJK font absent')
    brief = DiagnosticBrief({}, {})
    target = tmp_path/'report.pdf'
    with PdfPages(target) as pdf:
        brief.conclusions(pdf)
    with fitz.open(target) as doc:
        fonts = [f for f in doc[0].get_fonts() if f[2] == 'Type3']
        assert fonts
        for font in fonts:
            assert doc.xref_get_key(font[0], 'CharProcs')[0] in ('xref', 'dict')
    assert '阶段结论' in brief.notes[0]['title']
    assert not any(p['issues'] for p in brief.qa)

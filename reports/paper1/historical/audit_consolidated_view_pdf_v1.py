#!/usr/bin/env python3
"""Render and inspect the actual published PDF without touching experiment data."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

from pathlib import Path
import json
import hashlib

import pymupdf as fitz

ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/consolidated-analysis-v1')
def main():
    latest=json.loads((ROOT/'latest.json').read_text())
    archive=Path(latest['archive']); pdf=archive/'consolidated_view_analysis.pdf'; out=archive/'pdf-qa'; out.mkdir(exist_ok=True)
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==latest['sha256']
    titles=['候选视角空间','完整状态','几何距离','同一状态的真实相机位置','指标筛选收益','指标可靠性','候选空间收益与搜索兑现','阶段结论与训练覆盖对照']
    rows=[]
    with fitz.open(pdf) as document:
        assert len(document)==8
        for i,page in enumerate(document):
            text=page.get_text()
            assert titles[i] in text,(i+1,'missing extracted Chinese title')
            assert '\ufffd' not in text,(i+1,'replacement character in PDF text')
            image=out/f'page-{i+1:02d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(image)
            rows.append({'page':i+1,'title_present':True,'extracted_characters':len(text),'width':page.rect.width,'height':page.rect.height,'render':str(image)})
    receipt={'status':'PASS_EIGHT_PAGE_PDF_STRUCTURE_AND_TEXT','pdf':str(pdf),'sha256':latest['sha256'],'pages':rows,
             'visual_review':'Rendered pages require human/model visual inspection; structure pass alone does not certify layout.'}
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'pages':len(rows),'output':str(out),'bytes':pdf.stat().st_size}))


if __name__ == "__main__":
    main()

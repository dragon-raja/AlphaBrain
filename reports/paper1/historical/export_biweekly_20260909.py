"""Export font-independent, lossless 220-dpi reading copy and editable DOCX."""


def main():
    from pathlib import Path
    import sys, shutil, json, hashlib
    import pymupdf as fitz

    root=Path(__file__).resolve().parents[3] / 'docs/dsol_paper1/reports/biweekly/20260909'
    source=root/'科研进展双周报_20260827-20260909.pdf'
    src=fitz.open(source)
    assert len(src)==2 and not src.is_repaired and not src.is_encrypted
    out=fitz.open()
    render_hashes=[]
    for pg in src:
        pm=pg.get_pixmap(dpi=220,alpha=False)
        png=pm.tobytes('png')
        render_hashes.append(hashlib.sha256(pm.samples).hexdigest())
        dest=out.new_page(width=pg.rect.width,height=pg.rect.height)
        dest.insert_image(dest.rect,stream=png)
    out.set_metadata({'title':'科研进展周报 2026.08.27—09.09','author':'龙君昱',
                      'subject':'轻量阅读版；文字编辑请使用配套 Word 文档'})
    target=root/'weekly_report_20260827_20260909.pdf'
    out.save(target,garbage=4,deflate=True)
    check=fitz.open(target)
    assert len(check)==2 and not check.is_repaired
    for i,pg in enumerate(check):
        imgs=pg.get_images()
        assert len(imgs)==1
        pm=fitz.Pixmap(check,imgs[0][0])
        assert hashlib.sha256(pm.samples).hexdigest()==render_hashes[i]
    exports=Path('/mnt/data');exports.mkdir(parents=True,exist_ok=True)
    shutil.copy2(target,exports/target.name)
    shutil.copy2(root/'科研进展双周报_20260827-20260909.docx',exports/'weekly_report_20260827_20260909.docx')
    print(json.dumps({'source_bytes':source.stat().st_size,'download_bytes':target.stat().st_size,
     'pages':len(check),'lossless_render_check':'PASS','pdf':str(exports/target.name),
     'docx':str(exports/'weekly_report_20260827_20260909.docx')},ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded report-only housekeeping. Never touches experiments or frozen code.

Preserve all delivered PDFs and their paths; replace exact duplicate backups by
links to immutable archived PDFs. Pack noncurrent build previews with verified
member hashes before removing their loose copies. A receipt records restoration.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from datetime import datetime, timezone

REPO=Path(__file__).resolve().parents[2]
DOCS=REPO/'docs/dsol_paper1'
ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/unified-research-progress-v1')
CURRENT=DOCS/'reports/current/vla_view_research_progress_zh.pdf'
OLD=DOCS/'vla_view_research_focus_20260907_zh.pdf'
COMPAT=DOCS/'vla_view_landscape_and_metrics_20260908_zh.pdf'


def digest(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def alias(path,target):
    temp=path.with_name(path.name+'.tidy-link')
    if temp.exists() or temp.is_symlink():raise RuntimeError('Stale temporary alias: '+str(temp))
    temp.symlink_to(os.path.relpath(target,path.parent));os.replace(temp,path)


def main():
    with (ROOT/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        latest=json.loads((ROOT/'latest.json').read_text());active=Path(latest['archive'])
        if active.parent!=ROOT/'history':raise RuntimeError('Unexpected archive scope')
        original_digest=latest['pdf_sha256']
        for p in [OLD,COMPAT]:
            if digest(p)!=original_digest:raise RuntimeError('Public report changed: '+str(p))
        report={'utc':datetime.now(timezone.utc).isoformat(),'scope':'generated research-report artifacts only',
                'current_pdf_sha256':original_digest,'moves':[],'deduplicated':[],'packed':[],
                'experiment_files_changed':0,'loose_bytes_removed':0}
        CURRENT.parent.mkdir(parents=True,exist_ok=True)
        if not CURRENT.exists():
            OLD.rename(CURRENT);alias(OLD,CURRENT)
            report['moves'].append({'from':str(OLD),'to':str(CURRENT)})
        elif digest(CURRENT)!=original_digest:raise RuntimeError('Different current report exists')
        if not COMPAT.is_symlink():
            report['loose_bytes_removed']+=COMPAT.stat().st_size
            report['deduplicated'].append({'path':str(COMPAT),'target':str(CURRENT),'sha256':original_digest})
        alias(COMPAT,CURRENT)
        old_md=OLD.with_suffix('.md');new_md=CURRENT.with_suffix('.md')
        if not new_md.exists():
            old_md.rename(new_md);alias(old_md,new_md)
            report['moves'].append({'from':str(old_md),'to':str(new_md)})
        archives=sorted(p for p in (ROOT/'history').iterdir() if p.is_dir() and re.fullmatch(r'\d{8}T\d{12}Z',p.name))
        # Only immutable delivered PDFs are targets, never the mutable public copy.
        canonical={}
        for directory in archives:
            p=directory/'vla_research_progress.pdf'
            if p.is_file():canonical.setdefault(digest(p),p)
        for directory in archives:
            for p in sorted(directory.glob('previous-*.pdf')):
                if p.is_symlink():continue
                h=digest(p);target=canonical.get(h)
                if target is None:canonical[h]=p;continue
                size=p.stat().st_size;alias(p,target)
                if digest(p)!=h:raise RuntimeError('Duplicate conversion changed bytes')
                report['deduplicated'].append({'path':str(p),'target':str(target),'sha256':h})
                report['loose_bytes_removed']+=size
            if directory==active:continue
            qa=directory/'pdf-qa';files=[]
            if qa.exists():
                for p in qa.iterdir():
                    if p.is_symlink() or not p.is_file():raise RuntimeError('Unexpected preview member')
                    if not (re.fullmatch(r'page-\d{2}\.png',p.name) or p.name in ['contact.jpg','receipt.json']):
                        raise RuntimeError('Unknown preview file: '+str(p))
                    files.append(p)
            panel=directory/'research-panels.pdf'
            if panel.is_file() and not panel.is_symlink():files.append(panel)
            if not files:continue
            target=directory/'build-previews.tar.gz'
            if target.exists():raise RuntimeError('Archive already exists with loose files')
            hashes={str(p.relative_to(directory)):digest(p) for p in files}
            before=sum(p.stat().st_size for p in files)
            with tarfile.open(target,'x:gz',compresslevel=6) as tar:
                for p in sorted(files):tar.add(p,arcname=str(p.relative_to(directory)),recursive=False)
            with tarfile.open(target,'r:gz') as tar:
                if set(tar.getnames())!=set(hashes):raise RuntimeError('Packed member set differs')
                for name,h in hashes.items():
                    with tar.extractfile(name) as stream:
                        if hashlib.file_digest(stream,'sha256').hexdigest()!=h:raise RuntimeError('Packed bytes differ')
            for p in files:
                if digest(p)!=hashes[str(p.relative_to(directory))]:raise RuntimeError('Preview changed during packing')
            for p in files:p.unlink()  # Exact, validated, archived generated files only.
            if qa.exists():qa.rmdir()  # Must be empty; never recursive deletion.
            report['packed'].append({'archive':str(target),'members_sha256':hashes,'archive_sha256':digest(target),
                                     'restore_directory':str(directory)})
            report['loose_bytes_removed']+=before-target.stat().st_size
        for p in [OLD,COMPAT,CURRENT,active/'vla_research_progress.pdf']:
            if digest(p)!=original_digest:raise RuntimeError('Final PDF identity changed')
        latest['public']=str(CURRENT);latest['legacy_public']=str(OLD)
        temp=ROOT/'latest.tidy.json';temp.write_text(json.dumps(latest,ensure_ascii=False,indent=2)+'\n');os.replace(temp,ROOT/'latest.json')
        destination=DOCS/'reports/cleanup-receipt-20260908.json'
        if destination.exists():raise RuntimeError('Housekeeping receipt already exists')
        report['status']='PASS_REPORT_ONLY_HOUSEKEEPING'
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'deduplicated_files':len(report['deduplicated']),
                          'packed_builds':len(report['packed']),'net_bytes_removed':report['loose_bytes_removed'],
                          'receipt':str(destination)},ensure_ascii=False))


if __name__=='__main__':main()

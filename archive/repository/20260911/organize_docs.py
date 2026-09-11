"""One-shot document relocation with link-only prose edits and an audit chain."""
from pathlib import Path
import importlib.util
import os
import re
from urllib.parse import unquote

SPEC = importlib.util.spec_from_file_location("maintenance_transaction", Path(__file__).with_name("migrate.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
ROOT, Transaction = module.ROOT, module.Transaction
BASE = ROOT / "docs/dsol_paper1"


def destination(p):
    n = p.name
    if n in {"README.md", "HISTORY.md", "vla_view_research_focus_20260907_zh.pdf", "vla_view_landscape_and_metrics_20260908_zh.pdf"}:
        return p
    if n.endswith("_previews") or p.suffix == ".pdf":
        return BASE / "reports/archive/imported" / n
    if n == "report_sources_20260907":
        return BASE / "reports/sources/20260907"
    if p.is_dir():
        return p
    if p.suffix == ".json":
        return BASE / ("audits" if "audit" in n else "reports/sources") / n
    if p.suffix != ".md":
        return p
    if any(k in n for k in ("audit", "root_cause", "hygiene", "argument_review", "historical_validity")):
        group = "audits"
    elif any(k in n for k in ("master_plan", "convergence_plan", "research_synthesis", "next_stage")):
        group = "planning"
    elif any(k in n for k in ("protocol", "core_v1")):
        group = "protocols"
    elif any(k in n for k in ("execution", "status", "takeover", "parallel", "program_progress")):
        group = "execution"
    elif n.startswith("vla_view_"):
        group = "reports/notes"
    else:
        group = "results"
    return BASE / group / n


def main():
    tx = Transaction("docs")
    mapping = {}
    directory_map = {}
    for old in list(BASE.iterdir()):
        new = destination(old)
        if old == new:
            continue
        if old.is_dir():
            directory_map[old] = new
            for p in old.rglob("*"):
                if p.is_file():
                    mapping[p] = new / p.relative_to(old)
        else:
            mapping[old] = new
    all_map = {**mapping, **directory_map}
    originals = {p: p.read_text() for p in (ROOT / "docs").rglob("*.md")}
    originals[ROOT / "README.md"] = (ROOT / "README.md").read_text()
    link = re.compile(r"(\]\()([^\n)]*)(\))")

    def fix_links(text, old, new):
        def replace(match):
            value = match[2]
            if not value or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or value.startswith("#"):
                return match[0]
            angle = value.startswith("<") and ">" in value
            token = value[1:value.index(">")] if angle else value.split(" ", 1)[0]
            tail = value[value.index(">") + 1:] if angle else value[len(token):]
            bare, sep, anchor = token.partition("#")
            target = Path(os.path.normpath(str(old.parent / unquote(bare))))
            target = all_map.get(target, target)
            relative = os.path.relpath(target, new.parent)
            updated = relative + (sep + anchor if sep else "")
            return match[1] + ("<" + updated + ">" if angle else updated) + tail + match[3]
        return link.sub(replace, text)

    for old, new in sorted(mapping.items()):
        tx.save(old, new, fix_links(originals[old], old, new) if old in originals else None)
    for old, text in originals.items():
        if old in mapping:
            continue
        new = fix_links(text, old, old)
        if new != text:
            tx.save(old, old, new)
    # Remove only now-empty, explicitly identified former preview/source dirs.
    for directory in directory_map:
        for p in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        if not any(directory.iterdir()):
            directory.rmdir()
    # Update local readers/builders; do not touch frozen source identities or archives.
    import json
    protected = {ROOT / r['path'] for r in json.loads((ROOT / 'archive/paper1/maintenance/script-layout-20260911/manifest.json').read_text())['protected']}
    for area in ("scripts", "reports", "tests", "docs/dsol_paper1/reports/biweekly"):
        for p in (ROOT / area).rglob("*"):
            if not p.is_file() or p.suffix not in {".py", ".sh"} or p in protected:
                continue
            text = p.read_text()
            updated = text
            for old, new in mapping.items():
                updated = updated.replace(str(old.relative_to(ROOT)), str(new.relative_to(ROOT)))
            if updated != text:
                tx.save(p, p, updated)
    tx.finish(scope="Human documents classified; scientific prose unchanged except relative Markdown links; report/source payload bytes preserved",
              compatibility_pdf_paths_retained=2)


if __name__ == "__main__":
    main()

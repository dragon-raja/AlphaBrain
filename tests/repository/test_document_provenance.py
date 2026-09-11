import hashlib
import json
from pathlib import Path
import re

from tools.repository.evidence import current_path

ROOT = Path(__file__).resolve().parents[2]


def test_classification_preserves_prose_and_binary_payloads():
    m = json.loads((ROOT / "archive/repository/20260911/docs/manifest.json").read_text())
    links = re.compile(r"\]\([^\n)]*\)")
    for row in m["records"]:
        before = ROOT / row["before"]
        if before.exists():
            assert hashlib.sha256(before.read_bytes()).hexdigest() == row["before_sha256"]
        if row["old"] == row["new"]:
            continue  # Mutable indices/readers have their own later revisions.
        after = current_path(ROOT / row["new"])
        if row["old"].endswith(".md"):
            assert links.sub("](LINK)", before.read_text()) == links.sub("](LINK)", after.read_text())
        else:
            # This one pre-existing local-only PDF was never part of Git.
            local_only = "docs/dsol_paper1/reports/archive/imported/vla_view_research_unified_20260907_zh.pdf"
            if not after.exists() and row["new"] == local_only:
                continue
            assert hashlib.sha256(after.read_bytes()).hexdigest() == row["before_sha256"]
            if before.exists():
                assert before.read_bytes() == after.read_bytes()


def test_published_pdf_was_not_regenerated():
    p = ROOT / "docs/dsol_paper1/reports/current/vla_view_research_progress_zh.pdf"
    if not p.exists():
        import pytest
        pytest.skip("Published PDF is an external/local artifact")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "dbd9c2b6906fee9a120734291dda9dc4b56a16a9194e1852074fb796bfdb95a7"


def test_classified_document_links_resolve_except_declared_local_pdf_outputs():
    import os
    from urllib.parse import unquote
    local_only = {
        "docs/dsol_paper1/reports/current/vla_view_research_progress_zh.pdf",
        "docs/dsol_paper1/vla_view_landscape_and_metrics_20260908_zh.pdf",
        "docs/dsol_paper1/vla_view_research_focus_20260907_zh.pdf",
        "docs/dsol_paper1/reports/archive/imported/vla_view_research_unified_20260907_zh.pdf",
    }
    for p in (ROOT / "docs/dsol_paper1").rglob("*.md"):
        if p.is_symlink():
            assert p.resolve().is_file()
            continue
        for value in re.findall(r"\]\(([^\n)]*)\)", p.read_text()):
            if not value or value.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
                continue
            token = value[1:value.index(">")] if value.startswith("<") and ">" in value else value.split(" ", 1)[0]
            target = Path(os.path.normpath(str(p.parent / unquote(token.split("#")[0]))))
            if target.is_relative_to(ROOT) and str(target.relative_to(ROOT)) not in local_only:
                assert target.exists(), (p, token)

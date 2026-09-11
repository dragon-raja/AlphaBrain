import hashlib
import json
from pathlib import Path

from tools.repository.check import check

ROOT = Path(__file__).resolve().parents[2]


def test_repository_source_ownership_and_dependency_directions():
    result = check()
    assert result["errors"] == []


def test_cabi_archive_bytes_and_retained_consumers():
    m = json.loads((ROOT / "archive/repository/20260911/cabi/manifest.json").read_text())
    assert len(m["records"]) == 193
    for row in m["records"]:
        assert not (ROOT / row["old"]).exists()
        assert hashlib.sha256((ROOT / row["new"]).read_bytes()).hexdigest() == row["before_sha256"]
    for row in m["protected"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_document_root_contains_only_navigation_and_live_compatibility_links():
    files = {p.name for p in (ROOT / "docs/dsol_paper1").iterdir() if p.is_file()}
    assert {"README.md", "HISTORY.md"} <= files
    assert files <= {"README.md", "HISTORY.md", "vla_view_landscape_and_metrics_20260908_zh.pdf", "vla_view_research_focus_20260907_zh.pdf"}
    assert not list((ROOT / "docs").rglob("*.py"))


def test_shared_modules_do_not_require_outer_layers():
    from AlphaBrain.common import images, pair_records
    from deployment.model_server.tools import image_tools
    from scripts.dsol_paper1.training import libero_pair_records
    assert image_tools is images
    assert libero_pair_records is pair_records


def test_relocated_source_and_document_paths_are_not_accidentally_ignored():
    import subprocess
    for name in ("core", "cabi", "docs", "reports", "verify-tests"):
        m = json.loads((ROOT / f"archive/repository/20260911/{name}/manifest.json").read_text())
        names = [r["new"] for r in m["records"] if Path(r["new"]).suffix in {".py", ".sh", ".md"}]
        result = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(names) + "\n", cwd=ROOT, text=True, capture_output=True)
        assert not result.stdout, result.stdout
